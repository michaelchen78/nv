import numpy as np
import pycce as pc
from ase.build import bulk

from config import SPIN_TYPES, D, E, INTERACTION_DEFAULTS, DIST_TOLERANCE, JT_P1_INTERNAL_HF, JT_P1_QUADRUPOLE, P, \
    NV_HYPERFINE
from sim import logger, is_root
from mpi4py import MPI  # for debugging
from itertools import combinations, product
import warnings
from pathlib import Path
import logging

# for custom polarization
import pycce.run.mc as pycce_mc
from pycce.utilities import gen_state_list


# +++++++++++++++++++++++ small utility helpers +++++++++++++++++++++++ #
def current_ensemble():
    """
    read ensemble index off the shared logger object
    :return: ensemble index, an attribute of the logger in sim.py
    """
    return getattr(logger, "current_ensemble", None)


def _edge_tuple(i, j):
    return tuple(sorted((int(i), int(j))))


def canonical_spin_pair(a, b):
    """
    Return an unordered spin-name pair.
    """
    return tuple(sorted((str(a), str(b))))


def _has_edge(edge_set, i, j):
    return _edge_tuple(i, j) in edge_set


def _as_cluster_array(clusters, cluster_order):
    """
    Normalize a PyCCE cluster array to shape (n_clusters, cluster_order).
    """
    cluster_order = int(cluster_order)
    arr = np.asarray(clusters, dtype=int)

    if arr.size == 0:
        return np.empty((0, cluster_order), dtype=int)

    if arr.ndim == 1:
        if arr.size % cluster_order != 0:
            raise ValueError(
                f"Cluster array of size {arr.size} cannot be reshaped into order {cluster_order}"
            )
        arr = arr.reshape(-1, cluster_order)

    if arr.ndim != 2 or arr.shape[1] != cluster_order:
        raise ValueError(
            f"Expected order-{cluster_order} clusters to have shape (n, {cluster_order}), "
            f"got {arr.shape}"
        )

    return arr


def _pair_desc(pair, names):
    i, j = map(int, pair)
    return f"({i}={names[i]}, {j}={names[j]})"


def _file_only_logger(filename, mode="a"):
    """
    Logger that writes only to <logger.run_dir>/<filename>.
    Does not propagate to the main output log.
    """
    side_logger = logging.getLogger(f"{logger.name}.{filename}")
    side_logger.setLevel(logging.INFO)
    side_logger.propagate = False

    for h in list(side_logger.handlers):
        side_logger.removeHandler(h)
        h.close()

    path = Path(logger.run_dir) / filename
    fh = logging.FileHandler(path, mode=mode, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    side_logger.addHandler(fh)

    return side_logger


def _strict_colocated_en_pair(i, j, names, xyz, atol=DIST_TOLERANCE):
    i = int(i)
    j = int(j)

    pair_names = sorted([str(names[i]), str(names[j])])
    if pair_names != ["14N", "e"]:
        return False

    return bool(np.allclose(xyz[i], xyz[j], atol=atol, rtol=0.0))


def _is_strict_p1_en_pair(group, names):
    """
    True only if this same-location group is exactly one P1 electron
    and one P1 nitrogen.
    """
    if len(group) != 2:
        return False

    pair_names = sorted(str(names[int(i)]) for i in group)
    return pair_names == ["14N", "e"]


def count_same_loc_en_order2(calc):
    clusters2 = _as_cluster_array(calc.clusters[2], 2)
    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)

    count = 0
    for i, j in clusters2:
        pair_names = sorted([str(names[int(i)]), str(names[int(j)])])
        same_loc = np.allclose(xyz[int(i)], xyz[int(j)], atol=1e-8, rtol=0.0)
        if pair_names == ["14N", "e"] and same_loc:
            count += 1
    return count


def _unique_cluster_rows(rows, order):
    if len(rows) == 0:
        return np.empty((0, order), dtype=int)

    arr = np.asarray([sorted(map(int, row)) for row in rows], dtype=int)
    return np.unique(arr, axis=0).reshape(-1, order)


def _set_or_remove_cluster_order(calc, order, rows):
    rows = np.asarray(rows, dtype=int).reshape(-1, order)

    if rows.shape[0] == 0:
        calc.clusters.pop(order, None)
    else:
        calc.clusters[order] = rows


# +++++++++++++++++++++++ custom polarization +++++++++++++++++++++++ #
def install_custom_polarization():
    """
    PyCCE MC polarization patch

    Keeps PyCCE's monte carlo bath state sampling, but changes the MC sampling distribution.

    If pycce_mc.POLARIZATION_GAMMA is None:
        exact default behavior: uniform Iz sampling for all bath spins.

    If pycce_mc.POLARIZATION_GAMMA is a float:
        e and 13C spins are sampled with Gaussian polarization:
            P(+1/2) = 0.5 + 0.5 * exp[-(r/gamma)^2]
            P(-1/2) = 0.5 - 0.5 * exp[-(r/gamma)^2]
        all other spins remain uniformly sampled.
    """
    if getattr(pycce_mc, "_polarized_sampler_installed", False):
        return

    default_generate_bath_state = pycce_mc.generate_bath_state
    pycce_mc.POLARIZATION_GAMMA = None

    def generate_polarized_bath_state(bath, nbstates, seed=None, parallel=False):
        gamma = getattr(pycce_mc, "POLARIZATION_GAMMA", None)

        # original PyCCE behavior when no polarization is active.
        if gamma is None:
            yield from default_generate_bath_state(bath, nbstates, seed=seed, parallel=parallel,)
            return

        rgen = np.random.default_rng(seed)

        rank = 0
        comm = None

        if parallel:
            try:
                import mpi4py
                comm = mpi4py.MPI.COMM_WORLD
                rank = comm.Get_rank()
            except ImportError:
                raise Exception("parallel broken")

        for _ in range(nbstates):
            bath_state = np.empty(bath.shape, dtype=np.float64)
            dimensions = np.empty(bath.shape, dtype=np.int32)

            if rank == 0:
                dist = bath.dist()

                for raw_name in np.unique(bath.N):
                    name = str(raw_name)
                    mask = np.equal(bath.N, raw_name)
                    count = int(np.count_nonzero(mask))

                    spin = float(bath.types[name].s)
                    dim = int(round(2 * spin + 1))
                    dimensions[mask] = dim

                    if name == "e" or name == "13C":
                        # polos = np.exp(-(dist[mask] / float(gamma)) ** 2) * 0.5
                        # p_up = 0.5 + polos
                        # bath_state[mask] = np.where(rgen.random(count) < p_up, spin, -spin,)
                        p_up = float(gamma)
                        if not (0.0 <= p_up <= 1.0):
                            raise Exception(f"polarization must be a probability in [0, 1], got {p_up}")
                        bath_state[mask] = np.where(
                            rgen.random(count) < p_up,
                            spin,
                            -spin,
                        )
                    else:
                        bath_state[mask] = rgen.integers(dim, size=count) - spin

            if parallel:
                comm.Bcast(bath_state, root=0)
                comm.Bcast(dimensions, root=0)

            yield gen_state_list(bath_state, dimensions)

    pycce_mc.generate_bath_state = generate_polarized_bath_state
    pycce_mc._polarized_sampler_installed = True


# +++++++++++++++++++++++ run_experiment() helpers +++++++++++++++++++++++ #
def get_pulses(pulse_id):
    """
    Given some string pulse_id, return the pulse sequence or number of pulses to put into Simulator.compute.
    Right now it is simply 'hahn' is x-axis pi Hahn echo and 'FID' is FID.
    :param pulse_id: the string id
    :return: the pulse sequence or number of pulses
    """

    if pulse_id == "hahn":
        return [pc.Pulse('x', np.pi)]
    elif pulse_id == "FID":
        return 0
    elif isinstance(pulse_id, int):
        return pulse_id
    else:
        raise Exception("pulse_id not recognized")


def _read_imap_tensor_readonly(calc, i, j):
    """
    Read-only interaction tensor lookup.

    Does not call add_interaction and does not assign anything.
    It only indexes calc.bath.imap and copies the returned tensor for logging.
    """
    i = int(i)
    j = int(j)

    imap = getattr(calc.bath, "imap", None)
    if imap is None:
        return None, "calc.bath.imap is None"

    for a, b in ((i, j), (j, i)):
        try:
            return np.array(imap[a, b], dtype=float, copy=True), f"imap[{a}, {b}]"
        except KeyError:
            continue
        except Exception as exc:
            return None, f"error reading imap[{a}, {b}]: {exc!r}"

    return None, "no explicit imap tensor found in either direction"


def log_final_colocated_pair_diagnostics(calc, atol=DIST_TOLERANCE, tensor_atol=1e-8):
    """
    Read-only diagnostic into calc.bath and calc.clusters.
    Logs:
        1. all colocated calc.bath pairs
        2. all colocated order-2 clusters
        3. imap interaction tensors for colocated order-2 clusters

    Checks everything, but only prints detailed entries for the first 20.
    """
    diag_logger = _file_only_logger("log_final_colocated_pair_diagnostics_output.txt")
    max_diagnostic_print = 20

    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)

    '''all colocated calc.bath pairs'''
    groups = _same_location_groups(np.arange(len(names), dtype=int), xyz, atol=atol)

    bath_colocated_pairs = []
    for group in groups:
        if len(group) < 2:
            continue
        for i, j in combinations(group, 2):
            bath_colocated_pairs.append(_edge_tuple(i, j))

    bath_colocated_pairs = sorted(set(bath_colocated_pairs))

    bad_bath_pairs = [
        pair for pair in bath_colocated_pairs
        if not _strict_colocated_en_pair(pair[0], pair[1], names, xyz, atol=atol)
    ]

    diag_logger.info(
        "FINAL calc.bath colocated pairs: count=%d, all_strict_e_14N=%s. "
        "Printing first %d detailed entries.",
        len(bath_colocated_pairs),
        len(bad_bath_pairs) == 0,
        min(max_diagnostic_print, len(bath_colocated_pairs)),
    )

    for i, j in bath_colocated_pairs[:max_diagnostic_print]:
        diag_logger.info(
            "pair: %s, xyz=%s, strict_e_14N=%s",
            _pair_desc((i, j), names),
            np.array2string(xyz[int(i)], precision=12, separator=", "),
            _strict_colocated_en_pair(i, j, names, xyz, atol=atol),
        )

    if len(bath_colocated_pairs) > max_diagnostic_print:
        diag_logger.info(
            "FINAL calc.bath colocated pairs: skipped printing %d additional detailed entries.",
            len(bath_colocated_pairs) - max_diagnostic_print,
        )

    if len(bad_bath_pairs) > 0:
        diag_logger.warning(
            "FINAL calc.bath colocated pairs that are NOT strict e-14N pairs: count=%d, first_%d=%s",
            len(bad_bath_pairs),
            min(max_diagnostic_print, len(bad_bath_pairs)),
            [_pair_desc(pair, names) for pair in bad_bath_pairs[:max_diagnostic_print]],
        )

    '''all colocated order-2 clusters'''
    if hasattr(calc, "clusters") and calc.clusters is not None and 2 in calc.clusters:
        clusters2 = _as_cluster_array(calc.clusters[2], 2)
    else:
        clusters2 = np.empty((0, 2), dtype=int)

    colocated_cluster_rows = []
    for row_num, row in enumerate(clusters2):
        i, j = map(int, row)
        if np.allclose(xyz[i], xyz[j], atol=atol, rtol=0.0):
            colocated_cluster_rows.append((int(row_num), _edge_tuple(i, j)))

    cluster_pair_list = [pair for _, pair in colocated_cluster_rows]
    cluster_pair_set = set(cluster_pair_list)
    bath_pair_set = set(bath_colocated_pairs)

    missing_from_clusters = sorted(bath_pair_set - cluster_pair_set)
    extra_in_clusters = sorted(cluster_pair_set - bath_pair_set)
    duplicate_cluster_rows = len(cluster_pair_list) - len(cluster_pair_set)

    diag_logger.info(
        "FINAL colocated order-2 clusters: rows=%d, unique_pairs=%d, "
        "matches_calc_bath_colocated_pairs=%s, duplicate_rows=%d. "
        "Printing first %d detailed entries.",
        len(colocated_cluster_rows),
        len(cluster_pair_set),
        len(missing_from_clusters) == 0 and len(extra_in_clusters) == 0,
        duplicate_cluster_rows,
        min(max_diagnostic_print, len(colocated_cluster_rows)),
    )

    for row_num, pair in colocated_cluster_rows[:max_diagnostic_print]:
        diag_logger.info(
            "FINAL colocated order-2 cluster row=%d, pair=%s",
            row_num,
            _pair_desc(pair, names),
        )

    if len(colocated_cluster_rows) > max_diagnostic_print:
        diag_logger.info(
            "FINAL colocated order-2 clusters: skipped printing %d additional detailed entries.",
            len(colocated_cluster_rows) - max_diagnostic_print,
        )

    if len(missing_from_clusters) > 0:
        diag_logger.warning(
            "FINAL colocated pairs present in calc.bath but missing from calc.clusters[2]: "
            "count=%d, first_%d=%s",
            len(missing_from_clusters),
            min(max_diagnostic_print, len(missing_from_clusters)),
            [_pair_desc(pair, names) for pair in missing_from_clusters[:max_diagnostic_print]],
        )

    if len(extra_in_clusters) > 0:
        diag_logger.warning(
            "FINAL colocated order-2 clusters not found as colocated calc.bath pairs: "
            "count=%d, first_%d=%s",
            len(extra_in_clusters),
            min(max_diagnostic_print, len(extra_in_clusters)),
            [_pair_desc(pair, names) for pair in extra_in_clusters[:max_diagnostic_print]],
        )

    '''imap interaction tensors for colocated order-2 clusters'''
    tensor_missing = []
    tensor_zero = []
    tensor_nonzero = []

    for row_num, pair in colocated_cluster_rows:
        i, j = pair
        tensor, source = _read_imap_tensor_readonly(calc, i, j)

        if tensor is None:
            tensor_missing.append((row_num, pair, source))
            continue

        if np.allclose(tensor, 0.0, atol=tensor_atol, rtol=0.0):
            tensor_zero.append((row_num, pair, source, tensor))
        else:
            tensor_nonzero.append((row_num, pair, source, tensor))

    diag_logger.info(
        "FINAL interaction tensor check for colocated order-2 clusters: total_checked=%d, "
        "missing_or_error=%d, allclose_zero=%d, nonzero=%d. "
        "Printing first %d tensor details.",
        len(colocated_cluster_rows),
        len(tensor_missing),
        len(tensor_zero),
        len(tensor_nonzero),
        min(max_diagnostic_print, len(colocated_cluster_rows)),
    )

    printed = 0

    for row_num, pair, source in tensor_missing:
        if printed >= max_diagnostic_print:
            break

        diag_logger.warning(
            "FINAL interaction tensor for colocated order-2 cluster row=%d, pair=%s: %s",
            row_num,
            _pair_desc(pair, names),
            source,
        )
        printed += 1

    for row_num, pair, source, tensor in tensor_zero + tensor_nonzero:
        if printed >= max_diagnostic_print:
            break

        diag_logger.info(
            "FINAL interaction tensor for colocated order-2 cluster row=%d, pair=%s, "
            "source=%s, allclose_zero=%s, tensor=\n%s",
            row_num,
            _pair_desc(pair, names),
            source,
            np.allclose(tensor, 0.0, atol=tensor_atol, rtol=0.0),
            np.array2string(tensor, precision=8, suppress_small=False),
        )
        printed += 1

    skipped_tensor_details = len(colocated_cluster_rows) - printed
    if skipped_tensor_details > 0:
        diag_logger.info(
            "FINAL interaction tensor details: skipped printing %d additional tensor entries.",
            skipped_tensor_details,
        )


# +++++++++++++++++++++++ get_simulator() helpers +++++++++++++++++++++++ #
def add_p1_order2(calc, atol=1e-8):
    """
    Add missing order-2 clusters for colocated P1 electron/nitrogen pairs.

    Specifically adds pairs where:
        one spin is "e"
        one spin is "14N"
        xyz positions are equal within atol

    Edits calc.clusters[2] in place.
    """
    if not hasattr(calc, "clusters") or calc.clusters is None:
        raise Exception("calc.clusters does not exist.")

    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)

    if 2 in calc.clusters:
        order2 = _as_cluster_array(calc.clusters[2], 2)
    else:
        order2 = np.empty((0, 2), dtype=int)

    existing = {tuple(sorted(map(int, pair))) for pair in order2}

    e_idx = np.where(np.equal(names, "e"))[0]
    n_idx = np.where(np.equal(names, "14N"))[0]

    to_add = []

    for e in e_idx:
        for n in n_idx:
            if np.allclose(xyz[e], xyz[n], atol=atol, rtol=0.0):
                pair = tuple(sorted((int(e), int(n))))

                if pair not in existing:
                    to_add.append(pair)
                    existing.add(pair)

    if len(to_add) == 0:
        if is_root():
            logger.info("No missing colocated P1 e-14N order-2 clusters found.")
        return calc

    to_add = np.asarray(to_add, dtype=int).reshape(-1, 2)
    calc.clusters[2] = np.vstack([order2, to_add]).astype(int)

    if is_root():
        logger.info(
            "Added %d missing colocated P1 e-14N order-2 clusters. order-2 count: %d -> %d",
            to_add.shape[0],
            order2.shape[0],
            calc.clusters[2].shape[0],
        )

        # validate that the just added colocated e-14N pairs have the manually inputted hyperfines
        # this is read-only: it only indexes imap and copies the tensor for logging
        validation_logger = _file_only_logger("add_p1_order2_validation.txt")
        imap = getattr(calc.bath, "imap", None)
        if imap is None:
            validation_logger.warning(
                "Interaction validation: calc.bath.imap is None. "
                "Cannot verify explicit e-14N tensors for added P1 pairs."
            )
        else:
            max_validation_print = 20
            to_validate_print = to_add[:max_validation_print]

            validation_logger.info(
                "Interaction validation: printing first %d of %d added P1 pairs.",
                min(max_validation_print, to_add.shape[0]),
                to_add.shape[0],
            )

            for e, n in to_validate_print:
                e = int(e)
                n = int(n)

                try:
                    tensor_en = np.array(imap[e, n], dtype=float, copy=True)
                except KeyError:
                    validation_logger.warning(
                        "Interaction validation for added P1 pair (%d=%s, %d=%s): "
                        "NO explicit imap tensor found. PyCCE may use default point-dipole fallback.",
                        e, names[e],
                        n, names[n],
                    )
                    continue
                except Exception as exc:
                    validation_logger.warning(
                        "Interaction validation for added P1 pair (%d=%s, %d=%s): "
                        "could not read/copy imap tensor without error: %r",
                        e, names[e],
                        n, names[n],
                        exc,
                    )
                    continue

                validation_logger.info(
                    "Interaction validation for added P1 pair (%d=%s, %d=%s): "
                    "explicit imap tensor found; allclose_zero=%s; J=\n%s",
                    e, names[e],
                    n, names[n],
                    np.allclose(tensor_en, 0.0, atol=atol, rtol=0.0),
                    tensor_en,
                )

    return calc


def add_park_2e2n_clusters(calc, atol=DIST_TOLERANCE):
    """
    Starting from a final order-2 graph, manually add Park-style order-3
    and order-4 clusters.

    Requires calc.clusters[2] to already include the colocated e-14N P1 edges.

    Adds:
        order 3: one strict colocated (e, 14N) pair + one other spin,
                 connected through the order-2 graph.

        order 4: two strict colocated (e, 14N) pairs,
                 connected through at least one cross-pair order-2 edge.
    """
    if not hasattr(calc, "clusters") or calc.clusters is None:
        raise Exception("calc.clusters does not exist.")

    if 2 not in calc.clusters:
        raise Exception("add_park_2e2n_clusters requires calc.clusters[2].")

    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)

    order2 = _as_cluster_array(calc.clusters[2], 2)
    edge_set = {_edge_tuple(i, j) for i, j in order2}

    # Use order-1 cluster list if present; otherwise fall back to all bath indices.
    if 1 in calc.clusters:
        all_spins = _as_cluster_array(calc.clusters[1], 1).reshape(-1).astype(int)
    else:
        all_spins = np.arange(len(calc.bath), dtype=int)

    # Strict P1 pairs must already be order-2 edges.
    p1_pairs = []
    for i, j in order2:
        if _strict_colocated_en_pair(i, j, names, xyz, atol=atol):
            p1_pairs.append(_edge_tuple(i, j))

    p1_pairs = sorted(set(p1_pairs))

    if len(p1_pairs) == 0:
        raise Exception(
            "No colocated e-14N order-2 P1 pairs found. "
            "Run add_missing_p1_en_order2_clusters(calc) before add_park_2e2n_clusters(calc)."
        )

    # order 3: one P1 pair + one extra spin, connected by at least one edge to the pair.
    order3_rows = []

    for a, b in p1_pairs:
        for x in all_spins:
            x = int(x)

            if x == a or x == b:
                continue

            if _has_edge(edge_set, x, a) or _has_edge(edge_set, x, b):
                order3_rows.append((a, b, x))

    order3 = _unique_cluster_rows(order3_rows, 3)

    # order 4: two P1 pairs, connected by at least one cross edge.
    order4_rows = []

    for p, pair1 in enumerate(p1_pairs):
        a, b = pair1

        for pair2 in p1_pairs[p + 1:]:
            c, d = pair2

            if len({a, b, c, d}) != 4:
                continue

            connected = (
                _has_edge(edge_set, a, c) or
                _has_edge(edge_set, a, d) or
                _has_edge(edge_set, b, c) or
                _has_edge(edge_set, b, d)
            )

            if connected:
                order4_rows.append((a, b, c, d))

    order4 = _unique_cluster_rows(order4_rows, 4)

    old3 = calc.clusters[3].shape[0] if 3 in calc.clusters else 0
    old4 = calc.clusters[4].shape[0] if 4 in calc.clusters else 0

    _set_or_remove_cluster_order(calc, 3, order3)
    _set_or_remove_cluster_order(calc, 4, order4)

    if is_root():
        logger.info(
            "add_park_2e2n_clusters: P1 pairs=%d, order-3 clusters %d -> %d, order-4 clusters %d -> %d",
            len(p1_pairs),
            old3,
            order3.shape[0],
            old4,
            order4.shape[0],
        )

    return calc


def apply_custom_r_bath(atoms, custom_r_bath, central_spin):
    """
    Remove spins from the supercell according to spin type-specific r_bath cutoffs.

    custom_r_bath should be a dictionary like:
        {"13C": 80.0, "14N": 40.0, "e": 40.0}
    """
    center_pos = get_central_position(central_spin)
    xyz = np.asarray(atoms.xyz, dtype=float)
    n_arr = np.asarray(atoms.N, dtype=object)
    dist = np.linalg.norm(xyz - center_pos, axis=1)

    keep = np.ones(len(atoms), dtype=bool)
    removed_counts = {}

    for spin_type, new_r_bath in custom_r_bath.items():
        new_r_bath = float(new_r_bath)
        type_mask = np.equal(n_arr, spin_type)

        if not np.any(type_mask):
            raise ValueError(f"custom_r_bath spin type {spin_type!r} does not appear in atoms.N")

        remove_mask = type_mask & (dist > new_r_bath)
        removed_counts[spin_type] = int(np.count_nonzero(remove_mask))
        keep &= ~remove_mask

    n_removed = int(len(atoms) - np.count_nonzero(keep))

    if is_root():
        logger.info("custom_r_bath removed %d spins from supercell: %s", n_removed, removed_counts)

    return atoms[keep]


def apply_custom_r_dipole(calc, custom_r_dipole, r_dipole):
    """
    Edit calc.clusters in place after PyCCE has generated clusters normally to implement custom r_dipole.
    """
    if not hasattr(calc, "clusters") or calc.clusters is None:
        raise Exception("calc.clusters does not exist. Build Simulator with order and r_dipole first.")

    '''parse custom r_dipole's from yaml'''
    pair_cutoffs = parse_custom_r_dipole(custom_r_dipole=custom_r_dipole, default_r_dipole=r_dipole)
    if len(pair_cutoffs) == 0:
        warnings.warn("custom r_dipole was not None but had no effect")
        return None
    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)
    present_names = {str(x) for x in np.unique(names)}
    requested_names = {name for pair in pair_cutoffs for name in pair}
    missing_names = requested_names - present_names
    if missing_names:
        warnings.warn(
            f"custom_r_dipole mentions spin names not present in calc.bath: {sorted(missing_names)}",
            RuntimeWarning)

    '''implement custom r_dipole'''
    edge_cache = {}

    for cluster_order in list(calc.clusters.keys()):
        order_int = int(cluster_order)
        clusters = _as_cluster_array(calc.clusters[cluster_order], order_int)

        # Always keep 1-spin clusters.
        if order_int <= 1:
            calc.clusters[cluster_order] = clusters
            continue

        old_count = clusters.shape[0]

        keep = np.array([
            cluster_survives_custom_r_dipole(
                cluster=cluster,
                names=names,
                xyz=xyz,
                pair_cutoffs=pair_cutoffs,
                default_r_dipole=r_dipole,
                edge_cache=edge_cache,
            )
            for cluster in clusters
        ], dtype=bool)

        calc.clusters[cluster_order] = clusters[keep].reshape(-1, order_int)

        if is_root():
            logger.info(
                "custom_r_dipole filtered order-%d clusters: %d -> %d",
                order_int,
                old_count,
                calc.clusters[cluster_order].shape[0],
            )

    return calc


def validate_park_2e2n(calc):
    names = np.asarray(calc.bath.N, dtype=object)
    xyz = np.asarray(calc.bath.xyz, dtype=float)

    n1 = _as_cluster_array(calc.clusters[1], 1).shape[0] if 1 in calc.clusters else 0
    n2 = _as_cluster_array(calc.clusters[2], 2).shape[0] if 2 in calc.clusters else 0
    n3 = _as_cluster_array(calc.clusters[3], 3).shape[0] if 3 in calc.clusters else 0
    n4 = _as_cluster_array(calc.clusters[4], 4).shape[0] if 4 in calc.clusters else 0

    bad3 = 0
    bad4 = 0

    if 3 in calc.clusters:
        for row in _as_cluster_array(calc.clusters[3], 3):
            if not cluster_survives_park_2e2n(row, names, xyz):
                bad3 += 1

    if 4 in calc.clusters:
        for row in _as_cluster_array(calc.clusters[4], 4):
            if not cluster_survives_park_2e2n(row, names, xyz):
                bad4 += 1

    same_loc_pairs = count_same_loc_en_order2(calc)

    if is_root():
        logger.info(
            "park_2e2n validation: calc.order=%s, cluster keys=%s, "
            "n1=%d, n2=%d, n3=%d, n4=%d, same_loc_e_14N_order2=%d, bad3=%d, bad4=%d",
            calc.order,
            sorted(calc.clusters.keys()),
            n1, n2, n3, n4,
            same_loc_pairs,
            bad3, bad4,
        )

    if bad3 != 0 or bad4 != 0:
        raise RuntimeError(f"park_2e2n produced invalid clusters: bad3={bad3}, bad4={bad4}")

    if n4 == 0:
        warnings.warn("park_2e2n produced zero order-4 two-P1 clusters.", RuntimeWarning)


def _same_location_groups(indices, xyz, atol=DIST_TOLERANCE):
    """
    Group bath indices by same xyz position.
    :return a list of lists, where each inner list contains bath indices colocated within tolerance.
    """
    groups = []

    for idx in indices:
        idx = int(idx)
        placed = False

        for group in groups:
            ref = group[0]
            if np.allclose(xyz[idx], xyz[ref], atol=atol, rtol=0.0):
                group.append(idx)
                placed = True
                break

        if not placed:
            groups.append([idx])

    return groups


def cluster_survives_park_2e2n(cluster, names, xyz):
    """
    Park 2e2n cluster filter.

    Order-3 survives iff it contains exactly one colocated (e, 14N) pair
    plus one other spin.

    Order-4 survives iff it contains exactly two colocated (e, 14N) pairs.
    """
    cluster = np.asarray(cluster, dtype=int).reshape(-1)
    k = cluster.size

    if k not in (3, 4):
        return True

    if np.unique(cluster).size != k:
        return False

    groups = _same_location_groups(cluster, xyz)

    if k == 3:
        # Must be one strict P1 electron-nitrogen pair plus one singleton.
        group_sizes = sorted(len(g) for g in groups)
        if group_sizes != [1, 2]:
            return False

        pair_groups = [g for g in groups if len(g) == 2]
        return len(pair_groups) == 1 and _is_strict_p1_en_pair(pair_groups[0], names)

    if k == 4:
        # Must be exactly two strict P1 electron-nitrogen pairs.
        group_sizes = sorted(len(g) for g in groups)
        if group_sizes != [2, 2]:
            return False

        return all(_is_strict_p1_en_pair(g, names) for g in groups)

    return True


def get_central_position(central_spin):
    try:
        pos = np.asarray(central_spin.xyz, dtype=float)
    except AttributeError:
        try:
            pos = np.asarray(central_spin["xyz"], dtype=float)
        except Exception as exc:
            raise AttributeError(
                "Could not read central spin position from central_spin['xyz']."
            ) from exc

    pos = np.squeeze(pos)
    if pos.ndim == 2:
        if pos.shape[0] != 1:
            raise ValueError(f"Expected one central spin, got position array with shape {pos.shape}")
        pos = pos[0]

    pos = np.asarray(pos, dtype=float).reshape(-1)

    if pos.shape != (3,):
        raise ValueError(f"Expected central spin position to have shape (3,), got {pos.shape}")

    return pos


def parse_custom_r_dipole(custom_r_dipole, default_r_dipole):
    """
    Parse custom_r_dipole from YAML.
    YAML form:
        custom_r_dipole:
          "13C,13C": 8.0
          "13C,14N": 8.0
    If value is null or false, forbid that pair from forming a graph edge
    """
    if custom_r_dipole is None:
        return {}

    if not isinstance(custom_r_dipole, dict):
        raise TypeError(f"custom_r_dipole must be a dict or None, got {type(custom_r_dipole)}")

    default_r_dipole = float(default_r_dipole)

    if not np.isfinite(default_r_dipole) or default_r_dipole <= 0:
        raise ValueError(f"global r_dipole must be positive and finite, got {default_r_dipole}")

    parsed = {}

    for raw_key, raw_value in custom_r_dipole.items():
        if isinstance(raw_key, str):
            parts = [p.strip() for p in raw_key.split(",")]
        elif isinstance(raw_key, (tuple, list)):
            parts = [str(p).strip() for p in raw_key]
        else:
            raise TypeError(
                f"custom_r_dipole key must be a string like '13C,14N', got {raw_key!r}"
            )

        if len(parts) != 2 or parts[0] == "" or parts[1] == "":
            raise ValueError(
                f"Bad custom_r_dipole key {raw_key!r}; expected something like '13C,14N'"
            )

        pair = canonical_spin_pair(parts[0], parts[1])

        if pair in parsed:
            raise ValueError(
                f"Duplicate custom_r_dipole pair {pair}. "
                f"Check for repeated keys like '13C,14N' and '14N,13C'."
            )

        if raw_value is None or raw_value is False:
            cutoff = None
        elif raw_value is True:
            raise ValueError(
                f"custom_r_dipole[{raw_key!r}] cannot be True. "
                "Use a number, null, or false."
            )
        else:
            cutoff = float(raw_value)

            if not np.isfinite(cutoff):
                raise ValueError(f"custom_r_dipole[{raw_key!r}] must be finite, got {cutoff}")

            if cutoff < 0:
                raise ValueError(f"custom_r_dipole[{raw_key!r}] must be nonnegative, got {cutoff}")

            if cutoff > default_r_dipole:
                raise ValueError(
                    f"custom_r_dipole[{raw_key!r}]={cutoff} is larger than global "
                    f"r_dipole={default_r_dipole}. Since this method only deletes existing "
                    "PyCCE clusters, global r_dipole must be at least as large as every "
                    "custom cutoff."
                )

        parsed[pair] = cutoff

    return parsed


def _custom_edge_allowed(i, j, names, xyz, pair_cutoffs, default_r_dipole, cache):
    """
    Check whether bath spins i and j are connected under custom_r_dipole.
    Uses cache because the same pair appears in many higher-order clusters.
    """
    i = int(i)
    j = int(j)

    if i == j:
        return False

    key = (min(i, j), max(i, j))

    if key in cache:
        return cache[key]

    pair = canonical_spin_pair(names[i], names[j])
    cutoff = pair_cutoffs.get(pair, default_r_dipole)

    if cutoff is None:
        allowed = False
    else:
        rij = np.linalg.norm(xyz[i] - xyz[j])
        allowed = rij <= float(cutoff) + DIST_TOLERANCE

    cache[key] = allowed
    return allowed


def cluster_survives_custom_r_dipole(cluster, names, xyz, pair_cutoffs, default_r_dipole, edge_cache=None,):
    """
    Decide whether an existing PyCCE cluster should survive the new custom r_dipole.

    Uses PyCCE-style connected-graph logic:
        A cluster survives if the custom graph restricted to that cluster is connected.
    """
    cluster = np.asarray(cluster, dtype=int).reshape(-1)
    k = cluster.size

    if k <= 1:
        return True

    if np.unique(cluster).size != k:
        return False

    if edge_cache is None:
        edge_cache = {}

    adj = np.zeros((k, k), dtype=bool)

    for a in range(k):
        for b in range(a + 1, k):
            allowed = _custom_edge_allowed(
                cluster[a],
                cluster[b],
                names=names,
                xyz=xyz,
                pair_cutoffs=pair_cutoffs,
                default_r_dipole=default_r_dipole,
                cache=edge_cache,
            )

            if allowed:
                adj[a, b] = True
                adj[b, a] = True

    # Connected-graph test.
    seen = {0}
    stack = [0]

    while stack:
        u = stack.pop()
        for v in np.where(adj[u])[0]:
            v = int(v)
            if v not in seen:
                seen.add(v)
                stack.append(v)

    return len(seen) == k


def keep_existing_hyperfines(bath):
    """Do not modify bath['A']; prevents PyCCE from auto-generating point dipoles."""
    # used because we set custom hyperfines in get_supercell() and put in the point dipoles ourselves as part of that
    # which was out of convenience
    pass


# +++++++++++++++++++++++ get_supercell() helpers +++++++++++++++++++++++ #
def p1_hyperfine_quad(atoms, hf_on, quad_on, host_idx, jt_axis="A", seed=None):
    """
    Takes a BathArray with colocal (having the same location) P1 nuclei and electrons and adds in custom hyperfine
    and quadrupole dependent on jt_axis
    :param atoms: the BathArray
    :param hf_on: if True, puts in the hyperfine tensor; otherwise puts in zero matrix
    :param quad_on: whether the P1 quadrupole is on
    :param host_idx: None if no host 14N, otherwise is its index in atoms
    :param jt_axis: Jahn-Teller axis
    :param seed: if JT is mixed this is used
    """
    e_idx = np.where(np.equal(np.asarray(atoms.N, dtype=object), "e"))[0]
    p1_idx = np.where(np.equal(np.asarray(atoms.N, dtype=object), "14N"))[0]
    if host_idx is not None:  # remove the host NV nitrogen by index, if it exists
        p1_idx = p1_idx[p1_idx != int(host_idx)]

    if len(e_idx) != len(p1_idx):
        raise Exception(f"wrong number of p1 pairs: len(e_idx)={len(e_idx)}, len(p1_idx)={len(p1_idx)}")
    assert np.array_equal(atoms[e_idx].xyz, atoms[p1_idx].xyz), (
        f"arrays not equal: e_idx={e_idx} xyz={np.array2string(np.asarray(atoms[e_idx].xyz))}; "
        f"p1_idx={p1_idx} xyz={np.array2string(np.asarray(atoms[p1_idx].xyz))}; "
        f"len1={len(atoms[e_idx].xyz)} len2={len(atoms[p1_idx].xyz)}")  # ensure nuclei and electrons are colocated
    if not np.allclose(atoms[e_idx].xyz, atoms[p1_idx].xyz, atol=DIST_TOLERANCE, rtol=0.0):
        raise RuntimeError(f"P1 e/14N pairing mismatch: e_idx={e_idx}, p1_idx={p1_idx}")

    if jt_axis in ("A", "B", "C", "D"):
        if hf_on:
            if is_root():
                print("fixed jt:", jt_axis, "\n\n")
            for i, j in zip(e_idx, p1_idx):
                atoms.add_interaction(i, j, JT_P1_INTERNAL_HF[jt_axis])
        if quad_on:
            atoms["Q"][p1_idx] = JT_P1_QUADRUPOLE[jt_axis]
    elif jt_axis == "mixed":
        axes = np.array(["A", "B", "C", "D"], dtype=object)
        if is_root():
            rng = np.random.default_rng(None if seed is None else int(seed) & 0xFFFFFFFF)
            if seed is None:
                warnings.warn("JT axes were mixed without a deterministic seed.")
            chosen_axes = rng.choice(axes, size=len(e_idx), replace=True)
        else:
            chosen_axes = None
        chosen_axes = MPI.COMM_WORLD.bcast(chosen_axes, root=0)

        if hf_on and is_root():
            print("mixed jt")
            print(chosen_axes)
            print(JT_P1_INTERNAL_HF[str(chosen_axes[0])], "\n\n")

        for i, j, axis in zip(e_idx, p1_idx, chosen_axes):
            if hf_on:
                atoms.add_interaction(i, j, JT_P1_INTERNAL_HF[str(axis)])
            if quad_on:
                atoms["Q"][int(j)] = JT_P1_QUADRUPOLE[str(axis)]
    else:
        raise Exception(f"unknown jt_axis={jt_axis}")

    if not hf_on:
        if is_root():
            print("hyperfine off\n\n")
        zero_interaction = np.zeros((3, 3), dtype=float)
        for i, j in zip(e_idx, p1_idx):
            atoms.add_interaction(i, j, zero_interaction)


def nv_quadrupole(atoms, host_idx):
    quad = P * np.diag([-1 / 3, -1 / 3, 2 / 3])
    atoms['Q'][host_idx] = quad


def nv_hyperfine(atoms, nv_idx, host_hf):
    """
    Handle host NV nitrogen hyperfine.
    """
    if host_hf:
        atoms["A"][nv_idx] = np.array(NV_HYPERFINE, dtype=float)
    else:
        atoms["A"][nv_idx] = np.zeros((3, 3), dtype=float)


def host_bath_zero_dipoles(atoms, host_idx):
    if host_idx is None:
        raise Exception("host_idx cannot be None")

    zero = np.zeros((3, 3), dtype=float)
    n_arr = np.asarray(atoms.N, dtype=object)
    bath_idx = np.where(
        np.equal(n_arr, "14N") |
        np.equal(n_arr, "e") |
        np.equal(n_arr, "13C")
    )[0]

    bath_idx = bath_idx[bath_idx != int(host_idx)]

    for j in bath_idx:
        atoms.add_interaction(int(host_idx), int(j), zero)

    return atoms


def zero_dipoles(atoms, interactions, host_idx):
    zero = np.zeros((3, 3), dtype=float)

    n_arr = np.asarray(atoms.N, dtype=object)

    p1n_idx = np.where(np.equal(n_arr, "14N"))[0]
    p1e_idx = np.where(np.equal(n_arr, "e"))[0]
    c13_idx = np.where(np.equal(n_arr, "13C"))[0]

    # remove host NV nitrogen from P1 nitrogen list
    if host_idx is not None:
        p1n_idx = p1n_idx[p1n_idx != int(host_idx)]

    def zero_same_type(idx):
        for i, j in combinations(idx, 2):
            atoms.add_interaction(i, j, zero)

    def zero_cross_type(idx1, idx2):
        for i, j in product(idx1, idx2):
            if i != j:
                atoms.add_interaction(i, j, zero)

    if not interactions["p1n_p1e_dip"]:
        zero_cross_type(p1n_idx, p1e_idx)

    if not interactions["p1n_p1n_dip"]:
        zero_same_type(p1n_idx)

    if not interactions["p1e_p1e_dip"]:
        zero_same_type(p1e_idx)

    if not interactions["c13_c13_dip"]:
        zero_same_type(c13_idx)

    if not interactions["p1n_c13_dip"]:
        zero_cross_type(p1n_idx, c13_idx)

    if not interactions["p1e_c13_dip"]:
        zero_cross_type(p1e_idx, c13_idx)

    return atoms


def append_many_same_loc(atoms, ids, label):
    """
    Add a spin into a BathCell for every existing spin with index in some list of indices. The new spins are at the same
    locations as the existing spins.
    :param atoms: the BathCell
    :param ids: the list of indices of existing spins
    :param label: the label of the new spin; e.g., "e"
    :return: a new BathCell with the added spins
    """
    ids = np.asarray(ids, dtype=int)
    add = np.zeros(len(ids), dtype=atoms.dtype)
    add['N'] = label
    add['xyz'] = atoms['xyz'][ids]

    return atoms.__class__(array=np.concatenate([atoms.view(np.ndarray), add.view(np.ndarray)]))


def get_single_nv(nv_position, alpha, beta):
    return pc.CenterArray(spin=1, position=nv_position, D=D, E=E, alpha=alpha, beta=beta)


# +++++++++++++++++++++++ Primary functions +++++++++++++++++++++++ #
def get_supercell(supercell_params):
    """
    Builds a bath supercell of carbon 13 and P1s, and the NV center.
    :param supercell_params: see .yaml file and docs for details.
    :return: A PyCCE BathCell which is the supercell.
    """
    # ================= PARAMETERS (see yaml and docs for explanations) ================= #
    # required
    rm_host = supercell_params['rm_host']
    c13_conc = supercell_params['c13_conc']
    p1_conc = supercell_params['p1_conc']
    p1_mode = supercell_params['p1_mode']
    size = supercell_params['size']
    # optional
    jt_axis = supercell_params.get('jt_axis', 'mixed')
    zdir = supercell_params.get("zdir", [1, 1, 1])  # .get("yaml_name", default value)
    seed = supercell_params.get("seed", None)
    verbose = supercell_params.get("verbose", False)
    save_supercells = supercell_params.get("save_supercells", False)
    user_interactions = supercell_params.get("interactions", {}) or {}
    bad_keys = set(user_interactions) - set(INTERACTION_DEFAULTS)
    if bad_keys:
        raise KeyError(f"Unknown interaction keys: {sorted(bad_keys)}. "
                       f"Valid keys are: {sorted(INTERACTION_DEFAULTS)}")
    interactions = INTERACTION_DEFAULTS | user_interactions
    nv_position = supercell_params.get("nv_position", [0, 0, 0])  # these last 3 are NV parameters
    alpha = supercell_params.get("alpha", [0, 0, 1])
    beta = supercell_params.get("beta", [0, 1, 0])
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''Build supercell'''
    # unit cell
    diamond = pc.read_ase(bulk('C', 'diamond', cubic=True))
    diamond.add_isotopes(('13C', c13_conc))
    diamond.add_isotopes(('14C', p1_conc))  # recall: these are replaced with P1s later
    diamond.zdir = zdir

    # generate supercell
    if verbose and is_root():
        logger.info(f"Ensemble {current_ensemble()} starting now:")
        logger.info("Generating supercell: size=%s, p1_conc=%g, seed=%s", size, p1_conc, seed)
    if is_root():
        if seed is None:
            warnings.warn("Supercell was generated without a deterministic seed.")
    atoms = diamond.gen_supercell(size, seed=seed,
                        remove=[('C', [0., 0, 0]), ('C', [0.5, 0.5, 0.5])],  # remove NV carbons IF they're there
                        add=[('14N', [0.5, 0.5, 0.5]), ])  # add NV nitrogen nucleus
    nv_pos = atoms[atoms.N == '14N'].xyz[0].copy()  # position of NV nitrogen
    atoms.add_type(*SPIN_TYPES)

    # add P1s according to p1_mode
    idx = np.where(atoms.N == '14C')[0]
    atoms = append_many_same_loc(atoms, idx, 'e')  # add in electrons where 14Cs are
    atoms['N'][idx] = '14N'  # add in P1 nuclei by replacing the 14Cs with 14Ns
    if p1_mode == 'e':  # only P1 electrons
        keep = (atoms.N != '14N') | np.all(np.isclose(atoms.xyz, nv_pos), axis=1)
        atoms = atoms[keep]
    elif p1_mode == 'n':  # only P1 nuclei
        keep = (atoms.N != 'e')
        atoms = atoms[keep]
    elif p1_mode != 'b':
        raise Exception("\'p1_mode\' parameter not recognized, see yaml file")

    # remove host nitrogen according to rm_host
    host_idx = None  # None if we remove the host nitrogen, otherwise is the idx
    if rm_host:
        atoms = atoms[~np.all(np.isclose(atoms.xyz, nv_pos), axis=1)]
    else:
        matches = np.where(np.all(np.isclose(atoms.xyz, nv_pos), axis=1))[0]
        if len(matches) != 1:
            raise Exception(f"Expected precisely 1 atom at the NV host defect position={nv_pos}, found {len(matches)}")
        host_idx = matches[0]

    '''Add interactions'''
    # first add bath-central interactions
    atoms.from_point_dipole(nv_position)  # everything is automatically dipolar
    if not interactions["c13_dip"]:
        atoms["A"][atoms.N == "13C"] = np.zeros((3, 3), dtype=float)
    if not interactions["p1n_dip"]:
        p1n_mask = np.where(np.asarray(atoms.N) == "14N")[0]
        if host_idx is not None:
            p1n_mask = p1n_mask[p1n_mask != int(host_idx)]  # exclude the host NV nitrogen
        atoms["A"][p1n_mask] = np.zeros((3, 3), dtype=float)
    if not interactions["p1e_dip"]:
        atoms["A"][atoms.N == "e"] = np.zeros((3, 3), dtype=float)
    nv_hyperfine(atoms, host_idx, interactions['host_hf'])

    # then add bath intrinsic interactions
    zero_dipoles(atoms, interactions, host_idx)  # highly important this is before p1_hyperfine_quad

    # handling p1 internal hyperfines and quadrupoles (jt-axis dependent)
    if p1_conc > 0:
        p1_hyperfine_quad(atoms, interactions['p1_hf'], interactions['p1_quad'], host_idx, jt_axis=jt_axis, seed=seed)

    # handles host defect interactions with bad, and host defect quadrupole
    if not interactions['host_bath_dip'] and host_idx is not None:
        host_bath_zero_dipoles(atoms, host_idx)
    if interactions['host_quad'] and host_idx is not None:
        nv_quadrupole(atoms, host_idx)

    '''i/o and get NV center'''
    if verbose and is_root():
        # main logs
        logger.info("Built supercell: N_atoms=%d, N_P1=%d, N_13C=%d\n", len(atoms),
                    len(np.where(atoms.N == '14N')[0])-1, len(np.where(atoms.N == '13C')[0]))
    if save_supercells and is_root():
        # write atoms to a separate CSV via the run logger
        field_names = list(atoms.dtype.names)
        logger.save_csv(f"atoms_ens{current_ensemble()}", field_names,
                    ([a[name] for name in field_names] for a in atoms),
                    subdir="supercells")
    nv = get_single_nv(nv_position, alpha, beta)  # get NV center

    return atoms, nv


def get_simulator(atoms, central_spin, simulator_params):
    """
    Builds a PyCCE Simulator object given a BathCell and CenterArray.
    :param atoms: The PyCCE BathCell which is the supercell.
    :param central_spin: The PyCCE CenterArray to be used.
    :param simulator_params: This must contain at least order, r_bath, and r_dipole. See .yaml file for full details.
    :return: The PyCCE Simulator object.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    order = int(simulator_params['order'])
    r_bath = float(simulator_params['r_bath'])
    r_dipole = float(simulator_params['r_dipole'])
    # optional, defaults
    verbose = simulator_params.get("verbose", False)
    polarization = simulator_params.get("polarization", None)
    custom_r_bath = simulator_params.get("custom_r_bath", None)
    custom_r_dipole = simulator_params.get("custom_r_dipole", None)
    park_2e2n = simulator_params.get("park_2e2n", False)
    generalized_2e2n = simulator_params.get("generalized_2e2n", False)
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    '''build simulator'''
    # implement custom_r_bath
    if custom_r_bath is not None:
        atoms = apply_custom_r_bath(atoms, custom_r_bath, central_spin)

    # build Simulator object
    calc_params = dict(
        bath=atoms,
        spin=central_spin,
        order=order,
        r_bath=r_bath,
        r_dipole=r_dipole,
        hyperfine=keep_existing_hyperfines
    )
    calc = pc.Simulator(**calc_params)
    if verbose and is_root():
        logger.info(f"[ens num {current_ensemble()}] Built simulator:\n{calc}\n")

    '''make changes to simulator'''
    # custom polarization
    install_custom_polarization()
    if polarization is None:  # default behavior
        if is_root():
            print("\n\n\nno polarization\n\n\n")
    else:
        assert type(polarization) is float
        calc.polarization_gamma = float(polarization)

    # implement custom_r_dipole
    if custom_r_dipole is not None:
        apply_custom_r_dipole(calc, custom_r_dipole, r_dipole)

    # add same P1, e-nucleus order 2 clusters; this should be after custom_r_dipole implementation
    add_p1_order2(calc)

    # implement park_2e2n
    if park_2e2n:
        # First build Park-style order-4 clusters using the original spin-level order-2 graph.
        calc = add_park_2e2n_clusters(calc)

        names = np.asarray(calc.bath.N, dtype=object)
        xyz = np.asarray(calc.bath.xyz, dtype=float)
        old_keys = sorted(calc.clusters.keys())
        old1 = _as_cluster_array(calc.clusters[1], 1).shape[0] if 1 in calc.clusters else 0
        old2 = _as_cluster_array(calc.clusters[2], 2).shape[0] if 2 in calc.clusters else 0
        old3 = _as_cluster_array(calc.clusters[3], 3).shape[0] if 3 in calc.clusters else 0
        old4 = _as_cluster_array(calc.clusters[4], 4).shape[0] if 4 in calc.clusters else 0

        if generalized_2e2n:
            if is_root():
                logger.info("generalized_2e2n is on: skipping deletion of order 1, 2 and 3 clusters...\n\n")
        else:  # if generalized is on this is skipped and we keep all the clusters
            # If generalized is off then delete clusters to get Park-like cluster structure:
            #   order 2 = one full P1 center: colocated (e, 14N)
            #   order 4 = two full P1 centers: (e, 14N, e, 14N)
            # Remove standalone spin order-1 clusters, non-colocated order-2 clusters, and all order-3 clusters.
            if is_root():
                logger.info("generalized_2e2n is off: deleting order 1, 2 and 3 clusters...\n\n")
            p1_pair_rows = []
            for i, j in _as_cluster_array(calc.clusters[2], 2):
                if _strict_colocated_en_pair(i, j, names, xyz):
                    p1_pair_rows.append(_edge_tuple(i, j))

            p1_pair_rows = np.asarray(sorted(set(p1_pair_rows)), dtype=int).reshape(-1, 2)

            if p1_pair_rows.shape[0] == 0:
                raise RuntimeError("park_2e2n cluster protocol found zero colocated P1 e-14N order-2 pairs.")

            calc.clusters.pop(1, None)
            calc.clusters[2] = p1_pair_rows
            calc.clusters.pop(3, None)

            if count_same_loc_en_order2(calc) != _as_cluster_array(calc.clusters[2], 2).shape[0]:
                raise RuntimeError("After park_2e2n cluster protocol, not all order-2 clusters are colocated e-14N pairs.")

        # i/o
        validate_park_2e2n(calc)
        n1 = _as_cluster_array(calc.clusters[1], 1).shape[0] if 1 in calc.clusters else 0
        n2 = _as_cluster_array(calc.clusters[2], 2).shape[0] if 2 in calc.clusters else 0
        n3 = _as_cluster_array(calc.clusters[3], 3).shape[0] if 3 in calc.clusters else 0
        n4 = _as_cluster_array(calc.clusters[4], 4).shape[0] if 4 in calc.clusters else 0
        if is_root():
            logger.info(
                "Park cluster protocol: generalized_2e2n=%s; keys %s -> %s; "
                "order-1 %d -> %d; order-2 %d -> %d; order-3 %d -> %d; order-4 %d -> %d",
                generalized_2e2n,
                old_keys, sorted(calc.clusters.keys()),
                old1, n1,
                old2, n2,
                old3, n3,
                old4, n4,
            )

    return calc


def run_experiment(calc, experiment_params):
    """
    Calculates the coherence function of some PyCCE Simulator object.
    :param calc: The PyCCE Simulator object to run the simulations with.
    :param experiment_params: see .yaml file for details.
    :return: a 1D np array of the coherence trajectory
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    magnetic_field = np.array(experiment_params['magnetic_field'])
    pulses = experiment_params['pulses']
    if isinstance(pulses, str):  # see docs
        pulses = get_pulses(pulses)
        if is_root():
            logger.warn("Using a string id pulse.")
    time_space = experiment_params['time_space']
    cce_type = experiment_params['cce_type']
    parallel = experiment_params['parallel']
    # optional, defaults
    n_bath_states = experiment_params.get("n_bath_states", None)
    populations = experiment_params.get("populations", False)
    mc_seed_base = experiment_params.get("mc_seed", 8805)
    verbose = experiment_params.get("verbose", False)
    checkpoints = experiment_params.get("checkpoints", False)
    # ================= END OF PARAMETERS ================= #

    # ================= MAIN CODE ================= #
    # run the simulations
    calc_params = dict(
        quantity="coherence",
        magnetic_field=magnetic_field,
        pulses=pulses,
        parallel=parallel,
        masked=False,
    )

    if verbose and is_root():
        logger.info(f"\n[ens number {current_ensemble()}] Starting coherence experiment.\n")

    # change settings according to cce_type
    base_cce_type, *mc = cce_type.split("_", 1)
    if bool(mc):
        if parallel:
            calc_params['parallel_states'] = True
        calc_params['nbstates'] = n_bath_states

        if mc_seed_base is not None:
            ens = current_ensemble()
            ens = 0 if ens is None else int(ens)

            # Same ensemble index gets the same MC seed across multi sub-runs.
            # Different ensemble indices get different MC seeds.
            # seed is used for MC sampling
            calc_params['seed'] = (int(mc_seed_base) + ens) % (2 ** 32)
    if base_cce_type == 'cce':
        calc_params['method'] = 'cce'
    elif base_cce_type == 'gcce':
        calc_params['method'] = 'gcce'
    else:
        raise Exception(f"Unknown cce type: {cce_type}")

    # log validation for get_simulator() cluster operations
    if is_root() and verbose:
        logger.info(
            "FINAL clusters passed to calc.compute: keys=%s, counts=%s (starting compute now)",
            sorted(calc.clusters.keys()),
            {
                int(k): _as_cluster_array(v, int(k)).shape[0]
                for k, v in calc.clusters.items()
            },
        )

        try:
            log_final_colocated_pair_diagnostics(calc)
        except Exception as exc:
            logger.warning(
                "FINAL colocated-pair diagnostic failed before compute; "
                "continuing without changing calc: %r",
                exc,
            )

    # custom polarization
    pycce_mc.POLARIZATION_GAMMA = getattr(calc, "polarization_gamma", None)

    # # trying to 'polarize' the bath
    # if experiment_params.get("force_bath_up", False):
    #     proj = np.array([calc.bath.types[str(name)].s for name in calc.bath.N], dtype=float)
    #     calc.bath.state = gen_state_list(proj, calc.bath.dim)
    #     print(calc.bath.state)

    # print(calc.bath)
    # calculate coherence and add it to results
    if not populations:
        coherence = calc.compute(time_space, **calc_params)
    else:
        calc_params['normalized'] = False
        rho00 = calc.compute(time_space, i=calc.center.alpha, j=calc.center.alpha, **calc_params)
        rho11 = calc.compute(time_space, i=calc.center.beta, j=calc.center.beta, **calc_params)
        coherence = np.asarray(rho00) - np.asarray(rho11)

    if is_root():
        logger.info(f"\n[ens number {current_ensemble()}] Rank 0 mpi process has finished coherence experiment.\n")
    if checkpoints:
        logger.save_csv(f"L_ens{current_ensemble()}",
            ["time_space", "trajectory"], ((t, y) for t, y in zip(time_space, coherence)), subdir="checkpoints",
            ignore_mpi=False)  # ignore_mpi=False means only root writes, this is safe since compute returns the full
                               # property on each process

    return coherence
