import numpy as np
import pycce as pc
from ase.build import bulk
from sim import logger, is_root
from mpi4py import MPI  # for debugging

# Bath spin types
#              name    spin    gyro       quadrupole (for s>1/2)
# ZFS parameters of NV center in diamond
SPIN_TYPES = [('14N',  1,      1.9338,    20.44),
              ('13C',  1 / 2,  6.72828         ),
              ('e'  ,  1 / 2,  -17608.5962784  )
             ]
D = 2.88 * 1e6 # kHz
E = 0 # kHz


# +++++++++++++++++++++++ Auxiliary functions +++++++++++++++++++++++ #
def current_ensemble():
    """
    read ensemble index off the shared logger object
    :return: ensemble index, an attribute of the logger in sim.py
    """
    return getattr(logger, "current_ensemble", None)


def p1_hyperfine(atoms, on):
    """
    Takes a BathCell with colocal (having the same location) P1 nuclei and electrons and returns either the hyperfine
    tensor or the zero matrix as a Pycce InteractionMap.
    :param atoms: the BathCell
    :param on: if True, returns the hyperfine tensor; otherwise returns zero matrix
    :return: an InteractionMap
    """
    e_idx = np.where(atoms.N == 'e')[0]  # get indices
    p1_idx = np.where(atoms.N == '14N')[0]
    # to debug the NV's own electron add this
    if len(e_idx) == len(p1_idx) - 1:
        print("chopped off one 14N")
        p1_idx = p1_idx[1:]
    elif len(e_idx) == len(p1_idx):
        print("same length")
    else:
        raise Exception("wrong number of p1 pairs")

    assert np.array_equal(atoms[e_idx].xyz, atoms[p1_idx].xyz ) # ensure nuclei and electrons are colocated
    pairs = list(zip(e_idx, p1_idx))  # pair them up

    if on:
        # hyperfine tensor, from pg 17780 of https://pubs.acs.org/doi/pdf/10.1021/acs.jpcc.2c06145?ref=article_openPDF
        e_p1_interaction = [
            [82.0 * 1e3, 0, 0],
            [0, 82.0 * 1e3, 0],
            [0, 0, 114.0 * 1e3]
        ]
    else:  # in this case, the P1 HF is not considered
        e_p1_interaction = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]

    imap = pc.InteractionMap()
    for i, j in pairs:
        imap[i, j] = e_p1_interaction  # J_ij between bath spins i and j

    return imap


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
    return atoms.__class__(array=np.concatenate([atoms.view(np.ndarray),
                                                 add.view(np.ndarray)]))


def get_pulse(pulse_id):
    """
    Given some string pulse_id, return the pulse sequence or number of pulses to put into Simulator.compute. See yaml
    'pulse_id' parameter for details
    :param pulse_id: the string id
    :return: the pulse sequence or number of pulses
    """
    if pulse_id == "1":
        return 1
    elif pulse_id == "hahn":
        return [pc.Pulse('x', np.pi)]
    elif isinstance(pulse_id, int):
        return pulse_id
    else:
        raise Exception("pulse_id not recognized")


# +++++++++++++++++++++++ Primary functions +++++++++++++++++++++++ #
def get_supercell(supercell_params):
    """
    Builds a supercell with an NV center and a bath of P1 nuclei and electrons. The supercell is initially generated
    with 14C's, which utilizes PyCCE's native isotopic substitution protocols. These 14Cs are then replaced with P1
    nuclei and electrons. The P1 nuclei and electrons are colocated (identical locations).
    :param supercell_params: This must contain at least p1_conc and size. See .yaml file for full details.
    :return: A PyCCE BathCell which is the supercell.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    p1_conc = supercell_params['p1_conc']
    size = supercell_params['size']
    # optional, defaults
    seed = supercell_params.get("seed", None)
    zdir = supercell_params.get("zdir", [1, 1, 1])
    c13_conc = supercell_params.get("c13_conc", 0.011)
    verbose = supercell_params.get("verbose", False)
    temp_debug = supercell_params.get("temp_debug", True)

    # ================= MAIN CODE ================= #
    # build unit cell
    diamond = pc.read_ase(bulk('C', 'diamond', cubic=True))
    diamond.add_isotopes(('13C', c13_conc))
    diamond.add_isotopes(('14C', p1_conc))  # these are replaced with P1s later
    diamond.zdir = zdir  # set z direction of the defect

    # generate supercell
    if verbose and is_root():
        logger.info(f"Ensemble {current_ensemble()} starting now:")
        logger.info("Generating supercell: size=%s, p1_conc=%g, seed=%s", size, p1_conc, seed)

    atoms = diamond.gen_supercell(size, seed=seed,
                            remove=[('C', [0., 0, 0]), ('C', [0.5, 0.5, 0.5])],  # remove NV carbons IF they're there
                            add=[('14N', [0.5, 0.5, 0.5]), ])  # add NV nitrogen nuclei (electron added later)
    atoms.add_type(*SPIN_TYPES)
    assert np.allclose(atoms.A, atoms.A.flat[0])  # check that there are no hyperfine or quadrupoles
    assert np.allclose(atoms.Q, atoms.Q.flat[0])

    # add in P1 nuclei and electrons
    mask = (atoms.N == '14C')
    idx = np.where(mask)[0]
    atoms = append_many_same_loc(atoms, idx, 'e')  # add in electrons where 14Cs are
    if not temp_debug:
        print("placing electron")
        idx_14n = np.where(atoms.N == '14N')[0]
        atoms = append_many_same_loc(atoms, idx_14n, 'e')  # add in electron where NV nitrogen is
    else:
        print("no electron placed")
    atoms['N'][idx] = '14N'  # add in P1's by replacing the 14Cs with 14Ns
    assert '14C' not in atoms['N']  # make sure no 14Cs left

    if verbose and is_root():
        # main logs
        logger.info("Built supercell: N_atoms=%d, N_P1=%d, N_13C=%d\n", len(atoms), len(idx), len(atoms) - 2 * len(idx))

        # write atoms to a separate CSV via the run logger
        field_names = list(atoms.dtype.names)
        logger.save_csv(f"atoms_ens{current_ensemble()}", field_names,
                        ([a[name] for name in field_names] for a in atoms),
                        subdir="supercells")

        # for debugging 14C user warning:
        # names = atoms['N'].astype(str)  # make sure they’re strings
        # logger.info("\nH" + " ".join(names) + "\n\n")

    return atoms


def get_simulator(atoms, simulator_params):
    """
    Builds a PyCCE Simulator object given a BathCell which is a supercell with an NV center and a bath of P1s.
    :param atoms: The PyCCE BathCell which is the supercell.
    :param simulator_params: This must contain at least order, r_bath, and r_dipole. See .yaml file for full details.
    :return: The PyCCE Simulator object.
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    order = int(simulator_params['order'])
    r_bath = float(simulator_params['r_bath'])
    r_dipole = float(simulator_params['r_dipole'])
    # optional, defaults
    p1_hf = simulator_params.get("p1_hf", True)
    polarization = simulator_params.get("polarization", 0)
    nv_position = simulator_params.get("nv_position", [ 0, 0, 0 ])
    alpha = simulator_params.get("alpha", [ 0, 0, 1 ])
    beta = simulator_params.get("beta", [ 0, 1, 0 ])
    debug_hf = simulator_params.get("debug_hf", False)
    verbose = simulator_params.get("verbose", False)

    # ================= MAIN CODE ================= #
    # get NV center and imap
    nv = pc.CenterArray(spin=1, position=nv_position, D=D, E=E, alpha=alpha, beta=beta)
    imap = p1_hyperfine(atoms, p1_hf)  # get imap for P1 electron-nuclei hyperfine

    # build simulator
    calc_params = dict(
        bath=atoms,
        spin=nv,
        imap=imap,
        order=order,
        r_bath=r_bath,
        r_dipole=r_dipole,
    )
    if debug_hf:
        calc_params.pop("imap")
    calc = pc.Simulator(**calc_params)

    # if gamma is nonzero, will polarize the 13Cs in the Simulator's bath
    if polarization != 0:
        gamma = polarization
        polos = np.exp(-(calc.bath.dist() / gamma) ** 2) * 0.5

        for a, pol in zip(calc.bath, polos):
            if a.N != '13C':
                continue  # Skip 14N and electrons
            # Generate density matrix
            dm = np.zeros((2, 2), dtype=np.complex128)
            dm[0, 0] = 0.5 + pol
            dm[1, 1] = 0.5 - pol
            a.state = dm

    if verbose and is_root():
        logger.info(f"[ens num {current_ensemble()}] Built simulator:\n{calc}\n")

    return calc


def run_experiment(calc, experiment_params):
    """
    Calculates the coherence function of some PyCCE Simulator object using some specified cce method(s).
    :param calc: The PyCCE Simulator object to run the simulations with.
    :param experiment_params: This must contain at least magnetic_field, pulse_id, time_space, cce_types, and parallel.
    See .yaml file for full details.
    :return: a dictionary: each key is a cce_type, each value is a 1D np array of the coherence trajectory
    """
    # ================= PARAMETERS (see yaml for explanations) ================= #
    # required
    magnetic_field = experiment_params['magnetic_field']
    pulses = get_pulse(experiment_params['pulse_id'])
    time_space = experiment_params['time_space']
    cce_types = experiment_params['cce_types']
    parallel = experiment_params['parallel']
    # optional, defaults
    n_bath_states = experiment_params.get("n_bath_states", 20)
    verbose = experiment_params.get("verbose", False)
    checkpoints = experiment_params.get("checkpoints", True)

    # if CPMG, turn normalized off and update timespace
    cpmg = isinstance(experiment_params['pulse_id'], int)
    normalized = False if cpmg else True
    normalized_override = experiment_params.get("normalized_override", normalized)
    print("normalized", normalized_override)
    if cpmg:
        new_time_space = np.linspace(2.0*pulses*time_space[0], 2.0*pulses*time_space[-1], len(time_space))
        time_space = new_time_space

    # ================= MAIN CODE ================= #
    # initialize the dictionary to be returned
    results = dict()

    # run the simulations
    calc_params = dict(
        quantity="coherence",
        magnetic_field=magnetic_field,
        pulses=pulses,
        parallel=parallel,
    )
    # rank = MPI.COMM_WORLD.Get_rank()  # used for debugging mpi
    for cce_type in cce_types:
        if verbose and is_root():
            logger.info(f"\n[ens number {current_ensemble()}] Starting coherence experiment:{cce_type}.\n")

        # change settings according to cce_type
        base, *rest = cce_type.split("_", 1)
        if bool(rest):  # if 'mc' is included in the cce_type
            if parallel:
                calc_params['parallel_states'] = True
            calc_params['nbstates'] = n_bath_states
        if base == 'cce':
            calc_params['method'] = 'cce'
        elif base == 'gcce':
            calc_params['method'] = 'gcce'
            # if isinstance(pulses, int):
            print("normalized2", normalized_override)
            calc_params['normalized'] = normalized_override  # if CPMG, turns it off
        else: raise Exception(f"Unknown cce type: {cce_type}")

        # calculate coherence and add it to results
        coherence = calc.compute(time_space, **calc_params)
        results[cce_type] = coherence

        if verbose and is_root():
            logger.info(f"\nRank 0 mpi process has finished coherence experiment:{cce_type}.\n")
        if checkpoints:
            logger.save_csv(f"coherence_{cce_type}_ens{current_ensemble()}",
                            ["time_space", "trajectory"], ((t, y) for t, y in zip(time_space, coherence)),
                            subdir="checkpoints",
                            ignore_mpi=False)  # ignore_mpi=False means only root writes, this is safe since compute
                                               # returns the full property on each process

    return results
