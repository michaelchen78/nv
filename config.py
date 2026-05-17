"""config file"""


"""CONSTANTS"""
#             Bath spin types
#              name    spin    gyro       quadrupole (for s>1/2)
SPIN_TYPES = [('14N', 1, 1.9338, 20.44),
              ('13C',  1 / 2,  6.72828         ),
              ('e'  ,  1 / 2,  -17608.5962784  ),
              ]

# ZFS parameters of NV center in diamond
D = 2.88 * 1e6 # kHz
E = 0 # kHz


"""INTERACTIONS"""
# hyperfine tensors from Park 2022 npjqi SI ST2, experimental
JT_P1_INTERNAL_HF = {
    "A": [[81.31 * 1e3, 0, 0],
          [0, 81.31 * 1e3, 0],
          [0, 0, 114.03 * 1e3]],
    "B": [[103.12 * 1e3, 12.59 * 1e3, 8.90 * 1e3],
          [12.59 * 1e3, 88.58 * 1e3, 5.14 * 1e3],
          [8.90 * 1e3, 5.14 * 1e3, 84.95 * 1e3]],
    "C": [[103.12 * 1e3, -12.59 * 1e3, -8.90 * 1e3],
          [-12.59 * 1e3, 88.58 * 1e3, 5.14 * 1e3],
          [-8.90 * 1e3, 5.14 * 1e3, 84.95 * 1e3]],
    "D": [[81.31 * 1e3, 0, 0],
          [0, 110.39 * 1e3, -10.28 * 1e3],
          [0, -10.28 * 1e3, 84.95 * 1e3]]}

# quadrupole tensors from Park 2022 npjqi SI ST3
JT_P1_QUADRUPOLE = {
    "A": [
        [1.25 * 1e3, 0, 0],
        [0, 1.26 * 1e3, 0],
        [0, 0, -2.51 * 1e3],
    ],
    "B": [
        [-1.25 * 1e3, -1.45 * 1e3, -1.02 * 1e3],
        [-1.45 * 1e3, 0.42 * 1e3, -0.59 * 1e3],
        [-1.02 * 1e3, -0.59 * 1e3, 0.84 * 1e3],
    ],
    "C": [
        [-1.25 * 1e3, 1.45 * 1e3, 1.02 * 1e3],
        [1.45 * 1e3, 0.42 * 1e3, -0.59 * 1e3],
        [1.02 * 1e3, -0.59 * 1e3, 0.84 * 1e3],
    ],
    "D": [
        [1.26 * 1e3, 0, 0],
        [0, -2.09 * 1e3, 1.18 * 1e3],
        [0, 1.18 * 1e3, 0.84 * 1e3],
    ],
}

# NV quadrupole and hyperfine from Xie et al. 2021 PRL https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.127.053601
P = -4945.7549  # kHz
NV_HYPERFINE = [[-2.63 * 1e3, 0.0, 0.0],
                [0.0, -2.63 * 1e3, 0.0],
                [0.0, 0.0, -2.16 * 1e3]]

"""CODING"""
# registry of run types; if implementing a custom run must add to here
RUN_REGISTRY = {
    "unit": "unit_run",
    "multi": "multi_run",
    "grid": "grid_run",
    "cpmg": "cpmg",
}

# NPC bath interactions
INTERACTION_DEFAULTS = {
    # bath-central
    "host_hf": True,
    "c13_dip": True,
    "p1n_dip": True,
    "p1e_dip": True,
    # bath intrinsic
    "p1_hf": True,
    "p1n_p1e_dip": True,
    "p1n_p1n_dip": True,
    "p1e_p1e_dip": True,
    "c13_c13_dip": True,
    "p1n_c13_dip": True,
    "p1e_c13_dip": True,
    "host_bath_dip": True,
    "host_quad": True,
    "p1_quad": True,
}

# universal dist tolerance used in pycce bath operations
DIST_TOLERANCE = 1e-8
