from itertools import product
from pathlib import Path
import sys
import numpy as np
import whooie.pyplotdefs as pd

datadir = Path("output").joinpath("time_bin")
data = np.load(str(datadir.joinpath("final.npz")))
rho = data["rho"]

a = rho[4, 4]
b = rho[17, 17]
rho[4, 17] = (a + b) / 2
rho[17, 4] = ((a + b) / 2).conjugate()

atom_states = [
    "C0",
    "C1",
    "T0",
    "T1",
    "T2",
    "T3",
]
early_states = [
    0,
    1,
    2,
    3,
]
late_states = [
    0,
    1,
    2,
    3,
]
photon_states = list(product(early_states, late_states))

all_states = list(product(atom_states, early_states, late_states))

FS = pd.pp.rcParams["figure.figsize"]
(
    pd.Plotter.new(figsize=[3.5 * FS[0], 3.5 * FS[1]])
    .imshow(abs(rho), cmap=pd.colormaps["plasma"])
    .colorbar()
    .grid(False, which="both")
    .set_xticks([])
    .set_yticks(
        list(range(len(all_states))),
        all_states,
        fontsize="xx-small",
    )
    .set_clabel("$|\\rho_{ij}|$")
    .savefig(datadir.joinpath("time_bin_coherences.png"))
    .close()
)
print(np.diag(rho @ rho).sum())

comp_states = ["C0", "C1"]
single_photon_states = [(0, 1), (1, 0)]

iso_states = [
    (a, p)
    for (a, p) in product(atom_states, photon_states)
    if a in comp_states and p in single_photon_states
]
iso_selector = np.array([
    [1.0 if s == s0 else 0.0 for s in product(atom_states, photon_states)]
    for s0 in iso_states
])
rho_iso = iso_selector @ rho @ iso_selector.T

det_states = [
    "∅",
    "A@t0",
    "A@t1",
    "A@t2",
    "B@t0",
    "B@t1",
    "B@t2",
    "A@t0+A@t1",
    "A@t0+B@t1",
    "B@t0+A@t1",
    "B@t0+B@t1",
    "2xA@t1",
    "2xB@t1",
    "A@t0+A@t2",
    "A@t0+B@t2",
    "B@t0+A@t2",
    "B@t0+B@t2",
    "A@t1+A@t2",
    "A@t1+B@t2",
    "B@t1+A@t2",
    "B@t1+B@t2",
]

time_bin_states = ["t0", "t1", "t2"]
path1_states = ["a", "b"]
path2_states = ["A", "B"]
atom_bin_states = ["-z", "+x", "+z"]

def proj12(ph: float) -> np.ndarray:
    rt2 = np.sqrt(2.0)
    eiph = np.exp(1.0j * ph)
    return np.kron(
        np.eye(len(atom_states)),
        np.array([
            [ 1.0, 0.0,               0.0,               0.0               ],
            [ 0.0, 1.0j * eiph / 2.0, 0.0,               0.0               ],
            [ 0.0, 1.0j / 2.0,        1.0j * eiph / 2.0, 0.0               ],
            [ 0.0, 0.0,               1.0j / 2.0,        0.0               ],
            [ 0.0, eiph / 2.0,        0.0,               0.0               ],
            [ 0.0, -1.0 / 2.0,        eiph / 2.0,        0.0               ],
            [ 0.0, 0.0,               -1.0 / 2.0,        0.0               ],
            [ 0.0, 0.0,               0.0,               -eiph / 4.0       ],
            [ 0.0, 0.0,               0.0,               1.0j * eiph / 4.0 ],
            [ 0.0, 0.0,               0.0,               1.0j * eiph / 4.0 ],
            [ 0.0, 0.0,               0.0,               eiph / 4.0        ],
            [ 0.0, 0.0,               0.0,               -rt2 * eiph / 4.0 ],
            [ 0.0, 0.0,               0.0,               -rt2 * eiph / 4.0 ],
            [ 0.0, 0.0,               0.0,               -eiph / 4.0       ],
            [ 0.0, 0.0,               0.0,               -1.0j * eiph / 4. ],
            [ 0.0, 0.0,               0.0,               1.0j * eiph / 4.0 ],
            [ 0.0, 0.0,               0.0,               -eiph / 4.0       ],
            [ 0.0, 0.0,               0.0,               -1.0/ 4.0         ],
            [ 0.0, 0.0,               0.0,               -1.0j / 4.0       ],
            [ 0.0, 0.0,               0.0,               -1.0j / 4.0       ],
            [ 0.0, 0.0,               0.0,               1.0 / 4.0         ],
        ]),
    )

sys.exit(0)

# ph = np.linspace(0.0, 2.0 * np.pi, 1000)
# P = np.array([probs(f, rho_iso) for f in ph]).T
#
# (
#     pd.Plotter()
#     .colorplot(
#         ph / np.pi,
#         np.arange(len(analysis_states), dtype=int),
#         P,
#         cmap=pd.colormaps["vibrant"],
#     )
#     .colorbar()
#     .set_xlabel("$\\varphi / \\pi$")
#     .set_ylabel("Measurement result")
#     .set_yticks(
#         np.arange(len(analysis_states), dtype=int),
#         analysis_states,
#         fontsize="xx-small",
#     )
#     .set_clabel("Probability")
#     .grid(False, which="both")
#     .savefig(datadir.joinpath("time_bin_analysis_probs.png"))
# )

