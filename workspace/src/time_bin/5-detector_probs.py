from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

datadir = Path("output").joinpath("time_bin")
data = np.load(str(datadir.joinpath("final.npz")))
rho = data["rho"]

a = rho[4, 4]
b = rho[17, 17]
c = (a + b) / 2
rho[4, 17] = c
rho[17, 4] = c.conjugate()

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
mz_states = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]
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

def proj12(ph: float) -> np.ndarray:
    rt2 = np.sqrt(2)
    eiph = np.exp(1j * ph)
    return np.kron(
        np.eye(len(atom_states)),
        np.array([
            [ 1.0, 0.0,             0.0,             0.0               ],
            [ 0.0, 1j * eiph / 2.0, 0.0,             0.0               ],
            [ 0.0, 1j / 2.0,        1j * eiph / 2.0, 0.0               ],
            [ 0.0, 0.0,             1j / 2.0,        0.0               ],
            [ 0.0, eiph / 2.0,      0.0,             0.0               ],
            [ 0.0, -1.0 / 2.0,      eiph / 2.0,      0.0               ],
            [ 0.0, 0.0,             -1.0 / 2.0,      0.0               ],
            [ 0.0, 0.0,             0.0,             -eiph / 4.0       ],
            [ 0.0, 0.0,             0.0,             1j * eiph / 4.0   ],
            [ 0.0, 0.0,             0.0,             1j * eiph / 4.0   ],
            [ 0.0, 0.0,             0.0,             eiph / 4.0        ],
            [ 0.0, 0.0,             0.0,             -rt2 * eiph / 4.0 ],
            [ 0.0, 0.0,             0.0,             -rt2 * eiph / 4.0 ],
            [ 0.0, 0.0,             0.0,             -eiph / 4.0       ],
            [ 0.0, 0.0,             0.0,             -1j * eiph / 4.0  ],
            [ 0.0, 0.0,             0.0,             1j * eiph / 4.0   ],
            [ 0.0, 0.0,             0.0,             -eiph / 4.0       ],
            [ 0.0, 0.0,             0.0,             -1.0/ 4.0         ],
            [ 0.0, 0.0,             0.0,             -1j / 4.0         ],
            [ 0.0, 0.0,             0.0,             -1j / 4.0         ],
            [ 0.0, 0.0,             0.0,             1.0 / 4.0         ],
        ]),
    )

def rho_det(ph: float, rho: np.ndarray) -> np.ndarray:
    P12 = proj12(ph)
    return P12 @ rho @ P12.T.conj()

def det_probs(ph: float, rho: np.ndarray) -> np.ndarray:
    eipph = np.exp( 1j * ph)
    eimph = np.exp(-1j * ph)
    U = np.kron(
        np.array([
            [ 1.0,    eimph, 0.0, 0.0, 0.0, 0.0 ],
            [ -eipph, 1.0,   0.0, 0.0, 0.0, 0.0 ],
        ]) / np.sqrt(2),
        np.eye(len(det_states)),
    )
    detector = rho_det(ph, rho)
    rhox = U @ detector @ U.T.conj()
    return np.diag(rhox).real

single_photon_sel = np.array([
    [1.0 if el == mz else 0.0 for el in product(atom_states, photon_states)]
    for mz in product(atom_states, mz_states)
])
rho_reduced = single_photon_sel @ rho @ single_photon_sel.T

ph = np.linspace(0.0, 2 * np.pi, 1000)
probs = np.array([det_probs(f, rho_reduced) for f in ph]).T
print((probs <= 0.0).sum())
probs[np.where(probs <= 0.0)] = 1e-15

FS = pd.pp.rcParams["figure.figsize"]
(
    pd.Plotter.new(figsize=[FS[0], FS[1] * 2.0])
    .colorplot(
        ph / np.pi,
        np.arange(len(atom_states[:2]) * len(det_states), dtype=int),
        probs,
        cmap=pd.colormaps["vibrant"],
    )
    .colorbar()
    # .set_clim(-6, 0)
    .set_xlabel("$\\varphi / \\pi$")
    .set_ylabel("Measurement result")
    .set_yticks(
        list(range(len(atom_states[:2]) * len(det_states))),
        list(product(atom_states[:2], det_states)),
        fontsize="xx-small",
    )
    .set_clabel("Probability")
    .grid(False, which="both")
    .savefig(datadir.joinpath("time_bin_detector_probs.png"))
)


