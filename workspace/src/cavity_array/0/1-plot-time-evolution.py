from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

def qq(x):
    print(x)
    return x

def z_ord(natoms: int, mod: int, k0: int) -> str:
    return "".join("d" if k % mod == k0 else "u" for k in range(natoms))

def Sz(atom_state: str) -> int:
    return sum(+1 if a == "d" else -1 for a in atom_state)

outdir = Path("output/cavity_array")
data = np.load(str(outdir.joinpath("time_evolve_spinchain.npz")))
natoms = data["natoms"][0]
nmax = data["nmax"][0]

array_states = ["".join(atoms) for atoms in product(*(natoms * ["du"]))]
photon_states = list(range(nmax + 1))
states = list(product(array_states, photon_states))
print(f"#states = {len(states)}")

time = data["time"]
rho = data["rho"]
rho_diag = np.array([rho[n, n, :].real for n in range(len(states))])

nbar = rho_diag.T @ np.array([n for (_, n) in states])
mz = rho_diag.T @ np.array([Sz(s) / natoms for (s, _) in states])

(
    pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)
    [0]
    .plot(time, nbar, color="k")
    .ggrid()
    .set_ylabel("$\\langle a^\\dagger a \\rangle$")
    [1]
    .plot(time, mz, color="0.5")
    .ggrid()
    .set_ylabel("$\\langle \\sigma_z \\rangle$")
    .set_xlabel("Time")
    .savefig(outdir.joinpath("time_evolve_nbar_mz.png"))
    .close()
)

orderings = [
    ((m, k), z_ord(natoms, m, k))
    for (m, k) in [
        (1, -1),
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
]

P = pd.Plotter()
for ((m, k), z) in orderings:
    sel = np.array([1 if s == z else 0 for (s, _) in states])
    prob = rho_diag.T @ sel
    P.plot(time, prob, label=f"$P(\\mathbb{{Z}}_{{{m}}}^{{({k})}})$")
(
    P
    .ggrid()
    .legend()
    .set_xlabel("Time")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("time_evolve_states.png"))
    .close()
)

