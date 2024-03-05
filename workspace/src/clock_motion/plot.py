from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("clock_motion.npz")

data = np.load(str(infile))
time = data["time"]
rho = data["rho"]
rho_diag = rho[list(range(rho.shape[0])), list(range(rho.shape[0])), :]

atom_states = ["G0", "G1", "C0", "C1"]
nfock = rho.shape[0] // len(atom_states)

selector_G0 = np.array([
    1.0 if a == "G0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_G1 = np.array([
    1.0 if a == "G1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_C0 = np.array([
    1.0 if a == "C0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_C1 = np.array([
    1.0 if a == "C1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_nbar = np.array([
    n
    for (a, n) in product(atom_states, range(nfock))
])

P_G0 = rho_diag.T @ selector_G0
P_G1 = rho_diag.T @ selector_G1
P_C0 = rho_diag.T @ selector_C0
P_C1 = rho_diag.T @ selector_C1
nbar = rho_diag.T @ selector_nbar

P = pd.Plotter()
(
    P
    .plot(time, P_G0.real, label="∣G0⟩")
    .plot(time, P_G1.real, label="∣G1⟩")
    .plot(time, P_C0.real, label="∣C0⟩")
    .plot(time, P_C1.real, label="∣C1⟩")
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
)

P = pd.Plotter()
(
    P
    .plot(time, nbar.real, color="k")
    .ggrid()
    .set_xlabel("Time [μs]")
    .set_ylabel("$\\bar{n}$")
)

pd.show()

