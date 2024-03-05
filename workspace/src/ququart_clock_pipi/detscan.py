from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("ququart_clock_pipi_detscan.npz")

data = np.load(str(infile))
det = data["det"]
rho = data["scanned"]
rho_diag = rho[:, list(range(rho.shape[1])), list(range(rho.shape[1]))]

atom_states = [
    "G0",
    "G1",
    "C0",
    "C1",
]
nfock = rho.shape[1] // len(atom_states)

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

P_G0 = rho_diag @ selector_G0
P_G1 = rho_diag @ selector_G1
P_C0 = rho_diag @ selector_C0
P_C1 = rho_diag @ selector_C1
nbar = rho_diag @ selector_nbar

P = pd.Plotter()
(
    P
    .plot(det, P_G0.real, label="∣G0⟩")
    .plot(det, P_G1.real, label="∣G1⟩")
    .plot(det, P_C0.real, label="∣C0⟩")
    .plot(det, P_C1.real, label="∣C1⟩")
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Detuning [MHz]")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("ququart_clock_pipi_detscan_probs.png"))
)

P = pd.Plotter()
(
    P
    .plot(det, nbar.real, color="k")
    .ggrid()
    .set_xlabel("Detuning [MHz]")
    .set_ylabel("$\\bar{n}$")
    .savefig(outdir.joinpath("ququart_clock_pipi_detscan_nbar.png"))
)

pd.show()

