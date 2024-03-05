from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("ququart_clock_pipi_det2scan.npz")

data = np.load(str(infile))
det1 = data["det1"]
det2 = data["det2"]
rho = data["scanned"]
print(rho.shape)
rho_diag = rho[:, :, list(range(rho.shape[2])), list(range(rho.shape[3]))]

atom_states = [
    "G0",
    "G1",
    "C0",
    "C1",
]
nfock = rho.shape[2] // len(atom_states)

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

P_G0 = np.array([
    [scan_ij @ selector_G0 for scan_ij in scan_i] for scan_i in rho_diag
])
P_G1 = np.array([
    [scan_ij @ selector_G1 for scan_ij in scan_i] for scan_i in rho_diag
])
P_C0 = np.array([
    [scan_ij @ selector_C0 for scan_ij in scan_i] for scan_i in rho_diag
])
P_C1 = np.array([
    [scan_ij @ selector_C1 for scan_ij in scan_i] for scan_i in rho_diag
])
nbar = np.array([
    [scan_ij @ selector_nbar for scan_ij in scan_i] for scan_i in rho_diag
])

(
    pd.Plotter()
    .colorplot(det1, det2, P_G0.real)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣G0⟩")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(det1, det2, P_G1.real)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣G1⟩")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(det1, det2, P_C0.real)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣C0⟩")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(det1, det2, P_C1.real)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣C1⟩")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("Probability")
)

i0, j0 = np.unravel_index((P_C0.real * P_C1.real).argmax(), P_C0.shape)
(
    pd.Plotter()
    .colorplot(det1, det2, (P_C0.real * P_C1.real) / 0.25)
    .colorbar()
    .plot([det1[j0]], [det2[i0]], marker="o", color="w")
    .ggrid().grid(False, which="both")
    .set_title("$P_{|C_0\\rangle} \\times P_{|C_1\\rangle} / 0.25$")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("Probability")
    .savefig(outdir.joinpath("ququart_clock_pipi_det2scan_c0c1.png"))
)

(
    pd.Plotter()
    .colorplot(det1, det2, nbar.real)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_xlabel("Detuning 1 [MHz]")
    .set_ylabel("Detuning 2 [MHz]")
    .set_clabel("$\\bar{n}$")
)

pd.show()

