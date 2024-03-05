from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("ququart_clock_pi2pi2_pulselen_scan.npz")

data = np.load(str(infile))
t0 = data["t0"]
t1 = data["t1"]
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
    .colorplot(t0, t1, P_G0.real.T)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣G0⟩")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(t0, t1, P_G1.real.T)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣G1⟩")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(t0, t1, P_C0.real.T)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣C0⟩")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("Probability")
)

(
    pd.Plotter()
    .colorplot(t0, t1, P_C1.real.T)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_title("∣C1⟩")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("Probability")
)

i0, j0 = np.unravel_index((P_C0.real * P_C1.real).argmax(), P_C0.shape)
print(
f"""
t0 = {t0[i0]:.6f}
t1 = {t1[j0]:.6f}
P_clock = {(P_C0[i0, j0] + P_C1[i0, j0]).real:.6f}
"""[1:-1]
)
(
    pd.Plotter()
    .colorplot(t0, t1, (P_C0.real * P_C1.real).T / 0.25)
    .colorbar()
    .plot([t0[i0]], [t1[j0]], marker="o", color="w")
    .ggrid().grid(False, which="both")
    .set_title("$P_{|C_0\\rangle} \\times P_{|C_1\\rangle} / 0.25$")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("Probability")
    .savefig(outdir.joinpath("ququart_clock_pi2pi2_pulselen_scan_c0c1.png"))
)

(
    pd.Plotter()
    .colorplot(t0, t1, nbar.real.T)
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_xlabel("Pulse 0 length [μs]")
    .set_ylabel("Pulse 1 length [μs]")
    .set_clabel("$\\bar{n}$")
)

pd.show()

