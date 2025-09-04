from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("quoct_mpp.npz")

data = np.load(str(infile))
time = data["time"]
psi = data["psi"]
tmark = data["tmark"]
x_avg = data["x_avg"]
p_avg = data["p_avg"]

atom_states = ["G0", "G1", "E0", "E1"]
nfock = psi.shape[0] // len(atom_states)

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
for (k, (a, n)) in enumerate(product(atom_states, range(nfock))):
    P.plot(time, np.abs(psi[k, :]) ** 2, label=f"∣{a},{n}⟩")
(
    P
    .ggrid()
    .llegend()
    .set_ylim(-0.05, 1.05)
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("quoct_mpp.png"))
    .close()
)

selector_G0 = np.array([
    1.0 if a == "G0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_G1 = np.array([
    1.0 if a == "G1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_E0 = np.array([
    1.0 if a == "E0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_E1 = np.array([
    1.0 if a == "E1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_nbar = np.array([
    n
    for (a, n) in product(atom_states, range(nfock))
])

psi2 = (np.abs(psi.T) ** 2).real
P_G0 = psi2 @ selector_G0
P_G1 = psi2 @ selector_G1
P_E0 = psi2 @ selector_E0
P_E1 = psi2 @ selector_E1
nbar = psi2 @ selector_nbar

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
(
    P
    .plot(time, P_G0, label="∣g0⟩")
    .plot(time, P_G1, label="∣g1⟩")
    .plot(time, P_E0, label="∣e0⟩")
    .plot(time, P_E1, label="∣e1⟩")
    .ggrid()
    .llegend()
    .set_ylim(-0.05, 1.05)
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
)

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
(
    P
    .plot(time, nbar.real, color="k")
    .ggrid()
    .set_xlabel("Time [μs]")
    .set_ylabel("$\\bar{n}$")
)

(
    pd.Plotter()
    .plot(x_avg.real, p_avg.real, marker="", ls="-")
    .plot(x_avg[0].real, p_avg[0].real, marker="o", ls="", c="r")
    .ggrid()
)

pd.show()


