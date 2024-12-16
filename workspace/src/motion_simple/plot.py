from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("motion_simple.npz")

data = np.load(str(infile))
time = data["time"]
rho = data["rho"]
rho_diag = rho[list(range(rho.shape[0])), list(range(rho.shape[0])), :]
tmark = data["tmark"]

atom_states = ["G", "E"]
nfock = rho.shape[0] // len(atom_states)

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
for (k, (a, n)) in enumerate(product(atom_states, range(nfock))):
    P.plot(time, rho_diag[k, :].real, label=f"∣{a},{n}⟩")
(
    P
    .ggrid()
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_ylim(-0.05, 1.05)
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("motion_simple.png"))
)

selector_G = np.array([
    1.0 if a == "G" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_E = np.array([
    1.0 if a == "E" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_nbar = np.array([
    n
    for (a, n) in product(atom_states, range(nfock))
])

P_G = rho_diag.T @ selector_G
P_E = rho_diag.T @ selector_E
nbar = rho_diag.T @ selector_nbar

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
(
    P
    .plot(time, P_G.real, label="∣g⟩")
    .plot(time, P_E.real, label="∣e⟩")
    .ggrid()
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
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

pd.show()

