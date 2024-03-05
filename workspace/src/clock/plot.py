import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("clock.npz")

data = np.load(str(infile))
time = data["time"]
# psi = data["psi"]
rho = data["rho"]

atom_states = ["G0", "G1", "C0", "C1"]
states = ["∣" + s + "⟩" for s in atom_states]

P = pd.Plotter()
for (k, state) in enumerate(states):
    # P.plot(time, np.abs(psi[k, :])**2, label=state)
    P.plot(time, rho[k, k, :].real, label=state)
(
    P
    .ggrid()
    .legend(fontsize="xx-small")
    .show()
)

