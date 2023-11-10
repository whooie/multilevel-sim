from itertools import product
from math import log
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("rydberg_generic.npz")

data = np.load(str(infile))
time = data["time"]
psi = data["psi"]

atom_states = ["G0", "G1", "G2", "R"]
n_atoms = int(round(log(psi.shape[0], len(atom_states))))

states = [
    "∣" + ",".join(s) + "⟩"
    for s in product(*(n_atoms * [atom_states]))
]

P = pd.Plotter()
for (k, state) in enumerate(states):
    P.plot(time, np.abs(psi[k, :])**2, label=state)
(
    P
    .ggrid()
    .legend(fontsize="xx-small")
    .show()
)

