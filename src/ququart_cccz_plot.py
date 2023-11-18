from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("ququart_cccz.npz")

data = np.load(str(infile))
time = data["time"]
psi = data["psi"]

atom_states = [
    "C0",
    "C1",
    "R0",
    "R1",
    "R2",
    "R3",
]
n_atoms = 2

states = [",".join(s) for s in product(*(n_atoms * [atom_states]))]
state_labels = ["∣" + s + "⟩" for s in states]

P = pd.Plotter()
for (k, state) in enumerate(state_labels):
    P.plot(time, np.abs(psi[k, :])**2, label=state)
(
    P
    .ggrid()
    .legend(fontsize="xx-small")
)

s0 = psi[states.index("C0,C1"), :]
s1 = psi[states.index("C0,R3"), :]
ph = np.log(s1 / s0).imag % (2 * np.pi)
P = pd.Plotter()
(
    P
    .plot(time[1:-2], ph[1:-2] / np.pi)
    .savefig("output/cccz_phase.png")
)

pd.show()

