from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

datadir = Path("output/free_photon")
data = np.load(str(datadir.joinpath("data.npz")))
time = data["time"]
psi = data["psi"]

atom = ["G0", "G1", "E0", "E1"]
photon = [0, 1]
states = list(product(atom, photon))
probs = abs(psi)**2

P = pd.Plotter()
for state, state_prob in zip(states, probs):
    P.plot(time, state_prob, label=state)
(
    P
    .ggrid()
    .legend()
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
    .show()
)

