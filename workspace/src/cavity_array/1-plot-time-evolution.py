from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output/cavity_array")
data = np.load(str(outdir.joinpath("time_evolve_spinchain.npz")))
natoms = data["natoms"][0]
nmax = data["nmax"][0]
time = data["time"]
nbar = data["nbar"]
szbar = data["szbar"]

array_states = ["".join(atoms) for atoms in product(*(natoms * ["du"]))]
photon_states = list(range(nmax + 1))
states = list(product(array_states, photon_states))
print(f"#states = {len(states)}")

(
    pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)
    [0]
    .plot(time, nbar, color="k")
    .ggrid()
    .set_ylabel("$\\langle a^\\dagger a \\rangle$")
    [1]
    .plot(time, szbar, color="0.5")
    .ggrid()
    .set_ylabel("$\\langle \\sigma_z \\rangle$")
    .set_xlabel("Time")
    .savefig(outdir.joinpath("time_evolve_nbar_szbar.png"))
    .close()
)

