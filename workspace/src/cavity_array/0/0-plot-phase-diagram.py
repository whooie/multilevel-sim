from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output/cavity_array")
data = np.load(str(outdir.joinpath("phase_diagram_spinchain.npz")))
g = data["g"]
wz = data["wz"]
nbar = data["nbar"]
mz = data["mz"]

(
    pd.Plotter()
    .colorplot(g**2, wz, nbar, cmap=pd.colormaps["pix"])
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_xlabel("$g^2$")
    .set_ylabel("$\\omega_z$")
    .set_clabel("$\\langle a^\\dagger a \\rangle$")
    .savefig(outdir.joinpath("phase_diagram_nbar.png"))
    .close()
)

(
    pd.Plotter()
    .colorplot(g**2, wz, mz, cmap=pd.colormaps["hot-cold"])
    .colorbar()
    .ggrid().grid(False, which="both")
    .set_xlabel("$g^2$")
    .set_ylabel("$\\omega_z$")
    .set_clabel("$\\langle \\sigma^z \\rangle$")
    .savefig(outdir.joinpath("phase_diagram_mz.png"))
    .close()
)

