from pathlib import Path
import sys
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")

infile = outdir.joinpath("telecom-raman-single.npz")
data = np.load(str(infile))
time = data["time"]
psi = data["psi"]
states = [ "C0", "C1", "T0", "T1", "T2", "T3" ]
P = pd.Plotter()
for (state, psi_state) in zip(states, psi):
    P.plot(time, abs(psi_state)**2, label=state)
(
    P
    .set_xlabel("Time")
    .set_ylabel("Probability")
    .legend()
    .ggrid()
    .savefig(outdir.joinpath("telecom-raman-single.png"))
    .close()
)

infile = outdir.joinpath("telecom-raman.npz")
data = np.load(str(infile))
field = data["field"]
pol_angle = data["pol_angle"]
fidelity = data["fidelity"]
FIELD, POL_ANGLE = np.meshgrid(field, pol_angle)
P = pd.Plotter()
(
    P
    # .contourf(
    #     FIELD[:, 1:], POL_ANGLE[:, 1:] / np.pi, np.log10(1 - fidelity[:, 1:]),
    #     cmap=pd.colormaps["fire-ice"],
    # )
    .colorplot(
        field[1:], pol_angle / np.pi, np.log10(1 - fidelity[:, 1:]),
        cmap=pd.colormaps["fire-ice"],
    )
    .colorbar()
    .set_xlabel("Field [G]")
    .set_ylabel("θ/π")
    .set_clabel("$\\log_{10}(1 - P_{C_0})$")
    .ggrid().grid(False, which="both")
    .savefig(outdir.joinpath("telecom-raman-fidelity.png"))
    .close()
)

