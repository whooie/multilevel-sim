from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("ququart_clock_energies.npz")

data = np.load(str(infile))
B = data["B"]
g0 = data["G0"]
g1 = data["G1"]
c0 = data["C0"]
c1 = data["C1"]

(
    pd.Plotter()
    .plot(B, g0, label="∣G0⟩")
    .plot(B, g1, label="∣G1⟩")
    .plot(B, c0, label="∣C0⟩")
    .plot(B, c1, label="∣C1⟩")
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Magnetic field [G]")
    .set_ylabel("Shift [MHz]")
    .savefig(outdir.joinpath("ququart_clock_energies_shifts.png"))
)

(
    pd.Plotter()
    .plot(B, c0 - g0, label="E(∣C0⟩) - E(∣G0⟩)")
    .plot(B, c1 - g0, label="E(∣C1⟩) - E(∣G0⟩)")
    .plot(B, c0 - g1, label="E(∣C0⟩) - E(∣G1⟩)")
    .plot(B, c1 - g1, label="E(∣C1⟩) - E(∣G1⟩)")
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Magnetic field [G]")
    .set_ylabel("Diff. Shift [MHz]")
    .savefig(outdir.joinpath("ququart_clock_energies_diffshifts.png"))
)

pd.show()

