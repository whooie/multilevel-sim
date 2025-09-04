from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("quoct_mpp_rabi_scan.npz")
data = np.load(str(infile))
rabi = data["rabi"]
ndiff = data["ndiff"]
fidelity = data["fidelity"]

nmax = ndiff.shape[0] - 1
rabi0_ndiff = rabi[ndiff.mean(axis=0).argmin()]
rabi0_fidelity = rabi[fidelity.mean(axis=0).argmax()]

print(rabi0_ndiff)
print(rabi0_fidelity)

def qq(x):
    print(x)
    return x

P = pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)
# for legend
for m0 in range(nmax + 1):
    P[0].plot([], [], marker="", ls="-", c=f"C{m0 % 10}", label=f"$m_0 = {m0}$")
for (m0, (ndiff_m0, fidelity_m0)) in enumerate(zip(ndiff, fidelity)):
    P[0].semilogy(
        rabi, ndiff_m0,
        marker="", ls="-", c=f"C{m0 % 10}",
        dashes=[0, m0, 2, nmax - m0],
    )
    P[1].semilogy(
        rabi, 1 - fidelity_m0,
        marker="", ls="-", c=f"C{m0 % 10}",
    )
(
    P
    [0]
    .axvline(rabi0_ndiff, c="0.5")
    .ggrid().grid(False, which="both")
    .set_ylabel("$\\bar{m}' - m_0$")
    .legend(
        fontsize="small",
        frameon=False,
        # loc="upper left",
        loc="lower right",
        # bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    [1]
    .axvline(rabi0_fidelity, c="0.5")
    .ggrid().grid(False, which="both")
    .set_ylabel("$1 - P_{\\left| 1 0 m_0 \\right\\rangle}$")
    .set_xlabel("Rabi frequency [MHz]")
    .savefig(outdir.joinpath("quoct_mpp_rabi_scan.pdf"))
    .close()
)


