import numpy as np
import whooie.pyplotdefs as pd
from enum import IntEnum
pd.pp.rcParams["font.size"] = 5.0
FS = pd.pp.rcParams["figure.figsize"]

data = np.array([
    # Ω/2π (MHz), clock pop, atom pur, o pur, n pur, ground pur, clock pur, clock θ/π, clock φ/π, Δnbar
    [0.030, 0.851, 0.836572, 0.836572, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.531572],
    [0.040, 0.924, 0.913078, 0.913078, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.481596],
    [0.050, 0.969, 0.948867, 0.948867, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.277064],
    [0.060, 0.977, 0.955027, 0.955027, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.107625],
    [0.070, 0.975, 0.951032, 0.951032, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.030008],
    [0.080, 0.971, 0.944611, 0.944611, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.011175],
    [0.090, 0.968, 0.937930, 0.937930, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.021234],
    [0.100, 0.964, 0.931370, 0.931370, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.043404],
    [0.110, 0.961, 0.925019, 0.925019, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.069245],
    [0.120, 0.958, 0.918755, 0.918755, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.094949],
    [0.130, 0.954, 0.912607, 0.912607, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.118742],
    [0.140, 0.951, 0.906534, 0.906534, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.140017],
    [0.150, 0.947, 0.900554, 0.900554, 1.000000, 1.000000, 1.000000, 0.500000, 0.000000, 0.158780],
])

class PlotScale(IntEnum):
    Lin = 0
    Log = 1

dvar_labels = [
    ("1 - $P_{|C_0\\rangle} + P_{|C_1\\rangle}$", lambda x: 1 - x, PlotScale.Log),
    ("1 - $\\text{tr}(\\rho_{\\mathregular{atom}}^2)$", lambda x: 1 - x, PlotScale.Log),
    ("1 - $\\text{tr}(\\rho_{\\mathregular{o}}^2)$", lambda x: 1 - x, PlotScale.Log),
    ("1 - $\\text{tr}(\\rho_{\\mathregular{n}}^2)$", lambda x: 1 - x, PlotScale.Lin),
    ("1 - $\\text{tr}(\\rho_{\\mathregular{ground}}^2)$", lambda x: 1 - x, PlotScale.Lin),
    ("1 - $\\text{tr}(\\rho_{\\mathregular{clock}}^2)$", lambda x: 1 - x, PlotScale.Lin),
    ("$\\theta_{\\mathregular{clock}} / \\pi$", lambda x: x, PlotScale.Lin),
    ("$\\varphi_{\\mathregular{clock}} / \\pi$", lambda x: x, PlotScale.Lin),
    ("$\\Delta \\bar{n}$", lambda x: x, PlotScale.Log),
]
dvars = len(dvar_labels)
h_scale = dvars / 3.5

P = pd.Plotter.new(
    nrows=dvars,
    sharex=True,
    figsize=[FS[0], h_scale * FS[1]],
    as_plotarray=True,
)
for (k, (label, transform, scale)) in enumerate(dvar_labels):
    if scale == PlotScale.Lin:
        P[k].plot(
            data[:, 0], transform(data[:, k + 1]),
            marker="o", color=f"C{k % 10}",
        )
    elif scale == PlotScale.Log:
        P[k].semilogy(
            data[:, 0], transform(data[:, k + 1]),
            marker="o", color=f"C{k % 10}",
        )
    (
        P[k]
        .ggrid()
        # .tick_params(labelsize="xx-small")
        .set_ylabel(label)#, fontsize="xx-small")
    )
P[dvars - 1].set_xlabel("$\\Omega / 2 \\pi$ [MHz]")#, fontsize="xx-small")
P.savefig("output/ququart_clock_corpse_rabiscan_plot.png")
# pd.show()

