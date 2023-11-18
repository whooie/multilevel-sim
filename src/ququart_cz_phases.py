from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd
pd.pp.rcParams["legend.handlelength"] = 3.5

outdir = Path("output")
# infile = outdir.joinpath("ququart_cccz_phases.npz")
# data = np.load(str(infile))
# det = data["det"]
# ph01 = data["ph01"]
# ph11 = data["ph11"]

# (
#     pd.Plotter()
#     .plot(det, ph01, label="$\\varphi_{01}$")
#     .plot(det, ph11, label="$\\varphi_{11}$")
#     .plot(det, 2 * ph01 - np.pi, label="$2 \\varphi_{01} - \\pi$")
#     .ggrid()
#     .legend(fontsize="xx-small")
#     .set_xlabel("$\\Delta / \\Omega$")
#     .set_ylabel("Phase [rad]")
# )

data = np.array([
    # Ω = 1 MHz
    [ # Δ/Ω, ξ, φ0001, φ0010, ...
        [ 0.050, 0.960 * np.pi, -0.000, +0.000, +0.347, +0.352, +0.000, +0.000, +0.347, +0.352, +0.347, +0.347, +0.145, +0.137, +0.352, +0.352, +0.137, +0.127 ],
        [ 0.100, 0.919 * np.pi, -0.000, -0.000, +0.696, +0.701, -0.000, +0.000, +0.696, +0.702, +0.696, +0.696, +0.367, +0.359, +0.701, +0.702, +0.359, +0.349 ],
        [ 0.150, 0.881 * np.pi, -0.000, -0.000, +1.038, +1.036, -0.000, +0.000, +1.038, +1.036, +1.038, +1.038, +0.587, +0.579, +1.036, +1.036, +0.579, +0.570 ],
        [ 0.200, 0.847 * np.pi, -0.000, -0.000, +1.363, +1.360, -0.000, +0.000, +1.363, +1.360, +1.363, +1.363, +0.806, +0.798, +1.360, +1.360, +0.798, +0.788 ],
        [ 0.250, 0.817 * np.pi, -0.000, -0.000, +1.668, +1.672, -0.000, +0.000, +1.668, +1.672, +1.668, +1.668, +1.021, +1.013, +1.672, +1.672, +1.013, +1.003 ],
        [ 0.300, 0.791 * np.pi, -0.000, +0.000, +1.961, +1.963, +0.000, +0.000, +1.961, +1.963, +1.961, +1.961, +1.233, +1.225, +1.963, +1.963, +1.225, +1.215 ],
        [ 0.350, 0.768 * np.pi, -0.000, +0.000, +2.238, +2.239, +0.000, +0.000, +2.238, +2.239, +2.238, +2.238, +1.441, +1.433, +2.239, +2.239, +1.433, +1.423 ],
        [ 0.400, 0.750 * np.pi, -0.000, +0.000, +2.495, +2.496, +0.000, +0.000, +2.495, +2.496, +2.495, +2.495, +1.644, +1.635, +2.496, +2.496, +1.635, +1.626 ],
        [ 0.450, 0.737 * np.pi, -0.000, +0.000, +2.734, +2.732, +0.000, +0.000, +2.733, +2.733, +2.734, +2.733, +1.841, +1.833, +2.732, +2.733, +1.833, +1.823 ],
        [ 0.500, 0.725 * np.pi, -0.000, +0.000, +2.957, +2.959, +0.000, +0.000, +2.957, +2.959, +2.957, +2.957, +2.033, +2.024, +2.959, +2.959, +2.024, +2.015 ],
    ],

    # # Ω = 2 MHz
    # [ # Δ/Ω, ξ, φ0001, φ0010, ...
    # ],
    #
    # # Ω = 6 MHz          
    # [ # Δ/Ω, ξ, φ0001, φ0010, ...
    # ]
])
rabi_freq = np.array([1.0])
det = data[:, :, 0]
xi = data[:, :, 1]

state_labels = [
    "G0G0",
    "G0G1",
    "G0C0",
    "G0C1",
    "G1G0",
    "G1G1",
    "G1C0",
    "G1C1",
    "C0G0",
    "C0G1",
    "C0C0",
    "C0C1",
    "C1G0",
    "C1G1",
    "C1C0",
    "C1C1",
]

def dashes(idx: int) -> list[float]:
    binstr = f"{bin(idx)[2:]:>04}"
    return [
        *([0, 2] if binstr[0] == "0" else [1, 1]),
        *([0, 2] if binstr[1] == "0" else [1, 1]),
        *([0, 2] if binstr[2] == "0" else [1, 1]),
        *([0, 2] if binstr[3] == "0" else [1, 1]),
    ] + [10, 1]

P = pd.Plotter()
(
    P
    .plot([], [], marker="o", linestyle="-", color="C7", label="$\\xi$")
)
it = enumerate(zip(rabi_freq, det, xi))
for (k, (Wk, detk, xik)) in it:
    opacity = 1.0
    # opacity = 1.0 - 0.5 * k / (len(rabi_freq) - 1)
    (
        P
        .plot(
            [], [],
            marker="o", linestyle="-", color="k", alpha=opacity,
            label=f"Ω/2π = {Wk:.1f} MHz",
        )
        .plot(
            detk, xik / np.pi,
            marker="o", linestyle="-", color="C7", alpha=opacity,
        )
    )
    for (j, statej) in enumerate(state_labels):
        # P.plot(
        #     detk, data[k, :, j + 2] / np.pi,
        #     marker="o", linestyle="-", alpha=opacity,
        #     dashes=dashes(j),
        #     label=f"$\\varphi_{{{statej}}}$",
        # )

        count_C = statej.count("C")
        if count_C == 1:
            P.plot(
                detk, (2 * data[k, :, j + 2] - np.pi) / np.pi,
                marker="o", linestyle="-", alpha=opacity,
                dashes=dashes(j),
                label=f"$2 \\varphi_{{{statej}}} - \\pi$",
            )
        else:
            P.plot(
                detk, data[k, :, j + 2] / np.pi,
                marker="o", linestyle="-", alpha=opacity,
                dashes=dashes(j),
                label=f"$\\varphi_{{{statej}}}$",
            )

(
    P
    .ggrid()
    .legend(fontsize="xx-small")
	.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        columnspacing=1.0,
        fontsize="xx-small",
        framealpha=1.0,
        edgecolor="k",
    )
    .set_xlabel("$\\Delta / \\Omega$")
    .set_ylabel("Phase [$\\pi$]")
    .set_xlim(0.35, 0.385)
    .set_ylim(0.4, 0.55)
    .savefig(outdir.joinpath("ququart_cz_phases.png"))
)

pd.show()

