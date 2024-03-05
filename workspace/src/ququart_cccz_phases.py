from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

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
    [ # Δ/Ω, ξ, φ01, φ11
        [ 0.05, 0.960 * np.pi, 0.382,  0.140 ],
        [ 0.10, 0.918 * np.pi, 0.714,  0.362 ], # -0.042
        [ 0.15, 0.881 * np.pi, 1.050,  0.583 ], # -0.037
        [ 0.20, 0.846 * np.pi, 1.376,  0.801 ], # -0.035
        [ 0.25, 0.816 * np.pi, 1.685,  1.016 ], # -0.030
        [ 0.30, 0.790 * np.pi, 1.977,  1.228 ], # -0.026
        [ 0.35, 0.768 * np.pi, 2.251,  1.436 ], # -0.022
        [ 0.40, 0.750 * np.pi, 2.508,  1.639 ], # -0.018
        [ 0.45, 0.736 * np.pi, 2.747,  1.836 ], # -0.014
        [ 0.50, 0.725 * np.pi, 2.970,  2.028 ], # -0.011
    ],

    # Ω = 2 MHz
    [ # Δ/Ω, ξ, φ01, φ11
        [ 0.05, 0.960 * np.pi, 0.369,  0.056 ],
        [ 0.10, 0.918 * np.pi, 0.720,  0.279 ],  
        [ 0.15, 0.881 * np.pi, 1.057,  0.501 ],  
        [ 0.20, 0.846 * np.pi, 1.383,  0.721 ],  
        [ 0.25, 0.816 * np.pi, 1.691,  0.938 ],  
        [ 0.30, 0.790 * np.pi, 1.983,  1.152 ],  
        [ 0.35, 0.768 * np.pi, 2.257,  1.362 ],  
        [ 0.40, 0.750 * np.pi, 2.514,  1.568 ],  
        [ 0.45, 0.736 * np.pi, 2.753,  1.768 ],  
        [ 0.50, 0.725 * np.pi, 2.976,  1.962 ],  
    ],

    # Ω = 6 MHz          
    [ # Δ/Ω, ξ, φ01, φ11        
        [ 0.05, 0.960 * np.pi, 0.397, -0.278 ],  
        [ 0.10, 0.918 * np.pi, 0.748, -0.047 ],
        [ 0.15, 0.881 * np.pi, 1.084,  0.184 ],
        [ 0.20, 0.847 * np.pi, 1.409,  0.414 ],
        [ 0.25, 0.817 * np.pi, 1.717,  0.643 ],
        [ 0.30, 0.790 * np.pi, 2.010,  0.869 ],
        [ 0.35, 0.768 * np.pi, 2.285,  1.091 ],
        [ 0.40, 0.750 * np.pi, 2.541,  1.309 ],
        [ 0.45, 0.736 * np.pi, 2.780,  1.522 ],
        [ 0.50, 0.725 * np.pi, 3.003,  1.730 ],
    ]
])
rabi_freq = np.array([1.0, 2.0, 6.0])
det = data[:, :, 0]
xi = data[:, :, 1]
ph01 = data[:, :, 2]
ph11 = data[:, :, 3]

P = pd.Plotter()
(
    P
    .plot([], [], marker="o", linestyle="-", color="C0", label="$\\varphi_{01}$")
    .plot([], [], marker="o", linestyle="-", color="C1", label="$\\varphi_{11}$")
    .plot([], [], marker="o", linestyle="-", color="C3", label="$2 \\varphi_{01} - \\pi$")
    .plot([], [], marker="o", linestyle="-", color="C7", label="$\\xi$")
)
it = enumerate(zip(rabi_freq, det, xi, ph01, ph11))
for (k, (Wk, detk, xik, ph01k, ph11k)) in it:
    opacity = 1.0 - 0.5 * k / (len(rabi_freq) - 1)
    (
        P
        .plot(
            [], [],
            marker="o", linestyle="-", color="k", alpha=opacity,
            label=f"Ω/2π = {Wk:.1f} MHz",
        )
        .plot(
            detk, ph01k / np.pi,
            marker="o", linestyle="-", color="C0", alpha=opacity,
        )
        .plot(
            detk, ph11k / np.pi,
            marker="o", linestyle="-", color="C1", alpha=opacity,
        )
        .plot(
            detk, (2 * ph01k - np.pi) / np.pi,
            marker="o", linestyle="-", color="C3", alpha=opacity,
        )
        .plot(
            detk, xik / np.pi,
            marker="o", linestyle="-", color="C7", alpha=opacity,
        )
    )

(
    P
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("$\\Delta / \\Omega$")
    .set_ylabel("Phase [$\\pi$]")
    .savefig(outdir.joinpath("ququart_cccz_phases.png"))
)

pd.show()

