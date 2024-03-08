from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd
pd.pp.rcParams["legend.handlelength"] = 3.0
FS = pd.pp.rcParams["figure.figsize"]
pd.pp.rcParams["figure.figsize"] = [FS[0], FS[1] * 0.65]
pd.set_font("/usr/share/fonts/OTF/MyriadPro-Regular.otf", "MyriadPro")

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
    # # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 2.4 μm
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.350, +0.353, +0.000, +0.000, +0.350, +0.353, +0.350, +0.350, +0.218, +0.219, +0.353, +0.353, +0.219, +0.222 ],
    #     [ 0.075, 0.938 * np.pi, +0.000, +0.000, +0.528, +0.527, +0.000, +0.000, +0.528, +0.527, +0.528, +0.528, +0.328, +0.327, +0.527, +0.527, +0.327, +0.332 ],
    #     [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.701, +0.700, +0.000, +0.000, +0.701, +0.700, +0.701, +0.701, +0.439, +0.450, +0.700, +0.700, +0.450, +0.442 ],
    #     [ 0.125, 0.899 * np.pi, +0.000, +0.000, +0.871, +0.870, +0.000, +0.000, +0.871, +0.870, +0.871, +0.871, +0.549, +0.554, +0.870, +0.870, +0.554, +0.551 ],
    #     [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.039, +1.039, +0.000, +0.000, +1.039, +1.039, +1.039, +1.039, +0.658, +0.663, +1.039, +1.039, +0.663, +0.584 ],
    #     [ 0.175, 0.863 * np.pi, +0.000, +0.000, +1.201, +1.203, +0.000, +0.000, +1.201, +1.203, +1.201, +1.201, +0.767, +0.771, +1.203, +1.203, +0.771, +0.775 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, +0.875, +0.879, +1.365, +1.365, +0.879, +0.881 ],
    #     [ 0.225, 0.831 * np.pi, +0.000, +0.000, +1.518, +1.520, +0.000, +0.000, +1.518, +1.520, +1.518, +1.518, +0.982, +0.986, +1.520, +1.520, +0.986, +0.988 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.671, +1.673, +0.000, +0.000, +1.671, +1.673, +1.671, +1.671, +1.089, +1.092, +1.673, +1.673, +1.092, +1.094 ],
    #     [ 0.275, 0.802 * np.pi, +0.000, +0.000, +1.821, +1.820, +0.000, +0.000, +1.821, +1.820, +1.821, +1.821, +1.194, +1.198, +1.820, +1.820, +1.198, +1.199 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, +1.299, +1.302, +1.965, +1.965, +1.302, +1.303 ],
    #     [ 0.325, 0.778 * np.pi, +0.000, +0.000, +2.104, +2.104, +0.000, +0.000, +2.104, +2.104, +2.104, +2.104, +1.402, +1.405, +2.104, +2.104, +1.405, +1.406 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, +1.504, +1.507, +2.237, +2.237, +1.507, +1.508 ],
    #     [ 0.375, 0.758 * np.pi, +0.000, +0.000, +2.370, +2.369, +0.000, +0.000, +2.370, +2.369, +2.370, +2.370, +1.605, +1.608, +2.369, +2.369, +1.608, +1.609 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, +1.705, +1.707, +2.496, +2.496, +1.707, +1.708 ],
    #     [ 0.425, 0.743 * np.pi, +0.000, +0.000, +2.617, +2.615, +0.000, +0.000, +2.617, +2.615, +2.617, +2.617, +1.803, +1.805, +2.615, +2.615, +1.805, +1.806 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, +1.900, +1.902, +2.735, +2.735, +1.902, +1.902 ],
    #     [ 0.475, 0.730 * np.pi, +0.000, +0.000, +2.848, +2.848, +0.000, +0.000, +2.848, +2.848, +2.848, +2.848, +1.994, +2.000, +2.848, +2.848, +2.000, +1.997 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +2.091, +2.092, +2.957, +2.957, +2.092, +2.091 ],
    # ],

    # # Ω = 3.0 MHz, Jeff's interaction strength, r_sep = 2.4 μm
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.347, +0.356, +0.000, +0.000, +0.347, +0.356, +0.347, +0.347, +0.218, +0.216, +0.356, +0.356, +0.216, +0.210 ],
    #     [ 0.075, 0.938 * np.pi, +0.000, +0.000, +0.525, +0.530, +0.000, +0.000, +0.525, +0.530, +0.525, +0.525, +0.329, +0.326, +0.530, +0.530, +0.326, +0.321 ],
    #     [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.703, +0.698, +0.000, +0.000, +0.703, +0.698, +0.703, +0.703, +0.440, +0.437, +0.698, +0.698, +0.437, +0.431 ],
    #     [ 0.125, 0.898 * np.pi, +0.000, +0.000, +0.876, +0.869, +0.000, +0.000, +0.876, +0.869, +0.876, +0.876, +0.550, +0.547, +0.869, +0.869, +0.547, +0.541 ],
    #     [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.037, +1.041, +0.000, +0.000, +1.037, +1.041, +1.037, +1.037, +0.659, +0.657, +1.041, +1.041, +0.657, +0.650 ],
    #     [ 0.175, 0.863 * np.pi, +0.000, +0.000, +1.200, +1.205, +0.000, +0.000, +1.200, +1.205, +1.200, +1.200, +0.768, +0.765, +1.205, +1.205, +0.765, +0.759 ],
    #     [ 0.200, 0.845 * np.pi, +0.000, +0.000, +1.370, +1.361, +0.000, +0.000, +1.370, +1.361, +1.370, +1.370, +0.877, +0.874, +1.361, +1.361, +0.874, +0.867 ],
    #     [ 0.225, 0.830 * np.pi, +0.000, +0.000, +1.518, +1.525, +0.000, +0.000, +1.518, +1.525, +1.518, +1.518, +0.984, +0.981, +1.525, +1.525, +0.981, +0.974 ],
    #     [ 0.250, 0.815 * np.pi, +0.000, +0.000, +1.677, +1.670, +0.000, +0.000, +1.677, +1.670, +1.677, +1.677, +1.091, +1.088, +1.670, +1.670, +1.088, +1.081 ],
    #     [ 0.275, 0.801 * np.pi, +0.000, +0.000, +1.821, +1.825, +0.000, +0.000, +1.821, +1.825, +1.821, +1.821, +1.197, +1.193, +1.825, +1.825, +1.193, +1.186 ],
    #     [ 0.300, 0.789 * np.pi, +0.000, +0.000, +1.970, +1.961, +0.000, +0.000, +1.970, +1.961, +1.970, +1.970, +1.302, +1.298, +1.961, +1.961, +1.298, +1.291 ],
    #     [ 0.325, 0.777 * np.pi, +0.000, +0.000, +2.103, +2.109, +0.000, +0.000, +2.103, +2.109, +2.103, +2.103, +1.405, +1.402, +2.109, +2.109, +1.402, +1.394 ],
    #     [ 0.350, 0.766 * np.pi, +0.000, +0.000, +2.247, +2.237, +0.000, +0.000, +2.247, +2.237, +2.247, +2.247, +1.508, +1.504, +2.237, +2.237, +1.504, +1.496 ],
    #     [ 0.375, 0.758 * np.pi, +0.000, +0.000, +2.371, +2.369, +0.000, +0.000, +2.371, +2.369, +2.371, +2.371, +1.609, +1.605, +2.369, +2.369, +1.605, +1.597 ],
    #     [ 0.400, 0.748 * np.pi, +0.000, +0.000, +2.497, +2.501, +0.000, +0.000, +2.497, +2.501, +2.497, +2.497, +1.709, +1.704, +2.501, +2.501, +1.704, +1.697 ],
    #     [ 0.425, 0.740 * np.pi, +0.000, +0.000, +2.625, +2.617, +0.000, +0.000, +2.625, +2.617, +2.625, +2.625, +1.808, +1.803, +2.617, +2.617, +1.803, +1.795 ],
    #     [ 0.450, 0.735 * np.pi, +0.000, +0.000, +2.742, +2.731, +0.000, +0.000, +2.742, +2.731, +2.742, +2.742, +1.905, +1.900, +2.731, +2.731, +1.900, +1.892 ],
    #     [ 0.475, 0.729 * np.pi, +0.000, +0.000, +2.851, +2.849, +0.000, +0.000, +2.851, +2.849, +2.851, +2.851, +2.000, +1.995, +2.849, +2.849, +1.995, +1.987 ],
    #     [ 0.500, 0.722 * np.pi, +0.000, +0.000, +2.960, +2.963, +0.000, +0.000, +2.960, +2.963, +2.960, +2.960, +2.094, +2.089, +2.963, +2.963, +2.089, +2.081 ],
    # ],

    # Ω = 3.0 MHz, Jeff's interaction strength, r_sep = 3.6840314986404 μm
    [ # Δ/Ω, ξ, φ0000, φ0001, ...
        [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.347, +0.356, +0.000, +0.000, +0.347, +0.356, +0.347, +0.347, +0.211, +0.200, +0.356, +0.356, +0.200, +0.195 ],
        [ 0.075, 0.938 * np.pi, +0.000, +0.000, +0.525, +0.530, +0.000, +0.000, +0.525, +0.530, +0.525, +0.525, +0.322, +0.311, +0.530, +0.530, +0.311, +0.306 ],
        [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.704, +0.699, +0.000, +0.000, +0.704, +0.699, +0.704, +0.704, +0.433, +0.422, +0.699, +0.699, +0.422, +0.416 ],
        [ 0.125, 0.897 * np.pi, +0.000, +0.000, +0.877, +0.870, +0.000, +0.000, +0.877, +0.870, +0.877, +0.877, +0.543, +0.532, +0.870, +0.870, +0.532, +0.526 ],
        [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.038, +1.042, +0.000, +0.000, +1.038, +1.042, +1.038, +1.038, +0.653, +0.641, +1.042, +1.042, +0.641, +0.636 ],
        [ 0.175, 0.863 * np.pi, +0.000, +0.000, +1.200, +1.205, +0.000, +0.000, +1.200, +1.205, +1.200, +1.200, +0.762, +0.750, +1.205, +1.205, +0.750, +0.745 ],
        [ 0.200, 0.845 * np.pi, +0.000, +0.000, +1.370, +1.361, +0.000, +0.000, +1.370, +1.361, +1.370, +1.370, +0.870, +0.859, +1.361, +1.361, +0.859, +0.853 ],
        [ 0.225, 0.830 * np.pi, +0.000, +0.000, +1.518, +1.525, +0.000, +0.000, +1.518, +1.525, +1.518, +1.518, +0.978, +0.966, +1.525, +1.525, +0.966, +0.960 ],
        [ 0.250, 0.815 * np.pi, +0.000, +0.000, +1.677, +1.670, +0.000, +0.000, +1.677, +1.670, +1.677, +1.677, +1.085, +1.073, +1.670, +1.670, +1.073, +1.067 ],
        [ 0.275, 0.801 * np.pi, +0.000, +0.000, +1.821, +1.825, +0.000, +0.000, +1.821, +1.825, +1.821, +1.821, +1.190, +1.179, +1.825, +1.825, +1.179, +1.172 ],
        [ 0.300, 0.789 * np.pi, +0.000, +0.000, +1.970, +1.961, +0.000, +0.000, +1.970, +1.961, +1.970, +1.970, +1.295, +1.284, +1.961, +1.961, +1.284, +1.277 ],
        [ 0.325, 0.777 * np.pi, +0.000, +0.000, +2.103, +2.109, +0.000, +0.000, +2.103, +2.109, +2.103, +2.103, +1.399, +1.387, +2.109, +2.109, +1.387, +1.381 ],
        [ 0.350, 0.766 * np.pi, +0.000, +0.000, +2.247, +2.237, +0.000, +0.000, +2.247, +2.237, +2.247, +2.247, +1.502, +1.490, +2.237, +2.237, +1.490, +1.483 ],
        [ 0.375, 0.758 * np.pi, +0.000, +0.000, +2.371, +2.369, +0.000, +0.000, +2.371, +2.369, +2.371, +2.371, +1.603, +1.591, +2.369, +2.369, +1.591, +1.584 ],
        [ 0.400, 0.748 * np.pi, +0.000, +0.000, +2.497, +2.501, +0.000, +0.000, +2.497, +2.501, +2.497, +2.497, +1.703, +1.691, +2.501, +2.501, +1.691, +1.684 ],
        [ 0.425, 0.740 * np.pi, +0.000, +0.000, +2.625, +2.617, +0.000, +0.000, +2.625, +2.617, +2.625, +2.625, +1.802, +1.790, +2.617, +2.617, +1.790, +1.782 ],
        [ 0.450, 0.735 * np.pi, +0.000, +0.000, +2.742, +2.731, +0.000, +0.000, +2.742, +2.731, +2.742, +2.742, +1.899, +1.887, +2.731, +2.731, +1.887, +1.879 ],
        [ 0.475, 0.729 * np.pi, +0.000, +0.000, +2.851, +2.849, +0.000, +0.000, +2.851, +2.849, +2.851, +2.851, +1.995, +1.982, +2.849, +2.849, +1.982, +1.975 ],
        [ 0.500, 0.722 * np.pi, +0.000, +0.000, +2.960, +2.963, +0.000, +0.000, +2.960, +2.963, +2.960, +2.960, +2.089, +2.077, +2.963, +2.963, +2.077, +2.069 ],
    ],

    # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 2.4 μm
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.960 * np.pi, +0.000, +0.000, +0.348, +0.351, +0.000, +0.000, +0.348, +0.351, +0.348, +0.348, +0.229, +0.229, +0.351, +0.351, +0.229, +0.225 ],
    #     [ 0.100, 0.919 * np.pi, +0.000, +0.000, +0.699, +0.699, +0.000, +0.000, +0.699, +0.699, +0.699, +0.699, +0.450, +0.450, +0.699, +0.699, +0.450, +0.446 ],
    #     [ 0.150, 0.881 * np.pi, +0.000, +0.000, +1.037, +1.037, +0.000, +0.000, +1.037, +1.037, +1.037, +1.037, +0.669, +0.669, +1.037, +1.037, +0.669, +0.665 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, +0.885, +0.885, +1.365, +1.365, +0.885, +0.881 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.670, +1.673, +0.000, +0.000, +1.670, +1.673, +1.670, +1.670, +1.098, +1.098, +1.673, +1.673, +1.098, +1.095 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, +1.305, +1.305, +1.965, +1.965, +1.305, +1.304 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, +1.502, +1.504, +2.237, +2.237, +1.504, +1.497 ],
    #     [ 0.376, 0.758 * np.pi, +0.000, +0.000, +2.374, +2.375, +0.000, +0.000, +2.374, +2.375, +2.374, +2.374, +1.631, +1.622, +2.375, +2.375, +1.622, +1.613 ],
    #     [ 0.379, 0.757 * np.pi, +0.000, +0.000, +2.389, +2.390, +0.000, +0.000, +2.389, +2.390, +2.389, +2.389, +1.641, +1.634, +2.390, +2.390, +1.634, +1.625 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, +1.716, +1.716, +2.496, +2.496, +1.716, +1.708 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, +1.924, +1.909, +2.735, +2.735, +1.909, +1.901 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +2.104, +2.107, +2.957, +2.957, +2.107, +2.087 ],
    # ],

    # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 6.25 μm
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.350, +0.353, +0.000, +0.000, +0.350, +0.353, +0.350, +0.350, +0.123, +0.119, +0.353, +0.353, +0.119, +0.112 ],
    #     [ 0.100, 0.919 * np.pi, +0.000, +0.000, +0.699, +0.699, +0.000, +0.000, +0.699, +0.699, +0.699, +0.699, +0.345, +0.341, +0.699, +0.699, +0.341, +0.334 ],
    #     [ 0.150, 0.881 * np.pi, +0.000, +0.000, +1.037, +1.037, +0.000, +0.000, +1.037, +1.037, +1.037, +1.037, +0.566, +0.562, +1.037, +1.037, +0.562, +0.555 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, +0.785, +0.780, +1.365, +1.365, +0.780, +0.774 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.670, +1.673, +0.000, +0.000, +1.670, +1.673, +1.670, +1.670, +1.001, +0.996, +1.673, +1.673, +0.996, +0.989 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, +1.213, +1.208, +1.965, +1.965, +1.208, +1.202 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, +1.421, +1.416, +2.237, +2.237, +1.416, +1.410 ],
    #     [ 0.363, 0.763 * np.pi, +0.000, +0.000, +2.306, +2.307, +0.000, +0.000, +2.306, +2.307, +2.306, +2.306, +1.474, +1.470, +2.307, +2.307, +1.470, +1.463 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, +1.624, +1.620, +2.496, +2.496, +1.620, +1.613 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, +1.823, +1.818, +2.735, +2.735, +1.818, +1.811 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +2.015, +2.010, +2.957, +2.957, +2.010, +2.003 ],
    # ],

    # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 6.25 μm, altered interaction strength ratios
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.350, +0.353, +0.000, +0.000, +0.350, +0.353, +0.350, +0.350, +0.021, +0.119, +0.353, +0.353, +0.119, +0.079 ],
    #     [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.701, +0.701, +0.000, +0.000, +0.701, +0.701, +0.701, +0.701, +0.245, +0.341, +0.701, +0.701, +0.341, +0.301 ],
    #     [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.039, +1.039, +0.000, +0.000, +1.039, +1.039, +1.039, +1.039, +0.467, +0.562, +1.039, +1.039, +0.562, +0.523 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, +0.688, +0.780, +1.365, +1.365, +0.780, +0.742 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.670, +1.673, +0.000, +0.000, +1.670, +1.673, +1.670, +1.670, +0.906, +0.996, +1.673, +1.673, +0.996, +0.959 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, +1.121, +1.208, +1.965, +1.965, +1.208, +1.172 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, +1.333, +1.416, +2.237, +2.237, +1.416, +1.381 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, +1.539, +1.620, +2.496, +2.496, +1.620, +1.585 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, +1.740, +1.818, +2.735, +2.735, +1.818, +1.784 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +1.936, +2.010, +2.957, +2.957, +2.010, +1.977 ],
    # ],

    # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 12.0 μm
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.350, +0.353, +0.000, +0.000, +0.350, +0.353, +0.350, +0.350, +2.519, +2.542, +0.353, +0.353, +2.542, +2.525 ],
    #     [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.701, +0.701, +0.000, +0.000, +0.701, +0.701, +0.701, +0.701, +3.138, -3.129, +0.701, +0.701, -3.129, +3.136 ],
    #     [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.039, +1.039, +0.000, +0.000, +1.039, +1.039, +1.039, +1.039, -2.574, -2.559, +1.039, +1.039, -2.559, -2.576 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, -2.046, -2.031, +1.365, +1.365, -2.031, -2.044 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.670, +1.673, +0.000, +0.000, +1.670, +1.673, +1.670, +1.670, -1.554, -1.540, +1.673, +1.673, -1.540, -1.552 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, -1.095, -1.084, +1.965, +1.965, -1.084, -1.095 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, -0.665, -0.659, +2.237, +2.237, -0.659, -0.672 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, -0.270, -0.262, +2.496, +2.496, -0.262, -0.271 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, +0.103, +0.109, +2.735, +2.735, +0.109, +0.101 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +0.454, +0.457, +2.957, +2.957, +0.457, +0.446 ],
    # ],

    # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 12.0 μm, altered interaction strength ratios
    # [ # Δ/Ω, ξ, φ0000, φ0001, ...
    #     [ 0.050, 0.959 * np.pi, +0.000, +0.000, +0.350, +0.353, +0.000, +0.000, +0.350, +0.353, +0.350, +0.350, +1.628, +2.542, +0.353, +0.353, +2.542, +2.096 ],
    #     [ 0.100, 0.918 * np.pi, +0.000, +0.000, +0.701, +0.701, +0.000, +0.000, +0.701, +0.701, +0.701, +0.701, +2.305, -3.129, +0.701, +0.701, -3.129, +2.741 ],
    #     [ 0.150, 0.880 * np.pi, +0.000, +0.000, +1.039, +1.039, +0.000, +0.000, +1.039, +1.039, +1.039, +1.039, +2.945, -2.559, +1.039, +1.039, -2.559, -2.934 ],
    #     [ 0.200, 0.846 * np.pi, +0.000, +0.000, +1.362, +1.365, +0.000, +0.000, +1.362, +1.365, +1.362, +1.362, -2.740, -2.031, +1.365, +1.365, -2.031, -2.365 ],
    #     [ 0.250, 0.816 * np.pi, +0.000, +0.000, +1.670, +1.673, +0.000, +0.000, +1.670, +1.673, +1.670, +1.670, -2.177, -1.540, +1.673, +1.673, -1.540, -1.837 ],
    #     [ 0.300, 0.790 * np.pi, +0.000, +0.000, +1.963, +1.965, +0.000, +0.000, +1.963, +1.965, +1.963, +1.963, -1.650, -1.084, +1.965, +1.965, -1.084, -1.346 ],
    #     [ 0.350, 0.768 * np.pi, +0.000, +0.000, +2.239, +2.237, +0.000, +0.000, +2.239, +2.237, +2.239, +2.239, -1.156, -0.659, +2.237, +2.237, -0.659, -0.894 ],
    #     [ 0.400, 0.750 * np.pi, +0.000, +0.000, +2.494, +2.496, +0.000, +0.000, +2.494, +2.496, +2.494, +2.494, -0.705, -0.262, +2.496, +2.496, -0.262, -0.465 ],
    #     [ 0.450, 0.736 * np.pi, +0.000, +0.000, +2.734, +2.735, +0.000, +0.000, +2.734, +2.735, +2.734, +2.734, -0.282, +0.109, +2.735, +2.735, +0.109, -0.070 ],
    #     [ 0.500, 0.725 * np.pi, +0.000, +0.000, +2.959, +2.957, +0.000, +0.000, +2.959, +2.957, +2.959, +2.959, +0.115, +0.457, +2.957, +2.957, +0.457, +0.295 ],
    # ],

    # Ω = 1 MHz
    # [ # Δ/Ω, ξ, φ0001, φ0010, ...
    #     [ 0.050, 0.960 * np.pi, -0.000, +0.000, +0.347, +0.352, +0.000, +0.000, +0.347, +0.352, +0.347, +0.347, +0.145, +0.137, +0.352, +0.352, +0.137, +0.127 ],
    #     [ 0.100, 0.919 * np.pi, -0.000, -0.000, +0.696, +0.701, -0.000, +0.000, +0.696, +0.702, +0.696, +0.696, +0.367, +0.359, +0.701, +0.702, +0.359, +0.349 ],
    #     [ 0.150, 0.881 * np.pi, -0.000, -0.000, +1.038, +1.036, -0.000, +0.000, +1.038, +1.036, +1.038, +1.038, +0.587, +0.579, +1.036, +1.036, +0.579, +0.570 ],
    #     [ 0.200, 0.847 * np.pi, -0.000, -0.000, +1.363, +1.360, -0.000, +0.000, +1.363, +1.360, +1.363, +1.363, +0.806, +0.798, +1.360, +1.360, +0.798, +0.788 ],
    #     [ 0.250, 0.817 * np.pi, -0.000, -0.000, +1.668, +1.672, -0.000, +0.000, +1.668, +1.672, +1.668, +1.668, +1.021, +1.013, +1.672, +1.672, +1.013, +1.003 ],
    #     [ 0.300, 0.791 * np.pi, -0.000, +0.000, +1.961, +1.963, +0.000, +0.000, +1.961, +1.963, +1.961, +1.961, +1.233, +1.225, +1.963, +1.963, +1.225, +1.215 ],
    #     [ 0.350, 0.768 * np.pi, -0.000, +0.000, +2.238, +2.239, +0.000, +0.000, +2.238, +2.239, +2.238, +2.238, +1.441, +1.433, +2.239, +2.239, +1.433, +1.423 ],
    #     [ 0.400, 0.750 * np.pi, -0.000, +0.000, +2.495, +2.496, +0.000, +0.000, +2.495, +2.496, +2.495, +2.495, +1.644, +1.635, +2.496, +2.496, +1.635, +1.626 ],
    #     [ 0.450, 0.737 * np.pi, -0.000, +0.000, +2.734, +2.732, +0.000, +0.000, +2.733, +2.733, +2.734, +2.733, +1.841, +1.833, +2.732, +2.733, +1.833, +1.823 ],
    #     [ 0.500, 0.725 * np.pi, -0.000, +0.000, +2.957, +2.959, +0.000, +0.000, +2.957, +2.959, +2.957, +2.957, +2.033, +2.024, +2.959, +2.959, +2.024, +2.015 ],
    # ],
])
rabi_freq = np.array([0.63])
det = data[:, :, 0]
xi = data[:, :, 1]

opt = np.array([ # Δ/Ω, ξ, φ0000, φ0001, ...
    # # Ω = 0.63 MHz, Jeff's interaction strength, r_sep = 2.4 μm
    # [ 0.3767, 0.7577 * np.pi, +0.000, +0.000, +2.377, +2.379, +0.000, +0.000, +2.377, +2.379, +2.377, +2.377, +1.612, +1.615, +2.379, +2.379, +1.615, +1.615 ],

    # # Ω = 3.0 MHz, Jeff's interaction strength, r_sep = 2.4 μm
    # [ 0.376, 0.757 * np.pi, +0.000, +0.000, +2.377, +2.373, +0.000, +0.000, +2.377, +2.373, +2.377, +2.377, +1.613, +1.608, +2.373, +2.373, +1.608, +1.601 ],
    
    # Ω = 3.0 MHz, Jeff's interaction strength, r_sep = 3.6840314986404 μm
    [ 0.374, 0.758 * np.pi, +0.000, +0.000, +2.366, +2.364, +0.000, +0.000, +2.366, +2.364, +2.366, +2.366, +1.599, +1.587, +2.364, +2.364, +1.587, +1.580 ],
])

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
    return [6, 1] + [
        *([0, 2] if binstr[0] == "0" else [1, 1]),
        *([0, 2] if binstr[1] == "0" else [1, 1]),
        *([0, 2] if binstr[2] == "0" else [1, 1]),
        *([0, 2] if binstr[3] == "0" else [1, 1]),
    ]

P = pd.Plotter()
(
    P
    # .plot([], [], marker="", linestyle="-", color="C7", label="$\\xi$")
)
it = enumerate(zip(rabi_freq, det, xi, opt))
for (k, (Wk, detk, xik, optk)) in it:
    opacity = 1.0
    # opacity = 1.0 - 0.5 * k / (len(rabi_freq) - 1)
    # (
    #     P
    #     # .plot(
    #     #     [], [],
    #     #     marker="o", linestyle="-", color="k", alpha=opacity,
    #     #     label=f"Ω/2π = {Wk:.1f} MHz",
    #     # )
    #     .plot(
    #         detk, xik / np.pi,
    #         marker="o", linestyle="-", color="C7", alpha=opacity,
    #     )
    #     .plot(
    #         [optk[0]], [optk[1] / np.pi],
    #         marker="o", linestyle="-", color="r", alpha=opacity,
    #     )
    # )
    for (j, statej) in enumerate(state_labels):
        # P.plot(
        #     detk, data[k, :, j + 2] / np.pi,
        #     marker="o", linestyle="-", alpha=opacity,
        #     dashes=dashes(j),
        #     label=f"$\\varphi_{{{statej}}}$",
        # )

        count_C = statej.count("C")
        if count_C == 1:
            (
                P
                .plot(
                    detk, (2 * data[k, :, j + 2] - np.pi) / np.pi,
                    marker="", linestyle="-", color="C0", alpha=opacity,
                    dashes=dashes(j),
                    label=f"2φ$_{{\\mathregular{{{statej}}}}}$ - π",
                )
                .plot(
                    detk, (2 * data[k, :, j + 2] - np.pi) / np.pi,
                    marker="o", linestyle="-", color="C0", alpha=opacity,
                    dashes=dashes(j),
                )
                .plot(
                    [optk[0]], [(2 * optk[j + 2] - np.pi) / np.pi],
                    marker="o", linestyle="", color="r", alpha=opacity,
                )
            )
        elif count_C == 2:
            (
                P
                .plot(
                    detk, data[k, :, j + 2] / np.pi,
                    marker="", linestyle="-", color="C1", alpha=opacity,
                    dashes=dashes(j),
                    label=f"φ$_{{\\mathregular{{{statej}}}}}$",
                )
                .plot(
                    detk, data[k, :, j + 2] / np.pi,
                    marker="o", linestyle="-", color="C1", alpha=opacity,
                    dashes=dashes(j),
                )
                .plot(
                    [optk[0]], [optk[j + 2] / np.pi],
                    marker="o", linestyle="", color="r", alpha=opacity,
                )
            )
        else:
            (
                P
                .plot(
                    detk, data[k, :, j + 2] / np.pi,
                    marker="", linestyle="-", color="C3", alpha=opacity,
                    dashes=dashes(j),
                    label=f"φ$_{{\\mathregular{{{statej}}}}}$",
                )
                .plot(
                    detk, data[k, :, j + 2] / np.pi,
                    marker="o", linestyle="-", color="C3", alpha=opacity,
                    dashes=dashes(j),
                )
                .plot(
                    [optk[0]], [optk[j + 2] / np.pi],
                    marker="o", linestyle="", color="r", alpha=opacity,
                )
            )
(
    P
    .ggrid().grid(False, which="both")
	.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.025),
        ncol=1,
        columnspacing=1.0,
        fontsize="xx-small",
        framealpha=1.0,
        edgecolor="k",
    )
    .set_xlabel("Δ/Ω")
    .set_ylabel("Acc. phase [π]")
    # .set_xlim(0.357, 0.369)
    # .set_ylim(0.43, 0.50)
    .savefig(outdir.joinpath("ququart_cz_phases.png"))
    .savefig(outdir.joinpath("ququart_cz_phases.pdf"))
)

# pd.show()
