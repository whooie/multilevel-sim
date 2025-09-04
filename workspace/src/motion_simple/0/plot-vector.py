from itertools import product
import numpy as np
import matplotlib as mpl
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

outdir = Path("output")
infile = outdir.joinpath("motion_simple.npz")

data = np.load(str(infile))
time = data["time"]
psi = data["psi"]
tmark = data["tmark"]

atom_states = ["G", "E"]
nfock = psi.shape[0] // len(atom_states)

P = pd.Plotter()
for t0 in tmark:
    P.axvline(t0, color="k")
for (k, (a, n)) in enumerate(product(atom_states, range(nfock))):
    P.plot(time, abs(psi[k, :]) ** 2, label=f"∣{a},{n}⟩")
(
    P
    .ggrid()
    .legend(
        fontsize="x-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_ylim(-0.05, 1.05)
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("motion_simple.png"))
)

# 0 :: g,0
# 1 :: g,1
# 2 :: g,2
# 3 :: e,0
# 4 :: e,1
# 5 :: e,2

## emswap 1
k_up = 3
s_up = "$|1n0\\rangle$"
k_dn = 1
s_dn = "$|0n1\\rangle$"
view = (10.0, 18.0)

## emswap 2
# k_up = 4
# s_up = "$|1n1\\rangle$"
# k_dn = 2
# s_dn = "$|0n2\\rangle$"
# view = (10.0, 18.0)

## emshelve 1
# k_up = 5
# s_up = "$|1n2\\rangle$"
# k_dn = 1
# s_dn = "$|0n1\\rangle$"
# view = (10.0, 18.0)

## emshelve 2
# k_up = 4
# s_up = "$|1n1\\rangle$"
# k_dn = 0
# s_dn = "$|0n0\\rangle$"
# view = (10.0, 18.0)

## emcz 1
# k_up = 3
# s_up = "$|1n0\\rangle$"
# k_dn = 1
# s_dn = "$|0n1\\rangle$"
# view = (10.0, 18.0)

## emcz 2
# k_up = 4
# s_up = "$|1n1\\rangle$"
# k_dn = 2
# s_dn = "$|0n2\\rangle$"
# view = (10.0, 18.0)

th = np.linspace(0.0, np.pi, 35)
ph = np.linspace(0.0, 2.0 * np.pi, 71)
TH, PH = np.meshgrid(th, ph)
X = np.sin(TH) * np.cos(PH)
Y = np.sin(TH) * np.sin(PH)
Z = np.cos(TH)
P = (
    pd.Plotter.new_3d(figsize=[2.0, 2.0])
    .plot_surface(X, Y, Z, alpha=0.15, lw=0.0, color="#f8f8f8")
    .plot(np.cos(ph), np.sin(ph), np.zeros(ph.shape), color="#e8e8e8", lw=0.5)
    .plot(np.sin(ph), np.zeros(ph.shape), np.cos(ph), color="#e8e8e8", lw=0.5)
    .plot(np.zeros(ph.shape), np.sin(ph), np.cos(ph), color="#e8e8e8", lw=0.5)
    .plot([-1, 1], [0, 0], [0, 0], color="#e8e8e8", ls=":", lw=0.5)
    .text(1.1, 0.0, 0.0, "$x$", ha="center", va="center", fontsize="x-small", color="#555")
    .plot([0, 0], [-1, 1], [0, 0], color="#e8e8e8", ls=":", lw=0.5)
    .text(0.0, 1.1, 0.0, "$y$", ha="center", va="center", fontsize="x-small", color="#555")
    .plot([0, 0], [0, 0], [-1, 1], color="#e8e8e8", ls=":", lw=0.5)
    .set_box_aspect((1.0, 1.0, 0.906))
    .axis("off")
    .text(0.0, 0.0, +1.1, s_up, ha="center", va="bottom", fontsize="small")
    .text(0.0, 0.0, -1.1, s_dn, ha="center", va="top", fontsize="small")
)

a_up = psi[k_up, ::10]
a_dn = psi[k_dn, ::10]
n = np.sqrt(abs(a_up) ** 2 + abs(a_dn) ** 2)
a_up /= n
a_dn /= n
b_up = a_up / abs(a_up)
b_dn = a_dn / abs(a_dn)

th = 2.0 * np.acos(abs(a_up))
ph = np.arctan2(b_dn.imag, b_dn.real) - np.arctan2(b_up.imag, b_up.real)
x = 1.01 * np.sin(th) * np.cos(ph)
y = 1.01 * np.sin(th) * np.sin(ph)
z = 1.01 * np.cos(th)
cc = mpl.colormaps["vibrant"](np.linspace(0.0, 1.0, a_up.shape[0] - 1))
for (c, xx, yy, zz) in zip(cc, zip(x, x[1:]), zip(y, y[1:]), zip(z, z[1:])):
    P.plot(xx, yy, zz, color=c)
P.view_init(azim=view[0], elev=view[1])
P.savefig(
    infile.with_stem(infile.stem + "_bloch").with_suffix(".pdf"),
    pad_inches=0.0,
)

# selector_G = np.array([
#     1.0 if a == "G" else 0.0
#     for (a, n) in product(atom_states, range(nfock))
# ])
# selector_E = np.array([
#     1.0 if a == "E" else 0.0
#     for (a, n) in product(atom_states, range(nfock))
# ])
# selector_nbar = np.array([
#     n
#     for (a, n) in product(atom_states, range(nfock))
# ])
#
# P_G = (abs(psi.T) ** 2) @ selector_G
# P_E = (abs(psi.T) ** 2) @ selector_E
# nbar = (abs(psi.T) ** 2) @ selector_nbar
#
# P = pd.Plotter()
# for t0 in tmark:
#     P.axvline(t0, color="k")
# (
#     P
#     .plot(time, P_G.real, label="∣g⟩")
#     .plot(time, P_E.real, label="∣e⟩")
#     .ggrid()
#     .legend(
#         fontsize="xx-small",
#         frameon=False,
#         loc="upper left",
#         bbox_to_anchor=(1.0, 1.0),
#         framealpha=1.0,
#     )
#     .set_ylim(-0.05, 1.05)
#     .set_xlabel("Time [μs]")
#     .set_ylabel("Probability")
# )
#
# P = pd.Plotter()
# for t0 in tmark:
#     P.axvline(t0, color="k")
# (
#     P
#     .plot(time, nbar.real, color="k")
#     .ggrid()
#     .set_xlabel("Time [μs]")
#     .set_ylabel("$\\bar{n}$")
# )

pd.show()

