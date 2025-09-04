from itertools import product
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
import whooie.pyplotdefs as pd

outdir = Path("output")
infiles = [
    (outdir.joinpath(s), v)
    for (s, v) in [
        ("emcz.npz", (-8.0, 15.0)),
        ("emshelve.npz", (55.0, 15.0)),
        ("emswap.npz", (-130.0, 15.0)),
        ("enmccz.npz", (20.0, 15.0)),
        # ("emswap_he.npz", (125.0, 20.0)),
    ]
]

def gen_bloch_sphere(
    s_up: str,
    s_dn: str,
    view: tuple[float, float],
) -> pd.Plotter:
    th = np.linspace(0.0, np.pi, 35)
    ph = np.linspace(0.0, 2.0 * np.pi, 71)
    TH, PH = np.meshgrid(th, ph)
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    thv = np.pi / 2 - view[1] / 180 * np.pi
    phv = view[0] / 180 * np.pi
    nview = np.array([
        np.sin(thv) * np.cos(phv),
        np.sin(thv) * np.sin(phv),
        np.cos(thv),
    ])
    s0 = np.array([
        -np.sin(phv),
        np.cos(phv),
        0.0,
    ])
    s1 = np.array([
        -np.cos(thv) * np.cos(phv),
        -np.cos(thv) * np.sin(phv),
        np.sin(thv),
    ])
    circx = np.cos(ph) * s0[0] + np.sin(ph) * s1[0]
    circy = np.cos(ph) * s0[1] + np.sin(ph) * s1[1]
    circz = np.cos(ph) * s0[2] + np.sin(ph) * s1[2]
    circ = Poly3DCollection([list(zip(circx, circy, circz))], facecolor="#a8a8a8", lw=0.0, alpha=0.35, zorder=0)
    P = (
        pd.Plotter.new_3d(figsize=[2.0, 2.0])
        .plot_surface(X, Y, Z, alpha=0.15, lw=0.0, color="#f8f8f8", zorder=0)
        # .add_collection3d(circ)
        .plot(np.cos(ph), np.sin(ph), np.zeros(ph.shape), color="#e8e8e8", lw=0.5, zorder=0)
        .plot(np.sin(ph), np.zeros(ph.shape), np.cos(ph), color="#e8e8e8", lw=0.5, zorder=0)
        .plot(np.zeros(ph.shape), np.sin(ph), np.cos(ph), color="#e8e8e8", lw=0.5, zorder=0)
        .plot([-1, 1], [0, 0], [0, 0], color="#e8e8e8", ls=":", lw=0.5, zorder=0)
        .text(1.1, 0.0, 0.0, "$x$", ha="center", va="center", fontsize="x-small", color="#555", zorder=200)
        .plot([0, 0], [-1, 1], [0, 0], color="#e8e8e8", ls=":", lw=0.5, zorder=0)
        .text(0.0, 1.1, 0.0, "$y$", ha="center", va="center", fontsize="x-small", color="#555", zorder=200)
        .plot([0, 0], [0, 0], [-1, 1], color="#e8e8e8", ls=":", lw=0.5, zorder=0)
        .set_box_aspect((1.0, 1.0, 0.906))
        .axis("off")
        .text(0.0, 0.0, +1.1, s_up, ha="center", va="bottom", fontsize="small", zorder=200)
        .text(0.0, 0.0, -1.1, s_dn, ha="center", va="top", fontsize="small", zorder=200)
    )
    return P

def plot_bloch_sphere(
    P: pd.Plotter,
    a_up: np.ndarray[complex, 1],
    a_dn: np.ndarray[complex, 1],
    view: tuple[float, float],
) -> pd.Plotter:
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
    nview = np.array([
        np.cos(view[1] / 180 * np.pi) * np.cos(view[0] / 180 * np.pi),
        np.cos(view[1] / 180 * np.pi) * np.sin(view[0] / 180 * np.pi),
        np.sin(view[1] / 180 * np.pi),
    ])
    # print(view, nview)
    projs = list()
    for (c, xx, yy, zz) in zip(cc, zip(x, x[1:]), zip(y, y[1:]), zip(z, z[1:])):
        npoint = np.array([
            np.mean(xx),
            np.mean(yy),
            np.mean(zz),
        ])
        proj = nview @ npoint
        # print(proj)
        projs.append(proj)
        P.plot(xx, yy, zz, color=c, zorder=proj)
    # pd.Plotter().plot(projs).show().close()
    # print()
    return P

def process_gate(infile: Path, view: tuple[float, float]):
    data = np.load(str(infile))
    time = data["time"] # :: { t }
    psi = data["psi"] # :: { run, state, t }
    tmark = data["tmark"] # :: { run, mark }
    k_up = data["k_up"] # :: { run }
    s_up = ["".join(chr(b) for b in s) for s in data["s_up"]] # :: { run }
    k_dn = data["k_dn"] # :: { run }
    s_dn = ["".join(chr(b) for b in s) for s in data["s_dn"]] # :: { run }
    # P = pd.Plotter.new(
    #     nrows=psi.shape[0], sharex=True, as_plotarray=True, figsize=[2.25, 1.5])
    P = pd.Plotter.new_gridspec(
        dict(
            nrows=psi.shape[0] + 1,
            height_ratios=psi.shape[0] * [1] + [0.1],
            hspace=0.30,
        ),
        [pd.S[k] for k in range(psi.shape[0] + 1)],
        # shareax={k: {"x": 0} for k in range(1, psi.shape[0] + 1)},
        figsize=[2.25, 1.65],
    ).to_plotarray()
    it = enumerate(zip(psi, tmark, k_up, s_up, k_dn, s_dn))
    for (j, (psi_j, tmarks_j, kj_up, sj_up, kj_dn, sj_dn)) in it:
        a_up = psi_j[kj_up]
        a_dn = psi_j[kj_dn]
        for t in tmarks_j:
            P[j].axvline(t, color="0.35")
        (
            P[j]
            .plot(time, abs(a_dn) ** 2, label=sj_dn, clip_on=False, zorder=100)
            .plot(time, abs(a_up) ** 2, label=sj_up, clip_on=False, zorder=100)
            .ggrid().grid(False, which="both")
            .legend(
                fontsize="medium",
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                framealpha=1.0,
            )
            .tick_params(labelsize="medium")
            .set_xlim(time[0], time[-1])
            .set_xticks([], [])
            .set_ylim(0, 1)
            .set_yticks([0, 1], ["0", "1"])
        )
        sphere = gen_bloch_sphere(sj_up, sj_dn, view)
        plot_bloch_sphere(sphere, a_up[::10], a_dn[::10], view)
        sphere.view_init(azim=view[0], elev=view[1])
        sphere.savefig(
            infile.with_stem(infile.stem + f"_bloch_{j}").with_suffix(".pdf"))
    (
        P[psi.shape[0]]
        .imshow(
            np.array([time]),
            cmap="vibrant",
            interpolation="gaussian",
            aspect="auto",
            extent=[time[0] / 1000, time[-1] / 1000, 0, 1],
        )
        .ggrid().grid(False, which="both")
        .set_yticks([], [])
    )
    # pd.show()
    (
        P[0]
        .supylabel("Probability", fontsize="medium", x=-0.01)
        [psi.shape[0]]
        .set_xlabel("Time [ms]", fontsize="medium")
        .savefig(infile.with_stem(infile.stem + "_probs").with_suffix(".pdf"))
    )

for (infile, view) in infiles:
    print(infile)
    process_gate(infile, view)

