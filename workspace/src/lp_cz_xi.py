from dataclasses import dataclass
import numpy as np
import whooie.pyplotdefs as pd
pd.pp.rcParams["figure.dpi"] = 400.0
from whooie.analysis import ExpVal
import lmfit

def model(
    params: lmfit.Parameters,
    x: np.ndarray,
) -> np.ndarray:
    a = params["a"].value
    x1 = params["x1"].value
    x2 = params["x2"].value
    y1 = params["y1"].value
    y2 = params["y2"].value
    return (
        (
            a*np.sqrt(
                (
                    -a**2
                    + x1**2
                    - 2*x1*x2
                    + x2**2
                    + y1**2
                    - 2*y1*y2
                    + y2**2
                ) * (
                    -a**2
                    + 4*x**2
                    - 4*x*x1
                    - 4*x*x2
                    + x1**2
                    + 2*x1*x2
                    + x2**2
                    + y1**2
                    - 2*y1*y2
                    + y2**2
                )
            ) * (a - y1 + y2) * (a + y1 - y2)
            + (
                a**2
                - y1**2
                + 2*y1*y2
                - y2**2
            ) * (
                a**2*y1
                + a**2*y2
                + 2*x*x1*y1
                - 2*x*x1*y2
                - 2*x*x2*y1
                + 2*x*x2*y2
                - x1**2*y1
                + x1**2*y2
                + x2**2*y1
                - x2**2*y2
                - y1**3
                + y1**2*y2
                + y1*y2**2
                - y2**3
            )
        ) / (
            2 * (a - y1 + y2) * (a + y1 - y2)
            * (
                a**2
                - y1**2
                + 2*y1*y2
                - y2**2
            )
        )
    )
    # return a * np.exp(-b * x) + c

def residuals(
    params: lmfit.Parameters,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    m = model(params, x)
    return (m - y)**2

@dataclass
class Fit:
    x: np.ndarray
    y: np.ndarray
    a: ExpVal
    x1: ExpVal
    y1: ExpVal
    x2: ExpVal
    y2: ExpVal
    valuestr: str
    xplot: np.ndarray
    yplot: np.ndarray

def do_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Fit:
    theta = np.arctan2(0.84 * np.pi, -1.0)
    params = lmfit.Parameters()
    params.add("m", value=np.tan(theta / 2))
    params.add("cx", value=0.35, min=0.3, max=0.4)
    params.add("cy", value=0.708 * np.pi, min=0.706 * np.pi, max=0.710 * np.pi)
    params.add("a", value=2.0)
    params.add("x1", value=0.20)
    params.add("y1", expr="m * (x1 - cx) + cy")
    params.add("x2", value=0.50)
    params.add("y2", expr="m * (x2 - cx) + cy")
    fit = lmfit.minimize(residuals, params, args=(x, y))
    if not fit.success:
        raise Exception("error in fit")

    a = ExpVal(fit.params["a"].value, fit.params["a"].stderr)
    x1 = ExpVal(fit.params["x1"].value, fit.params["x1"].stderr)
    y1 = ExpVal(fit.params["y1"].value, fit.params["y1"].stderr)
    x2 = ExpVal(fit.params["x2"].value, fit.params["x2"].stderr)
    y2 = ExpVal(fit.params["y2"].value, fit.params["y2"].stderr)
    valuestr = f"""
a = {a.value_str()}
x1 = {x1.value_str()}
y1 = {y1.value_str()}
x2 = {x2.value_str()}
y2 = {y2.value_str()}
"""[1:-1]

    xplot = np.linspace(x.min(), x.max(), 1000)
    yplot = model(fit.params, xplot)

    return Fit(x, y, a, x1, y1, x2, y2, valuestr, xplot, yplot)

def main():
    data = np.array([
        # Δ/Ω, ξ
        [ 0.05, 0.960 * np.pi ],
        [ 0.10, 0.918 * np.pi ],
        [ 0.15, 0.881 * np.pi ],
        [ 0.20, 0.847 * np.pi ],
        [ 0.25, 0.817 * np.pi ],
        [ 0.30, 0.790 * np.pi ],
        [ 0.35, 0.768 * np.pi ],
        [ 0.40, 0.750 * np.pi ],
        [ 0.45, 0.736 * np.pi ],
        [ 0.50, 0.725 * np.pi ],
        [ 0.55, 0.717 * np.pi ],
        [ 0.60, 0.712 * np.pi ],
        [ 0.65, 0.709 * np.pi ],
        [ 0.70, 0.708 * np.pi ],
        [ 0.75, 0.708 * np.pi ],
        # [ 0.80, 0.708 * np.pi ],
        # [ 0.85, 0.708 * np.pi ],
        # [ 0.90, 0.708 * np.pi ],
        # [ 0.95, 0.708 * np.pi ],
        # [ 1.00, 0.708 * np.pi ],
    ])

    fit = do_fit(data[:, 0], data[:, 1])
    (
        pd.Plotter()
        .axline((0.05, 0.960 * np.pi), (0.10, 0.918 * np.pi), color="k")
        .axhline(0.708 * np.pi, color="k")
        .plot(fit.xplot, fit.yplot, marker="", linestyle="-", color="C0")
        .plot(fit.x, fit.y, marker="o", linestyle="", color="C0")
        .plot([fit.x1.val], [fit.y1.val], marker="o", linestyle="", color="r")
        .plot([fit.x2.val], [fit.y2.val], marker="o", linestyle="", color="b")
        .ggrid()
        .text_ax(
            1.01, 1.00, fit.valuestr,
            ha="left", va="top", fontsize="x-small",
        )
        .set_xlabel("Δ/Ω")
        .set_ylabel("ξ")
        .show()
        .close()
    )

if __name__ == "__main__":
    main()


