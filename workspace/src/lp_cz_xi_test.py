import numpy as np
import whooie.pyplotdefs as pd

def ag(W: float, D: float, t: float) -> complex:
    Weff = np.sqrt(W**2 + D**2)
    return (
        np.exp(1j * D * t / 2) * (
            (Weff - D) / (2 * Weff) * np.exp(1j * Weff * t / 2)
            + (Weff + D) / (2 * Weff) * np.exp(-1j * Weff * t / 2)
        )
    )

def ae(W: float, D: float, xi: float, t: float) -> complex:
    Weff = np.sqrt(W**2 + D**2)
    return (
        -1j * np.exp(-1j * D * t / 2)
        * (Weff**2 - D**2) / (2 * W * Weff) * np.sin(Weff * t / 2)
    )

def phi(W: float, D: float, xi: float, t: float) -> float:
    Weff = np.sqrt(W**2 + D**2)
    T = (2 * np.pi) / Weff / 2
    if isinstance(t, np.ndarray):
        n = np.array([tk // T for tk in t])
    else:
        n = int(t // T)
    return (
        -np.pi / 2 - np.pi * (n % 2)
        - D * t
        - np.arctan(-D / Weff * np.tan(Weff * t / 2))
        - xi
    )

W = 2.0 * np.pi
D = 0.35 * W
xi = 0.768 * np.pi
xi = 0
t0 = 2.0 * np.pi / np.sqrt(W**2 + D**2)
t1 = 2.0 * np.pi / np.sqrt(2 * W**2 + D**2)

t = np.linspace(0.0, 2.0 * t0, 1000)[2:-2]
php = phi(W, +D, 0.0, t) % (2 * np.pi)
phm = phi(W, +D, xi, t) % (2 * np.pi)
ph = np.log(ae(W, D, xi, t) / ag(W, D, t)).imag % (2 * np.pi)
(
    pd.Plotter()
    # .plot(t, php / np.pi)
    # .plot(t, phm / np.pi)
    .plot(t, ph / np.pi)
    .axvline(t1, linestyle="--", color="k")
    .ggrid()
    .show()
    .close()
)



