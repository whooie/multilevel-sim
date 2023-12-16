from itertools import product
import numpy as np
import whooie.pyplotdefs as pd
from pathlib import Path
import sys

def qq(X):
    print(X)
    return X

outdir = Path("output")
infile = outdir.joinpath("ququart_clock_pi2pi2.npz")

data = np.load(str(infile))
time = data["time"]
rho = data["rho"]
rho_diag = rho[list(range(rho.shape[0])), list(range(rho.shape[0])), :]

atom_states = [
    "G0",
    "G1",
    "C0",
    "C1",
]
nfock = rho.shape[0] // len(atom_states)

selector_G0 = np.array([
    1.0 if a == "G0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_G1 = np.array([
    1.0 if a == "G1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_C0 = np.array([
    1.0 if a == "C0" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_C1 = np.array([
    1.0 if a == "C1" else 0.0
    for (a, n) in product(atom_states, range(nfock))
])
selector_nbar = np.array([
    n
    for (a, n) in product(atom_states, range(nfock))
])

P_G0 = rho_diag.T @ selector_G0
P_G1 = rho_diag.T @ selector_G1
P_C0 = rho_diag.T @ selector_C0
P_C1 = rho_diag.T @ selector_C1
nbar = rho_diag.T @ selector_nbar

# k_pipulse = P_C1.argmax()
# k_pipulse = P_C0.argmax()
k_pipulse = (P_C0 * P_C1).argmax()

P = pd.Plotter()
(
    P
    .plot(time, P_G0.real, label="∣G0⟩")
    .plot(time, P_G1.real, label="∣G1⟩")
    .plot(time, P_C0.real, label="∣C0⟩")
    .plot(time, P_C1.real, label="∣C1⟩")
    .axvline(time[k_pipulse], color="C7")
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Time [μs]")
    .set_ylabel("Probability")
    .savefig(outdir.joinpath("ququart_clock_pi2pi2_probs.png"))
)

P = pd.Plotter()
(
    P
    .plot(time, nbar.real, color="k")
    .axvline(time[k_pipulse], color="C7")
    .ggrid()
    .set_xlabel("Time [μs]")
    .set_ylabel("$\\bar{n}$")
    .savefig(outdir.joinpath("ququart_clock_pi2pi2_nbar.png"))
)

rho_pipulse = rho[:, :, k_pipulse]

clock_population = (P_C0[k_pipulse] + P_C1[k_pipulse]).real

rho_atom = sum(
    (
        Tn := np.kron(
            np.eye(4),
            np.array([[1.0 if k == n else 0.0 for k in range(nfock)]]).T
        )
    ).T @ rho_pipulse @ Tn
    for n in range(nfock)
)
purity = np.diag(rho_atom @ rho_atom).sum().real

def multikron(*arrs: np.ndarray) -> np.ndarray:
    """
    Compute the tensor product of multiple matrices (since np.kron only takes
    two arguments)
    """
    if len(arrs) == 0:
        return np.array([[1]])
    else:
        return np.kron(arrs[0], multikron(*arrs[1:]))

def trace_qubit(n: int, k: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute the partial trace over the k-th qubit (indexed from zero) out of n
    total qubits for a given density matrix rho.
    """
    T0 = multikron(
        np.eye(2 ** k), np.array([[1, 0]]).T, np.eye(2 ** (n - k - 1))
    )
    T1 = multikron(
        np.eye(2 ** k), np.array([[0, 1]]).T, np.eye(2 ** (n - k - 1))
    )
    return (T0.T @ rho @ T0) + (T1.T @ rho @ T1)

def isolate_qubit(n: int, k: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute the density matrix for only the k-th qubit (indexed from zero) out
    of n total qubits by progressively tracing out all others.
    """
    if n == 1:
        return rho
    elif k == 0:
        return isolate_qubit(n - 1, k, trace_qubit(n, n - 1, rho))
    else:
        return isolate_qubit(n - 1, k - 1, trace_qubit(n, 0, rho))

rho_o = isolate_qubit(2, 0, rho_atom)
purity_o = np.diag(rho_o @ rho_o).sum().real

rho_n = isolate_qubit(2, 1, rho_atom)
purity_n = np.diag(rho_n @ rho_n).sum().real

rho_ground = (
    (PG := np.kron(np.array([[1.0, 0.0]]).T, np.eye(2))).T @ rho_atom @ PG
)
rho_ground /= np.diag(rho_ground).sum()
purity_ground = np.diag(rho_ground @ rho_ground).sum().real

rho_clock = (
    (PC := np.kron(np.array([[0.0, 1.0]]).T, np.eye(2))).T @ rho_atom @ PC
)
rho_clock /= np.diag(rho_clock).sum()
purity_clock = np.diag(rho_clock @ rho_clock).sum().real

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

alpha = np.array([
    np.diag(sigma_x @ rho_clock).sum().real,
    np.diag(sigma_y @ rho_clock).sum().real,
    np.diag(sigma_z @ rho_clock).sum().real,
])
alpha /= np.sqrt((alpha**2).sum())

theta = np.arctan2(np.sqrt(alpha[0]**2 + alpha[1]**2), alpha[2])
phi = np.arctan2(alpha[1], alpha[0])

print(f"clock population    = {clock_population:.6f}")
print(f"atom state purity   = {purity:.6f}")
print(f"o-qubit purity      = {purity_o:.6f}")
print(f"n-qubit purity      = {purity_n:.6f}")
print(f"ground state purity = {purity_ground:.6f}")
print(f"clock state purity  = {purity_clock:.6f}")
print(f"clock theta         = {theta / np.pi:.6f}π")
print(f"clock phi           = {phi / np.pi:.6f}π")

pd.show()

