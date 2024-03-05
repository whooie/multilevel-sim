import numpy as np
import scipy.linalg as la
import whooie.pyplotdefs as pd

states = ["C0", "C1", "T0", "T1", "T2", "T3"]

# H = np.array([
#     [0.000e0+0.000e0j, 0.000e0+0.000e0j, 4.712e1-2.886e-15j, 5.441e1+0.000e0j, 2.721e1+1.666e-15j, 0.000e0+0.000e0j],
#     [0.000e0+0.000e0j, -7.540e-1+0.000e0j, 0.000e0+0.000e0j, 2.721e1-1.666e-15j, 5.441e1+0.000e0j, 4.712e1+2.886e-15j],
#     [4.712e1+2.886e-15j, 0.000e0+0.000e0j, 1.534e3+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j],
#     [5.441e1+0.000e0j, 2.721e1+1.666e-15j, 0.000e0+0.000e0j, 1.885e3+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j],
#     [2.721e1-1.666e-15j, 5.441e1+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 2.236e3+0.000e0j, 0.000e0+0.000e0j],
#     [0.000e0+0.000e0j, 4.712e1-2.886e-15j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 2.586e3+0.000e0j],
# ])

H = np.array(
[[0.000e0+0.000e0j, 0.000e0+0.000e0j, 2.059e1-3.881e-15j, 7.319e1+0.000e0j, 1.189e1+2.241e-15j, 0.000e0+0.000e0j],
 [0.000e0+0.000e0j, -5.027e-1+0.000e0j, 0.000e0+0.000e0j, 1.189e1-2.241e-15j, 7.319e1+0.000e0j, 2.059e1+3.881e-15j],
 [2.059e1+3.881e-15j, 0.000e0+0.000e0j, 1.651e3+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j],
 [7.319e1+0.000e0j, 1.189e1+2.241e-15j, 0.000e0+0.000e0j, 1.885e3+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j],
 [1.189e1-2.241e-15j, 7.319e1+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 2.119e3+0.000e0j, 0.000e0+0.000e0j],
 [0.000e0+0.000e0j, 2.059e1-3.881e-15j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 0.000e0+0.000e0j, 2.353e3+0.000e0j]]
)

E, V = la.eigh(H)

psi0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
c = la.solve(V, psi0)

t = np.linspace(0.0, 1.33333, 400000)
psi = np.array([
    np.dot(V, c * np.exp(-1j * E * tk))
    for tk in t
]).T

kmax = np.argmax(abs(psi[0, :])**2)
Pmax = np.max(abs(psi[0, :])**2)
P = pd.Plotter()
for (state, psi_state) in zip(states, psi):
    P.plot(t, abs(psi_state)**2, label=state)
(
    P
    .axvline(t[kmax], color="k")
    .axhline(Pmax, color="k")
    .set_xlabel("Time")
    .set_ylabel("Probability")
    .legend()
    .ggrid()
    .savefig("output/telecom-raman-test.png")
    .close()
)
print(Pmax)
print(1 - Pmax)
print(np.log10(1 - Pmax))

Vp = V.conj().T
U = np.dot(V, np.dot(np.diag(np.exp(-1j * E * t[kmax])), Vp))
final = np.dot(U, psi0)
F = abs(final[0])**2
print(F)
print(1 - F)
print(np.log10(1 - F))

