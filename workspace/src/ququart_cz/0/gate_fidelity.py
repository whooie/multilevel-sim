from itertools import product
import re
import sys
import numpy as np
import sympy as sy
import whooie.pyplotdefs as pd
FS = pd.pp.rcParams["figure.figsize"]
pd.pp.rcParams["figure.figsize"] = [FS[0], FS[1] * 0.65]
pd.set_font("/usr/share/fonts/OTF/MyriadPro-Regular.otf", "MyriadPro")

try:
    mode = int(sys.argv[1])
except:
    print("missing integer mode argument")
    sys.exit(1)

if mode == 0:
    ph_ex = sy.symbols(" ".join(f"phi{bin(k)[2:]:>04}" for k in range(16)), real=True)
    ph0, ph1 = sy.symbols("phi0 phi1", real=True)

    atom_states = ["G0", "G1", "C0", "C1"]
    CZ = sy.diag([
        sy.S(1) if "".join(ss).count("C") == 0
        else sy.exp(sy.I * ph0) if "".join(ss).count("C") == 1
        else sy.exp(sy.I * (2 * ph0 - sy.pi))
        for ss in product(atom_states, atom_states)
    ], unpack=True)
    G = sy.Matrix([
        # 0000
        [+1.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0001
        [+0.000000+0.000000*sy.I, +1.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0010
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.228890+0.973373*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0011
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.229392+0.973232*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0100
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +1.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0101
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +1.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0110
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.228890+0.973373*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 0111
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.229392+0.973232*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1000
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.228890+0.973373*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1001
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.228890+0.973373*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1010
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.883510+0.458277*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1011
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.674400+0.733998*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1100
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.229392+0.973232*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1101
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.229392+0.973232*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I],
        # 1110
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, -0.674400+0.733998*sy.I, +0.000000+0.000000*sy.I],
        # 1111
        [+0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.000000+0.000000*sy.I, +0.897583+0.429540*sy.I],
        ]).T

    def clean_expr_str(expr: sy.Expr) -> str:
        s = str(expr.simplify())
        s = re.sub("I", "1j", s)
        s = re.sub(r"(sin|cos|exp|sqrt)", r"np.\1", s)
        return s

    def as_pyfunc(expr: sy.Expr, func_name: str) -> str:
        return (
f"""
def {func_name}(phi0: float) -> float:
    return (
        {clean_expr_str(expr)}
    ).real
"""[1:-1]
        )

    expr = abs(sy.trace(CZ.H * G)) / 16
    dexpr = expr.diff(ph0)
    d2expr = dexpr.diff(ph0)
    print(as_pyfunc(expr, "F"))
    print()
    print(as_pyfunc(dexpr, "dF"))
    print()
    print(as_pyfunc(d2expr, "d2F"))
elif mode == 1:
    def F(phi0: float) -> float:
        return (
            0.578675166630648*np.sqrt(((-0.0201710515112456*np.exp(2*1j*phi0) + 0.109923652184657*1j*np.exp(2*1j*phi0) - 0.290268756692487*np.exp(1j*phi0) - 0.452960143556552*1j*np.exp(1j*phi0) + 1 + 5.18036744029743e-18*1j)*np.exp(2*1j*phi0) + (-0.290268756692487 + 0.452960143556552*1j)*np.exp(1j*phi0) - 0.0201710515112456 - 0.109923652184657*1j)*np.exp(-2*1j*phi0))
        ).real

    def dF(phi0: float) -> float:
        return (
            ((-0.063610087744606*np.exp(1j*phi0) - 0.0116724865943854*1j*np.exp(1j*phi0) + 0.131058393274815 - 0.083985660573348*1j)*np.exp(3*1j*phi0) + (0.131058393274815 + 0.083985660573348*1j)*np.exp(1j*phi0) - 0.063610087744606 + 0.0116724865943854*1j)*np.exp(-2*1j*phi0)/np.sqrt(((-0.0201710515112456*np.exp(2*1j*phi0) + 0.109923652184657*1j*np.exp(2*1j*phi0) - 0.290268756692487*np.exp(1j*phi0) - 0.452960143556552*1j*np.exp(1j*phi0) + 1 + 5.18036744029743e-18*1j)*np.exp(2*1j*phi0) + (-0.290268756692487 + 0.452960143556552*1j)*np.exp(1j*phi0) - 0.0201710515112456 - 0.109923652184657*1j)*np.exp(-2*1j*phi0))
        ).real

    def d2F(phi0: float) -> float:
        return (
            np.sqrt(((-0.0201710515112456*np.exp(2*1j*phi0) + 0.109923652184657*1j*np.exp(2*1j*phi0) - 0.290268756692487*np.exp(1j*phi0) - 0.452960143556552*1j*np.exp(1j*phi0) + 1 + 5.18036744029743e-18*1j)*np.exp(2*1j*phi0) + (-0.290268756692487 + 0.452960143556552*1j)*np.exp(1j*phi0) - 0.0201710515112456 - 0.109923652184657*1j)*np.exp(-2*1j*phi0))*(((0.0233449731887709*np.exp(1j*phi0) - 0.127220175489212*1j*np.exp(1j*phi0) + 0.083985660573348 + 0.131058393274815*1j)*np.exp(3*1j*phi0) + (0.083985660573348 - 0.131058393274815*1j)*np.exp(1j*phi0) + 0.0233449731887709 + 0.127220175489212*1j)*((-0.0201710515112456*np.exp(2*1j*phi0) + 0.109923652184657*1j*np.exp(2*1j*phi0) - 0.290268756692487*np.exp(1j*phi0) - 0.452960143556552*1j*np.exp(1j*phi0) + 1 + 5.18036744029743e-18*1j)*np.exp(2*1j*phi0) + (-0.290268756692487 + 0.452960143556552*1j)*np.exp(1j*phi0) - 0.0201710515112456 - 0.109923652184657*1j) - 0.0296821143160256*((0.485356840986313*np.exp(1j*phi0) + 0.0890632511411119*1j*np.exp(1j*phi0) - 1 + 0.640826264344928*1j)*np.exp(3*1j*phi0) - (1 + 0.640826264344928*1j)*np.exp(1j*phi0) + 0.485356840986313 - 0.0890632511411119*1j)**2)/((-0.0201710515112456*np.exp(2*1j*phi0) + 0.109923652184657*1j*np.exp(2*1j*phi0) - 0.290268756692487*np.exp(1j*phi0) - 0.452960143556552*1j*np.exp(1j*phi0) + 1 + 5.18036744029743e-18*1j)*np.exp(2*1j*phi0) + (-0.290268756692487 + 0.452960143556552*1j)*np.exp(1j*phi0) - 0.0201710515112456 - 0.109923652184657*1j)**2
        ).real

    def newton_raphson(
        phi0_init: float,
        maxiters: int,
        epsilon: float,
    ) -> float:
        phi0 = phi0_init
        for _ in range(maxiters):
            dphi0 = -dF(phi0) / d2F(phi0)
            if abs(dphi0) < epsilon:
                return phi0
            phi0 += dphi0
            phi0 = phi0 % (2 * np.pi)
        print("reached maxiters")
        return phi0

    INVPHI = 0.6180339887498949 # inverse golden ratio
    INVPHI2 = 0.3819660112501051 # inverse golden ratio squared

    def gen_golden_point(bracket: (float, float), r: float) -> float:
        h = abs(bracket[1] - bracket[0])
        x0 = min(bracket)
        return x0 + r * h

    def golden_section(
        phi0_init: (float, float),
        maxiters: int,
        epsilon: float,
    ) -> float:
        x0, x3 = phi0_init
        x1 = gen_golden_point((x0, x3), INVPHI2)
        x2 = gen_golden_point((x0, x3), INVPHI)
        f1 = F(x1)
        f2 = F(x2)
        for _ in range(maxiters):
            if f1 > f2:
                if abs(x3 - x0) < epsilon:
                    return x1
                x3 = x2
                x2 = x1
                f2 = f1
                x1 = gen_golden_point((x0, x3), INVPHI2)
                f1 = F(x1)
            else:
                if abs(x3 - x0) < epsilon:
                    return x2
                x0 = x1
                x1 = x2
                f1 = f2
                x2 = gen_golden_point((x0, x3), INVPHI)
                f2 = F(x2)
        print("reached maxiters")
        if f1 > f2:
            return x1
        else:
            return x2

    # ph = np.linspace(0.0, 2.0 * np.pi, 1000)
    # (
    #     pd.Plotter()
    #     .plot(ph, F(ph))
    #     .plot(ph, dF(ph))
    #     .plot(ph, d2F(ph))
    #     .show()
    # )

    # phi0_init = +2.3
    # sol = newton_raphson(phi0_init, maxiters=1000, epsilon=1e-6)
    phi0_init = (0.0, np.pi)
    sol = golden_section(phi0_init, maxiters=1000, epsilon=1e-9)
    print(
f"""
phi0 = {sol:.6f}
F = {F(sol):.6f}
dF = {dF(sol):.6f}
d2F = {d2F(sol):.6f}
"""[1:-1]
    )
elif mode == 2:
    # # r_sep = 2.4 μm
    # data = np.array([ # ζ, F
    #     [0.000, 0.999996],
    #     [0.011, 0.999996],
    #     [0.050, 0.999996],
    #     [0.100, 0.999993],
    #     [0.150, 0.999990],
    #     [0.200, 0.999995],
    #     [0.250, 0.999993],
    #     [0.300, 0.999962],
    #     [0.350, 0.999993],
    #     [0.400, 0.999994],
    #     [0.450, 0.999084],
    #     [0.500, 0.999993],
    #     [0.550, 0.999995],
    #     [0.600, 0.999996],
    #     [0.650, 0.999994],
    #     [0.700, 0.999996],
    #     [0.750, 0.999996],
    #     [0.800, 0.999993],
    #     [0.850, 0.999970],
    #     [0.900, 0.999995],
    #     [0.950, 0.999994],
    #     [1.000, 0.249999],
    # ])
    # (
    #     pd.Plotter()
    #     .semilogy(data[:, 0], 1 - data[:, 1])
    #     .ggrid()
    #     .set_xlabel("$\\zeta$")
    #     .set_ylabel("$1 - \\mathcal{F}$")
    #     .savefig("output/ququart_cz_gate_fidelity_zeta.png")
    #     .close()
    # )

    # # r_sep = 6.25 μm
    # data = np.array([ # η, F
    #     [0.000010000, 0.749991],
    #     [0.000017783, 0.749992],
    #     [0.000031623, 0.749993],
    #     [0.000056234, 0.749995],
    #     [0.000100000, 0.750001],
    #     [0.000177828, 0.750020],
    #     [0.000316228, 0.750076],
    #     [0.000562341, 0.750246],
    #     [0.001000000, 0.750770],
    #     [0.001778279, 0.752360],
    #     [0.003162278, 0.757004],
    #     [0.005623413, 0.769377],
    #     [0.010000000, 0.797113],
    #     [0.017782794, 0.844887],
    #     [0.031622777, 0.902787],
    #     [0.056234133, 0.949490],
    #     [0.100000000, 0.980934],
    #     [0.177827941, 0.993877],
    #     [0.316227766, 0.998512],
    #     [0.562341325, 0.999776],
    #     [1.000000000, 0.999981],
    # ])

    # (
    #     pd.Plotter()
    #     .loglog(data[:, 0], 1 - data[:, 1])
    #     .ggrid()
    #     .set_xlabel("$\\eta$")
    #     .set_ylabel("$1 - \\mathcal{F}$")
    #     .savefig("output/ququart_cz_gate_fidelity_eta.png")
    #     .close()
    # )

    data_special = np.array([ # η, F
        # # r_sep = 2.4 μm
        # [0.000010000, 0.756438],
        # [0.000017783, 0.767990],
        # [0.000031623, 0.794152],
        # [0.000056234, 0.839752],
        # [0.000100000, 0.896055],
        # [0.000177828, 0.942963],
        # [0.000316228, 0.976167],
        # [0.000562341, 0.990914],
        # [0.001000000, 0.996877],
        # [0.001778279, 0.998968],
        # [0.003162278, 0.999667],
        # [0.005623413, 0.999893],
        # [0.010000000, 0.999964],
        # [0.017782794, 0.999986],
        # [0.031622777, 0.999993],
        # [0.056234133, 0.999994],
        # [0.100000000, 0.999995],
        # [0.177827941, 0.999989],
        # [0.316227766, 0.999996],
        # [0.562341325, 0.999978],
        # [1.000000000, 0.999996],

        # r_sep = 3.6840314986404 μm
        [0.000010000, 0.749950],
        [0.000017783, 0.749953],
        [0.000031623, 0.759964],
        [0.000056234, 0.750000],
        [0.000100000, 0.750119],
        [0.000177828, 0.750495],
        [0.000316228, 0.751667],
        [0.000562341, 0.755170],
        [0.001000000, 0.764825],
        [0.001778279, 0.787613],
        [0.003162278, 0.829737],
        [0.005623413, 0.885574],
        [0.010000000, 0.936375],
        [0.017782794, 0.972538],
        [0.031622777, 0.989503],
        [0.056234133, 0.996362],
        [0.100000000, 0.998814],
        [0.177827941, 0.999602],
        [0.316227766, 0.999838],
        [0.439284546, 0.999875],
        [0.562341325, 0.999899],
        [0.708227550, 0.999902],
        [0.854113775, 0.999905],
        [1.000000000, 0.999906],
    ])

    data_special_neg = np.array([ # η, F
        # # r_sep = 3.6840314986404 μm; opposite sign
        [0.000010000, 0.749950],
        [0.000017783, 0.749956],
        [0.000031623, 0.749973],
        [0.000056234, 0.750020],
        [0.000100000, 0.750158],
        [0.000177828, 0.750581],
        [0.000316228, 0.751889],
        [0.000562341, 0.755917],
        [0.001000000, 0.767790],
        [0.001778279, 0.799007],
        [0.003162278, 0.865621],
        [0.005623413, 0.955090],
        [0.010000000, 0.894798],
        [0.017782794, 0.903219],
        [0.031622777, 0.978528],
        [0.056234133, 0.994135],
        [0.100000000, 0.998129],
        [0.177827941, 0.999312],
        [0.316227766, 0.999690],
        [0.439284546, 0.999780],
        [0.562341325, 0.999820],
        [0.708227550, 0.999845],
        [0.854113775, 0.999860],
        [1.000000000, 0.999870],
    ])

    # data_scan = np.array([ # Ω/2π = 0.63 MHz
    #     # r_sep = 2.35033095 μm
    #     [ # η, F
    #         [0.000010000, 0.757989],
    #         [0.000017783, 0.771801],
    #         [0.000031623, 0.801680],
    #         [0.000056234, 0.850567],
    #         [0.000100000, 0.906738],
    #         [0.000177828, 0.949605],
    #         [0.000316228, 0.979451],
    #         [0.000562341, 0.992549],
    #         [0.001000000, 0.997425],
    #         [0.001778279, 0.999137],
    #         [0.003162278, 0.999711],
    #         [0.005623413, 0.999900],
    #         [0.010000000, 0.999962],
    #         [0.017782794, 0.999982],
    #         [0.031622777, 0.999989],
    #         [0.056234133, 0.999991],
    #         [0.100000000, 0.999992],
    #         [0.177827941, 0.999987],
    #         [0.316227766, 0.999966],
    #         [0.562341325, 0.999994],
    #         [1.000000000, 0.999994],
    #     ],
    #     # r_sep = 3.06821169 μm
    #     [ # η, F
    #         [0.000010000, 0.750344],
    #         [0.000017783, 0.751099],
    #         [0.000031623, 0.753402],
    #         [0.000056234, 0.760005],
    #         [0.000100000, 0.776728],
    #         [0.000177828, 0.811054],
    #         [0.000316228, 0.863226],
    #         [0.000562341, 0.918510],
    #         [0.001000000, 0.958057],
    #         [0.001778279, 0.983937],
    #         [0.003162278, 0.994284],
    #         [0.005623413, 0.998073],
    #         [0.010000000, 0.999385],
    #         [0.017782794, 0.999809],
    #         [0.031622777, 0.999942],
    #         [0.056234133, 0.999982],
    #         [0.100000000, 0.999993],
    #         [0.177827941, 0.999995],
    #         [0.316227766, 0.999996],
    #         [0.562341325, 0.999993],
    #         [1.000000000, 0.999997],
    #     ],
    #     # r_sep = 4.00536061 μm
    #     [ # η, F
    #         [0.000010000, 0.750018],
    #         [0.000017783, 0.750054],
    #         [0.000031623, 0.750163],
    #         [0.000056234, 0.751500],
    #         [0.000100000, 0.751500],
    #         [0.000177828, 0.754488],
    #         [0.000316228, 0.762798],
    #         [0.000562341, 0.782924],
    #         [0.001000000, 0.821764],
    #         [0.001778279, 0.876284],
    #         [0.003162278, 0.929308],
    #         [0.005623413, 0.966946],
    #         [0.010000000, 0.987095],
    #         [0.017782794, 0.995498],
    #         [0.031622777, 0.998535],
    #         [0.056234133, 0.999540],
    #         [0.100000000, 0.999860],
    #         [0.177827941, 0.999957],
    #         [0.316227766, 0.999984],
    #         [0.562341325, 0.999990],
    #         [1.000000000, 0.999991],
    #     ],
    #     # r_sep = 5.22875057 μm
    #     [ # η, F
    #         [0.000010000, 0.749998],
    #         [0.000017783, 0.750000],
    #         [0.000031623, 0.750004],
    #         [0.000056234, 0.750018],
    #         [0.000100000, 0.750060],
    #         [0.000177828, 0.750194],
    #         [0.000316228, 0.750611],
    #         [0.000562341, 0.751892],
    #         [0.001000000, 0.755689],
    #         [0.001778279, 0.766043],
    #         [0.003162278, 0.790124],
    #         [0.005623413, 0.833784],
    #         [0.010000000, 0.890263],
    #         [0.017782794, 0.940025],
    #         [0.031622777, 0.975087],
    #         [0.056234133, 0.990878],
    #         [0.100000000, 0.997038],
    #         [0.177827941, 0.999168],
    #         [0.316227766, 0.999806],
    #         [0.562341325, 0.999968],
    #         [1.000000000, 0.999994],
    #     ],
    #     # r_sep = 6.82581050 μm
    #     [ # η, F
    #         [0.000010000, 0.749980],
    #         [0.000017783, 0.749980],
    #         [0.000031623, 0.749980],
    #         [0.000056234, 0.749980],
    #         [0.000100000, 0.749982],
    #         [0.000177828, 0.749988],
    #         [0.000316228, 0.750007],
    #         [0.000562341, 0.750065],
    #         [0.001000000, 0.750248],
    #         [0.001778279, 0.750817],
    #         [0.003162278, 0.752556],
    #         [0.005623413, 0.757638],
    #         [0.010000000, 0.771096],
    #         [0.017782794, 0.800853],
    #         [0.031622777, 0.851050],
    #         [0.056234133, 0.910069],
    #         [0.100000000, 0.955464],
    #         [0.177827941, 0.984677],
    #         [0.316227766, 0.995927],
    #         [0.562341325, 0.999375],
    #         [1.000000000, 0.999958],
    #     ],
    # ])

    data_scan = np.array([ # Ω/2π = 3.0 MHz
        # r_sep = 2.35033095 μm
        [ # η, F
            [0.000010000, 0.750328],
            [0.000017783, 0.751146],
            [0.000031623, 0.753628],
            [0.000056234, 0.760693],
            [0.000100000, 0.778367],
            [0.000177828, 0.814007],
            [0.000316228, 0.866923],
            [0.000562341, 0.921657],
            [0.001000000, 0.960598],
            [0.001778279, 0.984858],
            [0.003162278, 0.994621],
            [0.005623413, 0.998136],
            [0.010000000, 0.999349],
            [0.017782794, 0.999744],
            [0.031622777, 0.999867],
            [0.056234133, 0.999902],
            [0.100000000, 0.999891],
            [0.177827941, 0.999900],
            [0.316227766, 0.999926],
            [0.439284546, 0.999926],
            [0.562341325, 0.999926],
            [0.708227550, 0.999926],
            [0.854113775, 0.999926],
            [1.000000000, 0.999926],
        ],
        # r_sep = 3.06821169 μm
        [ # η, F
            [0.000010000, 0.749945],
            [0.000017783, 0.749971],
            [0.000031623, 0.750066],
            [0.000056234, 0.750386],
            [0.000100000, 0.751408],
            [0.000177828, 0.754528],
            [0.000316228, 0.763293],
            [0.000562341, 0.784462],
            [0.001000000, 0.824751],
            [0.001778279, 0.880049],
            [0.003162278, 0.932407],
            [0.005623413, 0.969566],
            [0.010000000, 0.988181],
            [0.017782794, 0.995926],
            [0.031622777, 0.998653],
            [0.056234133, 0.999541],
            [0.100000000, 0.999808],
            [0.177827941, 0.999880],
            [0.316227766, 0.999890],
            [0.439284546, 0.999897],
            [0.562341325, 0.999827],
            [0.708227550, 0.999855],
            [0.854113775, 0.999910],
            [1.000000000, 0.999909],
        ],
        # r_sep = 4.00536061 μm
        [ # η, F
            [0.000010000, 0.749982],
            [0.000017783, 0.759989],
            [0.000031623, 0.750004],
            [0.000056234, 0.750035],
            [0.000100000, 0.750112],
            [0.000177828, 0.750309],
            [0.000316228, 0.750851],
            [0.000562341, 0.752382],
            [0.001000000, 0.756680],
            [0.001778279, 0.767958],
            [0.003162278, 0.793384],
            [0.005623413, 0.838181],
            [0.010000000, 0.894496],
            [0.017782794, 0.942536],
            [0.031622777, 0.976372],
            [0.056234133, 0.991269],
            [0.100000000, 0.997111],
            [0.177827941, 0.999065],
            [0.316227766, 0.999661],
            [0.439284546, 0.999765],
            [0.562341325, 0.999811],
            [0.708227550, 0.999823],
            [0.854113775, 0.999830],
            [1.000000000, 0.999831],
        ],
        # r_sep = 5.22875057 μm
        [ # η, F
            [0.000010000, 0.749963],
            [0.000017783, 0.749964],
            [0.000031623, 0.749964],
            [0.000056234, 0.749965],
            [0.000100000, 0.749967],
            [0.000177828, 0.749974],
            [0.000316228, 0.749995],
            [0.000562341, 0.750059],
            [0.001000000, 0.750259],
            [0.001778279, 0.750875],
            [0.003162278, 0.752748],
            [0.005623413, 0.758185],
            [0.010000000, 0.772430],
            [0.017782794, 0.803437],
            [0.031622777, 0.854613],
            [0.056234133, 0.913295],
            [0.100000000, 0.957145],
            [0.177827941, 0.985291],
            [0.316227766, 0.996196],
            [0.439284546, 0.998539],
            [0.562341325, 0.999376],
            [0.708227550, 0.999745],
            [0.854113775, 0.999870],
            [1.000000000, 0.999902],
        ],
        # r_sep = 6.82581050 μm
        [ # η, F
            [0.000010000, 0.749355],
            [0.000017783, 0.749355],
            [0.000031623, 0.749355],
            [0.000056234, 0.749355],
            [0.000100000, 0.749356],
            [0.000177828, 0.749356],
            [0.000316228, 0.749357],
            [0.000562341, 0.749360],
            [0.001000000, 0.749371],
            [0.001778279, 0.749403],
            [0.003162278, 0.749505],
            [0.005623413, 0.749824],
            [0.010000000, 0.750811],
            [0.017782794, 0.753803],
            [0.031622777, 0.762331],
            [0.056234133, 0.783759],
            [0.100000000, 0.827472],
            [0.177827941, 0.893086],
            [0.316227766, 0.955892],
            [0.439284546, 0.978911],
            [0.562341325, 0.987633],
            [0.708227550, 0.996102],
            [0.854113775, 0.998273],
            [1.000000000, 0.998725],
        ],
    ])

    data_scan_neg = np.array([ # Ω/2π = 3.0 MHz
        # r_sep = 2.35033095 μm
        [ # η, F
            [0.000010000, 0.750357],
            [0.000017783, 0.751237],
            [0.000031623, 0.754003],
            [0.000056234, 0.762404],
            [0.000100000, 0.785664],
            [0.000177828, 0.839597],
            [0.000316228, 0.929096],
            [0.000562341, 0.950340],
            [0.001000000, 0.828751],
            [0.001778279, 0.967981],
            [0.003162278, 0.991859],
            [0.005623413, 0.997560],
            [0.010000000, 0.999190],
            [0.017782794, 0.999685],
            [0.031622777, 0.999843],
            [0.056234133, 0.999895],
            [0.100000000, 0.999912],
            [0.177827941, 0.999904],
            [0.316227766, 0.999919],
            [0.439284546, 0.999921],
            [0.562341325, 0.999921],
            [0.708227550, 0.999922],
            [0.854113775, 0.999922],
            [1.000000000, 0.999923],
        ],
        # r_sep = 3.06821169 μm
        [ # η, F
            [0.000010000, 0.749954],
            [0.000017783, 0.749986],
            [0.000031623, 0.750094],
            [0.000056234, 0.750444],
            [0.000100000, 0.751568],
            [0.000177828, 0.755112],
            [0.000316228, 0.765762],
            [0.000562341, 0.794377],
            [0.001000000, 0.857200],
            [0.001778279, 0.948101],
            [0.003162278, 0.915241],
            [0.005623413, 0.882879],
            [0.010000000, 0.974115],
            [0.017782794, 0.993443],
            [0.031622777, 0.997869],
            [0.056234133, 0.999204],
            [0.100000000, 0.999641],
            [0.177827941, 0.999794],
            [0.316227766, 0.999852],
            [0.439284546, 0.999867],
            [0.562341325, 0.999876],
            [0.708227550, 0.999891],
            [0.854113775, 0.999851],
            [1.000000000, 0.999871],
        ],
        # r_sep = 4.00536061 μm
        [ # η, F
            [0.000010000, 0.749922],
            [0.000017783, 0.749920],
            [0.000031623, 0.749918],
            [0.000056234, 0.749921],
            [0.000100000, 0.749947],
            [0.000177828, 0.750057],
            [0.000316228, 0.750459],
            [0.000562341, 0.751823],
            [0.001000000, 0.756236],
            [0.001778279, 0.769494],
            [0.003162278, 0.804164],
            [0.005623413, 0.875924],
            [0.010000000, 0.961576],
            [0.017782794, 0.862752],
            [0.031622777, 0.923643],
            [0.056234133, 0.982034],
            [0.100000000, 0.994534],
            [0.177827941, 0.998049],
            [0.316227766, 0.999155],
            [0.439284546, 0.999437],
            [0.562341325, 0.999538],
            [0.708227550, 0.999627],
            [0.854113775, 0.999669],
            [1.000000000, 0.999682],
        ],
        # r_sep = 5.22875057 μm
        [ # η, F
            [0.000010000, 0.750432],
            [0.000017783, 0.750424],
            [0.000031623, 0.750410],
            [0.000056234, 0.750386],
            [0.000100000, 0.750343],
            [0.000177828, 0.750270],
            [0.000316228, 0.750148],
            [0.000562341, 0.749956],
            [0.001000000, 0.749694],
            [0.001778279, 0.749489],
            [0.003162278, 0.749980],
            [0.005623413, 0.753580],
            [0.010000000, 0.767619],
            [0.017782794, 0.807547],
            [0.031622777, 0.888534],
            [0.056234133, 0.964962],
            [0.100000000, 0.812755],
            [0.177827941, 0.941897],
            [0.316227766, 0.982937],
            [0.439284546, 0.988665],
            [0.562341325, 0.993459],
            [0.708227550, 0.994188],
            [0.854113775, 0.995370],
            [1.000000000, 0.996855],
        ],
        # r_sep = 6.82581050 μm
        [ # η, F
            [0.000010000, 0.759123],
            [0.000017783, 0.759117],
            [0.000031623, 0.759106],
            [0.000056234, 0.759087],
            [0.000100000, 0.759053],
            [0.000177828, 0.758992],
            [0.000316228, 0.758885],
            [0.000562341, 0.758694],
            [0.001000000, 0.758356],
            [0.001778279, 0.757760],
            [0.003162278, 0.756714],
            [0.005623413, 0.754901],
            [0.010000000, 0.751840],
            [0.017782794, 0.747018],
            [0.031622777, 0.741026],
            [0.056234133, 0.741789],
            [0.100000000, 0.777390],
            [0.177827941, 0.869329],
            [0.316227766, 0.945792],
            [0.439284546, 0.832520],
            [0.562341325, 0.776700],
            [0.708227550, 0.832660],
            [0.854113775, 0.876782],
            [1.000000000, 0.922450],
        ],
    ])

    U = np.logspace(np.log10(30000), np.log10(50), 5)
    N = data_scan.shape[0]
    P = pd.Plotter()
    for k in range(data_scan.shape[0]):
        opacity = 1 -  0.75 * k / (N - 1)
        Uk = U[k] if U[k] < 1000 else U[k] / 1000
        u = "MHz" if U[k] < 1000 else "GHz"
        P.semilogy(
            data_scan[k, :, 0], 1 - data_scan[k, :, 1],
            marker="o", linestyle="-", color="C4", alpha=opacity,
            zorder=2 * (N - k - 1) + 1,
            label=f"{Uk:.1f} {u}",
        )
        P.semilogy(
            -data_scan_neg[k, :, 0], 1 - data_scan_neg[k, :, 1],
            marker="o", linestyle="-", color="C4", alpha=opacity,
            zorder=2 * (N - k - 1),
        )
        # P.loglog(
        #     data_scan[k, :, 0], 1 - data_scan[k, :, 1],
        #     marker="o", linestyle="-", color="C4", alpha=opacity,
        #     zorder=2 * (N - k - 1) + 1,
        #     label=f"{Uk:.1f} {u}",
        # )
        # P.loglog(
        #     data_scan_neg[k, :, 0], 1 - data_scan_neg[k, :, 1],
        #     marker="o", linestyle="--", color="C4", alpha=opacity,
        #     zorder=2 * (N - k - 1),
        # )
    (
        P
        .semilogy(
            data_special[:, 0], 1 - data_special[:, 1],
            marker="o", linestyle="-", color="r",
            zorder=2 * N + 1,
            label=f"2.0 GHz",
        )
        .semilogy(
            -data_special_neg[:, 0], 1 - data_special_neg[:, 1],
            marker="o", linestyle="-", color="r",
            zorder=2 * N,
        )
        # .loglog(
        #     data_special[:, 0], 1 - data_special[:, 1],
        #     marker="o", linestyle="-", color="r",
        #     zorder=2 * N + 1,
        #     label=f"2.0 GHz",
        # )
        # .loglog(
        #     data_special_neg[:, 0], 1 - data_special_neg[:, 1],
        #     marker="o", linestyle="--", color="r",
        #     zorder=2 * N,
        # )
        .ggrid().grid(False, which="both")
        .legend()
        .set_xlabel("η")
        .set_ylabel("1 – $\\mathcal{F}$")
        .savefig("output/ququart_cz_gate_fidelity_eta_scan.png")
        .savefig("output/ququart_cz_gate_fidelity_eta_scan.pdf")
        .show()
        .close()
    )
#