use multilevel_sim::{
    // zm_br_fn,
    // gfactors::{
    //     G_1S0_F12,
    //     G_3P0_F12,
    //     G_3P1_F12,
    //     G_3D1_F12,
    //     G_3S1_F32,
    //     MU_B,
    // },
    hilbert::{
        BasisState,
        SpinState,
        RydbergState,
        CavityCoupling,
        PhotonLadder,
    },
    spin::Spin,
};

/// Cavity spacing in MHz
pub const OMEGA_C: f64 = 376.985e6;

/// G0 state energy in MHz
pub const OMEGA_0: f64 = 0.0;

/// G1 state energy in MHz
pub const OMEGA_1: f64 = 6.835e3;

/// E0 state energy in MHz
pub const OMEGA_E: f64 = 377.111e6;

/// E1 state energy in MHz
pub const OMEGA_D: f64 = 377.112e6;

/// R state energy in MHz
pub const OMEGA_R: f64 = DELTA_R + OMEGA_DELTA_R;

/// G0 ↔ E1 drive frequency in MHz
pub const OMEGA_DELTA_D: f64 = OMEGA_D - DELTA_D;

/// G1 ↔ E0 drive frequency in MHz
pub const OMEGA_DELTA_E: f64 = OMEGA_E - DELTA_E;

/// G1 ↔ R drive frequency in MHz
pub const OMEGA_DELTA_R: f64 = 1009.014e6;

/// G0 ↔ E1 drive strength in MHz
pub const OMEGA_DRIVE_D: f64 = 0.5;

/// G1 ↔ E0 drive strength in MHz
pub const OMEGA_DRIVE_E: f64 = 0.5;

/// G1 ↔ R drive strength in MHz
pub const OMEGA_DRIVE_R: f64 = 2.0;

/// G0 ↔ E0 cavity coupling in MHz
pub const G_E: f64 = 150.860;

/// G1 ↔ E1 cavity coupling in MHz
pub const G_D: f64 = 142.009;

/// Normalization factor for simulated atom array (number of atoms?)
pub const N: f64 = 2e5;

/// (?) in MHz
pub const DELTA_C: f64 = 127e3;

/// G0 ↔ E1 drive detuning in MHz
pub const DELTA_D: f64 = 100e3;

/// G1 ↔ E0 drive detuning in MHz
pub const DELTA_E: f64 = 112.854e3;

/// G1 ↔ R drive detuning in MHz
pub const DELTA_R: f64 = 20.0;

/// (?) in MHz
pub const DELTA_1: f64 = OMEGA_1 - OMEGA_1_PRIME;

/// (?) in MHz
pub const OMEGA_A_TILDE: f64 = OMEGA_C - (OMEGA_DELTA_D - OMEGA_1_PRIME);

/// (?) in MHz
pub const OMEGA_1_PRIME: f64 = (OMEGA_DELTA_D - OMEGA_DELTA_E) / 2.0;

/// (?) in MHz
pub const LAMBDA_D: f64 = 447.21359549995793 * G_D * OMEGA_DRIVE_D / DELTA_D / 2.0;
//                        ^ sqrt(N)

/// (?) in MHz
pub const LAMBDA_E: f64 = 447.21359549995793 * G_E * OMEGA_DRIVE_E / DELTA_E / 2.0;
//                        ^ sqrt(N)

/// ≈ 2 λd ≈ 2 λe
pub const G: f64 = 2.0 * LAMBDA_D;

/// Effective cavity frequency in MHz
pub const OMEGA_A: f64 = 1.0;

/// (?) in MHz
pub const J_Z: f64 = 0.05;

/// Spin-collective energy in MHz
pub const OMEGA_Z: f64 = -0.03;


pub const C6: f64 = 5_000_000.0; // MHz μm^6

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State { // see arxiv:2312.17385 page 11
    G0, // ∣0⟩
    G1, // ∣1⟩
    E0, // ∣e⟩
    E1, // ∣d⟩
    R,  // ∣Ryd⟩
}
use State::*;

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            | (G0, E0) | (E0, G0)
            | (G0, E1) | (E1, G0)
            | (G1, E0) | (E0, G1)
            | (G1, E1) | (E1, G1)
            | (G1, R ) | (R,  G1)
        )
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            G0 => (1_u32, -1_i32).into(),
            G1 => (1_u32,  1_i32).into(),
            E0 => (1_u32, -1_i32).into(),
            E1 => (1_u32,  1_i32).into(),
            R  => (3_u32,  1_i32).into(),
        }
    }
}

impl RydbergState for State {
    fn is_rydberg(&self) -> bool { matches!(*self, R) }

    fn c6_with(&self, other: &Self) -> Option<f64> {
        match (*self, *other) {
            (R, R) => Some(C6),
            _ => None,
        }
    }
}

impl CavityCoupling<1> for State {
    const MODE_SPACING: [f64; 1] = [OMEGA_C];

    fn coupling(&self, to_other: &Self) -> Option<PhotonLadder> {
        use PhotonLadder::*;

        match (&self, *to_other) {
            (G0, E0) => Some(Absorb(0, G_E)),
            (E0, G0) => Some(Emit(0, G_E)),
            (G1, E1) => Some(Absorb(0, G_D)),
            (E1, G1) => Some(Emit(0, G_D)),
            _ => None,
        }
    }
}

