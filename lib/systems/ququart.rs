use std::f64::consts::TAU;
use crate::{
    zm_br_fn,
    gfactors::{
        G_1S0_F12,
        G_3P0_F12,
        G_3P1_F12,
        G_3D1_F12,
        G_3S1_F32,
        MU_B,
    },
    hilbert::{ BasisState, SpinState, RydbergState, TrappedMagic },
    spin::Spin,
};

pub const MU_G: f64 = MU_B * G_1S0_F12;
pub const MU_C: f64 = MU_B * G_3P0_F12;
pub const MU_N: f64 = MU_B * G_3P1_F12;
pub const MU_T: f64 = MU_B * G_3D1_F12;
pub const MU_R: f64 = MU_B * G_3S1_F32;

// pub const C6_NOMINAL: f64 = 5_000_000.0 / 0.988723; // MHz μm^6

// pub const R0R0: f64 = -0.988723 * C6_NOMINAL; // calculated
// pub const R0R1: f64 = -0.979977 * C6_NOMINAL;
// pub const R0R2: f64 = -0.983736 * C6_NOMINAL;
// pub const R0R3: f64 = -1.000000 * C6_NOMINAL;
// pub const R1R1: f64 = -0.969186 * C6_NOMINAL;
// pub const R1R2: f64 = -0.970439 * C6_NOMINAL;
// pub const R1R3: f64 = -0.983736 * C6_NOMINAL;
// pub const R2R2: f64 = -0.969186 * C6_NOMINAL;
// pub const R2R3: f64 = -0.979977 * C6_NOMINAL;
// pub const R3R3: f64 = -0.988723 * C6_NOMINAL;

pub const C6_NOMINAL: f64 = 5_000_000.0; // MHz μm^6

// pub const ETA: f64 = 0.000010000; // logspace(-5.0, 0.0, 21)
// pub const ETA: f64 = 0.000017783;
// pub const ETA: f64 = 0.000031623;
pub const ETA: f64 = 0.000056234;
// pub const ETA: f64 = 0.000100000;
// pub const ETA: f64 = 0.000177828;
// pub const ETA: f64 = 0.000316228;
// pub const ETA: f64 = 0.000562341;
// pub const ETA: f64 = 0.001000000;
// pub const ETA: f64 = 0.001778279;
// pub const ETA: f64 = 0.003162278;
// pub const ETA: f64 = 0.005623413;
// pub const ETA: f64 = 0.010000000;
// pub const ETA: f64 = 0.017782794;
// pub const ETA: f64 = 0.031622777;
// pub const ETA: f64 = 0.056234133;
// pub const ETA: f64 = 0.100000000;
// pub const ETA: f64 = 0.177827941;
// pub const ETA: f64 = 0.316227766;
// pub const ETA: f64 = 0.562341325;
// pub const ETA: f64 = 1.000000000;

pub const ZETA: f64 = 1.0 - ETA;
pub const R0R0: f64 = (-1.000000 + ZETA) * C6_NOMINAL; // test
pub const R0R1: f64 = -0.979977 * C6_NOMINAL;
pub const R0R2: f64 = -0.983736 * C6_NOMINAL;
pub const R0R3: f64 = -1.000000 * C6_NOMINAL;
pub const R1R1: f64 = -0.969186 * C6_NOMINAL;
pub const R1R2: f64 = -0.970439 * C6_NOMINAL;
pub const R1R3: f64 = -0.983736 * C6_NOMINAL;
pub const R2R2: f64 = -0.969186 * C6_NOMINAL;
pub const R2R3: f64 = -0.979977 * C6_NOMINAL;
pub const R3R3: f64 = (-1.000000 + ZETA) * C6_NOMINAL;

// pub const R_SEP: f64 = 2.4; // μm; Jeff's value
// pub const R_SEP: f64 = 6.25; // μm
// pub const R_SEP: f64 = 12.0; // μm

pub const R_SEP: f64 = 2.35033095; // μm; U = logspace(log10(30000), log10(50), 5)
// pub const R_SEP: f64 = 2.40000000; // μm
// pub const R_SEP: f64 = 3.06821169; // μm
// pub const R_SEP: f64 = 4.00536061; // μm
// pub const R_SEP: f64 = 5.22875057; // μm
// pub const R_SEP: f64 = 6.82581050; // μm

pub const TRAP_FREQ: f64 = 10e-3; // MHz
// pub const TRAP_FREQ: f64 = 9.5e-3; // MHz; Kaufman omg paper

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    G0, //  1S0  F = 1/2  mF = -1/2
    G1, //  1S0  F = 1/2  mF = +1/2
    C0, //  3P0  F = 1/2  mF = -1/2
    C1, //  3P0  F = 1/2  mF = +1/2
    N0, //  3P1  F = 1/2  mF = -1/2
    N1, //  3P1  F = 1/2  mF = +1/2
    T0, //  3D1  F = 1/2  mF = -1/2
    T1, //  3D1  F = 1/2  mF = +1/2
    R0, // *3S1  F = 3/2  mF = -3/2
    R1, // *3S1  F = 3/2  mF = -1/2
    R2, // *3S1  F = 3/2  mF = +1/2
    R3, // *3S1  F = 3/2  mF = +3/2
}
use State::*;

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            // (G0, C0) | (C0, G0) // 3P0 couplings
            // | (G0, C1) | (C1, G0)
            // | (G1, C0) | (C0, G1)
            // | (G1, C1) | (C1, G1)
            | (G0, N0) | (N0, G0) // 3P1 couplings
            | (G0, N1) | (N1, G0)
            | (G1, N0) | (N0, G1)
            | (G1, N1) | (N1, G1)
            | (C0, T0) | (T0, C0) // 3D1 couplings
            | (C0, T1) | (T1, C0)
            | (C1, T0) | (T0, C1)
            | (C1, T1) | (T1, C1)
            | (C0, R0) | (R0, C0) // *3S1 couplings
            | (C0, R1) | (R1, C0)
            | (C0, R2) | (R2, C0)
            | (C1, R1) | (R1, C1)
            | (C1, R2) | (R2, C1)
            | (C1, R3) | (R3, C1)
        )
    }
}

zm_br_fn!(
    zm_br_g0 : { // MHz / G
        -1.10727922e-08,
         3.74955336e-04,
         7.63360649e-09,
         1.94851695e-02,
         1.39214067e-08,
    }
);
zm_br_fn!(
    zm_br_g1 : { // MHz / G
        -1.41632243e-06,
        -3.78301182e-04,
         1.41632838e-06,
         4.72462887e+00,
         5.58055706e+00,
    }
);
zm_br_fn!(
    zm_br_c0 : { // MHz / G
         1.01151916e+02,
         9.40966754e-05,
        -1.01151915e+02,
        -1.99260954e-06,
         2.02381457e-09,
    }
);
zm_br_fn!(
    zm_br_c1 : { // MHz / G
         4.64712454e+00,
        -2.98286729e-04,
        -4.64711085e+00,
        -4.43638992e-05,
         4.39216548e-08,
    }
);

// MHz / G
pub fn zm(state: State, B: f64) -> f64 {
    match state {
        // G0 => zm_br_g0(B),
        // G1 => zm_br_g1(B),
        // C0 => zm_br_c0(B),
        // C1 => zm_br_c1(B),
        G0 => -1.1e-3 / 1.5 * state.spin().proj().f() * B,
        G1 => -1.1e-3 / 1.5 * state.spin().proj().f() * B,
        C0 => -1.7e-3 / 1.5 * state.spin().proj().f() * B,
        C1 => -1.7e-3 / 1.5 * state.spin().proj().f() * B,
        N0 => MU_N * state.spin().proj().f() * B,
        N1 => MU_N * state.spin().proj().f() * B,
        T0 => MU_T * state.spin().proj().f() * B,
        T1 => MU_T * state.spin().proj().f() * B,
        R0 => MU_R * state.spin().proj().f() * B,
        R1 => MU_R * state.spin().proj().f() * B,
        R2 => MU_R * state.spin().proj().f() * B,
        R3 => MU_R * state.spin().proj().f() * B,
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            G0 => (1_u32, -1_i32).into(),
            G1 => (1_u32,  1_i32).into(),
            C0 => (1_u32, -1_i32).into(),
            C1 => (1_u32,  1_i32).into(),
            N0 => (1_u32, -1_i32).into(),
            N1 => (1_u32,  1_i32).into(),
            T0 => (1_u32, -1_i32).into(),
            T1 => (1_u32,  1_i32).into(),
            R0 => (3_u32, -3_i32).into(),
            R1 => (3_u32, -1_i32).into(),
            R2 => (3_u32,  1_i32).into(),
            R3 => (3_u32,  3_i32).into(),
        }
    }
}

impl RydbergState for State {
    fn is_rydberg(&self) -> bool { matches!(self, R0 | R1 | R2 | R3) }

    fn c6_with(&self, other: &Self) -> Option<f64> {
        match (*self, *other) {
            (R0, R0)            => Some(R0R0),
            (R0, R1) | (R1, R0) => Some(R0R1),
            (R0, R2) | (R2, R0) => Some(R0R2),
            (R0, R3) | (R3, R0) => Some(R0R3),
            (R1, R1)            => Some(R1R1),
            (R1, R2) | (R2, R1) => Some(R1R2),
            (R1, R3) | (R3, R1) => Some(R1R3),
            (R2, R2)            => Some(R2R2),
            (R2, R3) | (R3, R2) => Some(R2R3),
            (R3, R3)            => Some(R3R3),
            _ => None,
        }
    }
}

impl TrappedMagic for State {
    const TRAP_FREQ: f64 = TAU * TRAP_FREQ;
}

