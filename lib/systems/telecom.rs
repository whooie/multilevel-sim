use crate::{
    zm_br_fn,
    gfactors::{
        G_3P0_F12,
        G_3D1_F32,
        MU_B,
    },
    hilbert::{ BasisState, SpinState },
    spin::Spin,
};

pub const MU_C: f64 = MU_B * G_3P0_F12;
pub const MU_T: f64 = MU_B * G_3D1_F32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    C0, // 3P0  F = 1/2  mF = -1/2
    C1, // 3P0  F = 1/2  mF = +1/2
    T0, // 3D1  F = 3/2  mF = -3/2
    T1, // 3D1  F = 3/2  mF = -1/2
    T2, // 3D1  F = 3/2  mF = +1/2
    T3, // 3D1  F = 3/2  mF = +3/2
}
use State::*;

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            | (C0, T0) | (T0, C0) // 3D1 couplings
            | (C0, T1) | (T1, C0)
            | (C0, T2) | (T2, C0)
            | (C1, T1) | (T1, C1)
            | (C1, T2) | (T2, C1)
            | (C1, T3) | (T3, C1)
        )
    }
}

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
        // C0 => -1.7e-3 / 1.5 * state.spin().proj().f() * B,
        // C1 => -1.7e-3 / 1.5 * state.spin().proj().f() * B,
        C0 => -1.5e-3 / 1.5 * state.spin().proj().f() * B,
        C1 => -1.5e-3 / 1.5 * state.spin().proj().f() * B,
        T0 => MU_T * state.spin().proj().f() * B,
        T1 => MU_T * state.spin().proj().f() * B,
        T2 => MU_T * state.spin().proj().f() * B,
        T3 => MU_T * state.spin().proj().f() * B,
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            C0 => (1_u32, -1_i32).into(),
            C1 => (1_u32,  1_i32).into(),
            T0 => (3_u32, -3_i32).into(),
            T1 => (3_u32, -1_i32).into(),
            T2 => (3_u32,  1_i32).into(),
            T3 => (3_u32,  3_i32).into(),
        }
    }
}

