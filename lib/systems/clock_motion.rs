use std::f64::consts::TAU;
use crate::{
    zm_br_fn,
    hilbert::{ BasisState, SpinState, TrappedMagic },
    spin::Spin,
    gfactors::{ G_3P0_F12, G_1S0_F12, MU_B },
};

pub const MU_C: f64 = MU_B * G_3P0_F12; // MHz G^-1
pub const MU_G: f64 = MU_B * G_1S0_F12; // MHz G^-1
pub const TRAP_FREQ: f64 = 10e-3; // MHz

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    G0,
    G1,
    C0,
    C1,
}

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            (Self::G0, Self::C0)
            | (Self::C0, Self::G0)
            | (Self::G0, Self::C1)
            | (Self::C1, Self::G0)
            | (Self::G1, Self::C0)
            | (Self::C0, Self::G1)
            | (Self::G1, Self::C1)
            | (Self::C1, Self::G1)
        )
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            Self::G0 => (1_u32, -1_i32).into(),
            Self::G1 => (1_u32,  1_i32).into(),
            Self::C0 => (1_u32, -1_i32).into(),
            Self::C1 => (1_u32,  1_i32).into(),
        }
    }
}

impl TrappedMagic for State {
    const TRAP_FREQ: f64 = TAU * TRAP_FREQ;
}

zm_br_fn!(
    zm_br_g0 : {
        -1.10727922e-08,
         3.74955336e-04,
         7.63360649e-09,
         1.94851695e-02,
         1.39214067e-08,
    }
);
zm_br_fn!(
    zm_br_g1 : {
        -1.41632243e-06,
        -3.78301182e-04,
         1.41632838e-06,
         4.72462887e+00,
         5.58055706e+00,
    }
);
zm_br_fn!(
    zm_br_c0 : {
         1.01151916e+02,
         9.40966754e-05,
        -1.01151915e+02,
        -1.99260954e-06,
         2.02381457e-09,
    }
);
zm_br_fn!(
    zm_br_c1 : {
         4.64712454e+00,
        -2.98286729e-04,
        -4.64711085e+00,
        -4.43638992e-05,
         4.39216548e-08,
    }
);

pub fn zm(state: State, B: f64) -> f64 {
    match state {
        State::G0 => zm_br_g0(B),
        State::G1 => zm_br_g1(B),
        State::C0 => zm_br_c0(B),
        State::C1 => zm_br_c1(B),
    }
}


