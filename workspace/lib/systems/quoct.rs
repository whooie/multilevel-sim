use std::f64::consts::TAU;
use multilevel_sim::{
    hilbert::{ BasisState, SpinState, TrappedMagic },
    spin::Spin,
};

pub const TRAP_FREQ: f64 = 30e-3; // MHz

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State { G0, G1, E0, E1 }

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            (Self::G0, Self::E0) | (Self::E0, Self::G0)
            | (Self::G0, Self::E1) | (Self::E1, Self::G0)
            | (Self::G1, Self::E0) | (Self::E0, Self::G1)
            | (Self::G1, Self::E1) | (Self::E1, Self::G1)
        )
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            Self::G0 => (1_u32, -1_i32).into(),
            Self::G1 => (1_u32,  1_i32).into(),
            Self::E0 => (1_u32, -1_i32).into(),
            Self::E1 => (1_u32,  1_i32).into(),
        }
    }
}

impl TrappedMagic for State {
    const TRAP_FREQ: f64 = TAU * TRAP_FREQ;
}

