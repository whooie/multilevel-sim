use std::f64::consts::TAU;
use multilevel_sim::{
    hilbert::{ BasisState, SpinState, TrappedMagic },
    spin::Spin,
};

pub const TRAP_FREQ: f64 = 30e-3; // MHz

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    G,
    E,
}

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            (Self::G, Self::E) | (Self::E, Self::G)
        )
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            Self::G => (1_u32, 1_i32).into(),
            Self::E => (1_u32, 1_i32).into(),
        }
    }
}

impl TrappedMagic for State {
    const TRAP_FREQ: f64 = TAU * TRAP_FREQ;
}

