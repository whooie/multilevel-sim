use std::f64::consts::TAU;
use multilevel_sim::{
    hilbert::{ BasisState, SpinState, TrappedMagic },
    spin::Spin,
};

pub const TRAP_FREQ: f64 = 10e-3; // MHz
// pub const TRAP_FREQ: f64 = 30e-3; // MHz
// pub const TRAP_FREQ: f64 = 50e-3; // MHz
// pub const TRAP_FREQ: f64 = 70e-3; // MHz
// pub const TRAP_FREQ: f64 = 90e-3; // MHz
// pub const TRAP_FREQ: f64 = 100e-3; // MHz
// pub const TRAP_FREQ: f64 = 200e-3; // MHz
// pub const TRAP_FREQ: f64 = 400e-3; // MHz
// pub const TRAP_FREQ: f64 = 800e-3; // MHz
// pub const TRAP_FREQ: f64 = 1000e-3; // MHz
// pub const TRAP_FREQ: f64 = 0.400; // MHz
// pub const TRAP_FREQ: f64 = 2.0; // MHz

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

