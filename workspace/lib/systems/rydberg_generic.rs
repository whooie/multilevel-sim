use multilevel_sim::{
    hilbert::{ BasisState, SpinState, RydbergState },
    spin::Spin,
};

pub const C6: f64 = 300000.0; // MHz μm^6
pub const R_SEP: f64 = 3.5; // μm

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    G0,
    G1,
    G2,
    R,
}

impl BasisState for State {
    fn couples_to(&self, other: &Self) -> bool {
        matches!(
            (*self, *other),
            (Self::G0, Self::R)
            | (Self::R, Self::G0)
            | (Self::G1, Self::R)
            | (Self::R, Self::G1)
            | (Self::G2, Self::R)
            | (Self::R, Self::G2),
        )
    }
}

impl SpinState for State {
    fn spin(&self) -> Spin {
        match *self {
            Self::G0 => (2_u32, -2_i32).into(),
            Self::G1 => (2_u32,  0_i32).into(),
            Self::G2 => (2_u32,  2_i32).into(),
            Self::R  => (0_u32,  0_i32).into(),
        }
    }
}

impl RydbergState for State {
    fn is_rydberg(&self) -> bool { matches!(self, Self::R) }

    fn c6_with(&self, other: &Self) -> Option<f64> {
        match (*self, *other) {
            (Self::R, Self::R) => Some(C6),
            _ => None,
        }
    }
}

