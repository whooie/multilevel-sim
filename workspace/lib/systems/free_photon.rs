use multilevel_sim::{
    hilbert::{ BasisState, SpinState },
    spin::Spin,
};

pub const GAMMA: f64 = 200e4 / 1e6;
pub const NMAX: usize = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    G0,
    G1,
    E0,
    E1,
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
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Photon(pub State, pub usize);

impl BasisState for Photon {
    fn couples_to(&self, other: &Self) -> bool {
        self.0.couples_to(&other.0) && self.1 == other.1
    }
}

impl SpinState for Photon {
    fn spin(&self) -> Spin { self.0.spin() }
}

impl AsRef<State> for State {
    fn as_ref(&self) -> &State { self }
}

impl AsRef<State> for Photon {
    fn as_ref(&self) -> &State { &self.0 }
}

pub fn zm<S>(state: S, B: f64) -> f64
where S: AsRef<State>
{
    let state = *state.as_ref();
    match state {
        G0 => 1e-3 * state.spin().proj().f() * B,
        G1 => 1e-3 * state.spin().proj().f() * B,
        E0 => 3e-3 * state.spin().proj().f() * B,
        E1 => 3e-3 * state.spin().proj().f() * B,
    }
}

