use multilevel_sim::{
    dynamics::transition_cg,
    gfactors::{ G_3D1_F32, MU_B },
    hilbert::{ BasisState, SpinState, SpontaneousDecay },
    spin::Spin,
};

pub const MU_C: f64 = -1.5e-3 / 1.5;
pub const MU_T: f64 = MU_B * G_3D1_F32;
pub const GAMMA: f64 = 200e4 / 1e6; // taken from group wiki; includes angular factors
pub const NMAX: usize = 3;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    /// <sup>3</sup>P<sub>0</sub>  F = 1/2  m<sub>F</sub> = -1/2
    C0,
    /// <sup>3</sup>P<sub>0</sub>  F = 1/2  m<sub>F</sub> = +1/2
    C1,
    /// <sup>3</sup>D<sub>1</sub>  F = 3/2  m<sub>F</sub> = -3/2
    T0,
    /// <sup>3</sup>D<sub>1</sub>  F = 3/2  m<sub>F</sub> = -1/2
    T1,
    /// <sup>3</sup>D<sub>1</sub>  F = 3/2  m<sub>F</sub> = +1/2
    T2,
    /// <sup>3</sup>D<sub>1</sub>  F = 3/2  m<sub>F</sub> = +3/2
    T3,
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

impl SpontaneousDecay for State {
    fn decay_rate(&self, other: &Self) -> Option<f64> {
        let cg = || transition_cg(other.spin(), self.spin());
        match (*self, *other) {
            (T0, C0)
            | (T1, C0) | (T1, C1)
            | (T2, C0) | (T2, C1)
            | (T3, C1)
            => cg().map(|c| GAMMA * c),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Early(pub State, pub usize);

impl BasisState for Early {
    fn couples_to(&self, other: &Self) -> bool {
        self.0.couples_to(&other.0) && self.1 == other.1
    }
}

impl SpinState for Early {
    fn spin(&self) -> Spin { self.0.spin() }
}

impl SpontaneousDecay for Early {
    fn decay_rate(&self, other: &Self) -> Option<f64> {
        // ((other.1 == self.1 + 1) ^ (self.1 == NMAX && other.1 == NMAX))
        (other.1 == self.1 + 1)
            .then(|| self.0.decay_rate(&other.0))
            .flatten()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Late(pub Early, pub usize);

impl BasisState for Late {
    fn couples_to(&self, other: &Self) -> bool {
        self.0.couples_to(&other.0) && self.1 == other.1
    }
}

impl SpinState for Late {
    fn spin(&self) -> Spin { self.0.spin() }
}

impl SpontaneousDecay for Late {
    fn decay_rate(&self, other: &Self) -> Option<f64> {
        // ((other.1 == self.1 + 1) ^ (self.1 == NMAX && other.1 == NMAX))
        (other.1 == self.1 + 1 && self.0.1 == other.0.1)
            .then(|| self.0.0.decay_rate(&other.0.0))
            .flatten()
    }
}

impl AsRef<State> for State {
    fn as_ref(&self) -> &State { self }
}

impl AsRef<State> for Early {
    fn as_ref(&self) -> &State { &self.0 }
}

impl AsRef<State> for Late {
    fn as_ref(&self) -> &State { &self.0.0 }
}

pub fn zm<S>(state: S, B: f64) -> f64
where S: AsRef<State>
{
    let state = state.as_ref();
    match *state {
        C0 => MU_C * state.spin().proj().f() * B,
        C1 => MU_C * state.spin().proj().f() * B,
        T0 => MU_T * state.spin().proj().f() * B,
        T1 => MU_T * state.spin().proj().f() * B,
        T2 => MU_T * state.spin().proj().f() * B,
        T3 => MU_T * state.spin().proj().f() * B,
    }
}

