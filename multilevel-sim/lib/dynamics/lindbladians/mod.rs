//! Lindbladian operators for various systems.

use std::{
    fmt,
    rc::Rc,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use crate::hilbert::{ Basis, SpontaneousDecay, StateIter };

pub mod generic;
pub use generic::{ LOperator, LParams };

pub mod cavity;
pub use cavity::{ LOperatorCavity, LCavityParams };

pub mod transverse_ising;
pub use transverse_ising::{ LOperatorTransverseIsing, LTransverseIsingParams };

/// Basic requirements for any implementation of a Lindbladian operator.
pub trait LOp<'a, S> {
    /// Initialization data type.
    type Params;

    /// Basis type containing state energies.
    type Basis: PartialEq + StateIter<'a, State = S>;

    /// Initialize `self`.
    fn new_operator(params: Self::Params) -> Self;

    /// Operate on a density matrix.
    fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64>;

    /// Return a reference to the basis.
    fn get_basis(&self) -> &Self::Basis;
}

/// Builder for non-Hermitian, real matrices giving spontaneous decay rates in a
/// single-atom system.
///
/// The `(i, j)`-th entry of the generated matrix gives the decay rate, in units
/// of angular frequency, of the `i`-th state to the `j`-th state. Any desired
/// weighting on decay rates (e.g. Clebsch-Gordan coefficients) should be
/// performed within the [`SpontaneousDecay`] impl.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct YBuilder<'a, S>
where S: SpontaneousDecay
{
    basis: &'a Basis<S>,
    f_decay: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, S> fmt::Debug for YBuilder<'a, S>
where S: SpontaneousDecay
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "YBuilder {{ basis: {:?}, f_decay: ", self.basis)?;
        if self.f_decay.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, S> YBuilder<'a, S>
where S: SpontaneousDecay
{
    fn def_decay(s1: &S, s2: &S) -> Option<f64> { s1.decay_rate(s2) }

    /// Create a new `YBuilder`.
    pub fn new(basis: &'a Basis<S>) -> Self {
        Self { basis, f_decay: None }
    }

    /// Use a provided decay rate function instead of the [`SpontaneousDecay`]
    /// implementation.
    pub fn with_decay<F>(mut self, f_decay: F) -> Self
    where F: Fn(&S, &S) -> Option<f64> + 'a
    {
        self.f_decay = Some(Rc::new(f_decay));
        self
    }

    /// Get a reference to the basis.
    pub fn basis(&self) -> &Basis<S> { self.basis }

    /// Compute the decay rate coupling matrix.
    pub fn gen(&self) -> nd::Array2<f64> {
        if let Some(f) = &self.f_decay {
            self.gen_with(f.as_ref())
        } else {
            self.gen_with(Self::def_decay)
        }
    }

    /// Like [`Self::gen`], but using a provided decay rate function rather than
    /// that of [`SpontaneousDecay`].
    pub fn gen_with<F>(&self, f_decay: F) -> nd::Array2<f64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let n = self.basis.len();
        let Y: nd::Array2<f64>
            = nd::Array1::from_iter(
                self.basis.keys()
                .cartesian_product(self.basis.keys())
                .map(|(s1, s2)| f_decay(s1, s2).unwrap_or(0.0))
            )
            .into_shape((n, n))
            .expect("YBuilder::gen_with: error reshaping array");
        Y
    }
}

