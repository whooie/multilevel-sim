//! A generic multilevel system.
//!
//! See also [`hamiltonians::generic`][super::super::hamiltonians::generic].

use std::{ fmt, rc::Rc };
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use num_traits::Zero;
use crate::{
    dynamics::lindbladians::LOp,
    hilbert::{ Basis, SpontaneousDecay },
};

/// Implements a Lindbladian operator for a given single-atom system.
///
/// Any desired weighting on decay rates (e.g. Clebsch-Gordan coefficients)
/// should be done within the [`SpontaneousDecay`] impl.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct LOperator<'a, S>
where S: SpontaneousDecay
{
    pub(crate) basis: &'a Basis<S>,
    pub(crate) f_decay: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, S> fmt::Debug for LOperator<'a, S>
where S: SpontaneousDecay
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LOperator {{ basis: {:?}, f_decay: ", self.basis)?;
        if self.f_decay.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, S> LOperator<'a, S>
where S: SpontaneousDecay
{
    fn def_decay(s1: &S, s2: &S) -> Option<f64> { s1.decay_rate(s2) }

    /// Create a new `LOperator`.
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
    ///
    /// The `(i, j)`-th entry in this matrix is the decay rate from `i`-th state
    /// to the `j`-th state.
    pub fn decay_matrix(&self) -> nd::Array2<f64> {
        if let Some(f) = &self.f_decay {
            self.decay_matrix_with(f.as_ref())
        } else {
            self.decay_matrix_with(Self::def_decay)
        }
    }

    /// Like [`Self::decay_matrix`], but using a provided decay rate function
    /// rather than that of [`SpontaneousDecay`].
    pub fn decay_matrix_with<F>(&self, f_decay: F) -> nd::Array2<f64>
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
            .expect("LOperator::decay_matrix_with: error reshaping array");
        Y
    }

    /// Perform the operator action on a density matrix.
    pub fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64> {
        if let Some(f) = &self.f_decay {
            self.op_with(rho, f.as_ref())
        } else {
            self.op_with(rho, Self::def_decay)
        }
    }

    /// Like [`Self::op`], but using a provided decay rate function rather than
    /// that of [`SpontaneousDecay`].
    pub fn op_with<F>(&self, rho: &nd::Array2<C64>, f_decay: F)
        -> nd::Array2<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let n = self.basis.len();
        let mut L: nd::Array2<C64> = nd::Array2::zeros((n, n));
        let mut term: C64;
        let z = C64::zero();
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((a, s1), (b, s2)) in iter {
            if let Some(y) = f_decay(s1, s2) {
                for ((i, j), l) in L.indexed_iter_mut() {
                    term
                        = if i == j {
                            y * (
                                if i == b { rho[[a, a]] } else { z }
                                - if i == a { rho[[a, a]] } else { z }
                            )
                        } else {
                            -y * if i == a || j == a
                                { rho[[i, j]] / 2.0 } else { z }
                        };
                    *l += term;
                }
            } else {
                continue;
            }
        }
        L
    }

    /// Return a closure capturing `self` that implements the Lindbladian
    /// operator.
    pub fn op_fn(&self) -> impl Fn(&nd::Array2<C64>) -> nd::Array2<C64> + '_ {
        |rho: &nd::Array2<C64>| {
            if let Some(f) = &self.f_decay {
                self.op_with(rho, f.as_ref())
            } else {
                self.op_with(rho, Self::def_decay)
            }
        }
    }

    /// Like [`Self::op`], but using a provided decay rate function rather than
    /// that of [`SpontaneousDecay`].
    pub fn op_fn_with<F>(&self, f_decay: F)
        -> impl Fn(&nd::Array2<C64>) -> nd::Array2<C64> + '_
    where F: Fn(&S, &S) -> Option<f64> + 'a
    {
        move |rho: &nd::Array2<C64>| self.op_with(rho, &f_decay)
    }
}

/// Initialization data for [`LOperator`].
#[derive(Copy, Clone, Debug)]
pub struct LParams<'a, S>
where S: SpontaneousDecay
{
    pub basis: &'a Basis<S>
}

impl<'a, S> LOp<'a, S> for LOperator<'a, S>
where S: SpontaneousDecay
{
    type Params = LParams<'a, S>;
    type Basis = Basis<S>;

    fn new_operator(params: Self::Params) -> Self {
        let LParams { basis } = params;
        Self::new(basis)
    }

    fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64> {
        self.op(rho)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis }
}

