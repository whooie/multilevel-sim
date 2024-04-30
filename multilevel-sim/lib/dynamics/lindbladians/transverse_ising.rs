//! A linear spin chain with a σ<sub>*z*</sub>σ<sub>*z*</sub> Rydberg-like
//! interaction, coupled to a single optical cavity mode.
//!
//! See also [`hamiltonians::transverse_ising`][super::super::hamiltonians::transverse_ising].

use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use crate::{
    dynamics::{
        hamiltonians::HBuilderTransverseIsing,
        lindbladians::LOp,
        arraykron,
    },
    hilbert::{ Basis, Cavity, HSpin },
    rabi::anti_commutator,
};

/// Specialized Lindbladian operator for a `N`-site Rydberg spin chain coupled
/// to a single optical cavity mode.
///
/// See also [`HBuilderTransverseIsing`].
#[derive(Clone, Debug)]
pub struct LOperatorTransverseIsing<'a, const N: usize> {
    pub(crate) basis: &'a Basis<Cavity<N, 1, HSpin>>,
    pub(crate) gamma: f64,
    pub(crate) kappa: f64,
    pub(crate) nmax: usize,
}

impl<'a, const N: usize> LOperatorTransverseIsing<'a, N> {
    /// Create a new `LOperatorTransverseIsing`.
    pub fn new_raw(
        basis: &'a Basis<Cavity<N, 1, HSpin>>,
        gamma: f64,
        kappa: f64,
        nmax: usize,
    ) -> Self
    {
        Self { basis, gamma, kappa, nmax }
    }

    /// Create a new `LOperatorTransverseIsing` using data from a borrowed
    /// [`HBuilderTransverseIsing`].
    pub fn new(
        builder: &'a HBuilderTransverseIsing<N>,
        gamma: f64,
        kappa: f64,
    ) -> Self {
        Self::new_raw(&builder.basis, gamma, kappa, builder.nmax)
    }

    /// Return a reference to the full spin-cavity basis.
    pub fn basis(&self) -> &Basis<Cavity<N, 1, HSpin>> { self.basis }

    /// Return all model parameters.
    pub fn params(&self) -> LTransverseIsingParams<'a, N> {
        LTransverseIsingParams::Raw {
            basis: self.basis,
            gamma: self.gamma,
            kappa: self.kappa,
            nmax: self.nmax,
        }
    }

    /// Perform the operator action on a density matrix.
    pub fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64> {
        let n = self.basis.len();
        let mut L: nd::Array2<C64> = nd::Array2::zeros((n, n));
        let mut L_term: nd::Array2<C64>;
        let atomic_sp: nd::Array2<C64>
            = nd::array![[0.0.into(), 0.0.into()], [1.0.into(), 0.0.into()]];
        let mut sp: nd::Array2<C64>;
        let mut sm: nd::ArrayView2<C64>;
        for atom_idx in 0..N {
            sp
                = kron(
                    &arraykron(N, 2, atom_idx, &atomic_sp),
                    &nd::Array2::eye(self.nmax + 1),
                );
            sm = sp.t();
            L_term
                = (
                    sm.dot(rho).dot(&sp)
                    - anti_commutator(&sp.dot(&sm), rho) / 2.0
                ) * self.gamma;
            L += &L_term;
        }
        let mut mode_ap: nd::Array2<C64>
            = nd::Array2::zeros((self.nmax + 1, self.nmax + 1));
        mode_ap.diag_mut().iter_mut().enumerate()
            .for_each(|(n, elem)| { *elem += (n as f64 + 1.0).sqrt(); });
        let ap: nd::Array2<C64>
            = kron(&nd::Array2::eye(2_usize.pow(N as u32)), &mode_ap);
        let am: nd::ArrayView2<C64> = ap.t();
        L_term
            = (
                am.dot(rho).dot(&ap) * 2.0
                - anti_commutator(&ap.dot(&am), rho)
            ) * (2.0 * self.kappa);
        L += &L_term;
        L
    }
}

/// Initialization data for [`LOperatorTransverseIsing`].
#[derive(Copy, Clone, Debug)]
pub enum LTransverseIsingParams<'a, const N: usize> {
    Raw {
        basis: &'a Basis<Cavity<N, 1, HSpin>>,
        gamma: f64,
        kappa: f64,
        nmax: usize,
    },
    Builder {
        builder: &'a HBuilderTransverseIsing<N>,
        gamma: f64,
        kappa: f64,
    },
}

impl<'a, const N: usize> LOp<'a, Cavity<N, 1, HSpin>>
    for LOperatorTransverseIsing<'a, N>
{
    type Params = LTransverseIsingParams<'a, N>;
    type Basis = Basis<Cavity<N, 1, HSpin>>;

    fn new_operator(params: Self::Params) -> Self {
        match params {
            LTransverseIsingParams::Raw { basis, gamma, kappa, nmax } => {
                Self::new_raw(basis, gamma, kappa, nmax)
            },
            LTransverseIsingParams::Builder { builder, gamma, kappa } => {
                Self::new(builder, gamma, kappa)
            },
        }
    }

    fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64> {
        self.op(rho)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis }
}

