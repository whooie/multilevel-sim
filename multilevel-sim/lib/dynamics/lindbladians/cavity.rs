//! A collection of atoms uniformly coupled to a number of optical cavity modes.
//!
//! See also [`hamiltonians::cavity`][super::super::hamiltonians::cavity].

use std::{ fmt, rc::Rc };
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use num_complex::Complex64 as C64;
use crate::{
    dynamics::{
        hamiltonians::{ HBuilderCavity, HBuilderCavityRydberg },
        lindbladians::LOp,
        arraykron,
        cavitykron,
    },
    hilbert::{
        Basis,
        Cavity,
        CavityCoupling,
        RydbergState,
        SpinState,
        SpontaneousDecay,
    },
    rabi::anti_commutator,
};

/// Implements a Lindbladian operator for an array of `N` atoms coupled to `P`
/// cavity modes.
///
/// Here, [`SpontaneousDecay`] gives the rate at which a transition emits
/// photons *not* into any of the cavity modes, and the initialization parameter
/// `kappa` (see [`Self::new`]) gives the rate at which the photons in the
/// cavity modes are leaked to the environment.
#[derive(Clone)]
pub struct LOperatorCavity<'a, const N: usize, const P: usize, S>
where S: SpontaneousDecay
{
    pub(crate) atom_basis: &'a Basis<S>,
    pub(crate) basis: &'a Basis<Cavity<N, P, S>>,
    pub(crate) kappa: [f64; P],
    pub(crate) nmax: [usize; P],
    pub(crate) f_decay: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, const N: usize, const P: usize, S> fmt::Debug
    for LOperatorCavity<'a, N, P, S>
where S: SpontaneousDecay
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
            "LOperatorCavity {{ basis: {:?}, kappa: {:?}, f_decay: ",
            self.basis, self.kappa,
        )?;
        if self.f_decay.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, const N: usize, const P: usize, S> LOperatorCavity<'a, N, P, S>
where S: SpontaneousDecay
{
    fn def_decay(s1: &S, s2: &S) -> Option<f64> { s1.decay_rate(s2) }

    /// Create a new `LOperatorCavity`.
    pub fn new_raw(
        atom_basis: &'a Basis<S>,
        basis: &'a Basis<Cavity<N, P, S>>,
        kappa: [f64; P],
        nmax: [usize; P],
    ) -> Self {
        Self { atom_basis, basis, kappa, nmax, f_decay: None }
    }

    /// Create a new `LOperatorCavity` using data from a borrowed
    /// [`HBuilderCavity`].
    pub fn new(
        hbuilder: &'a HBuilderCavity<'a, N, P, S>,
        kappa: [f64; P],
    ) -> Self
    where S: SpinState + CavityCoupling<P>
    {
        Self::new_raw(
            hbuilder.atom_basis,
            &hbuilder.basis,
            kappa,
            hbuilder.nmax,
        )
    }

    /// Create a new `LOperatorCavity` using data from a borrowed
    /// [`HBuilderCavityRydberg`].
    pub fn from_hbuilder_cavity_rydberg(
        hbuilder: &'a HBuilderCavityRydberg<'a, N, P, S>,
        kappa: [f64; P],
    ) -> Self
    where S: SpinState + RydbergState + CavityCoupling<P>
    {
        Self::new(&hbuilder.builder, kappa)
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
    pub fn basis(&self) -> &Basis<Cavity<N, P, S>> { self.basis }

    /// Compute the total decay rate coupling matrix.
    ///
    /// The `(i, j)`-th entry in this matrix is the decay rate from the `i`-th
    /// atom-cavity state to the `j`-th such state.
    ///
    /// **Note**: Lindbladian operators are *not* linear, so this matrix should
    /// not be used in e.g. [`crate::rabi::lindbladian`] to directly generate
    /// the action of the total operator on this system.
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
                .map(|(sn1, sn2)| {
                    let atom_decay: f64
                        = sn1.atomic_states().iter()
                        .zip(sn2.atomic_states())
                        .map(|(s1, s2)| f_decay(s1, s2).unwrap_or(0.0))
                        .sum();
                    let photon_decay: f64
                        = sn1.photons().iter()
                        .zip(sn2.photons())
                        .zip(&self.kappa)
                        .map(|((&n1, &n2), &k)| {
                            (n1 == n2 + 1).then_some((n1 as f64).sqrt() * k)
                                .unwrap_or(0.0)
                        })
                        .sum();
                    atom_decay + photon_decay
                })
            )
            .into_shape((n, n))
            .unwrap();
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
        let n_atom = self.atom_basis.len();
        let n_atom_space = n_atom.pow(N as u32);
        let n_photon_space = self.nmax.iter().copied().map(|m| m + 1).product();
        let mut L: nd::Array2<C64> = nd::Array2::zeros((n, n));
        let mut L_term: nd::Array2<C64>;
        let mut atomic_sp: nd::Array2<C64>;
        let mut sp: nd::Array2<C64>;
        let mut sm: nd::ArrayView2<C64>;
        let iter
            = self.atom_basis.keys().enumerate()
            .cartesian_product(self.atom_basis.keys().enumerate());
        for ((i, s1), (j, s2)) in iter {
            if let Some(y) = f_decay(s1, s2) {
                atomic_sp = nd::Array2::zeros((n_atom, n_atom));
                atomic_sp[[i, j]] += 1.0;
                for atom_idx in 0..N {
                    sp
                        = kron(
                            &arraykron(N, n_atom, atom_idx, &atomic_sp),
                            &nd::Array2::eye(n_photon_space),
                        );
                    sm = sp.t();
                    L_term
                        = (
                            sm.dot(rho).dot(&sp)
                            - anti_commutator(&sp.dot(&sm), rho) / 2.0
                        ) * y;
                    L += &L_term;
                }
            }
        }
        let mut mode_ap: nd::Array2<C64>;
        let mut ap: nd::Array2<C64>;
        let mut am: nd::ArrayView2<C64>;
        let iter
            = self.nmax.iter()
            .zip(&self.kappa)
            .enumerate();
        for (mode_idx, (&mode_max, &mode_kappa)) in iter {
            mode_ap = nd::Array2::zeros((mode_max + 1, mode_max + 1));
            mode_ap.slice_mut(s![1.., ..mode_max])
                .diag_mut().iter_mut().enumerate()
                .for_each(|(n, elem)| { *elem += (n as f64 + 1.0).sqrt(); });
            ap
                = kron(
                    &nd::Array2::eye(n_atom_space),
                    &cavitykron(&self.nmax, mode_idx, &mode_ap),
                );
            am = ap.t();
            L_term
                = (
                    am.dot(rho).dot(&ap) * 2.0
                    - anti_commutator(&ap.dot(&am), rho)
                ) * (2.0 * mode_kappa);
            L += &L_term;
        }
        L
    }
}

/// Initialization data for [`LOperatorCavity`]
#[derive(Copy, Clone, Debug)]
pub enum LCavityParams<'a, const N: usize, const P: usize, S>
where S: SpinState + CavityCoupling<P> + SpontaneousDecay
{
    Raw {
        atom_basis: &'a Basis<S>,
        basis: &'a Basis<Cavity<N, P, S>>,
        kappa: [f64; P],
        nmax: [usize; P],
    },
    Cavity {
        builder: &'a HBuilderCavity<'a, N, P, S>,
        kappa: [f64; P],
    },
}

impl<'a, const N: usize, const P: usize, S> LOp<'a, Cavity<N, P, S>>
    for LOperatorCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P> + SpontaneousDecay
{
    type Params = LCavityParams<'a, N, P, S>;
    type Basis = Basis<Cavity<N, P, S>>;

    fn new_operator(params: Self::Params) -> Self {
        match params {
            LCavityParams::Raw { atom_basis, basis, kappa, nmax } => {
                Self::new_raw(atom_basis, basis, kappa, nmax)
            },
            LCavityParams::Cavity { builder, kappa } => {
                Self::new(builder, kappa)
            },
        }
    }

    fn op(&self, rho: &nd::Array2<C64>) -> nd::Array2<C64> {
        self.op(rho)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis }
}

