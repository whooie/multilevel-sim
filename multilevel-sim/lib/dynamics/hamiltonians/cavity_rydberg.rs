//! A linear chain of Rydberg atoms uniformly coupled to a number of optical
//! cavity modes.
//!
//! See also [`lindbladians::cavity`][super::super::lindbladians::cavity].

use std::{ fmt, rc::Rc };
use itertools::Itertools;
use ndarray::{ self as nd, s };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::{
            cavity::HBuilderCavity,
            rydberg::RydbergCoupling,
            HBuild,
        },
        DriveParams,
        PolarizationParams,
    },
    hilbert::{
        Basis,
        Cavity,
        CavityCoupling,
        PhotonLadder,
        RydbergState,
        SpinState,
    },
};

/// Hamiltonian builder for a collectively driven `N`-site linear array of
/// Rydberg atoms coupled to `P` cavity modes.
#[derive(Clone)]
pub struct HBuilderCavityRydberg<'a, const N: usize, const P: usize, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    pub(crate) builder: HBuilderCavity<'a, N, P, S>,
    pub(crate) ryd: RydbergCoupling,
    pub(crate) f_c6: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, const N: usize, const P: usize, S> fmt::Debug
    for HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBuilderCavityRydberg {{ \
            builder: {:?}, \
            ryd: {:?}, \
            f_c6: ",
            self.builder,
            self.ryd,
        )?;
        if self.f_c6.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, const N: usize, const P: usize, S> AsRef<HBuilderCavity<'a, N, P, S>>
    for HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    fn as_ref(&self) -> &HBuilderCavity<'a, N, P, S> { &self.builder }
}

impl<'a, const N: usize, const P: usize, S> HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    fn def_coupling(s1: &S, s2: &S) -> Option<PhotonLadder> {
        s1.coupling(s2)
    }

    fn def_c6(s1: &S, s2: &S) -> Option<f64> { s1.c6_with(s2) }

    pub(crate) fn do_f_coupling(&self, s1: &S, s2: &S) -> Option<PhotonLadder> {
        self.builder.do_f_coupling(s1, s2)
    }

    fn f_c6(&self, s1: &S, s2: &S) -> Option<f64> {
        if let Some(f) = &self.f_c6 {
            f(s1, s2)
        } else {
            Self::def_c6(s1, s2)
        }
    }

    /// Create a new `HBuilderCavityRydberg`.
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        nmax: [usize; P],
        spacing: f64,
    ) -> Self
    {
        let ryd = RydbergCoupling::Chain(spacing);
        let mut builder
            = HBuilderCavity::new(atom_basis, drive, polarization, nmax);
        builder.basis.iter_mut()
            .for_each(|(ss, e)| *e += ryd.compute_shift(ss.atomic_states()));
        Self { builder, ryd, f_c6: None }
    }

    /// Create a new `HBuilderCavityRydberg` from a [`HBuilderCavity`] and an
    /// array spacing.
    pub fn from_cavity_builder(
        mut builder: HBuilderCavity<'a, N, P, S>,
        spacing: f64,
    ) -> Self
    {
        let ryd = RydbergCoupling::Chain(spacing);
        builder.basis.iter_mut()
            .for_each(|(ss, e)| *e += ryd.compute_shift(ss.atomic_states()));
        Self { builder, ryd, f_c6: None }
    }

    /// Use a provided cavity coupling function instead of the
    /// [`CavityCoupling`] implementation.
    pub fn with_g<F>(mut self, f_coupling: F) -> Self
    where F: Fn(&S, &S) -> Option<PhotonLadder> + 'a
    {
        self.builder = self.builder.with_g(f_coupling);
        self
    }

    /// Use a provided C6 function instead of the [`RydbergState`]
    /// implementation.
    pub fn with_c6<F>(self, f_c6: F) -> Self
    where F: Fn(&S, &S) -> Option<f64> + 'a
    {
        let new_f_c6 = Rc::new(f_c6);
        let Self {
            mut builder,
            ryd,
            f_c6: old_f_c6,
        } = self;
        if let Some(f) = &old_f_c6 {
            builder.basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= ryd.compute_shift_with(
                        ss.atomic_states(), f.as_ref());
                    *e += ryd.compute_shift_with(
                        ss.atomic_states(), new_f_c6.as_ref());
                });
        } else {
            builder.basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= ryd.compute_shift(ss.atomic_states());
                    *e += ryd.compute_shift_with(
                        ss.atomic_states(), new_f_c6.as_ref());
                });
        }
        Self { builder, ryd, f_c6: Some(new_f_c6) }
    }

    /// Return a reference to the atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.builder.atom_basis() }

    /// Return a reference to the full atom-cavity basis.
    pub fn basis(&self) -> &Basis<Cavity<N, P, S>> { self.builder.basis() }

    /// Return the maximum cavity mode numbers for each mode.
    pub fn nmax(&self) -> &[usize; P] { self.builder.nmax() }

    /// Generate the state vector for coherent states over all cavity modes for
    /// a single state of the atomic array.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_vector(&self, atomic_state: &[S; N], alpha: &[C64; P])
        -> Option<nd::Array1<C64>>
    {
        self.builder.coherent_state_vector(atomic_state, alpha)
    }

    /// Generate the state vector for coherent states over all cavity modes for
    /// an arbitrary admixture of atomic array states.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_atomic<F>(&self, atomic_amps: F, alpha: &[C64; P])
        -> nd::Array1<C64>
    where F: Fn(&[S; N], usize, f64) -> C64
    {
        self.builder.coherent_state_atomic(atomic_amps, alpha)
    }

    /// Compute a time-independent Hamiltonian if the uniform drive is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        if let Some(f) = &self.f_c6 {
            self.gen_static_with(Self::def_coupling, f.as_ref())
        } else {
            self.gen_static_with_g(Self::def_coupling)
        }
    }

    /// Like [`Self::gen_static`], but using a provided cavity coupling function
    /// rather than that of the [`CavityCoupling`] implementation.
    pub fn gen_static_with_g<F>(&self, f_coupling: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        if let Some(f) = &self.f_c6 {
            self.gen_static_with(f_coupling, f.as_ref())
        } else {
            let mut H: nd::Array2<C64>
                = self.builder.gen_static_with(f_coupling)?;
            H.diag_mut().iter_mut()
                .zip(self.builder.basis.values())
                .for_each(|(h, e)| { *h = (*e).into(); });
            Some(H)
        }
    }

    /// Like [`Self::gen_static`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn gen_static_with_c6<F>(&self, f_c6: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<f64>
    {
        self.gen_static_with(Self::def_coupling, f_c6)
    }

    /// Like [`Self::gen_static`], but using provided cavity coupling and C6
    /// functions rather than those of the [`CavityCoupling`] and
    /// [`RydbergState`] implementations.
    pub fn gen_static_with<F, G>(&self, f_coupling: F, f_c6: G)
        -> Option<nd::Array2<C64>>
    where
        F: Fn(&S, &S) -> Option<PhotonLadder>,
        G: Fn(&S, &S) -> Option<f64>,
    {
        let mut H: nd::Array2<C64> = self.builder.gen_static_with(f_coupling)?;
        H.diag_mut().iter_mut()
            .zip(self.builder.basis.keys())
            .for_each(|(h, sn)| {
                *h += self.ryd.compute_shift_with(sn.atomic_states(), &f_c6);
            });
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        if let Some(f) = &self.f_c6 {
            self.diagonalize_with(Self::def_coupling, f.as_ref())
        } else {
            self.diagonalize_with(Self::def_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn diagonalize_with_g<F>(&self, f_coupling: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        if let Some(f) = &self.f_c6 {
            self.diagonalize_with(f_coupling, f.as_ref())
        } else {
            self.diagonalize_with(f_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn diagonalize_with_c6<F>(&self, f_c6: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        self.diagonalize_with(Self::def_coupling, f_c6)
    }

    /// Like [`Self::diagonalize`], but using provided cavity coupling and C6
    /// functions rather than those of the [`CavityCoupling`] and
    /// [`RydbergState`] implementations.
    pub fn diagonalize_with<F, G>(&self, f_coupling: F, f_c6: G)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where
        F: Fn(&S, &S) -> Option<PhotonLadder>,
        G: Fn(&S, &S) -> Option<f64>,
    {
        match self.gen_static_with(f_coupling, f_c6)?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// ground state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        if let Some(f) = &self.f_c6 {
            self.ground_state_with(Self::def_coupling, f.as_ref())
        } else {
            self.ground_state_with(Self::def_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::ground_state`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn ground_state_with_g<F>(&self, f_coupling: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        if let Some(f) = &self.f_c6 {
            self.ground_state_with(f_coupling, f.as_ref())
        } else {
            self.ground_state_with(f_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::ground_state`], but using a provided C6 function rather
    /// than that of the [`RydbergState`] implementation.
    pub fn ground_state_with_c6<F>(&self, f_c6: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        self.ground_state_with(Self::def_coupling, f_c6)
    }

    /// Like [`Self::ground_state`], but using provided cavity coupling and C6
    /// functions rather than those of the [`CavityCoupling`] and
    /// [`RydbergState`] implementations.
    pub fn ground_state_with<F, G>(&self, f_coupling: F, f_c6: G)
        -> Option<(f64, nd::Array1<C64>)>
    where
        F: Fn(&S, &S) -> Option<PhotonLadder>,
        G: Fn(&S, &S) -> Option<f64>
    {
        let (E, V) = self.diagonalize_with(f_coupling, f_c6)?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        if let Some(f) = &self.f_c6 {
            self.gen_at_with(t, Self::def_coupling, f.as_ref())
        } else {
            self.gen_at_with(t, Self::def_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::gen_at`], but using a provided cavity coupling function
    /// rather than that of the [`CavityCoupling`] implementation.
    pub fn gen_at_with_g<F>(&self, t: f64, f_coupling: F) -> nd::Array2<C64>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        if let Some(f) = &self.f_c6 {
            self.gen_at_with(t, f_coupling, f.as_ref())
        } else {
            self.gen_at_with(t, f_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::gen_at`], but using a provided C6 function rather than that
    /// of the the [`RydbergState`] implementation.
    pub fn gen_at_with_c6<F>(&self, t: f64, f_c6: F) -> nd::Array2<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        self.gen_at_with(t, Self::def_coupling, f_c6)
    }

    /// Like [`Self::gen_at`], but using provided cavity coupling and C6
    /// functions rather than those of the [`CavityCoupling`] and
    /// [`RydbergState`] implementations.
    pub fn gen_at_with<F, G>(&self, t: f64, f_coupling: F, f_c6: G)
        -> nd::Array2<C64>
    where
        F: Fn(&S, &S) -> Option<PhotonLadder>,
        G: Fn(&S, &S) -> Option<f64>,
    {
        let mut H: nd::Array2<C64> = self.builder.gen_at_with(t, f_coupling);
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: C64;
        let iter
            = self.basis().keys().enumerate()
            .cartesian_product(self.basis().keys().enumerate());
        for ((j, sn1), (i, sn2)) in iter {
            if visited.contains(&(sn2, sn1)) { continue; }
            shift1 = self.ryd.compute_shift_with(sn1.atomic_states(), &f_c6);
            shift2 = self.ryd.compute_shift_with(sn2.atomic_states(), &f_c6);
            if (shift2 - shift1).abs() < 1e-15 { continue; }
            shift_phase = (C64::i() * (shift2 - shift1) * t).exp();
            H[[i, j]] *= shift_phase;
            H[[j, i]] *= shift_phase.conj();
            visited.insert((sn1, sn2));
        }
        H
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        if let Some(f) = &self.f_c6 {
            self.gen_with(time, Self::def_coupling, f.as_ref())
        } else {
            self.gen_with(time, Self::def_coupling, Self::def_c6)
        }
    }

    /// Like [`Self::gen`], but using a provided cavity coupling function rather
    /// than that of the [`CavityCoupling`] implementation.
    pub fn gen_with_g<F>(&self, time: &nd::Array1<f64>, f_coupling: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        if let Some(f) = &self.f_c6 {
            self.gen_with(time, f_coupling, f.as_ref())
        } else {
            self.gen_with(time, f_coupling, Self::def_c6)
        }
    }

    /// Like[`Self::gen`], but using a provided C6 function rather than that of
    /// the [`RydbergState`] implementation.
    pub fn gen_with_c6<F>(&self, time: &nd::Array1<f64>, f_c6: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        self.gen_with(time, Self::def_coupling, f_c6)
    }

    /// Like [`Self::gen`], but using provided cavity coupling and C6 functions
    /// rather than those of the [`CavityCoupling`] and [`RydbergState`]
    /// implementations.
    pub fn gen_with<F, G>(
        &self,
        time: &nd::Array1<f64>,
        f_coupling: F,
        f_c6: G,
    ) -> nd::Array3<C64>
    where
        F: Fn(&S, &S) -> Option<PhotonLadder>,
        G: Fn(&S, &S) -> Option<f64>,
    {
        let mut H: nd::Array3<C64> = self.builder.gen_with(time, f_coupling);
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: nd::Array1<C64>;
        let mut shift_phase_conj: nd::Array1<C64>;
        let iter
            = self.basis().keys().enumerate()
            .cartesian_product(self.basis().keys().enumerate());
        for ((j, sn1), (i, sn2)) in iter {
            if visited.contains(&(sn2, sn1)) { continue; }
            shift1 = self.ryd.compute_shift_with(sn1.atomic_states(), &f_c6);
            shift2 = self.ryd.compute_shift_with(sn2.atomic_states(), &f_c6);
            if (shift2 - shift1).abs() < 1e-15 { continue; }
            shift_phase
                = time.mapv(|t| (C64::i() * (shift2 - shift1) * t).exp());
            shift_phase_conj = shift_phase.mapv(|a| a.conj());

            H.slice_mut(s![i, j, ..]).iter_mut()
                .zip(shift_phase)
                .for_each(|(Hijk, shiftk)| { *Hijk *= shiftk; });
            H.slice_mut(s![j, i, ..]).iter_mut()
                .zip(shift_phase_conj)
                .for_each(|(Hjik, shiftk)| { *Hjik *= shiftk; });
            visited.insert((sn1, sn2));
        }
        H
    }
}

/// Initialization data for [`HBuilderCavityRydberg`].
#[derive(Clone, Debug)]
pub struct HCavityRydbergParams<'a, const P: usize, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    pub atom_basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub nmax: [usize; P],
    pub spacing: f64,
}

impl<'a, const N: usize, const P: usize, S> HBuild<'a, Cavity<N, P, S>>
    for HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    type Params = HCavityRydbergParams<'a, P, S>;
    type Basis = Basis<Cavity<N, P, S>>;

    fn new_builder(params: Self::Params) -> Self {
        let HCavityRydbergParams {
            atom_basis,
            drive,
            polarization,
            nmax,
            spacing,
        } = params;
        Self::new(atom_basis, drive, polarization, nmax, spacing)
    }

    fn build_static(&self) -> Option<nd::Array2<C64>> {
        self.gen_static()
    }

    fn build_at(&self, t: f64) -> nd::Array2<C64> {
        self.gen_at(t)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis() }
}

