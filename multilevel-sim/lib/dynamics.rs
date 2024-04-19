//! Constructs to calculate quantities relevant to driven multilevel dynamics
//! and associated initial states.

use std::{
    f64::consts::TAU,
    fmt,
    marker::PhantomData,
    ops::{ Deref, DerefMut },
    rc::Rc,
};
// use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use ndarray_linalg::{ EighInto, InverseInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rand::{ prelude as rnd, Rng };
use rustc_hash::{ FxHashSet as HashSet };
use crate::{
    hilbert::{
        BasisState,
        SpinState,
        RydbergState,
        TrappedMagic,
        CavityCoupling,
        SpontaneousDecay,
        Fock,
        Cavity,
        PhotonLadder,
        Basis,
        ProdBasis,
        outer_prod,
    },
    rabi::state_norm,
    spin::Spin,
};

/// Compute the "progressive" integral of the function `y` using the trapezoidal
/// rule for uniform step size `dx`.
///
/// The progressive integral is defined as
/// ```text
/// I(x) = \int_a^x y(x') dx'
/// ```
fn trapz_prog(y: &nd::Array1<f64>, dx: f64) -> nd::Array1<f64> {
    let mut acc: f64 = 0.0;
    [0.0].into_iter()
        .chain(
            y.iter()
            .zip(y.iter().skip(1))
            .map(|(yk, ykp1)| {
                acc += dx * (*yk + *ykp1) / 2.0;
                acc
            })
        )
        .collect()
}

/// Compute the "progressive" integral of the function `y` using the trapezoidal
/// rule for nonuniform step size `dx`.
///
/// The progressive integral is defined as
/// ```text
/// I(x) = \int_a^x y(x') dx'
/// ```
///
/// *Panics* if the arrays have unequal lengths.
fn trapz_prog_nonuniform(y: &nd::Array1<f64>, x: &nd::Array1<f64>)
    -> nd::Array1<f64>
{
    if y.len() != x.len() {
        panic!("trapz_prog_nonuniform: unequal array lengths");
    }
    let dx: Vec<f64>
        = x.iter().zip(x.iter().skip(1))
        .map(|(xk, xkp1)| *xkp1 - *xk)
        .collect();
    let mut acc: f64 = 0.0;
    [0.0].into_iter()
        .chain(
            y.iter()
            .zip(y.iter().skip(1))
            .zip(dx)
            .map(|((yk, ykp1), dxk)| {
                acc += dxk * (*yk + *ykp1) / 2.0;
                acc
            })
        )
        .collect()
}

/// Calculate the absolute value of the Clebsch-Gordan coefficient for an
/// electric dipole transition.
pub fn transition_cg<S1, S2>(ground: S1, excited: S2) -> Option<f64>
where
    S1: Into<Spin>,
    S2: Into<Spin>,
{
    let ground = ground.into();
    let excited = excited.into();
    let spin_diff = ground.proj().halves() - excited.proj().halves();
    match spin_diff.abs() {
        0 | 2 => {
            let photon: Spin = (2_u32, spin_diff).into();
            let cg
                = (2.0 * ground.total().f() + 1.0).sqrt()
                * crate::spin::w3j(excited, photon, ground.reflected()).abs();
            Some(cg)
        },
        _ => None,
    }
}

/// Calculate the polarization impurity weighting factor for a transition driven
/// with impurity parameter `chi`.
pub fn pol_impurity<S1, S2>(ground: S1, excited: S2, q: Transition, chi: f64)
    -> Option<f64>
where
    S1: Into<Spin>,
    S2: Into<Spin>,
{
    let ground = ground.into();
    let excited = excited.into();
    let spin_diff = excited.proj().halves() - ground.proj().halves();
    let target_diff = q.to_spin_halves();
    match (spin_diff - target_diff).abs() {
        0 => Some((1.0 - chi).sqrt()),
        2 => Some((chi / 2.0).sqrt()),
        _ => None,
    }
}

/// Calculate the polarization weighting factor for a transition driven with
/// well-defined polarization.
///
/// `alpha` is the mixing angle between horizontal and vertical components,
/// `beta` is their relative phase angle (both relative to horizontal), and
/// `theta` is the angle of the incoming *k*-vector relative to the magnetic
/// field/quantization axis.
pub fn pol_poincare<S1, S2>(
    ground: S1,
    excited: S2,
    alpha: f64,
    beta: f64,
    theta: f64,
) -> Option<C64>
where
    S1: Into<Spin>,
    S2: Into<Spin>,
{
    let ground = ground.into();
    let excited = excited.into();
    match excited.proj().halves() - ground.proj().halves() {
        -2 => Some(
            (
                alpha.cos()
                + C64::i() * (C64::i() * beta).exp() * alpha.sin() * theta.cos()
            ) / 2.0_f64.sqrt()
        ),
        0 => Some((C64::i() * beta).exp() * alpha.sin() * theta.sin()),
        2 => Some(
            (
                alpha.cos()
                - C64::i() * (C64::i() * beta).exp() * alpha.sin() * theta.cos()
            ) / 2.0_f64.sqrt()
        ),
        _ => None,
    }
}

/// Names a specific electric dipole transition.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transition {
    SigmaMinus,
    Pi,
    SigmaPlus,
}

impl Transition {
    /// Convert to an equivalent Δm value, quantified as a number of halves.
    ///
    /// See also [`Spin::halves`].
    pub fn to_spin_halves(&self) -> i32 {
        match *self {
            Self::SigmaMinus => -2,
            Self::Pi => 0,
            Self::SigmaPlus => 2,
        }
    }
}

/// Parameterization of the driving beam's polarization.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PolarizationParams {
    /// The driving beam does not have well-defined polarization, and is instead
    /// characterized by only an impurity parameter `chi`, which describes the
    /// relative proportions of power distributed across the possible σ± and π
    /// polarizations. Polarizations other than `principal` are assumed to have
    /// equal power.
    Impurity {
        /// Principal polarization.
        principal: Transition,
        /// Polarization impurity parameter; must be between 0 and 1.
        chi: f64,
    },
    /// The driving beam has well-defined polarization that is resolved in the
    /// H/V basis giving two characterizing angles, `alpha` and `beta`, along
    /// with a third, `theta`, that describes the angle of incidence of the
    /// incoming beam (assumed to be a plane wave).
    Poincare {
        /// Mixing angle, in radians, between horizontal and vertical, relative
        /// to horizontal.
        alpha: f64,
        /// Phase angle, in radians, between horizontal and vertical, relative
        /// to horizontal.
        beta: f64,
        /// Angle, in radians, of the incoming *k*-vector relative to the
        /// magnetic field/quantization axis.
        theta: f64,
    }
}

impl PolarizationParams {
    /// Calculates the appropriate weighting factor for a transition.
    pub fn drive_weight<S1, S2>(&self, ground: S1, excited: S2) -> Option<C64>
    where
        S1: Into<Spin>,
        S2: Into<Spin>,
    {
        match *self {
            Self::Impurity { principal, chi }
                => pol_impurity(ground, excited, principal, chi).map(C64::from),
            Self::Poincare { alpha, beta, theta }
                => pol_poincare(ground, excited, alpha, beta, theta),
        }
    }
}

/// Parameterization of the driving beam's power and frequency.
#[derive(Clone)]
pub enum DriveParams<'a> {
    /// Constant drive parameters.
    Constant {
        /// Drive frequency ω (radians)
        frequency: f64,
        /// Drive strength Ω (radians)
        strength: f64,
        /// Phase offset φ (radians)
        phase: f64,
    },
    /// Time-varied drive parameters.
    Variable {
        /// Drive frequency ω(t) (radians)
        frequency: Rc<dyn Fn(f64) -> f64 + 'a>,
        /// Drive strength Ω(t) (radians)
        strength: Rc<dyn Fn(f64) -> f64 + 'a>,
        /// Phase offset φ (radians)
        phase: f64,
    },
}

impl<'a> std::fmt::Debug for DriveParams<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant { frequency, strength, phase } => {
                write!(f,
                    "Constant {{ \
                    frequency: {:?}, \
                    strength: {:?}, \
                    phase: {:?} \
                    }}",
                    frequency, strength, phase,
                )
            },
            Self::Variable { frequency: _, strength: _, phase } => {
                write!(f,
                    "Variable {{ \
                    frequency: Rc<...>, \
                    strength: Rc<...>, \
                    phase: {:?} \
                    }}",
                    phase,
                )
            },
        }
    }
}

impl<'a> DriveParams<'a> {
    /// Create a new `DriveParams::Constant`.
    pub fn new_constant(frequency: f64, strength: f64, phase: f64) -> Self {
        Self::Constant { frequency, strength, phase }
    }

    /// Create a new `DriveParams::Variable`.
    pub fn new_variable<F1, F2>(frequency: F1, strength: F2, phase: f64)
        -> Self
    where
        F1: Fn(f64) -> f64 + 'a,
        F2: Fn(f64) -> f64 + 'a,
    {
        Self::Variable {
            frequency: Rc::new(frequency),
            strength: Rc::new(strength),
            phase,
        }
    }

    /// Calculate the time dependence of the drive phase (integral of frequency
    /// with phase offset) and strength over an array of time coordinates.
    pub fn gen_time_dep(&self, time: &nd::Array1<f64>)
        -> (nd::Array1<f64>, nd::Array1<f64>)
    {
        match self {
            Self::Constant { frequency, strength, phase } => {
                let ph = *phase + *frequency * time;
                let W = time.mapv(|_| *strength);
                (ph, W)
            },
            Self::Variable { frequency, strength, phase } => {
                let w = time.mapv(|t| (frequency.as_ref())(t));
                let ph = *phase + trapz_prog_nonuniform(&w, time);
                let W = time.mapv(|t| (strength.as_ref())(t));
                (ph, W)
            },
        }
    }
}

/// Hamiltonian builder for a driven single-atom system.
#[derive(Clone, Debug)]
pub struct HBuilder<'a, S>
where S: SpinState
{
    basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
}

impl<'a, S> HBuilder<'a, S>
where S: SpinState
{
    /// Create a new `HBuilder`.
    pub fn new(
        basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
    ) -> Self
    {
        Self { basis, drive, polarization }
    }

    /// Get a reference to the basis.
    pub fn basis(&self) -> &Basis<S> { self.basis }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        let DriveParams::Constant { frequency, strength, phase }
            = &self.drive else { return None; };
        let mut H: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis.values().map(|e| C64::from(*e))
                    .collect::<nd::Array1<C64>>()
            );
        let mut visited: HashSet<(&S, &S)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: C64;
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, s1), (i, s2)) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            drive = 0.5 * drive_weight * C64::from_polar(*strength, *phase);

            H[[i, j]] = drive;
            H[[j, i]] = drive.conj();
            H[[i.max(j); 2]] -= frequency;
            visited.insert((s1, s2));
        }
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        match self.gen_static()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        let (E, V) = self.diagonalize()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let (drive_phase, drive_strength) = self.drive.gen_time_dep(time);
        if drive_strength.iter().map(|Wk| Wk.abs()).sum::<f64>() < 1e-15 {
            return H;
        }
        let state_phases = self.basis.gen_time_dep(time);
        let mut visited: HashSet<(&S, &S)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: nd::Array1<C64>;
        let mut drive_conj: nd::Array1<C64>;
        let iter
            = state_phases.iter().enumerate()
            .cartesian_product(state_phases.iter().enumerate());
        for ((j, (s1, ph1)), (i, (s2, ph2))) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            drive
                = ph1.iter().zip(ph2).zip(&drive_phase).zip(&drive_strength)
                .map(|(((ph1k, ph2k), phk), Wk)| {
                    *Wk * 0.5 * drive_weight
                        * (C64::i() * (*phk - (*ph2k - *ph1k))).exp()
                })
                .collect();
            drive_conj = drive.mapv(|a| a.conj());

            drive.move_into(H.slice_mut(s![i, j, ..]));
            drive_conj.move_into(H.slice_mut(s![j, i, ..]));
            visited.insert((s1, s2));
        }
        H
    }
}

fn multiatom_kron<'a, I>(HH: I) -> nd::Array2<C64>
where I: IntoIterator<Item = nd::ArrayView2<'a, C64>>
{
    let HH: Vec<nd::ArrayView2<C64>> = HH.into_iter().collect();
    if !HH.iter().all(|H| H.is_square()) {
        panic!("multiatom_kron: encountered non-square matrix");
    }
    let nsize: usize = HH.iter().map(|H| H.shape()[0]).product();
    let mut eyesize1: usize;
    let mut eye1: nd::Array2<C64>;
    let mut eyesize2: usize;
    let mut eye2: nd::Array2<C64>;
    let mut term: nd::Array2<C64>;
    let mut acc: nd::Array2<C64> = nd::Array::zeros((nsize, nsize));
    for (k, Hk) in HH.iter().enumerate() {
        eyesize1 = HH.iter().take(k).map(|H| H.shape()[0]).product();
        eye1 = nd::Array2::eye(eyesize1);
        eyesize2 = HH.iter().skip(k + 1).map(|H| H.shape()[0]).product();
        eye2 = nd::Array2::eye(eyesize2);
        term = kron(&kron(&eye1, Hk), &eye2);
        acc += &term;
    }
    acc
}

/// Connectivity between atoms.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RydbergCoupling {
    /// All-to-all coupling, where every atom is taken to be a fixed distance
    /// from all others.
    AllToAll(f64),
    /// Arrangement in a 1D chain with fixed spacing.
    Chain(f64),
}

impl RydbergCoupling {
    /// Compute the total Rydberg shift for a single multi-atom state.
    pub fn compute_shift<S>(&self, state: &[S]) -> f64
    where S: RydbergState
    {
        self.compute_shift_with(state, |s1, s2| s1.c6_with(s2))
    }

    /// Compute the total Rydberg shift for a single multi-atom state using a
    /// provided C6 function rather than that of the [`RydbergState`]
    /// implementation.
    pub fn compute_shift_with<S, F>(&self, state: &[S], f_c6: F) -> f64
    where F: Fn(&S, &S) -> Option<f64>
    {
        match self {
            Self::AllToAll(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| f_c6(s1, s2))
                            .map(|c6| c6 / r.powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
            Self::Chain(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| f_c6(s1, s2))
                            .map(|c6| c6 / (*r * (j as f64 - i as f64)).powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
        }
    }
}

/// Hamiltonian builder for a driven multi-atom system including ~1/r^6 Rydberg
/// interactions.
#[derive(Clone)]
pub struct HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    sites: Vec<HBuilder<'a, S>>,
    prod_basis: ProdBasis<S>,
    pub coupling: RydbergCoupling,
    f_c6: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, S> fmt::Debug for HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBuilderRydberg {{ \
            sites: {:?}, \
            prod_basis: {:?}, \
            coupling: {:?}, \
            f_c6: ",
            self.sites,
            self.prod_basis,
            self.coupling,
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

impl<'a, S> HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    fn def_c6(s1: &S, s2: &S) -> Option<f64> { s1.c6_with(s2) }

    /// Create a new `HBuilderRydberg` where the drives for each atom are
    /// specified individually.
    pub fn new<I>(
        sites: I,
        coupling: RydbergCoupling,
    ) -> Self
    where I: IntoIterator<Item = HBuilder<'a, S>>
    {
        let sites: Vec<HBuilder<S>> = sites.into_iter().collect();
        let mut prod_basis
            = ProdBasis::from_kron(
                sites.iter().map(|builder| builder.basis()));
        prod_basis.iter_mut()
            .for_each(|(ss, e)| *e += coupling.compute_shift(ss));
        Self { sites, prod_basis, coupling, f_c6: None }
    }

    /// Create a new `HBuilderRydberg` where all atoms are driven globally.
    pub fn new_nsites(
        hbuilder: HBuilder<'a, S>,
        nsites: usize,
        coupling: RydbergCoupling,
    ) -> Self
    {
        let sites: Vec<HBuilder<S>>
            = (0..nsites).map(|_| hbuilder.clone()).collect();
        let mut prod_basis
            = ProdBasis::from_kron(
                sites.iter().map(|builder| builder.basis()));
        prod_basis.iter_mut()
            .for_each(|(ss, e)| *e += coupling.compute_shift(ss));
        Self { sites, prod_basis, coupling, f_c6: None }
    }

    /// Use a provided C6 function instead of the [`RydbergState`]
    /// implementation.
    pub fn with_c6<F>(self, f_c6: F) -> Self
    where F: Fn(&S, &S) -> Option<f64> + 'a
    {
        let new_f_c6 = Rc::new(f_c6);
        let Self {
            sites,
            mut prod_basis,
            coupling,
            f_c6: old_f_c6,
        } = self;
        if let Some(f) = &old_f_c6 {
            prod_basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= coupling.compute_shift_with(ss, f.as_ref());
                    *e += coupling.compute_shift_with(ss, new_f_c6.as_ref());
                });
        } else {
            prod_basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= coupling.compute_shift(ss);
                    *e += coupling.compute_shift_with(ss, new_f_c6.as_ref());
                });
        }
        Self { sites, prod_basis, coupling, f_c6: Some(new_f_c6) }
    }

    /// Return a reference to the [`HBuilder`] for site `index`.
    pub fn hbuilder(&self, index: usize) -> Option<&HBuilder<S>> {
        self.sites.get(index)
    }

    /// Return a reference to the basis for site `index`.
    pub fn basis(&self, index: usize) -> Option<&Basis<S>> {
        self.sites.get(index).map(|builder| builder.basis())
    }

    /// Compute and return a reference to a [`ProdBasis`] holding all n-atom
    /// states and energies including Rydberg shifts.
    pub fn prod_basis(&self) -> &ProdBasis<S> { &self.prod_basis }

    /// Compute a time-independent Hamiltonian if all drives are
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive(s) in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        if let Some(f) = &self.f_c6 {
            self.gen_static_with(f.as_ref())
        } else {
            self.gen_static()
        }
    }

    /// Like [`Self::gen_static`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn gen_static_with<F>(&self, f_c6: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let sites_H: Vec<nd::Array2<C64>>
            = self.sites.iter()
            .map(|site| site.gen_static())
            .collect::<Option<Vec<_>>>()?;
        let mut H: nd::Array2<C64>
            = multiatom_kron(sites_H.iter().map(|h| h.view()));
        self.prod_basis.keys()
            .zip(H.diag_mut())
            .for_each(|(ss, e)| {
                *e += self.coupling.compute_shift_with(ss, &f_c6);
            });
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        if let Some(f) = &self.f_c6 {
            self.diagonalize_with(f.as_ref())
        } else {
            self.diagonalize_with(Self::def_c6)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn diagonalize_with<F>(&self, f_c6: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        match self.gen_static_with(f_c6)?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        if let Some(f) = &self.f_c6 {
            self.ground_state_with(f.as_ref())
        } else {
            self.ground_state_with(Self::def_c6)
        }
    }

    /// Like [`Self::ground_state`], but using a provided C6 function rather
    /// than that of the [`RydbergState`] implementation.
    pub fn ground_state_with<F>(&self, f_c6: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let (E, V) = self.diagonalize_with(f_c6)?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        if let Some(f) = &self.f_c6 {
            self.gen_with(time, f.as_ref())
        } else {
            self.gen_with(time, Self::def_c6)
        }
    }

    /// Like [`Self::gen`], but using a provided C6 function rather than that of
    /// the [`RydbergState`] implementation.
    pub fn gen_with<F>(&self, time: &nd::Array1<f64>, f_c6: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let sites_H: Vec<nd::Array3<C64>>
            = self.sites.iter().map(|site| site.gen(time)).collect();
        let H: Vec<nd::Array2<C64>>
            = (0..time.len())
            .map(|k| {
                multiatom_kron(
                    sites_H.iter().map(|H| H.slice(s![.., .., k])))
            })
            .collect();
        let mut H: nd::Array3<C64>
            = nd::stack(
                nd::Axis(2),
                &H.iter().map(|Ht| Ht.view()).collect::<Vec<_>>()
            ).unwrap();
        let mut visited: HashSet<(&Vec<S>, &Vec<S>)> = HashSet::default();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: nd::Array1<C64>;
        let mut shift_phase_conj: nd::Array1<C64>;
        let iter
            = self.prod_basis.keys().enumerate()
            .cartesian_product(self.prod_basis.keys().enumerate());
        for ((j, ss1), (i, ss2)) in iter {
            if visited.contains(&(ss2, ss1)) { continue; }
            shift1 = self.coupling.compute_shift_with(ss1, &f_c6);
            shift2 = self.coupling.compute_shift_with(ss2, &f_c6);
            if (shift2 - shift1).abs() < 1e-15 { continue; }
            shift_phase
                = time.mapv(|t| (C64::i() * (shift2 - shift1) * t).exp());
            shift_phase_conj = shift_phase.mapv(|a| a.conj());

            H.slice_mut(s![i, j, ..]).iter_mut()
                .zip(shift_phase)
                .for_each(|(Hijk, shiftk)| *Hijk *= shiftk);
            H.slice_mut(s![j, i, ..]).iter_mut()
                .zip(shift_phase_conj)
                .for_each(|(Hjik, shiftk)| *Hjik *= shiftk);
            visited.insert((ss1, ss2));
        }
        H
    }
}

/// Determines the maximum Fock state index included in the simulation.
#[derive(Copy, Clone, Debug)]
pub enum FockCutoff {
    /// Set the maximum Fock state as that for which the Boltzmann distribution
    /// crosses this threshold.
    Boltzmann(f64),
    /// Set the maximum Fock state index explicitly.
    NMax(usize),
}

/// Parameters for [`HBuilderMagicTrap`].
#[derive(Copy, Clone, Debug)]
pub struct MotionalParams {
    /// Atomic mass (see [`HBuilderMagicTrap`] for info on units)
    pub mass: f64,
    /// Wavelength of the drive in meters, assumed constant over the range of
    /// possible drive frequencies.
    pub wavelength: f64,
    /// Initial atom temperature (see [`HBuilderMagicTrap`] for info on
    /// units).
    pub temperature: f64,
    /// Boltzmann distribution cutoff to determine how many motional states are
    /// included in the model.
    pub fock_cutoff: Option<FockCutoff>,
}

/// Hamiltonian builder for a driven single-atom system including effects from
/// motion in a magic, harmonic trap.
#[derive(Clone, Debug)]
pub struct HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    atom_basis: &'a Basis<S>,
    basis: Basis<Fock<S>>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    mass: f64,
    wavelength: f64,
    temperature: f64,
    nmax: usize,
    x: f64,
    Z: f64,
    lamb_dicke: f64,
}

impl<'a, S> HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    const HBAR: f64 = 1.0545718001391127e-34; // J / Hz / 2π
    const KB: f64 = 1.38064852e-23; // J / K

    /// Trap frequency, in units of angular frequency, for all states.
    const TRAP_FREQ: f64 = S::TRAP_FREQ;

    /// Create a new `HBuilderMagicTrap`.
    ///
    /// The maximum Fock state included in the model, if not [set
    /// explicitly][FockCutoff::NMax], will be that for which the associated
    /// probability in a thermal distribution at `temperature` is at or below
    /// `boltzmann_threshold` (default 10^-6). The driving wavelength is assumed
    /// to be approximately constant over the range of possible driving
    /// frequencies.
    ///
    /// # Units
    /// To keep the ratio ħω/kT invariant, temperature should be provided in
    /// units that are commensurate with those used for frequency/energy: For
    /// units where `10^n ħ = 1`, the temperature should be provided in units
    /// where `10^n = 1` (i.e. units of `10^n K`). To likewise keep the
    /// Lamb-Dicke parameter invariant, the atomic mass should be provided
    /// in units where `10^-n = 1` (i.e. units of `10^-n kg`). The wavelength
    /// should be in meters.
    ///
    /// For example, when working with frequency/energy described in MHz, `n =
    /// 6`, so temperature should be provided in units of 1 MK (`1 MK = 10^6
    /// K`) and the atomic mass should be provided in units of 1 mg (`1 mg =
    /// 10^-6 kg`).
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        params: MotionalParams,
    ) -> Self
    {
        let MotionalParams {
            mass,
            wavelength,
            temperature,
            fock_cutoff,
        } = params;
        let x: f64
            = Self::HBAR * Self::TRAP_FREQ / (Self::KB * temperature);
        let Z: f64 = (2.0 * (x / 2.0).sinh()).recip();
        let nmax: f64
            = match fock_cutoff.unwrap_or(FockCutoff::Boltzmann(1e-6)) {
                FockCutoff::Boltzmann(p) => {
                    (-(Z * p).ln() / x - 0.5).ceil()
                },
                FockCutoff::NMax(nmax) => {
                    nmax as f64
                },
            };
        if nmax > usize::MAX as f64 {
            panic!(
                "HBuilderMagicTrap::new: maximum Fock index overflows usize"
            );
        }
        if atom_basis.len() as f64 * nmax > usize::MAX as f64 {
            panic!(
                "HBuilderMagicTrap::new: maximum atom*Fock index overflows \
                usize"
            );
        }
        let nmax = nmax as usize;
        let basis: Basis<Fock<S>>
            = atom_basis.iter()
            .flat_map(|(s, e)| {
                (0..=nmax)
                    .map(|n| {
                        (
                            Fock::from((s.clone(), n)),
                            *e + Self::TRAP_FREQ * (n as f64 + 0.5),
                        )
                    })
            })
            .collect();
        let lamb_dicke: f64
            = TAU / wavelength
            * (Self::HBAR / 2.0 / mass / Self::TRAP_FREQ).sqrt();
        Self {
            atom_basis,
            basis,
            drive,
            polarization,
            mass,
            wavelength,
            temperature,
            nmax,
            x,
            Z,
            lamb_dicke,
        }
    }

    /// Return a reference to the atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.atom_basis }

    /// Return a reference to the full Fock basis.
    pub fn basis(&self) -> &Basis<Fock<S>> { &self.basis }

    /// Return the atomic mass in original units.
    pub fn mass(&self) -> f64 { self.mass }

    /// Return the laser wavelength in original units.
    pub fn wavelength(&self) -> f64 { self.wavelength }

    /// Return the atomic temperature in original units.
    pub fn temperature(&self) -> f64 { self.temperature }

    /// Return the maximum Fock state index.
    pub fn nmax(&self) -> usize { self.nmax }

    /// Return the Lamb-Dicke parameter.
    pub fn lamb_dicke(&self) -> f64 { self.lamb_dicke }

    /// Generate a thermal state vector following a Boltzmann distribution over
    /// the motional states of a single atomic state.
    ///
    /// The resulting state is given random phases on all motional states,
    /// sampled uniformly over [0, 2π).
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the Boltzmann distribution
    /// cutoff is not sufficiently low, this can cause the average Fock index to
    /// disagree with theory.
    pub fn thermal_state_vector(&self, atomic_state: &S)
        -> Option<nd::Array1<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_state() == atomic_state) {
            return None;
        }
        let mut rng = rnd::thread_rng();
        let mut psi: nd::Array1<C64>
            = self.basis.keys()
            .map(|s| {
                if s.atomic_state() == atomic_state {
                    C64::from_polar(
                        (
                            (-self.x * (s.fock_index() as f64 + 0.5)).exp()
                            / self.Z
                        ).sqrt(),
                        TAU * rng.gen::<f64>(),
                    )
                } else {
                    C64::zero()
                }
            })
            .collect();
        let N: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= N;
        Some(psi)
    }

    /// Generate a thermal state density matrix following a Boltzmann
    /// distribution over the motional states of a single atomic state.
    ///
    /// The resulting state has no off-diagonal elements.
    ///
    /// **Note**: the returned matrix is renormalized such that its trace is
    /// equal to 1; in cases where the Boltzmann distribution cutoff is not
    /// sufficiently low, this can cause the average Fock index to disagree with
    /// theory.
    pub fn thermal_state_density(&self, atomic_state: &S)
        -> Option<nd::Array2<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_state() == atomic_state) {
            return None;
        }
        let mut rho: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis.keys()
                    .map(|s| {
                        if s.atomic_state() == atomic_state {
                            C64::from(
                                (-self.x * (s.fock_index() as f64 + 0.5)).exp()
                                / self.Z
                            )
                        } else {
                            C64::zero()
                        }
                    })
                    .collect::<nd::Array1<C64>>()
            );
        let N: C64 = rho.diag().iter().sum::<C64>();
        rho /= N;
        Some(rho)
    }

    /// Generate a thermal state density matrix with coherence only in the
    /// atomic subspace. The motional subspace is taken to follow a (decohered)
    /// Boltzmann distribution.
    ///
    /// **Note**: the returned matrix is renormalized such that its trace is
    /// equal to 1; in cases where the Boltzmann distribution cutoff is not
    /// sufficiently low, this can cause the average Fock index to disagree with
    /// theory.
    ///
    /// *Panics* if the length of `atomic_state` disagrees with the size of the
    /// atomic basis.
    pub fn thermal_density_atomic<F>(&self, amps: F) -> nd::Array2<C64>
    where F: Fn(&S, usize, f64) -> C64
    {
        let mut atom_state: nd::Array1<C64>
            = self.atom_basis.get_vector_weighted(amps);
        let N: C64 = state_norm(&atom_state);
        atom_state /= N;
        let atom_density: nd::Array2<C64>
            = outer_prod(&atom_state, &atom_state);

        let mut motional_density: nd::Array2<C64>
            = nd::Array2::from_diag(
                &(0..=self.nmax)
                    .map(|n| {
                        C64::from(
                            (-self.x * (n as f64 + 0.5)).exp() / self.Z
                        )
                    })
                    .collect::<nd::Array1<C64>>()
            );
        let N: C64 = motional_density.diag().iter().sum::<C64>();
        motional_density /= N;

        kron(&atom_density, &motional_density)
    }

    fn make_x(&self) -> nd::Array2<C64> {
        let n_x = self.nmax + 1;
        let mut x_op: nd::Array2<C64> = nd::Array2::zeros((n_x, n_x));
        let coeff: f64 = self.wavelength * self.lamb_dicke / TAU;
        x_op // a
            .slice_mut(s![..n_x - 1, 1..n_x])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(coeff * (n as f64 + 1.0).sqrt());
            });
        x_op // a^\dagger
            .slice_mut(s![1..n_x, ..n_x - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(coeff * (n as f64 + 1.0).sqrt());
            });
        x_op
    }

    /// Compute the *x* (position) operator over the entire atom-motion basis.
    pub fn gen_x(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_x())
    }

    fn make_p(&self) -> nd::Array2<C64> {
        let n_p = self.nmax + 1;
        let mut p_op: nd::Array2<C64> = nd::Array2::zeros((n_p, n_p));
        let coeff: C64
            = C64::i() * self.wavelength / TAU
            * self.mass * Self::TRAP_FREQ
            * self.lamb_dicke;
        p_op // a
            .slice_mut(s![..n_p - 1, 1..n_p])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = -coeff * (n as f64 + 1.0).sqrt();
            });
        p_op // a^\dagger
            .slice_mut(s![1..n_p, ..n_p - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem =  coeff * (n as f64 + 1.0).sqrt();
            });
        p_op
    }

    /// Compute the *p* (momentum) operator over the entire atom-motion basis.
    pub fn gen_p(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_p())
    }

    fn make_exp_ikx(&self) -> nd::Array2<C64> {
        let n_kx = self.nmax + 1;
        let mut kx_op: nd::Array2<C64> = nd::Array2::zeros((n_kx, n_kx));
        kx_op // a
            .slice_mut(s![..n_kx - 1, 1..n_kx])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(self.lamb_dicke * (n as f64 + 1.0).sqrt());
            });
        kx_op // a^\dagger
            .slice_mut(s![1..n_kx, ..n_kx - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(self.lamb_dicke * (n as f64 + 1.0).sqrt());
            });
        let (evals, evects): (nd::Array1<f64>, nd::Array2<C64>)
            = kx_op.eigh_into(UPLO::Lower)
            .expect("HBuilderMagicTrap::make_exp_ikx: error diagonalizing");
        let L = nd::Array2::from_diag(
            &evals.mapv(|lk| C64::from_polar(1.0, lk)));
        let V = evects.clone();
        let U = evects.inv_into()
            .expect("HBuilderMagicTrap::make_exp_ikx: error inverting");
        V.dot(&L).dot(&U)
    }

    /// Compute the <i>e<sup>ikx</sup></i> (laser phase) operator over the
    /// entire atom-motion basis.
    pub fn gen_eikx(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_exp_ikx())
    }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        let DriveParams::Constant { frequency, strength, phase }
            = &self.drive else { return None; };
        let mut H: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis().values().map(|e| C64::from(*e))
                    .collect::<nd::Array1<C64>>()
            );
        let eikx = self.make_exp_ikx();
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut eikx_weight: C64;
        let mut drive: C64;
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, s1), (i, s2)) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            eikx_weight = eikx[[s2.fock_index(), s1.fock_index()]].conj();
            drive
                = 0.5 * drive_weight * eikx_weight
                * C64::from_polar(*strength, *phase);

            H[[i, j]] = drive;
            H[[j, i]] = drive.conj();
            H[[i.max(j); 2]] -= frequency;
            visited.insert((s1, s2));
        }
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        match self.gen_static()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        let (E, V) = self.diagonalize()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let (drive_phase, drive_strength) = self.drive.gen_time_dep(time);
        if drive_strength.iter().map(|Wk| Wk.abs()).sum::<f64>() < 1e-15 {
            return H;
        }
        let state_phases = self.basis.gen_time_dep(time);
        let eikx = self.make_exp_ikx();
        let mut eikx_weight: C64;
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: nd::Array1<C64>;
        let mut drive_conj: nd::Array1<C64>;
        let iter
            = state_phases.iter().enumerate()
            .cartesian_product(state_phases.iter().enumerate());
        for ((j, (s1, ph1)), (i, (s2, ph2))) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            eikx_weight = eikx[[s2.fock_index(), s1.fock_index()]].conj();
            drive
                = ph1.iter().zip(ph2).zip(&drive_phase).zip(&drive_strength)
                .map(|(((ph1k, ph2k), phk), Wk)| {
                    *Wk * 0.5 * drive_weight * eikx_weight
                        * (C64::i() * (*phk - (*ph2k - *ph1k))).exp()
                })
                .collect();
            drive_conj = drive.mapv(|a| a.conj());

            drive.move_into(H.slice_mut(s![i, j, ..]));
            drive_conj.move_into(H.slice_mut(s![j, i, ..]));
            visited.insert((s1, s2));
        }
        H
    }
}

/// Hamiltonian builder for a collectively driven `N`-site linear array of atoms
/// coupled to `P` cavity modes.
#[derive(Clone)]
pub struct HBuilderCavity<'a, const N: usize, const P: usize, S>
where S: SpinState + CavityCoupling<P>
{
    atom_basis: &'a Basis<S>,
    basis: Basis<Cavity<N, P, S>>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    nmax: [usize; P],
    f_coupling: Option<Rc<dyn Fn(&S, &S) -> Option<PhotonLadder> + 'a>>,
}

impl<'a, const N: usize, const P: usize, S> fmt::Debug
    for HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBuilderCavity {{ \
            atom_basis: {:?}, \
            basis: {:?}, \
            drive: {:?}, \
            polarization: {:?}, \
            nmax: {:?}, \
            f_coupling: ",
            self.atom_basis,
            self.basis,
            self.drive,
            self.polarization,
            self.nmax,
        )?;
        if self.f_coupling.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, const N: usize, const P: usize, S> HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    fn def_coupling(s1: &S, s2: &S) -> Option<PhotonLadder> {
        s1.coupling(s2)
    }

    /// Create a new `HBuilderCavity`.
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        nmax: [usize; P],
    ) -> Self
    {
        let atom_iter
            = (0..N).map(|_| atom_basis.iter()).multi_cartesian_product();
        let cavity_iter
            = nmax.iter().map(|p| 0..=*p).multi_cartesian_product();
        let basis: Basis<Cavity<N, P, S>>
            = atom_iter.cartesian_product(cavity_iter)
            .map(|(ss, nn)| {
                let atoms: [S; N]
                    = ss.iter()
                    .map(|(s, _)| (*s).clone())
                    .collect::<Vec<S>>()
                    .try_into()
                    .unwrap();
                let atom_energy: f64
                    = ss.iter()
                    .map(|(_, e)| *e)
                    .sum();
                let photons: [usize; P] = nn.try_into().unwrap();
                let photon_energy: f64
                    = photons.iter()
                    .zip(&S::MODE_SPACING)
                    .map(|(n, e)| (*n as f64) * *e)
                    .sum();
                ((atoms, photons).into(), atom_energy + photon_energy)
            })
            .collect();
        Self {
            atom_basis,
            basis,
            drive,
            polarization,
            nmax,
            f_coupling: None,
        }
    }

    /// Use a provided cavity coupling function instead of the
    /// [`CavityCoupling`] implementation.
    pub fn with_g<F>(mut self, f_coupling: F) -> Self
    where F: Fn(&S, &S) -> Option<PhotonLadder> + 'a
    {
        self.f_coupling = Some(Rc::new(f_coupling));
        self
    }

    /// Return a reference to the atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.atom_basis }

    /// Return a reference to the full atom-cavity basis.
    pub fn basis(&self) -> &Basis<Cavity<N, P, S>> { &self.basis }

    /// Return the maximum cavity mode numbers for each mode.
    pub fn nmax(&self) -> &[usize; P] { &self.nmax }

    /// Generate the state vector for coherent states over all cavity modes for
    /// a single state of the atomic array.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can casue average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_vector(
        &self,
        atomic_states: &[S; N],
        alpha: &[C64; P],
    ) -> Option<nd::Array1<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_states() == atomic_states) {
            return None;
        }
        let pref: Vec<C64>
            = alpha.iter().map(|a| (-0.5 * a * a.conj()).exp()).collect();
        let mut psi: nd::Array1<C64>
            = self.basis.keys()
            .map(|s| {
                if s.atomic_states() == atomic_states {
                    s.photons().iter()
                        .zip(&pref)
                        .zip(alpha)
                        .map(|((n, p), a)| {
                            let n = *n as i32;
                            let fact_n: f64
                                = (1..=n).map(f64::from).product();
                            p / fact_n.sqrt() * a.powi(n)
                        })
                        .product()
                } else {
                    C64::zero()
                }
            })
            .collect();
        let norm: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= norm;
        Some(psi)
    }

    /// Generate the state vector for coherent states over all cavity modes for
    /// an arbitrary admixture of atomic array states.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_atomic<F>(&self, atom_amps: F, alpha: &[C64; P])
        -> nd::Array1<C64>
    where F: Fn(&[S; N], usize, f64) -> C64
    {
        let pref: Vec<C64>
            = alpha.iter().map(|a| (-0.5 * a * a.conj()).exp()).collect();
        let mut psi: nd::Array1<C64>
            = self.basis.iter().enumerate()
            .map(|(index, (sn, energy))| {
                let atom = atom_amps(sn.atomic_states(), index, *energy);
                let photon: C64
                    = sn.photons().iter()
                    .zip(&pref)
                    .zip(alpha)
                    .map(|((n, p), a)| {
                        let n = *n as i32;
                        let fact_n: f64
                            = (1..=n).map(f64::from).product();
                        p / fact_n.sqrt() * a.powi(n)
                    })
                    .product();
                atom * photon
            })
            .collect();
        let norm: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= norm;
        psi
    }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        if let Some(f) = &self.f_coupling {
            self.gen_static_with(f.as_ref())
        } else {
            self.gen_static_with(Self::def_coupling)
        }
    }

    /// Like [`Self::gen_static`], but using a provided cavity coupling function
    /// rather than that of the [`CavityCoupling`] implementation.
    pub fn gen_static_with<F>(&self, f_coupling: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let H_site: nd::Array2<C64>
            = HBuilder::new(
                self.atom_basis,
                self.drive.clone(),
                self.polarization,
            )
            .gen_static()?;
        let photon_eye: nd::Array2<C64>
            = nd::Array2::eye(self.nmax.iter().map(|&n| n + 1).product());
        let mut H: nd::Array2<C64>
            = kron(
                &multiatom_kron((0..N).map(|_| H_site.view())),
                &photon_eye,
            );
        H.diag_mut().iter_mut()
            .zip(self.basis.values())
            .for_each(|(h, e)| { *h = (*e).into(); });
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut ss1: &[S; N];
        let mut ss2: &[S; N];
        let mut nn1: &[usize; P];
        let mut nn2: &[usize; P];
        let mut coupling: f64 = 0.0;
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, sn1), (i, sn2)) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            nn1 = sn1.photons();
            nn2 = sn2.photons();

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match f_coupling(s1, s2) {
                    None => { continue; }
                    Some(PhotonLadder::Emit(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 + 1 == n2).then_some(g * (n2 as f64).sqrt())
                            })
                        {
                            coupling = gn;
                        }
                    },
                    Some(PhotonLadder::Absorb(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 == n2 + 1).then_some(g * (n1 as f64).sqrt())
                            })
                        {
                            coupling = gn;
                        }
                    },
                }
            }

            H[[i, j]] += coupling;
            H[[j, i]] += coupling;
            visited.insert((sn2, sn1));
        }
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        if let Some(f) = &self.f_coupling {
            self.diagonalize_with(f.as_ref())
        } else {
            self.diagonalize_with(Self::def_coupling)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn diagonalize_with<F>(&self, f_coupling: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        match self.gen_static_with(f_coupling)?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        if let Some(f) = &self.f_coupling {
            self.ground_state_with(f.as_ref())
        } else {
            self.ground_state_with(Self::def_coupling)
        }
    }

    /// Like [`Self::ground_state`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn ground_state_with<F>(&self, f_coupling: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let (E, V) = self.diagonalize_with(f_coupling)?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        if let Some(f) = &self.f_coupling {
            self.gen_with(time, f.as_ref())
        } else {
            self.gen_with(time, Self::def_coupling)
        }
    }

    /// Like [`Self::gen`], but using a provided cavity coupling function rather
    /// than that of the [`CavityCoupling`] implementation.
    pub fn gen_with<F>(&self, time: &nd::Array1<f64>, f_coupling: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let H_site: nd::Array3<C64>
            = HBuilder::new(
                self.atom_basis,
                self.drive.clone(),
                self.polarization,
            )
            .gen(time);
        let photon_eye: nd::Array2<C64>
            = nd::Array2::eye(self.nmax.iter().map(|&n| n + 1).product());
        let H: Vec<nd::Array2<C64>>
            = (0..time.len())
            .map(|k| {
                kron(
                    &multiatom_kron(
                        (0..N).map(|_| H_site.slice(s![.., .., k]))),
                    &photon_eye,
                )
            })
            .collect();
        let mut H: nd::Array3<C64>
            = nd::stack(
                nd::Axis(2),
                &H.iter().map(|Ht| Ht.view()).collect::<Vec<_>>()
            ).unwrap();
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut ss1: &[S; N];
        let mut ss2: &[S; N];
        let mut nn1: &[usize; P];
        let mut nn2: &[usize; P];
        let mut freq: f64;
        let mut coupling: nd::Array1<C64> = nd::Array1::zeros(time.len());
        let mut coupling_conj: nd::Array1<C64>;
        let iter
            = self.basis.iter().enumerate()
            .cartesian_product(self.basis.iter().enumerate());
        for ((j, (sn1, e1)), (i, (sn2, e2))) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            nn1 = sn1.photons();
            nn2 = sn2.photons();

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match f_coupling(s1, s2) {
                    None => { continue; },
                    Some(PhotonLadder::Emit(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 + 1 == n2).then_some(g * (n2 as f64).sqrt())
                            })
                        {
                            freq = *e2 - *e1;
                            coupling.iter_mut()
                                .zip(time)
                                .for_each(|(c, t)| {
                                    *c = C64::from_polar(gn, freq * t);
                                });
                        }
                    },
                    Some(PhotonLadder::Absorb(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 == n2 + 1).then_some(g * (n1 as f64).sqrt())
                            })
                        {
                            freq = *e2 - *e1;
                            coupling.iter_mut()
                                .zip(time)
                                .for_each(|(c, t)| {
                                    *c = C64::from_polar(gn, freq * t);
                                });
                        }
                    },
                }
            }
            coupling_conj = coupling.mapv(|a| a.conj());

            H.slice_mut(s![i, j, ..]).iter_mut()
                .zip(&coupling)
                .for_each(|(Hijk, couplingk)| { *Hijk += *couplingk; });
            H.slice_mut(s![j, i, ..]).iter_mut()
                .zip(coupling_conj)
                .for_each(|(Hjik, couplingk)| { *Hjik += couplingk; });
            visited.insert((sn2, sn1));
        }
        H
    }
}

/// Hamiltonian builder for a collectively driven `N`-site linear array of
/// Rydberg atoms coupled to `P` cavity modes.
#[derive(Clone)]
pub struct HBuilderCavityRydberg<'a, const N: usize, const P: usize, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    builder: HBuilderCavity<'a, N, P, S>,
    ryd: RydbergCoupling,
    f_c6: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
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

impl<'a, const N: usize, const P: usize, S> HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    fn def_coupling(s1: &S, s2: &S) -> Option<PhotonLadder> {
        s1.coupling(s2)
    }

    fn def_c6(s1: &S, s2: &S) -> Option<f64> { s1.c6_with(s2) }

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
    /// the cavity modes are not sufficiently high, this can casue average
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
    /// state is returned.
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
        let v: nd::Array1<C64> = V.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
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

    /// Like[`Self::gen`], but using a provided C6 function rahter than that of
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
            if visited.contains(&(sn1, sn2)) { continue; }
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

/// Basic requirements for any Hamiltonian builder.
pub trait HBuild<'a, S>
where S: BasisState
{
    /// Initialization data type.
    type Params;

    /// Basis type containing state energies.
    type Basis: PartialEq;

    /// Initialize `self`.
    fn new_builder(params: Self::Params) -> Self;

    /// Build the Hamiltonian array.
    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64>;

    /// Return a reference to the basis.
    fn get_basis(&self) -> &Self::Basis;
}

/// Initialization data for [`HBuilder`].
#[derive(Clone, Debug)]
pub struct HBuilderParams<'a, S>
where S: SpinState
{
    pub basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
}

impl<'a, S> HBuild<'a, S> for HBuilder<'a, S>
where S: SpinState
{
    type Params = HBuilderParams<'a, S>;
    type Basis = Basis<S>;

    fn new_builder(params: Self::Params) -> Self {
        let HBuilderParams { basis, drive, polarization } = params;
        Self::new(basis, drive, polarization)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis() }
}

/// Initialization data for [`HBuilderRydberg`].
#[derive(Clone, Debug)]
pub struct HBuilderRydbergParams<'a, I, S>
where
    I: IntoIterator<Item = HBuilder<'a, S>>,
    S: SpinState + RydbergState + 'a,
{
    pub sites: I,
    pub coupling: RydbergCoupling,
}

impl<'a, S> HBuild<'a, S> for HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    type Params = HBuilderRydbergParams<'a, Vec<HBuilder<'a, S>>, S>;
    type Basis = ProdBasis<S>;

    fn new_builder(params: Self::Params) -> Self {
        let HBuilderRydbergParams { sites, coupling } = params;
        Self::new(sites, coupling)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.prod_basis() }
}

/// Initialization data for [`HBuilderMagicTrap`].
#[derive(Clone, Debug)]
pub struct HBuilderMagicTrapParams<'a, S>
where S: SpinState + TrappedMagic
{
    pub basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub motion: MotionalParams,
}

impl<'a, S> HBuild<'a, S> for HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    type Params = HBuilderMagicTrapParams<'a, S>;
    type Basis = Basis<Fock<S>>;

    fn new_builder(params: Self::Params) -> Self {
        let HBuilderMagicTrapParams { basis, drive, polarization, motion }
            = params;
        Self::new(basis, drive, polarization, motion)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis() }
}

/// Collection of objects implementing [`HBuild`] for easy initialization and
/// composition of a series of Hamiltonians.
pub struct SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    builders: Vec<H>,
    state_lifetime: PhantomData<&'a S>,
}

impl<'a, H, S> AsRef<Vec<H>> for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    fn as_ref(&self) -> &Vec<H> { &self.builders }
}

impl<'a, H, S> AsMut<Vec<H>> for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    fn as_mut(&mut self) -> &mut Vec<H> { &mut self.builders }
}

impl<'a, H, S> Deref for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    type Target = Vec<H>;

    fn deref(&self) -> &Self::Target { &self.builders }
}

impl<'a, H, S> DerefMut for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.builders }
}

impl<'a, H, S> From<Vec<H>> for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    fn from(builders: Vec<H>) -> Self {
        Self { builders, state_lifetime: PhantomData }
    }
}

impl<'a, H, S> FromIterator<H::Params> for SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = H::Params>
    {
        Self {
            builders: iter.into_iter().map(H::new_builder).collect(),
            state_lifetime: PhantomData,
        }
    }
}

impl<'a, H, S> SequenceBuilder<'a, H, S>
where
    H: HBuild<'a, S>,
    S: BasisState,
{
    /// Return `Some(true`) if all builders' bases are equal; `None` if no
    /// builders are present.
    pub fn equal_bases(&self) -> Option<bool> {
        self.builders.first()
            .map(|b| b.get_basis())
            .map(|comp| self.builders.iter().all(|b| b.get_basis() == comp))
    }

    /// Return `Some` if at least one builder is present and all builders' bases
    /// are equal.
    pub fn get_basis(&self) -> Option<&H::Basis> {
        if let Some(true) = self.equal_bases() {
            self.builders.first().map(|builder| builder.get_basis())
        } else {
            None
        }
    }

    /// Return the basis of the `n`-th builder.
    pub fn get_basis_at(&self, n: usize) -> Option<&H::Basis> {
        self.builders.get(n).map(|builder| builder.get_basis())
    }

    /// Build the total hamiltonian as the sum of all builders' generated
    /// Hamiltonians.
    pub fn build(&self, time: &nd::Array1<f64>) -> Option<nd::Array3<C64>> {
        self.builders.iter()
            .fold(
                None,
                |acc: Option<nd::Array3<C64>>, builder: &H| {
                    if let Some(h) = acc {
                        Some(h + builder.build(time))
                    } else {
                        Some(builder.build(time))
                    }
                }
            )
    }
}

/// Builder for non-Hermitian, real matrices giving spontaneous decay rates in a
/// single-atom system.
///
/// The `i,j`-th entry of the generated matrix gives the decay rate, in units of
/// angular frequency, of the `j`-th excited state to the `i`-th ground state.
/// Any desired weighting on decay rates (e.g. Clebsch-Gordan coefficients)
/// should be performed within the [`SpontaneousDecay`] impl.
#[derive(Clone)]
pub struct YBuilder<'a, S>
where S: SpontaneousDecay
{
    basis: &'a Basis<S>,
    f_decay: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
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
                .map(|(sg, se)| f_decay(sg, se).unwrap_or(0.0))
            )
            .into_shape((n, n))
            .expect("YBuilder::gen_with: error reshaping array");
        Y
    }
}

