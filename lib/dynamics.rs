//! Constructs to calculate quantities relevant to driven multilevel dynamics
//! and associated initial states.

use std::{
    collections::HashSet,
    f64::consts::TAU,
};
// use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use ndarray_linalg::{ EighInto, InverseInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rand::{ prelude as rnd, Rng };
use crate::{
    hilbert::{
        BasisState,
        SpinState,
        RydbergState,
        TrappedMagic,
        Fock,
        SpontaneousDecay,
        Basis,
        ProdBasis,
    },
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
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DriveParams {
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
        frequency: fn(f64) -> f64,
        /// Drive strength Ω(t) (radians)
        strength: fn(f64) -> f64,
        /// Phase offset φ (radians)
        phase: f64,
    },
}

impl DriveParams {
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
                let w = time.mapv(frequency);
                let ph = *phase + trapz_prog_nonuniform(&w, time);
                let W = time.mapv(strength);
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
    pub drive: DriveParams,
    pub polarization: PolarizationParams,
}

impl<'a, S> HBuilder<'a, S>
where S: SpinState
{
    /// Create a new `HBuilder`.
    pub fn new(
        basis: &'a Basis<S>,
        drive: DriveParams,
        polarization: PolarizationParams,
    ) -> Self
    {
        Self { basis, drive, polarization }
    }

    /// Get a reference to the basis.
    pub fn basis(&self) -> &Basis<S> { self.basis }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let (drive_phase, drive_strength) = self.drive.gen_time_dep(time);
        if drive_strength.iter().map(|Wk| Wk.abs()).sum::<f64>() < 1e-12 {
            return H;
        }
        let state_phases = self.basis.gen_time_dep(time);
        let mut visited: HashSet<(&S, &S)> = HashSet::new();
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
                    *Wk * drive_weight
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
        match self {
            Self::AllToAll(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| s1.c6_with(s2))
                            .map(|c6| c6 / r.powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
            Self::Chain(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| s1.c6_with(s2))
                            .map(|c6| c6 / (*r * (j as f64 - i as f64)).powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
        }
    }
}

/// Hamiltonian builder for a driven multi-atom system including ~1/r^6 Rydberg
/// interactions.
#[derive(Clone, Debug)]
pub struct HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    sites: Vec<HBuilder<'a, S>>,
    prod_basis: ProdBasis<S>,
    pub coupling: RydbergCoupling,
}

impl<'a, S> HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
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
        Self { sites, prod_basis, coupling }
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
        Self { sites, prod_basis, coupling }
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

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
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
        let mut visited: HashSet<(&Vec<S>, &Vec<S>)> = HashSet::new();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: nd::Array1<C64>;
        let mut shift_phase_conj: nd::Array1<C64>;
        let iter
            = self.prod_basis.keys().enumerate()
            .cartesian_product(self.prod_basis.keys().enumerate());
        for ((j, ss1), (i, ss2)) in iter {
            if visited.contains(&(ss2, ss1)) { continue; }
            shift1 = self.coupling.compute_shift(ss1);
            shift2 = self.coupling.compute_shift(ss2);
            shift_phase
                = time.mapv(|t| (C64::i() * (shift2 - shift1) * t).exp());
            shift_phase_conj = shift_phase.mapv(|a| a.conj());

            H.slice_mut(s![i, j, ..])
                .iter_mut()
                .zip(shift_phase)
                .for_each(|(Hijk, shiftk)| *Hijk *= shiftk);
            H.slice_mut(s![j, i, ..])
                .iter_mut()
                .zip(shift_phase_conj)
                .for_each(|(Hijk, shiftk)| *Hijk *= shiftk);
            visited.insert((ss1, ss2));
        }
        H
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FockCutoff {
    Boltzmann(f64),
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
    pub drive: DriveParams,
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
    /// The maximum Fock state included in the model will be that for which the
    /// associated probability in a thermal distribution at `temperature` is at
    /// or below `boltzmann_threshold` (default 10^-6). The driving wavelength
    /// is assumed to be approximately constant over the range of possible
    /// driving frequencies.
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
        drive: DriveParams,
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
            = match fock_cutoff {
                Some(FockCutoff::Boltzmann(p)) => {
                    (-(Z * p).ln() / x - 0.5).ceil()
                },
                Some(FockCutoff::NMax(nmax)) => {
                    nmax as f64
                },
                None => {
                    let p: f64 = 1e-6;
                    (-(Z * p).ln() / x - 0.5).ceil()
                },
            };
        println!("T={:.3e} x={:.3e} Z={:.3e} nmax={:.3e}", temperature, x, Z, nmax);
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

    fn make_exp_ikx(&self) -> nd::Array2<C64> {
        let n_kx = self.nmax + 1;
        let mut kx_op: nd::Array2<C64> = nd::Array2::zeros((n_kx, n_kx));
        kx_op
            .slice_mut(s![..n_kx - 1, 1..n_kx])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(self.lamb_dicke * (n as f64 + 1.0).sqrt());
            });
        kx_op
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

    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let (drive_phase, drive_strength) = self.drive.gen_time_dep(time);
        if drive_strength.iter().map(|Wk| Wk.abs()).sum::<f64>() < 1e-12 {
            return H;
        }
        let state_phases = self.basis.gen_time_dep(time);
        let eikx = self.make_exp_ikx();
        let mut eikx_weight: C64;
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::new();
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
                    *Wk * drive_weight * eikx_weight
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

/// Builder for non-Hermitian, real matrices giving spontaneous decay rates in a
/// single-atom system.
///
/// The `i,j`-th entry of the generated matrix gives the decay rate, in units of
/// angular frequency, of the `j`-th excited state to the `i`-th ground state.
/// Any desired weighting on decay rates should be performed within the
/// [`SpontaneousDecay`] trait.
#[derive(Clone, Debug)]
pub struct YBuilder<S>
where S: SpontaneousDecay
{
    basis: Basis<S>,
    decay: f64,
}

impl<S> YBuilder<S>
where S: SpontaneousDecay
{
    /// Create a new `YBuilder`.
    pub fn new(basis: Basis<S>, decay: f64) -> Self {
        Self { basis, decay }
    }

    /// Get a reference to the basis.
    pub fn basis(&self) -> &Basis<S> { &self.basis }

    /// Get a mutable reference to the basis.
    pub fn basis_mut(&mut self) -> &mut Basis<S> { &mut self.basis }

    /// Compute the decay rate coupling matrix.
    pub fn gen(&self) -> nd::Array2<f64> {
        let n = self.basis.len();
        let Y: nd::Array2<f64>
            = nd::Array1::from_iter(
                self.basis.keys()
                .cartesian_product(self.basis.keys())
                .map(|(sg, se)| se.decay_rate(sg).unwrap_or(0.0))
            )
            .into_shape((n, n))
            .expect("YBuilder::gen: error reshaping array");
        Y
    }
}

