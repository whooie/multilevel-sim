//! Constructs to calculate quantities relevant to driven multilevel dynamics
//! and associated initial states.

use std::rc::Rc;
use ndarray::{ self as nd, s, linalg::kron };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rustc_hash::FxHashMap as HashMap;
use crate::spin::Spin;

pub mod hamiltonians;
pub use hamiltonians::{
    generic::{ HBuilder, HParams },
    rydberg::{ HBuilderRydberg, HRydbergParams, RydbergCoupling },
    magic_trap::{
        HBuilderMagicTrap,
        HMagicTrapParams,
        MotionalParams,
        FockCutoff,
    },
    motional_rydberg::{ HBuilderMotionalRydberg, HMotionalRydbergParams },
    cavity::{ HBuilderCavity, HCavityParams },
    cavity_rydberg::{ HBuilderCavityRydberg, HCavityRydbergParams },
    transverse_ising::{ HBuilderTransverseIsing, HTransverseIsingParams },
    HBuild,
    OverlayBuilder,
    SequenceBuilder,
};

pub mod lindbladians;
pub use lindbladians::{
    generic::{ LOperator, LParams },
    cavity::{ LOperatorCavity, LCavityParams },
    transverse_ising::{ LOperatorTransverseIsing, LTransverseIsingParams },
    LOp,
    YBuilder,
};

/// Compute the "progressive" integral of the function `y` using the trapezoidal
/// rule for uniform step size `dx`.
///
/// The progressive integral is defined as
/// ```text
/// I(x) = \int_a^x y(x') dx'
/// ```
pub(crate) fn trapz_prog(y: &nd::Array1<f64>, dx: f64) -> nd::Array1<f64> {
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
pub(crate) fn trapz_prog_nonuniform(y: &nd::Array1<f64>, x: &nd::Array1<f64>)
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

/// Apply trapezoidal rule to a 1D array sampled at even intervals.
pub(crate) fn trapz<A, X>(y: &nd::Array1<A>, dx: &X) -> A
where
    A: Clone + std::ops::Add<Output = A> + std::ops::Mul<X, Output = A> + Zero,
    X: num_traits::Float + std::ops::Mul<f64, Output = X>,
{
    let n: usize = y.len();
    y[0].clone() * (*dx * 0.5)
        + y.slice(s![1..n - 1]).sum() * *dx
        + y[n - 1].clone() * (*dx * 0.5)
}

pub(crate) fn romberg<F>(
    integrand: F,
    t: f64,
    n_max: Option<usize>,
    epsilon: Option<f64>,
) -> f64
where F: Fn(f64) -> f64
{
    let mut terms: HashMap<(usize, usize), f64> = HashMap::default();
    let n_max = n_max.unwrap_or(31);
    let epsilon = epsilon.unwrap_or(1e-9);
    let mut integ: f64;
    let mut extrap: f64;
    let mut four_m: f64;
    let mut extrap_diff: f64;
    let mut time: nd::Array1<f64>;
    let mut dt: f64;
    let mut y: nd::Array1<f64>;
    for n in 0..=n_max {
        time = nd::Array1::linspace(0.0, t, 2_usize.pow(n as u32) + 1);
        dt = time[1] - time[0];
        y = time.mapv(&integrand);
        integ = trapz(&y, &dt);
        terms.insert((n, 0), integ);
        for m in 1..=n {
            four_m = 4.0_f64.powi(m as i32);
            extrap
                = (four_m - 1.0).recip() * (
                    four_m * *terms.get(&(n, m - 1)).unwrap()
                    - *terms.get(&(n - 1, m - 1)).unwrap()
                );
            terms.insert((n, m), extrap);
        }
        if n > 0 {
            extrap_diff
                = *terms.get(&(n, n)).unwrap()
                - *terms.get(&(n, n - 1)).unwrap();
            if extrap_diff.abs() < epsilon {
                return *terms.get(&(n, n)).unwrap();
            }
        }
    }
    panic!("romberg: failed to converge");
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

    /// Compute the drive phase and strength at a given time.
    pub fn gen_at(&self, t: f64) -> (f64, f64) {
        match self {
            Self::Constant { frequency, strength, phase } => {
                (*phase + *frequency * t, *strength)
            },
            Self::Variable { frequency, strength, phase } => {
                let W = strength(t);
                let ph = *phase + romberg(frequency.as_ref(), t, None, None);
                (ph, W)
            },
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

pub(crate) fn arraykron(
    array_size: usize,
    atom_size: usize,
    atom_idx: usize,
    a: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    let eyesize1 = atom_size.pow(atom_idx as u32);
    let eyesize2 = atom_size.pow((array_size - atom_idx - 1) as u32);
    kron(&kron(&nd::Array2::eye(eyesize1), a), &nd::Array2::eye(eyesize2))
}

pub(crate) fn cavitykron<const P: usize>(
    nmax: &[usize; P],
    mode_idx: usize,
    a: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    let eyesize1 = nmax.iter().take(mode_idx).copied().product();
    let eyesize2 = nmax.iter().skip(mode_idx + 1).copied().product();
    kron(&kron(&nd::Array2::eye(eyesize1), a), &nd::Array2::eye(eyesize2))
}

pub(crate) fn multiatom_kron<'a, I>(HH: I) -> nd::Array2<C64>
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

