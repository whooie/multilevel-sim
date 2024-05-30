//! Functions for numerical integration of the Schrödinger, Liouville, and
//! Lindblad equations.
//!
//! Where unspecified, the last index of a 2D or 3D array corresponds to time,
//! all Hamiltonians decay rates should be in units of angular frequency, and
//! integration is via fourth-order Runge-Kutta.

use std::rc::Rc;
use itertools::Itertools;
use ndarray::{ self as nd, s, Dimension };
use ndarray_linalg::{ self as la, Eigh, Solve };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use crate::{
    dynamics::{ HBuild, LOp },
    hilbert::{ StateIter, outer_prod },
};

pub mod schrodinger;
pub mod liouville;
pub mod lindblad;

pub(crate) trait NewAxis: nd::Dimension {
    fn new_axis(self, size: usize) -> <Self::Larger as Dimension>::Pattern;
}

impl NewAxis for nd::Ix1 {
    fn new_axis(self, new_size: usize) -> (usize, usize) {
        let n = self.into_pattern();
        (n, new_size)
    }
}

impl NewAxis for nd::Ix2 {
    fn new_axis(self, new_size: usize) -> (usize, usize, usize) {
        let (a, b) = self.into_pattern();
        (a, b, new_size)
    }
}

pub(crate) trait TimeView<D>
where D: nd::Dimension
{
    type View<'a> where Self: 'a;
    type ViewMut<'a> where Self: 'a;

    fn view_time(&self, k: usize) -> Self::View<'_>;
    fn view_time_mut(&mut self, k: usize) -> Self::ViewMut<'_>;
}

impl TimeView<nd::Ix1> for nd::Array2<C64> {
    type View<'a> = nd::ArrayView1<'a, C64> where Self: 'a;
    type ViewMut<'a> = nd::ArrayViewMut1<'a, C64> where Self: 'a;

    fn view_time(&self, k: usize) -> Self::View<'_> {
        self.slice(s![.., k])
    }

    fn view_time_mut(&mut self, k: usize) -> Self::ViewMut<'_> {
        self.slice_mut(s![.., k])
    }
}

impl TimeView<nd::Ix2> for nd::Array3<C64> {
    type View<'a> = nd::ArrayView2<'a, C64> where Self: 'a;
    type ViewMut<'a> = nd::ArrayViewMut2<'a, C64> where Self: 'a;

    fn view_time(&self, k: usize) -> Self::View<'_> {
        self.slice(s![.., .., k])
    }

    fn view_time_mut(&mut self, k: usize) -> Self::ViewMut<'_> {
        self.slice_mut(s![.., .., k])
    }
}

/// Compute a "norm" of an objcect, treating it as a representation of a quantum
/// state.
pub trait StateNorm {
    fn norm(&self) -> C64;
}

/// The norm of an `ndarray::Array1<num_complex::Complex64>` is the quadrature
/// sum of its elements.
impl StateNorm for nd::Array1<C64> {
    fn norm(&self) -> C64 { self.mapv(|a| a * a.conj()).sum().sqrt() }
}

/// The norm of an `ndarray::Array2<num_complex::Complex64>` is the sum of its
/// main diagonal.
impl StateNorm for nd::Array2<C64> {
    fn norm(&self) -> C64 { self.diag().iter().sum() }
}

/// Heap-allocated [`Fn`] trait object computing a pure state amplitude, given a
/// particular basis state.
pub type AmplitudeFn<'a, S> = Rc<dyn Fn(&S) -> C64 + 'a>;

/// Different descriptions for a pure state vector, convertible to the standard
/// 1D complex-valued array representation.
#[derive(Clone)]
pub enum Pure<'a, S> {
    /// A single basis state.
    Single(S),
    /// A pre-constructed array. Will be renormalized.
    Array(nd::Array1<C64>),
    /// A functional form giving basis state amplitudes. Normalized upon
    /// instantiation as an array.
    Function(AmplitudeFn<'a, S>),
}

impl<'a, S> std::fmt::Debug for Pure<'a, S>
where S: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(s) => write!(f, "Single({:?})", s),
            Self::Array(a) => write!(f, "Array({:?})", a),
            Self::Function(_) => write!(f, "Function(...)"),
        }
    }
}

impl<S> From<nd::Array1<C64>> for Pure<'_, S> {
    fn from(a: nd::Array1<C64>) -> Self { Self::Array(a) }
}

impl<'a, F, S> From<F> for Pure<'a, S>
where F: Fn(&S) -> C64 + 'a
{
    fn from(f: F) -> Self { Self::Function(Rc::new(f)) }
}

impl<'a, S> From<AmplitudeFn<'a, S>> for Pure<'a, S> {
    fn from(f: AmplitudeFn<'a, S>) -> Self { Self::Function(f) }
}

impl<'a, S> Pure<'a, S> {
    /// Create a new [`Self::Single`].
    pub fn from_single(state: S) -> Self { Self::Single(state) }

    /// Create a new [`Self::Array`].
    pub fn from_array(array: nd::Array1<C64>) -> Self { array.into() }

    /// Create a new [`Self::Function`].
    pub fn from_func<F: Fn(&S) -> C64 + 'a>(f: F) -> Self { f.into() }

    /// Create a new [`Self::Function`].
    pub fn from_ampfn(f: AmplitudeFn<'a, S>) -> Self { f.into() }

    /// Convert to a 1D complex-valued array, if possible.
    ///
    /// The following conditions must be met by the resulting array.
    /// - must have length equal to that of `basis`
    /// - must have elements summing in quadrature to a non-zero value
    pub fn into_array<'b, I>(self, basis: &'b I) -> Option<nd::Array1<C64>>
    where
        S: PartialEq + 'b,
        I: StateIter<'b, State = S>,
    {
        match self {
            Self::Single(s0) => {
                let a: nd::Array1<C64>
                    = basis.state_iter()
                    .map(|s| if s == &s0 { 1.0.into() } else { 0.0.into() })
                    .collect();
                (a.sum() != 0.0.into()).then_some(a)
            },
            Self::Array(a) => {
                (
                    a.len() == basis.num_states()
                    && a.sum() != 0.0.into()
                )
                .then_some(a)
                .map(|mut a| { a /= a.norm(); a })
            },
            Self::Function(f) => {
                let a: nd::Array1<C64>
                    = basis.state_iter()
                    .map(f.as_ref())
                    .collect();
                let norm = a.norm();
                (norm != 0.0.into()).then_some(a)
                    .map(|mut a| { a /= norm; a })
            },
        }
    }
}

/// Heap-allocated [`Fn`] trait object computing a density matrix element, given
/// a particular pair of states.
pub type DensityFn<'a, S> = Rc<dyn Fn(&S, &S) -> C64 + 'a>;

/// Heap-allocated [`Fn`] trait object computing an main-diagonal density matrix
/// element, given a particular basis state.
pub type MixedFn<'a, S> = Rc<dyn Fn(&S) -> f64 + 'a>;

/// Different descriptions for a density matrix, convertible to the standard
/// 2D complex-valued array representation.
#[derive(Clone)]
pub enum Density<'a, S> {
    /// A single basis state.
    Single(S),
    /// A pre-constructed array. Will be renormalized.
    Array(nd::Array2<C64>),
    /// A functional form giving density matrix elements. Normalized upon
    /// instantiation as an array.
    Function(DensityFn<'a, S>),
    /// A description of a pure state.
    Pure(Pure<'a, S>),
    /// A functional form giving a (completely) classical mixture of single
    /// basis states. Normalized upon instantiation as an array.
    Mixed(MixedFn<'a, S>),
}

impl<'a, S> std::fmt::Debug for Density<'a, S>
where S: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(s) => write!(f, "Single({:?})", s),
            Self::Array(a) => write!(f, "Array({:?})", a),
            Self::Function(_) => write!(f, "Function(...)"),
            Self::Pure(p) => write!(f, "Pure({:?})", p),
            Self::Mixed(_) => write!(f, "Mixed(...)"),
        }
    }
}

impl<S> From<nd::Array2<C64>> for Density<'_, S> {
    fn from(a: nd::Array2<C64>) -> Self { Self::Array(a) }
}

impl<'a, F, S> From<F> for Density<'a, S>
where F: Fn(&S, &S) -> C64 + 'a
{
    fn from(f: F) -> Self { Self::Function(Rc::new(f)) }
}

impl<'a, S> From<DensityFn<'a, S>> for Density<'a, S> {
    fn from(f: DensityFn<'a, S>) -> Self { Self::Function(f) }
}

impl<'a, S> From<Pure<'a, S>> for Density<'a, S> {
    fn from(pure: Pure<'a, S>) -> Self { Self::Pure(pure) }
}

impl<'a, S> From<MixedFn<'a, S>> for Density<'a, S> {
    fn from(f: MixedFn<'a, S>) -> Self { Self::Mixed(f) }
}

impl<'a, S> Density<'a, S> {
    /// Create a new [`Self::Single`].
    pub fn from_single(state: S) -> Self { Self::Single(state) }

    /// Create a new [`Self::Array`].
    pub fn from_array(array: nd::Array2<C64>) -> Self { array.into() }

    /// Create a new [`Self::Function`].
    pub fn from_func<F: Fn(&S, &S) -> C64 + 'a>(f: F) -> Self { f.into() }

    /// Create a new [`Self::Function`].
    pub fn from_densfn(f: DensityFn<'a, S>) -> Self { f.into() }

    /// Create a new [`Self::Pure`].
    pub fn from_pure(pure: Pure<'a, S>) -> Self { pure.into() }

    /// Create a new [`Self::Pure`].
    pub fn from_into_pure<P: Into<Pure<'a, S>>>(pure: P) -> Self {
        pure.into().into()
    }

    /// Create a new [`Self::Mixed`].
    pub fn from_mixture<F: Fn(&S) -> f64 + 'a>(f: F) -> Self {
        Self::Mixed(Rc::new(f))
    }

    /// Create a new [`Self::Mixed`].
    pub fn from_mixfn(f: MixedFn<'a, S>) -> Self { Self::Mixed(f) }

    /// Convert to a 2D complex-valued array, if possible.
    ///
    /// The following conditions must be met by the resulting array:
    /// - must be square with dimension equal to the length of `basis`
    /// - must have all real, non-negative main-diagonal elements
    /// - must have trace not equal to zero
    /// - must be Hermitian
    pub fn into_array<'b, I>(self, basis: &'b I) -> Option<nd::Array2<C64>>
    where
        S: Clone + PartialEq + 'b,
        I: StateIter<'b, State = S>,
    {
        match self {
            Self::Single(s0) => {
                let a: nd::Array1<C64>
                    = basis.state_iter()
                    .map(|s| if s == &s0 { 1.0.into() } else { 0.0.into() })
                    .collect();
                let norm = a.sum();
                (norm != 0.0.into()).then_some(a)
                    .map(|a| nd::Array2::from_diag(&a))
            },
            Self::Array(a) => {
                let norm = a.norm();
                (
                    a.shape() == [basis.num_states(); 2]
                    && a.diag().iter().all(|p| p.im == 0.0 && p.re >= 0.0)
                    && norm != 0.0.into()
                    && a == a.t().mapv(|p| p.conj())
                )
                .then_some(a)
                .map(|mut a| { a /= norm; a })
            },
            Self::Function(f) => {
                let n = basis.num_states();
                let a: nd::Array2<C64>
                    = basis.state_iter()
                    .cartesian_product(basis.state_iter())
                    .map(|(si, sj)| f(si, sj))
                    .collect::<nd::Array1<C64>>()
                    .into_shape((n, n))
                    .unwrap();
                let norm = a.norm();
                (
                    a.diag().iter().all(|p| p.im == 0.0 && p.re >= 0.0)
                    && norm != 0.0.into()
                    && a == a.t().mapv(|p| p.conj())
                )
                .then_some(a)
                .map(|mut a| { a /= norm; a })
            },
            Self::Pure(pure) => {
                pure.into_array(basis)
                    .map(|a| outer_prod(&a, &a))
            },
            Self::Mixed(f) => {
                let a: nd::Array1<f64>
                    = basis.state_iter()
                    .map(f.as_ref())
                    .collect();
                let norm = a.sum();
                (
                    a.iter().all(|p| *p >= 0.0)
                    && norm != 0.0
                )
                .then_some(a)
                .map(|a| {
                    nd::Array2::from_diag(&a.mapv(|p| C64::from(p / norm)))
                })
            },
        }
    }
}

/// Compute the commutator `[A, B] = A B - B A`.
pub fn commutator<SA, SB>(
    A: &nd::ArrayBase<SA, nd::Ix2>,
    B: &nd::ArrayBase<SB, nd::Ix2>,
) -> nd::Array2<C64>
where
    SA: nd::Data<Elem = C64>,
    SB: nd::Data<Elem = C64>,
{
    A.dot(B) - B.dot(A)
}

/// Compute the anti-commutator `{A, B} = A B + B A`.
pub fn anti_commutator<SA, SB>(
    A: &nd::ArrayBase<SA, nd::Ix2>,
    B: &nd::ArrayBase<SB, nd::Ix2>,
) -> nd::Array2<C64>
where
    SA: nd::Data<Elem = C64>,
    SB: nd::Data<Elem = C64>,
{
    A.dot(B) + B.dot(A)
}

/// Compute the non-Hermitian part of the RHS of the Lindblad master equation.
///
/// Assumes that `Y` and `rho` are both square, and that the system's decay
/// rates can be characterized by a single matrix `Y`, whose `(i, j)`-th element
/// is the total decay rate from the `i`-th state to the `j`-th state. Note that
/// this criterion pretty much excludes all systems comprising more than one
/// particle. For multi-particle systems, see [`LOp`].
pub fn lindbladian<SA, SB>(
    Y: &nd::ArrayBase<SA, nd::Ix2>,
    rho: &nd::ArrayBase<SB, nd::Ix2>,
) -> nd::Array2<C64>
where
    SA: nd::Data<Elem = f64>,
    SB: nd::Data<Elem = C64>,
{
    let mut L: nd::Array2<C64> = nd::Array2::zeros(Y.raw_dim());
    let mut term: C64;
    for ((a, b), &y) in Y.indexed_iter() {
        if y.abs() <= f64::EPSILON { continue; }
        for ((i, j), l) in L.indexed_iter_mut() {
            term
                = if i == j {
                    y * (
                        if i == b { rho[[a, a]] } else { C64::zero() }
                        - if i == a { rho[[a, a]] } else { C64::zero() }
                    )
                } else {
                    -y * if i == a || j == a
                        { rho[[i, j]] / 2.0 } else { C64::zero() }
                };
            *l += term;
        }
    }
    L
}

/// Stack a series of arrays.
pub fn stack_arrays<A, D>(axis: nd::Axis, arrays: &[nd::Array<A, D>])
    -> Result<nd::Array<A, D::Larger>, nd::ShapeError>
where
    A: Clone,
    D: nd::Dimension,
    D::Larger: nd::RemoveAxis,
{
    nd::stack(
        axis,
        &arrays.iter().map(|arr| arr.view()).collect::<Vec<_>>(),
    )
}

fn array_diff<A>(arr: &nd::Array1<A>) -> nd::Array1<A>
where A: Copy + std::ops::Sub<A, Output = A>
{
    arr.iter().zip(arr.iter().skip(1))
        .map(|(ak, akp1)| *akp1 - *ak)
        .collect()
}

// fourth-order Runge-Kutta for time-independent Hamiltonian, generic over the
// dimension of the quantum state array and the RHS of the ODE being solved
pub(crate) fn do_evolve<D, F>(
    z0: &nd::Array<C64, D>,
    h: &nd::Array2<C64>,
    rhs: F,
    t: &nd::Array1<f64>,
) -> nd::Array<C64, D::Larger>
where
    D: nd::Dimension + NewAxis + Copy,
    D::Larger: Copy,
    nd::Array<C64, D>: StateNorm,
    for<'a> nd::Array<C64, D::Larger>:
        TimeView<
            D,
            View<'a> = nd::ArrayView<'a, C64, D>,
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>,
        >,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old: nd::Array<C64, D> = z0.clone();
    let mut k1: nd::Array<C64, D>;
    let mut k2: nd::Array<C64, D>;
    let mut k3: nd::Array<C64, D>;
    let mut k4: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    z.view_time_mut(0).assign(z0);
    let iter = dt.iter().enumerate();
    for (k, &dtk) in iter {
        k1 = rhs(h, &z_old);
        k2 = rhs(h, &(&z_old + &k1 * (dtk / 2.0)));
        k3 = rhs(h, &(&z_old + &k2 * (dtk / 2.0)));
        k4 = rhs(h, &(&z_old + &k3 * dtk));
        z_new = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dtk / 6.0);
        norm = z_new.norm();
        z_old = z_new / norm;
        z_old.clone().move_into(z.view_time_mut(k + 1));
    }
    z
}

// fourth-order Runge-Kutta for time-dependent Hamiltonian, generic over the
// dimension of the quantum state array and the RHS of the ODE being solved
pub(crate) fn do_evolve_t<D, F>(
    z0: &nd::Array<C64, D>,
    h: &nd::Array3<C64>,
    rhs: F,
    t: &nd::Array1<f64>,
) -> nd::Array<C64, D::Larger>
where
    D: nd::Dimension + NewAxis + Copy,
    D::Larger: Copy,
    nd::Array<C64, D>: StateNorm,
    for<'a> nd::Array<C64, D::Larger>:
        TimeView<
            D,
            View<'a> = nd::ArrayView<'a, C64, D>,
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>,
        >,
    F: Fn(nd::ArrayView2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old: nd::Array<C64, D> = z0.clone();
    let mut k1: nd::Array<C64, D>;
    let mut k2: nd::Array<C64, D>;
    let mut k3: nd::Array<C64, D>;
    let mut k4: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    z.view_time_mut(0).assign(z0);
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(
            h.axis_iter(nd::Axis(2))
            .zip(h.axis_iter(nd::Axis(2)).skip(1))
            .zip(h.axis_iter(nd::Axis(2)).skip(2))
        )
        .enumerate()
        .step_by(2);
    for (k, ((&dtk, &dtkp1), ((hk, hkp1), hkp2))) in iter {
        k1 = rhs(hk, &z_old);
        k2 = rhs(hkp1, &(&z_old + &k1 * dtk));
        k3 = rhs(hkp1, &(&z_old * &k2 * dtk));
        k4 = rhs(hkp2, &(&z_old * &k3 * (dtk + dtkp1)));
        z_new
            = &z_old * (k1 + k2 * 2.0 + k3 * 2.0 + k4) * ((dtk + dtkp1) / 6.0);
        norm = z_new.norm();
        z_old = z_new / norm;
        z_old.clone().move_into(z.view_time_mut(k));
    }
    for k in (1..n - 1).step_by(2) {
        z_new = (&z.view_time(k - 1) + &z.view_time(k + 1)) / 2.0;
        z_new.move_into(z.view_time_mut(k));
    }
    if n % 2 == 0 {
        z_new = z.view_time(n - 2).to_owned();
        z_new.move_into(z.view_time_mut(n - 1));
    }
    z
}

// fourth-order Runge-Kutta for time-dependent Hamiltonian given by a function,
// generic over the dimension of the quantum state array and the RHS of the ODE
// being solved
pub(crate) fn do_evolve_fn<D, H, F>(
    z0: &nd::Array<C64, D>,
    h: H,
    rhs: F,
    t: &nd::Array1<f64>,
) -> nd::Array<C64, D::Larger>
where
    D: nd::Dimension + NewAxis + Copy,
    D::Larger: Copy,
    nd::Array<C64, D>: StateNorm,
    for<'a> nd::Array<C64, D::Larger>:
        TimeView<
            D,
            View<'a> = nd::ArrayView<'a, C64, D>,
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>,
        >,
    H: Fn(f64) -> nd::Array2<C64>,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old: nd::Array<C64, D> = z0.clone();
    let mut hk: nd::Array2<C64>;
    let mut hkp1h: nd::Array2<C64>;
    let mut hkp1: nd::Array2<C64>;
    let mut k1: nd::Array<C64, D>;
    let mut k2: nd::Array<C64, D>;
    let mut k3: nd::Array<C64, D>;
    let mut k4: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    z.view_time_mut(0).assign(z0);
    let iter = dt.iter().zip(t).enumerate();
    for (k, (&dtk, &tk)) in iter {
        hk = h(tk);
        hkp1h = h(tk + dtk / 2.0);
        hkp1 = h(tk + dtk);
        k1 = rhs(&hk, &z_old);
        k2 = rhs(&hkp1h, &(&z_old + &k1 * (dtk / 2.0)));
        k3 = rhs(&hkp1h, &(&z_old + &k2 * (dtk / 2.0)));
        k4 = rhs(&hkp1, &(&z_old + &k3 * dtk));
        z_new = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dtk / 6.0);
        norm = z_new.norm();
        z_old = z_new / norm;
        z_old.clone().move_into(z.view_time_mut(k));
    }
    z
}

// fourth-order Runge-Kutta for time-independent Hamiltonian with reduced
// integration output, generic over the dimension of the quantum state array and
// the RHS of the ODE being solved
pub(crate) fn do_evolve_reduced<D, F, X, T>(
    z0: &nd::Array<C64, D>,
    h: &nd::Array2<C64>,
    rhs: F,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    D: nd::Dimension,
    nd::Array<C64, D>: StateNorm,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
    X: Fn(&nd::Array<C64, D>) -> T,
{
    let dt = array_diff(t);
    let mut z_old: nd::Array<C64, D> = z0.clone();
    let mut k1: nd::Array<C64, D>;
    let mut k2: nd::Array<C64, D>;
    let mut k3: nd::Array<C64, D>;
    let mut k4: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    let mut x_t: Vec<T> = Vec::with_capacity(t.len());
    x_t.push(x(&z_old));
    let iter = dt.iter();
    for &dtk in iter {
        k1 = rhs(h, &z_old);
        k2 = rhs(h, &(&z_old + &k1 * (dtk / 2.0)));
        k3 = rhs(h, &(&z_old + &k2 * (dtk / 2.0)));
        k4 = rhs(h, &(&z_old + &k3 * dtk));
        z_new = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dtk / 6.0);
        norm = z_new.norm();
        z_new /= norm;
        x_t.push(x(&z_new));
        z_old = z_new;
    }
    x_t
}

// fourth-order Runge-Kutta for the time-dependent Hamiltonian given by a
// function with reduced integration output, generic over the dimension of the
// quantum state array and the RHS of the ODE being solved
pub(crate) fn do_evolve_fn_reduced<D, H, F, X, T>(
    z0: &nd::Array<C64, D>,
    h: H,
    rhs: F,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    D: nd::Dimension,
    nd::Array<C64, D>: StateNorm,
    H: Fn(f64) -> nd::Array2<C64>,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
    X: Fn(&nd::Array<C64, D>) -> T,
{
    let dt = array_diff(t);
    let mut z_old: nd::Array<C64, D> = z0.clone();
    let mut hk: nd::Array2<C64>;
    let mut hkp1h: nd::Array2<C64>;
    let mut hkp1: nd::Array2<C64>;
    let mut k1: nd::Array<C64, D>;
    let mut k2: nd::Array<C64, D>;
    let mut k3: nd::Array<C64, D>;
    let mut k4: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    let mut x_t: Vec<T> = Vec::with_capacity(t.len());
    x_t.push(x(&z_old));
    let iter = dt.iter().zip(t);
    for (&dtk, &tk) in iter {
        hk = h(tk);
        hkp1h = h(tk + dtk / 2.0);
        hkp1 = h(tk + dtk);
        k1 = rhs(&hk, &z_old);
        k2 = rhs(&hkp1h, &(&z_old + &k1 * (dtk / 2.0)));
        k3 = rhs(&hkp1h, &(&z_old * &k2 * (dtk / 2.0)));
        k4 = rhs(&hkp1, &(&z_old * &k3 * dtk));
        z_new = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dtk / 6.0);
        norm = z_new.norm();
        z_new /= norm;
        x_t.push(x(&z_new));
        z_old = z_new;
    }
    x_t
}

