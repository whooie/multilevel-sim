//! Functions for numerical integration of the Schrödinger and Lindblad master
//! equations using a variety of methods.
//!
//! Where unspecified, the last index of an array corresponds to time.

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

trait NewAxis: nd::Dimension {
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

trait TimeView<D>
where D: nd::Dimension
{
    type View<'a> where Self: 'a;
        // = nd::ArrayView<'a, C64, D::Smaller> where Self: 'a;
    type ViewMut<'a> where Self: 'a;
        // = nd::ArrayViewMut<'a, C64, D::Smaller> where Self: 'a;

    fn view_time<'a>(&'a self, k: usize) -> Self::View<'a>;
    fn view_time_mut<'a>(&'a mut self, k: usize) -> Self::ViewMut<'a>;
}

impl TimeView<nd::Ix1> for nd::Array2<C64> {
    type View<'a> = nd::ArrayView1<'a, C64> where Self: 'a;
    type ViewMut<'a> = nd::ArrayViewMut1<'a, C64> where Self: 'a;

    fn view_time<'a>(&'a self, k: usize) -> Self::View<'a> {
        self.slice(s![.., k])
    }

    fn view_time_mut<'a>(&'a mut self, k: usize) -> Self::ViewMut<'a> {
        self.slice_mut(s![.., k])
    }
}

impl TimeView<nd::Ix2> for nd::Array3<C64> {
    type View<'a> = nd::ArrayView2<'a, C64> where Self: 'a;
    type ViewMut<'a> = nd::ArrayViewMut2<'a, C64> where Self: 'a;

    fn view_time<'a>(&'a self, k: usize) -> Self::View<'a> {
        self.slice(s![.., .., k])
    }

    fn view_time_mut<'a>(&'a mut self, k: usize) -> Self::ViewMut<'a> {
        self.slice_mut(s![.., .., k])
    }
}

/// Compute the norm of an object, treating it as a representation of a quantum
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

/// Different descriptions for a pure state vector, convertible to the standard
/// 1D complex-valued array representation.
#[derive(Clone)]
pub enum PureMaker<'a, S> {
    /// A single basis state.
    Single(S),
    /// A pre-constructed array. Will be renormalized.
    Array(nd::Array1<C64>),
    /// A functional form giving basis state amplitudes. Normalized upon
    /// instantiation as an array.
    Function(Rc<dyn Fn(&S) -> C64 + 'a>),
}

impl<S> From<nd::Array1<C64>> for PureMaker<'_, S> {
    fn from(a: nd::Array1<C64>) -> Self { Self::Array(a) }
}

impl<'a, F, S> From<F> for PureMaker<'a, S>
where F: Fn(&S) -> C64 + 'a
{
    fn from(f: F) -> Self { Self::Function(Rc::new(f)) }
}

impl<'a, S> From<Rc<dyn Fn(&S) -> C64 + 'a>> for PureMaker<'a, S> {
    fn from(rc: Rc<dyn Fn(&S) -> C64 + 'a>) -> Self { Self::Function(rc) }
}

impl<'a, S> PureMaker<'a, S> {
    /// Create a new [`Self::Single`].
    pub fn from_single(state: S) -> Self { Self::Single(state) }

    /// Create a new [`Self::Array`].
    pub fn from_array(array: nd::Array1<C64>) -> Self { array.into() }

    /// Create a new [`Self::Function`].
    pub fn from_func<F>(f: F) -> Self
    where F: Fn(&S) -> C64 + 'a
    {
        f.into()
    }

    /// Create a new [`Self::Function`].
    pub fn from_func_rc(rc: Rc<dyn Fn(&S) -> C64 + 'a>) -> Self { rc.into() }

    /// Convert to a 1D complex-valued array, if possible.
    ///
    /// The following conditions must be met by the produced array:
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

/// Different representations for a density matrix, convertible to the standard
/// 2D complex-valued array representation.
#[derive(Clone)]
pub enum DensityMaker<'a, S> {
    /// A single basis state.
    Single(S),
    /// A pre-constructed array. Will be renormalized.
    Array(nd::Array2<C64>),
    /// A functional form giving density matrix elements. Normalized upon
    /// instantiation as an array.
    Function(Rc<dyn Fn(&S, &S) -> C64 + 'a>),
    /// A description of a pure state.
    Pure(PureMaker<'a, S>),
    /// A functional form giving a classical mixture of single basis states.
    /// Normalized upon instantiation as an array.
    Mixed(Rc<dyn Fn(&S) -> f64 + 'a>),
}

impl<S> From<nd::Array2<C64>> for DensityMaker<'_, S> {
    fn from(a: nd::Array2<C64>) -> Self { Self::Array(a) }
}

impl<'a, F, S> From<F> for DensityMaker<'a, S>
where F: Fn(&S, &S) -> C64 + 'a
{
    fn from(f: F) -> Self { Self::Function(Rc::new(f)) }
}

impl<'a, S> From<Rc<dyn Fn(&S, &S) -> C64 + 'a>> for DensityMaker<'a, S> {
    fn from(rc: Rc<dyn Fn(&S, &S) -> C64 + 'a>) -> Self { Self::Function(rc) }
}

impl<'a, S> From<PureMaker<'a, S>> for DensityMaker<'a, S> {
    fn from(pure: PureMaker<'a, S>) -> Self { Self::Pure(pure) }
}

impl<'a, S> From<Rc<dyn Fn(&S) -> f64 + 'a>> for DensityMaker<'a, S> {
    fn from(rc: Rc<dyn Fn(&S) -> f64 + 'a>) -> Self { Self::Mixed(rc) }
}

impl<'a, S> DensityMaker<'a, S> {
    /// Create a new [`Self::Single`].
    pub fn from_single(state: S) -> Self { Self::Single(state) }

    /// Create a new [`Self::Array`].
    pub fn from_array(array: nd::Array2<C64>) -> Self { array.into() }

    /// Create a new [`Self::Function`].
    pub fn from_func<F>(f: F) -> Self
    where F: Fn(&S, &S) -> C64 + 'a
    {
        f.into()
    }

    /// Create a new [`Self::Function`].
    pub fn from_func_rc(rc: Rc<dyn Fn(&S, &S) -> C64 + 'a>) -> Self {
        rc.into()
    }

    /// Create a new [`Self::Pure`].
    pub fn from_pure(pure: PureMaker<'a, S>) -> Self { pure.into() }

    /// Create a new [`Self::Pure`].
    pub fn from_into_pure<P>(pure: P) -> Self
    where P: Into<PureMaker<'a, S>>
    {
        pure.into().into()
    }

    /// Create a new [`Self::Mixed`].
    pub fn from_mixture<F>(f: F) -> Self
    where F: Fn(&S) -> f64 + 'a
    {
        Self::Mixed(Rc::new(f))
    }

    /// Create a new [`Self::Mixed`].
    pub fn from_mixture_rc(rc: Rc<dyn Fn(&S) -> f64 + 'a>) -> Self { rc.into() }

    /// Convert to a 2D complex-valued array, if possible.
    ///
    /// The following conditions must be met by the produced array:
    /// - must be square, with dimension equal to the length of `basis`
    /// - must have all real, non-negative on-diagonal elements
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
                    a.shape() == &[basis.num_states(); 2]
                    && a.diag().iter().all(|p| p.im == 0.0 && p.re >= 0.0)
                    && norm != 0.0.into()
                    && a == a.t().mapv(|r| r.conj())
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
                    && a == a.t().mapv(|r| r.conj())
                )
                .then_some(a)
                .map(|mut a| { a /= a.norm(); a })
            },
            Self::Pure(pure) => {
                pure.into_array(basis)
                    .map(|arr| outer_prod(&arr, &arr))
            },
            Self::Mixed(f) => {
                let a: nd::Array1<f64>
                    = basis.state_iter()
                    .map(|s| f(s).into())
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
/// Assumes `Y` and `rho` are both square, with the `(i, j)`-th element of `Y`
/// giving the decay rate from the `i`-th state to the `j`-th state.
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

/// Compute the Schrödinger coherent evolution of the initial state `psi0` for
/// Hamiltonian `H`.
///
/// Requires `H` to be in units of angular frequency.
pub fn eigen_evolve(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    let (E, V): (nd::Array1<f64>, nd::Array2<C64>)
        = H.eigh(la::UPLO::Lower)
        .expect("eigen_evolve: diagonalization error");
    let c: nd::Array1<C64> = V.solve(psi0)
        .expect("eigen_evolve: linalg solve error");
    let mut psi: nd::Array2<C64> = nd::Array::zeros((psi0.len(), t.len()));
    let iter
        = t.iter()
        .zip(psi.axis_iter_mut(nd::Axis(1)));
    for (&tk, psik) in iter {
        V.dot(&(&c * &E.mapv(|e| (-C64::i() * e * tk).exp())))
            .move_into(psik);
    }
    psi
}

/// Compute the Schrödinger coherent evolution of an initial state for a
/// time-independent Hamiltonian, if possible.
pub fn eigen_evolve_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    Some(eigen_evolve(&psi0, &h, t))
}

/// Compute the Schrödinger coherent evolution of the initial state `psi0` for
/// time-dependent Hamiltonian `H` by diagonalizing at each time step.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn eigen_evolve_t(
    psi0: &nd::Array1<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    let dt = array_diff(t);
    let mut EV: (nd::Array1<f64>, nd::Array2<C64>);
    let mut c: nd::Array1<C64>;
    let mut psi: nd::Array2<C64> = nd::Array::zeros((psi0.len(), t.len()));
    psi.slice_mut(s![.., 0]).assign(psi0);
    let iter
        = dt.into_iter()
        .zip(H.axis_iter(nd::Axis(2)))
        .enumerate();
    for (k, (dtk, Hk)) in iter {
        EV = Hk.eigh(la::UPLO::Lower)
            .expect("eigen_evolve_t: diagonalization error");
        c = EV.1.solve(&psi.slice(s![.., k]))
            .expect("eigen_evolve_t: linalg solve error");
        EV.1.dot(&(&c * EV.0.mapv(|e| (-C64::i() * e * dtk).exp())))
            .move_into(psi.slice_mut(s![.., k + 1]));
    }
    psi
}

/// Compute the Schrödinger coherent evolution of an initial state for a
/// time-dependent Hamiltonian by diagonalizing at each time step.
pub fn eigen_evolve_t_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array3<C64> = hbuilder.build(t);
    Some(eigen_evolve_t(&psi0, &h, t))
}

fn evolve_t_static<D, F>(
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
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>
        >,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old0: nd::Array<C64, D>;
    let mut z_old1: nd::Array<C64, D>;
    let mut dz: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    z.view_time_mut(0).assign(z0);
    z.view_time_mut(1).assign(z0);
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .enumerate();
    z_old0 = z0.clone();
    z_old1 = z0.clone();
    for (k, (&dtk, &dtkp1)) in iter {
        dz = rhs(h, &z_old1) * (dtk + dtkp1);
        z_new = &z_old0 + dz;
        norm = z_new.norm();
        z_old0 = z_old1;
        z_old1 = z_new / norm;
        z_old1.clone().move_into(z.view_time_mut(k + 2));
    }
    z
}

fn evolve_t<D, F>(
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
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>
        >,
    F: Fn(nd::ArrayView2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old0: nd::Array<C64, D>;
    let mut z_old1: nd::Array<C64, D>;
    let mut dz: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    z.view_time_mut(0).assign(z0);
    z.view_time_mut(1).assign(z0);
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(h.axis_iter(nd::Axis(2)).skip(1))
        .enumerate();
    z_old0 = z0.clone();
    z_old1 = z0.clone();
    for (k, ((&dtk, &dtkp1), hkp1)) in iter {
        dz = rhs(hkp1, &z_old1) * (dtk + dtkp1);
        z_new = &z_old0 + dz;
        norm = z_new.norm();
        z_old0 = z_old1;
        z_old1 = z_new / norm;
        z_old1.clone().move_into(z.view_time_mut(k + 2));
    }
    z
}

fn evolve_t_reduced_static<D, F, X, T>(
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
    let mut z_old0: nd::Array<C64, D> = z0.clone();
    let mut z_old1: nd::Array<C64, D> = z0.clone();
    let mut dz: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    let mut x_t: Vec<T> = Vec::with_capacity(t.len());
    x_t.push(x(&z_old0));
    x_t.push(x(&z_old1));
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1));
    for (&dtk, &dtkp1) in iter {
        dz = rhs(h, &z_old1) * (dtk + dtkp1);
        z_new = &z_old0 + dz;
        norm = z_new.norm();
        z_new /= norm;
        x_t.push(x(&z_new));
        z_old0 = z_old1;
        z_old1 = z_new;
    }
    x_t
}

fn evolve_t_reduced<D, H, F, X, T>(
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
    let mut z_old0: nd::Array<C64, D> = z0.clone();
    let mut z_old1: nd::Array<C64, D> = z0.clone();
    let mut dz: nd::Array<C64, D>;
    let mut z_new: nd::Array<C64, D>;
    let mut norm: C64;
    let mut x_t: Vec<T> = Vec::with_capacity(t.len());
    x_t.push(x(&z_old0));
    x_t.push(x(&z_old1));
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(t.iter().skip(1));
    for ((&dtk, &dtkp1), &tk) in iter {
        dz = rhs(&h(tk), &z_old1) * (dtk + dtkp1);
        z_new = &z_old0 + dz;
        norm = z_new.norm();
        z_new /= norm;
        x_t.push(x(&z_new));
        z_old0 = z_old1;
        z_old1 = z_new;
    }
    x_t
}

fn evolve_rk4_static<D, F>(
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
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>
        >,
    F: Fn(&nd::Array2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old: nd::Array<C64, D>;
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
        .enumerate()
        .step_by(2);
    z_old = z0.clone();
    for (k, (&dtk, &dtkp1)) in iter {
        k1 = rhs(h, &z_old);
        k2 = rhs(h, &(&z_old + &k1 * dtk));
        k3 = rhs(h, &(&z_old + &k2 * dtk));
        k4 = rhs(h, &(&z_old + &k3 * (dtk + dtkp1)));
        z_new
            = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * ((dtk + dtkp1) / 6.0);
        norm = z_new.norm();
        z_old = z_new / norm;
        z_old.clone().move_into(z.view_time_mut(k + 2));
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

fn evolve_rk4<D, F>(
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
            ViewMut<'a> = nd::ArrayViewMut<'a, C64, D>
        >,
    F: Fn(nd::ArrayView2<C64>, &nd::Array<C64, D>) -> nd::Array<C64, D>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut z: nd::Array<C64, D::Larger>
        = nd::Array::zeros(z0.raw_dim().new_axis(n));
    let mut z_old: nd::Array<C64, D>;
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
    z_old = z0.clone();
    for (k, ((&dtk, &dtkp1), ((hk, hkp1), hkp2))) in iter {
        k1 = rhs(hk, &z_old);
        k2 = rhs(hkp1, &(&z_old + &k1 * dtk));
        k3 = rhs(hkp1, &(&z_old + &k2 * dtk));
        k4 = rhs(hkp2, &(&z_old + &k3 * (dtk + dtkp1)));
        z_new
            = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * ((dtk + dtkp1) / 6.0);
        norm = z_new.norm();
        z_old = z_new / norm;
        z_old.clone().move_into(z.view_time_mut(k + 2));
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

fn evolve_rk4_reduced_static<D, F, X, T>(
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

fn evolve_rk4_reduced<D, H, F, X, T>(
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
        k3 = rhs(&hkp1h, &(&z_old + &k2 * (dtk / 2.0)));
        k4 = rhs(&hkp1, &(&z_old + &k3 * dtk));
        z_new = &z_old + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dtk / 6.0);
        norm = z_new.norm();
        z_new /= norm;
        x_t.push(x(&z_new));
        z_old = z_new;
    }
    x_t
}

fn rhs_schrodinger(H: nd::ArrayView2<C64>, psi: &nd::Array1<C64>)
    -> nd::Array1<C64>
{
    -C64::i() * H.dot(psi)
}

/// Numerically integrate the Schrödinger equation using the midpoint rule for a
/// time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn schrodinger_evolve_t(
    psi0: &nd::Array1<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    evolve_t(psi0, H, rhs_schrodinger, t)
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta for a time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn schrodinger_evolve_rk4(
    psi0: &nd::Array1<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    evolve_rk4(psi0, H, rhs_schrodinger, t)
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta.
pub fn schrodinger_evolve_rk4_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array3<C64> = hbuilder.build(t);
    Some(schrodinger_evolve_rk4(&psi0, &h, t))
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta for a time-dependent Hamiltonian and apply a function to the
/// state at each time step, recording only the output of the function.
pub fn schrodinger_evolve_rk4_reduced<'a, P, H, S, X, T>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
    X: Fn(&nd::Array1<C64>) -> T,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h_fn = |t| hbuilder.build_at(t);
    let res = evolve_rk4_reduced(&psi0, h_fn, rhs_schrodinger_static, t, x);
    Some(res)
}

fn rhs_schrodinger_static(H: &nd::Array2<C64>, psi: &nd::Array1<C64>)
    -> nd::Array1<C64>
{
    -C64::i() * H.dot(psi)
}

/// Numerically integrate the Schrödinger equation using the midpoint rule for a
/// time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn schrodinger_evolve_t_static(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    evolve_t_static(psi0, H, rhs_schrodinger_static, t)
}

/// Numerically integrate the Schrödinger equation using the midpoint rule for a
/// time-independent Hamiltonian and apply a function to the state at each time
/// step, recording only the output of the function.
///
/// Fails if a time-independent Hamiltonian cannot be generated.
pub fn schrodinger_evolve_t_reduced_static<'a, P, H, S, X, T>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
    X: Fn(&nd::Array1<C64>) -> T,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    Some(evolve_t_reduced_static(&psi0, &h, rhs_schrodinger_static, t, x))
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn schrodinger_evolve_rk4_static(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    evolve_rk4_static(psi0, H, rhs_schrodinger_static, t)
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian.
///
/// Fails if a time-independent Hamiltonian cannot be generated.
pub fn schrodinger_evolve_rk4_static_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    Some(schrodinger_evolve_rk4_static(&psi0, &h, t))
}

/// Numerically integrate the Schrödinger equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian and apply a function to the
/// state at each time step, recording only the output of the function.
///
/// Fails if a time-independent Hamiltonian cannot be generated.
pub fn schrodinger_evolve_rk4_reduced_static<'a, P, H, S, X, T>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    P: Into<PureMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
    X: Fn(&nd::Array1<C64>) -> T,
{
    let psi0: nd::Array1<C64> = psi0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    Some(evolve_rk4_reduced_static(&psi0, &h, rhs_schrodinger_static, t, x))
}

fn rhs_liouville(H: nd::ArrayView2<C64>, rho: &nd::Array2<C64>)
    -> nd::Array2<C64>
{
    -C64::i() * commutator(&H, rho)
}

/// Numerically integrate the Liouville equation using the midpoint rule for a
/// time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn liouville_evolve_t(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_t(rho0, H, rhs_liouville, t)
}

/// Numerically integrate the Liouville equation using fourth-order Runge-Kutta
/// for a time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn liouville_evolve_rk4(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_rk4(rho0, H, rhs_liouville, t)
}

/// Numerically integrate the Liouville equation using fourth-order
/// Runge-Kutta.
pub fn liouville_evolve_rk4_with<'a, D, H, S>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    D: Into<DensityMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + Clone + 'a,
{
    let rho0: nd::Array2<C64> = rho0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array3<C64> = hbuilder.build(t);
    Some(liouville_evolve_rk4(&rho0, &h, t))
}

fn rhs_liouville_static(H: &nd::Array2<C64>, rho: &nd::Array2<C64>)
    -> nd::Array2<C64>
{
    -C64::i() * commutator(H, rho)
}

/// Numerically integrate the Liouville equation using the midpoint rule for a
/// time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn liouville_evolve_t_static(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_t_static(rho0, H, rhs_liouville_static, t)
}

/// Numerically integrate the Liouville equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn liouville_evolve_rk4_static(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_rk4_static(rho0, H, rhs_liouville_static, t)
}

/// Numerically integrate the Liouville equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian.
///
/// Fails if a time-independent Hamiltonian cannot be generated.
pub fn liouville_evolve_rk4_static_with<'a, D, H, S>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    D: Into<DensityMaker<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + Clone + 'a,
{
    let rho0: nd::Array2<C64> = rho0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    Some(liouville_evolve_rk4_static(&rho0, &h, t))
}

fn rhs_lindblad(
    H: nd::ArrayView2<C64>,
    Y: &nd::Array2<f64>,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    -C64::i() * commutator(&H, rho) + lindbladian(Y, rho)
}

/// Numerically integrate the Lindblad equation using the midpoint rule for a
/// time-dependent Hamiltonian.
///
/// Requires `H` and `Y` to be in units of angular frequency. The third index of
/// `H` should correspond to time.
pub fn lindblad_evolve_t(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_t(rho0, H, |H, rho| rhs_lindblad(H, Y, rho), t)
}

/// Numerically integrate the Liouville equation using fourth-order Runge-Kutta
/// for a time-dependent Hamiltonian.
///
/// Requires `H` and `Y` to be in units of angular frequency. The third index of
/// `H` should correspond to time.
pub fn lindblad_evolve_rk4(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_rk4(rho0, H, |H, rho| rhs_lindblad(H, Y, rho), t)
}

fn rhs_lindblad_op<'a, L, S>(
    H: nd::ArrayView2<C64>,
    loperator: &'a L,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
where L: LOp<'a, S>
{
    -C64::i() * commutator(&H, rho) + loperator.op(rho)
}

/// Numerically integrate the Liouville equation using fourth-order
/// Runge-Kutta.
pub fn lindblad_evolve_rk4_with<'a, D, H, L, S>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    D: Into<DensityMaker<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
    S: PartialEq + Clone + 'a,
{
    let rho0: nd::Array2<C64> = rho0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array3<C64> = hbuilder.build(t);
    Some(evolve_rk4(&rho0, &h, |H, rho| rhs_lindblad_op(H, loperator, rho), t))
}

fn rhs_lindblad_static(
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    -C64::i() * commutator(H, rho) + lindbladian(Y, rho)
}

/// Numerically integrate the Lindblad equation using the midpoint rule for a
/// time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn lindblad_evolve_t_static(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_t_static(rho0, H, |H, rho| rhs_lindblad_static(H, Y, rho), t)
}

/// Numerically integrate the Lindblad equation using fourth-order Runge-Kutta
/// for a time-independent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency.
pub fn lindblad_evolve_rk4_static(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    evolve_rk4_static(rho0, H, |H, rho| rhs_lindblad_static(H, Y, rho), t)
}

fn rhs_lindblad_static_op<'a, L, S>(
    H: &nd::Array2<C64>,
    loperator: &'a L,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
where L: LOp<'a, S>
{
    -C64::i() * commutator(H, rho) + loperator.op(rho)
}

/// Numerically integrate the Liouville equation using fourth-order
/// Runge-Kutta for a time-independent Hamiltonian.
///
/// Fails if a time-independent Hamiltonian cannot be generated.
pub fn lindblad_evolve_rk4_static_with<'a, D, H, L, S>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    D: Into<DensityMaker<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
    S: PartialEq + Clone + 'a,
{
    let rho0: nd::Array2<C64> = rho0.into().into_array(hbuilder.get_basis())?;
    let h: nd::Array2<C64> = hbuilder.build_static()?;
    let res
        = evolve_rk4_static(
            &rho0, &h, |H, rho| rhs_lindblad_static_op(H, loperator, rho), t);
    Some(res)
}

/// Find the index of the `M`-th local maximum in an array of oscillating
/// values.
pub fn find_2pi_idx(X: &nd::Array1<f64>, M: usize, eps: f64) -> (usize, f64) {
    let mut x0: f64 = *X.first()
        .expect("find_2pi_idx: array must be non-empty");
    let mut k0: usize = 0;
    let switch: Vec<f64>
        = (0..2 * M).map(|m| (1 - 2 * (m % 2)) as f64).collect();
    let mut m: usize = 0;
    for (k, x) in X.iter().enumerate() {
        if switch[m] * (x - x0) <= -eps {
            x0 = *x;
            k0 = k;
        } else if switch[m] * (x - x0) > eps {
            if m < 2 * M - 1 {
                m += 1;
            } else {
                break;
            }
        }
    }
    (k0, x0)
}

// /// Numerically integrate the Schrödinger equation using the midpoint rule for a
// /// time-dependent Hamiltonian.
// ///
// /// Requires `H` to be in units of angular frequency. The third index of `H`
// /// should correspond to time.
// pub fn schrodinger_evolve_t(
//     psi0: &nd::Array1<C64>,
//     H: &nd::Array3<C64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array2<C64>
// {
//     let dt = array_diff(t);
//     let mut psi: nd::Array2<C64> = nd::Array::zeros((psi0.len(), t.len()));
//     let mut dpsi: nd::Array1<C64>;
//     let mut psi_new: nd::Array1<C64>;
//     let mut N: C64;
//     psi.slice_mut(s![.., 0]).assign(psi0);
//     psi.slice_mut(s![.., 1]).assign(psi0);
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(H.axis_iter(nd::Axis(2)).skip(1))
//         .enumerate();
//     for (k, ((&dtk, &dtkp1), Hkp1)) in iter {
//         dpsi = (-(dtk + dtkp1) * C64::i())
//             * Hkp1.dot(&psi.slice(s![.., k + 1]));
//         psi_new = &psi.slice(s![.., k]) + dpsi;
//         N = psi_new.mapv(|a| a * a.conj()).sum().sqrt();
//         (psi_new / N).move_into(psi.slice_mut(s![.., k + 2]));
//     }
//     psi
// }
//
// /// Numerically integrate the Schrödinger equation using fourth-order
// /// Runge-Kutta for a time-dependent Hamiltonian.
// ///
// /// Requires `H` to be in units of angular frequency. The third index of `H`
// /// should correspond to time.
// pub fn schrodinger_evolve_rk4(
//     psi0: &nd::Array1<C64>,
//     H: &nd::Array3<C64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array2<C64>
// {
//     let n = t.len();
//     let dt = array_diff(t);
//     let mut psi: nd::Array2<C64> = nd::Array::zeros((psi0.len(), t.len()));
//     let mut psi_old: nd::ArrayView1<C64>;
//     let mut phi1: nd::Array1<C64>;
//     let mut phi2: nd::Array1<C64>;
//     let mut phi3: nd::Array1<C64>;
//     let mut phi4: nd::Array1<C64>;
//     let mut psi_new: nd::Array1<C64>;
//     let mut N: C64;
//     psi.slice_mut(s![.., 0]).assign(psi0);
//     let rhs = |H: nd::ArrayView2<C64>, p: nd::ArrayView1<C64>| {
//         -C64::i() * H.dot(&p)
//     };
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(
//             H.axis_iter(nd::Axis(2))
//             .zip(H.axis_iter(nd::Axis(2)).skip(1))
//             .zip(H.axis_iter(nd::Axis(2)).skip(2))
//         )
//         .enumerate()
//         .step_by(2);
//     for (k, ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2))) in iter {
//         psi_old = psi.slice(s![.., k]);
//         phi1 = rhs(Hk, psi_old);
//         phi2 = rhs(Hkp1, (&psi_old + &phi1 * dtk).view());
//         phi3 = rhs(Hkp1, (&psi_old + &phi2 * dtk).view());
//         phi4 = rhs(Hkp2, (&psi_old + &phi3 * (dtk + dtkp1)).view());
//         psi_new
//             = &psi_old
//             + (phi1 + phi2 * 2.0 + phi3 * 2.0 + phi4) * ((dtk + dtkp1) / 6.0);
//         N = psi_new.mapv(|a| a * a.conj()).sum().sqrt();
//         (psi_new / N).move_into(psi.slice_mut(s![.., k + 2]));
//     }
//     for k in (1..n - 1).step_by(2) {
//         psi_new = (&psi.slice(s![.., k - 1]) + &psi.slice(s![.., k + 1])) / 2.0;
//         psi_new.move_into(psi.slice_mut(s![.., k]));
//     }
//     if n % 2 == 0 {
//         psi_new = psi.slice(s![.., n - 2]).to_owned();
//         psi_new.move_into(psi.slice_mut(s![.., n - 1]));
//     }
//     psi
// }

// /// Numerically integrate the Liouville equation using the midpoint rule for a
// /// time-dependent Hamiltonian.
// ///
// /// Requires `H` to be in units of angular frequency. The third index of `H`
// /// should correspond to time.
// pub fn liouville_evolve_t(
//     rho0: &nd::Array2<C64>,
//     H: &nd::Array3<C64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array3<C64>
// {
//     let dt = array_diff(t);
//     let mut rho: nd::Array3<C64>
//         = nd::Array::zeros((rho0.shape()[0], rho0.shape()[1], t.len()));
//     let mut drho: nd::Array2<C64>;
//     let mut rho_new: nd::Array2<C64>;
//     let mut N: C64;
//     rho.slice_mut(s![.., .., 0]).assign(rho0);
//     rho.slice_mut(s![.., .., 1]).assign(rho0);
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(H.axis_iter(nd::Axis(2)).skip(1))
//         .enumerate();
//     for (k, ((&dtk, &dtkp1), Hkp1)) in iter {
//         drho = (-(dtk + dtkp1) * C64::i())
//             * commutator(&Hkp1, &rho.slice(s![.., .., k + 1]));
//         rho_new = &rho.slice(s![.., .., k]) + drho;
//         N = trace(&rho_new);
//         (rho_new / N).move_into(rho.slice_mut(s![.., .., k + 2]));
//     }
//     rho
// }
//
// /// Numerically integrate the Liouville equation using fourth-order Runge-Kutta
// /// for a time-dependent Hamiltonian.
// ///
// /// Requires `H` to be in units of angular frequency. The third index of `H`
// /// should correspond to time.
// pub fn liouville_evolve_rk4(
//     rho0: &nd::Array2<C64>,
//     H: &nd::Array3<C64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array3<C64>
// {
//     let n = t.len();
//     let dt = array_diff(t);
//     let mut rho: nd::Array3<C64>
//         = nd::Array::zeros((rho0.shape()[0], rho0.shape()[1], t.len()));
//     let mut rho_old: nd::ArrayView2<C64>;
//     let mut r1: nd::Array2<C64>;
//     let mut r2: nd::Array2<C64>;
//     let mut r3: nd::Array2<C64>;
//     let mut r4: nd::Array2<C64>;
//     let mut rho_new: nd::Array2<C64>;
//     let mut N: C64;
//     rho.slice_mut(s![.., .., 0]).assign(rho0);
//     let rhs = |H: nd::ArrayView2<C64>, p: nd::ArrayView2<C64>| {
//         -C64::i() * commutator(&H, &p)
//     };
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(
//             H.axis_iter(nd::Axis(2))
//             .zip(H.axis_iter(nd::Axis(2)).skip(1))
//             .zip(H.axis_iter(nd::Axis(2)).skip(2))
//         )
//         .enumerate()
//         .step_by(2);
//     for (k, ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2))) in iter {
//         rho_old = rho.slice(s![.., .., k]);
//         r1 = rhs(Hk, rho_old);
//         r2 = rhs(Hkp1, (&rho_old + &r1 * dtk).view());
//         r3 = rhs(Hkp1, (&rho_old + &r2 * dtk).view());
//         r4 = rhs(Hkp2, (&rho_old + &r3 * (dtk + dtkp1)).view());
//         rho_new
//             = &rho_old
//             + (r1 + r2 * 2.0 + r3 * 2.0 + r4) * ((dtk + dtkp1) / 6.0);
//         N = rho_new.mapv(|a| a * a.conj()).sum().sqrt();
//         (rho_new / N).move_into(rho.slice_mut(s![.., .., k + 2]));
//     }
//     for k in (1..n - 1).step_by(2) {
//         rho_new
//             = (
//                 &rho.slice(s![.., .., k - 1])
//                 + &rho.slice(s![.., .., k + 1])
//             ) / 2.0;
//         rho_new.move_into(rho.slice_mut(s![.., .., k]));
//     }
//     if n % 2 == 0 {
//         rho_new = rho.slice(s![.., .., n - 2]).to_owned();
//         rho_new.move_into(rho.slice_mut(s![.., .., n - 1]));
//     }
//     rho
// }

// /// Numerically integrate the Lindblad equation using the midpoint rule for a
// /// time-dependent Hamiltonian.
// ///
// /// Requires `H` and `Y` to be in units of angular frequency. The third index of
// /// `H` should correspond to time.
// pub fn lindblad_evolve_t(
//     rho0: &nd::Array2<C64>,
//     H: &nd::Array3<C64>,
//     Y: &nd::Array2<f64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array3<C64>
// {
//     let dt = array_diff(t);
//     let mut rho: nd::Array3<C64>
//         = nd::Array::zeros((rho0.shape()[0], rho0.shape()[1], t.len()));
//     let mut rho_cur: nd::ArrayView2<C64>;
//     let mut drho: nd::Array2<C64>;
//     let mut rho_new: nd::Array2<C64>;
//     let mut N: C64;
//     rho.slice_mut(s![.., .., 0]).assign(rho0);
//     rho.slice_mut(s![.., .., 1]).assign(rho0);
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(H.axis_iter(nd::Axis(2)).skip(1))
//         .enumerate();
//     for (k, ((&dtk, &dtkp1), Hkp1)) in iter {
//         rho_cur = rho.slice(s![.., .., k + 1]);
//         drho = C64::from(dtk + dtkp1)
//             * (
//                 -C64::i() * commutator(&Hkp1, &rho_cur)
//                 + lindbladian(Y, &rho_cur)
//             );
//         rho_new = &rho.slice(s![.., .., k]) + drho;
//         N = trace(&rho_new);
//         (rho_new / N).move_into(rho.slice_mut(s![.., .., k + 2]));
//     }
//     rho
// }
//
// /// Numerically integrate the Liouville equation using fourth-order Runge-Kutta
// /// for a time-dependent Hamiltonian.
// ///
// /// Requires `H` and `Y` to be in units of angular frequency. The third index of
// /// `H` should correspond to time.
// pub fn lindblad_evolve_rk4(
//     rho0: &nd::Array2<C64>,
//     H: &nd::Array3<C64>,
//     Y: &nd::Array2<f64>,
//     t: &nd::Array1<f64>,
// ) -> nd::Array3<C64>
// {
//     let n = t.len();
//     let dt = array_diff(t);
//     let mut rho: nd::Array3<C64>
//         = nd::Array::zeros((rho0.shape()[0], rho0.shape()[1], t.len()));
//     let mut rho_old: nd::ArrayView2<C64>;
//     let mut r1: nd::Array2<C64>;
//     let mut r2: nd::Array2<C64>;
//     let mut r3: nd::Array2<C64>;
//     let mut r4: nd::Array2<C64>;
//     let mut rho_new: nd::Array2<C64>;
//     let mut N: C64;
//     rho.slice_mut(s![.., .., 0]).assign(rho0);
//     let rhs = |H: nd::ArrayView2<C64>, p: nd::ArrayView2<C64>| {
//         -C64::i() * commutator(&H, &p) + lindbladian(Y, &p)
//     };
//     let iter
//         = dt.iter()
//         .zip(dt.iter().skip(1))
//         .zip(
//             H.axis_iter(nd::Axis(2))
//             .zip(H.axis_iter(nd::Axis(2)).skip(1))
//             .zip(H.axis_iter(nd::Axis(2)).skip(2))
//         )
//         .enumerate()
//         .step_by(2);
//     for (k, ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2))) in iter {
//         rho_old = rho.slice(s![.., .., k]);
//         r1 = rhs(Hk, rho_old);
//         r2 = rhs(Hkp1, (&rho_old + &r1 * dtk).view());
//         r3 = rhs(Hkp1, (&rho_old + &r2 * dtk).view());
//         r4 = rhs(Hkp2, (&rho_old + &r3 * (dtk + dtkp1)).view());
//         rho_new
//             = &rho_old
//             + (r1 + r2 * 2.0 + r3 * 2.0 + r4) * ((dtk + dtkp1) / 6.0);
//         N = trace(&rho_new);
//         (rho_new / N).move_into(rho.slice_mut(s![.., .., k + 2]));
//     }
//     for k in (1..n - 1).step_by(2) {
//         rho_new
//             = (
//                 &rho.slice(s![.., .., k - 1])
//                 + &rho.slice(s![.., .., k + 1])
//             ) / 2.0;
//         rho_new.move_into(rho.slice_mut(s![.., .., k]));
//     }
//     if n % 2 == 0 {
//         rho_new = rho.slice(s![.., .., n - 2]).to_owned();
//         rho_new.move_into(rho.slice_mut(s![.., .., n - 1]));
//     }
//     rho
// }

