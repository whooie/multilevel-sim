//! Functions for numerical integration of the Schrödinger and Lindblad master
//! equations using a variety of methods.
//!
//! Where unspecified, the last index of an array corresponds to time.

#![allow(unused_imports)]

use ndarray::{ self as nd, s };
use ndarray_linalg::{ self as la, Eigh, Solve };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use crate::hilbert::outer_prod;

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
pub fn lindbladian(Y: &nd::Array2<f64>, rho: &nd::Array2<C64>)
    -> nd::Array2<C64>
{
    let n: usize = Y.shape()[0];
    let mut L: nd::Array2<C64> = nd::Array2::zeros(Y.raw_dim());
    let mut M: nd::Array2<C64>;
    for ((a, b), &y) in Y.indexed_iter() {
        if y <= f64::EPSILON { continue; }
        M = nd::Array2::from_shape_fn(
            (n, n),
            |(i, j)| {
                y * (
                    if i == a && j == a { rho[[b, b]] } else { C64::zero() }
                    - if i == b { rho[[i, j]] / 2.0 } else { C64::zero() }
                    - if j == b { rho[[i, j]] / 2.0 } else { C64::zero() }
                )
            }
        );
        L += &M;
    }
    L
}

/// Compute the trace of a square matrix `A`.
pub fn trace(A: &nd::Array2<C64>) -> C64 {
    A.diag().iter().copied().sum()
}

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

pub fn state_norm(state: &nd::Array1<C64>) -> C64 {
    state.iter().map(|a| *a * a.conj()).sum::<C64>().sqrt()
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
    let psi: Vec<nd::Array1<C64>>
        = t.iter()
        .map(|tk| V.dot(&(&c * E.mapv(|e| (-C64::i() * e * tk).exp()))))
        .collect();
    stack_arrays(nd::Axis(1), &psi)
        .expect("eigen_evolve: array stacking error")
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
    let mut psi: Vec<nd::Array1<C64>> = Vec::with_capacity(t.len());
    psi.push(psi0.clone());
    let mut psi_new: nd::Array1<C64>;
    for (dtk, Hk) in dt.into_iter().zip(H.axis_iter(nd::Axis(2))) {
        EV = Hk.eigh(la::UPLO::Lower)
            .expect("eigen_evolve_t: diagonalization error");
        c = EV.1.solve(psi.last().unwrap())
            .expect("eigen_evolve_t: linalg solve error");
        psi_new = EV.1.dot(&(c * EV.0.mapv(|e| (-C64::i() * e * dtk).exp())));
        psi.push(psi_new);
    }
    stack_arrays(nd::Axis(1), &psi)
        .expect("eigen_evolve_t: array stacking error")
}

/// Numerically integrate the Schrödinger equation using the midpoint rule.
///
/// Requires `H` to be in units of angular frequency.
pub fn schrodinger_evolve(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    let dt = array_diff(t);
    let mut psi: Vec<nd::Array1<C64>> = Vec::with_capacity(t.len());
    psi.push(psi0.clone());
    psi.push(psi0.clone());
    let mut len: usize = 2;
    let mut psi_old: &nd::Array1<C64>;
    let mut psi_cur: &nd::Array1<C64>;
    let mut dpsi: nd::Array1<C64>;
    let mut psi_new: nd::Array1<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1));
    for (&dtk, &dtkp1) in iter {
        psi_old = psi.get(len - 2).unwrap();
        psi_cur = psi.last().unwrap();
        dpsi = (-(dtk + dtkp1) * C64::i()) * H.dot(psi_cur);
        psi_new = psi_old + dpsi;
        N = state_norm(&psi_new);
        psi.push(psi_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(1), &psi)
        .expect("schrodinger_evolve: array stacking error")
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
    let dt = array_diff(t);
    let mut psi: Vec<nd::Array1<C64>> = Vec::with_capacity(t.len());
    psi.push(psi0.clone());
    psi.push(psi0.clone());
    let mut len: usize = 2;
    let mut psi_old: &nd::Array1<C64>;
    let mut psi_cur: &nd::Array1<C64>;
    let mut dpsi: nd::Array1<C64>;
    let mut psi_new: nd::Array1<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(H.axis_iter(nd::Axis(2)).skip(1));
    for ((&dtk, &dtkp1), Hk) in iter {
        psi_old = psi.get(len - 2).unwrap();
        psi_cur = psi.last().unwrap();
        dpsi = (-(dtk + dtkp1) * C64::i()) * Hk.dot(psi_cur);
        psi_new = psi_old + dpsi;
        N = state_norm(&psi_new);
        psi.push(psi_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(1), &psi)
        .expect("schrodinger_evolve_t: array stacking error")
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
    let dt = array_diff(t);
    let mut psi: Vec<nd::Array1<C64>> = Vec::with_capacity(t.len());
    psi.push(psi0.clone());
    let mut psi_old: &nd::Array1<C64>;
    let mut phi1: nd::Array1<C64>;
    let mut phi2: nd::Array1<C64>;
    let mut phi3: nd::Array1<C64>;
    let mut phi4: nd::Array1<C64>;
    let mut psi_mid: nd::Array1<C64>;
    let mut psi_new: nd::Array1<C64>;
    let mut N: C64;
    let rhs
        = |H: nd::ArrayView2<C64>, ps: &nd::Array1<C64>| {
            -C64::i() * H.dot(ps)
        };
    let iter
        = dt.iter().step_by(2)
        .zip(dt.iter().skip(1).step_by(2))
        .zip(
            H.axis_iter(nd::Axis(2)).step_by(2)
            .zip(H.axis_iter(nd::Axis(2)).skip(1).step_by(2))
            .zip(H.axis_iter(nd::Axis(2)).skip(2).step_by(2))
        );
    for ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2)) in iter {
        psi_old = psi.last().unwrap();
        phi1 = rhs(Hk, psi_old);
        phi2 = rhs(Hkp1, &(psi_old + &phi1 * dtk));
        phi3 = rhs(Hkp1, &(psi_old + &phi2 * dtk));
        phi4 = rhs(Hkp2, &(psi_old + &phi3 * (dtk + dtkp1)));
        psi_new
            = psi_old
            + (phi1 + phi2 * 2.0 + phi3 * 2.0 + phi4) * ((dtk + dtkp1) / 12.0);
        N = state_norm(&psi_new);
        psi_new /= N;
        psi_mid = (psi_old + &psi_new) * 0.5;
        psi.push(psi_mid);
        psi.push(psi_new);
    }
    if psi.len() < t.len() {
        psi.push(psi.last().unwrap().clone());
    }
    stack_arrays(nd::Axis(1), &psi)
        .expect("schrodinger_evolve_rk4: array stacking error")
}

/// Numerically integrate the Liouville equation using the midpoint rule.
///
/// Requires `H` to be in units of angular frequency.
pub fn liouville_evolve(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    rho.push(rho0.clone());
    let mut len: usize = 2;
    let mut rho_old: &nd::Array2<C64>;
    let mut rho_cur: &nd::Array2<C64>;
    let mut drho: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1));
    for (&dtk, &dtkp1) in iter {
        rho_old = rho.get(len - 2).unwrap();
        rho_cur = rho.last().unwrap();
        drho = (-(dtk + dtkp1) * C64::i()) * commutator(H, rho_cur);
        rho_new = rho_old + drho;
        N = trace(&rho_new);
        rho.push(rho_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("liouville_evolve: array stacking error")
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
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    rho.push(rho0.clone());
    let mut len: usize = 2;
    let mut rho_old: &nd::Array2<C64>;
    let mut rho_cur: &nd::Array2<C64>;
    let mut drho: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(H.axis_iter(nd::Axis(2)).skip(1));
    for ((&dtk, &dtkp1), Hk) in iter {
        rho_old = rho.get(len - 2).unwrap();
        rho_cur = rho.last().unwrap();
        drho = (-(dtk + dtkp1) * C64::i()) * commutator(&Hk, rho_cur);
        rho_new = rho_old + drho;
        N = trace(&rho_new);
        rho.push(rho_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("liouville_evolve_t: array stacking error")
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
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    let mut rho_old: &nd::Array2<C64>;
    let mut r1: nd::Array2<C64>;
    let mut r2: nd::Array2<C64>;
    let mut r3: nd::Array2<C64>;
    let mut r4: nd::Array2<C64>;
    let mut rho_mid: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let rhs
        = |H: nd::ArrayView2<C64>, r: &nd::Array2<C64>| {
            -C64::i() * commutator(&H, r)
        };
    let iter
        = dt.iter().step_by(2)
        .zip(dt.iter().skip(1).step_by(2))
        .zip(
            H.axis_iter(nd::Axis(2)).step_by(2)
            .zip(H.axis_iter(nd::Axis(2)).skip(1).step_by(2))
            .zip(H.axis_iter(nd::Axis(2)).skip(2).step_by(2))
        );
    for ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2)) in iter {
        rho_old = rho.last().unwrap();
        r1 = rhs(Hk, rho_old);
        r2 = rhs(Hkp1, &(rho_old + &r1 * dtk));
        r3 = rhs(Hkp1, &(rho_old + &r2 * dtk));
        r4 = rhs(Hkp2, &(rho_old + &r3 * (dtk + dtkp1)));
        rho_new
            = rho_old
            + (r1 + r2 * 2.0 + r3 * 2.0 + r4) * ((dtk + dtkp1) / 12.0);
        N = trace(&rho_new);
        rho_new /= N;
        rho_mid = (rho_old + &rho_new) * 0.5;
        rho.push(rho_mid);
        rho.push(rho_new);
    }
    if rho.len() < t.len() {
        rho.push(rho.last().unwrap().clone());
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("liouville_evolve_rk4: array stacking error")
}

/// Numerically integrate the Lindblad equation using the midpoint rule.
///
/// Requires `H` to be in units of angular frequency.
pub fn lindblad_evolve(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    rho.push(rho0.clone());
    let mut len: usize = 2;
    let mut rho_old: &nd::Array2<C64>;
    let mut rho_cur: &nd::Array2<C64>;
    let mut drho: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1));
    for (&dtk, &dtkp1) in iter {
        rho_old = rho.get(len - 2).unwrap();
        rho_cur = rho.last().unwrap();
        drho
            = (-C64::i() * commutator(H, rho_cur) + lindbladian(Y, rho_cur))
            * (dtk + dtkp1);
        rho_new = rho_old + drho;
        N = trace(&rho_new);
        rho.push(rho_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("lindblad_evolve: array stacking error")
}

/// Numerically integrate the Lindblad equation using the midpoint rule for a
/// time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn lindblad_evolve_t(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    rho.push(rho0.clone());
    let mut len: usize = 2;
    let mut rho_old: &nd::Array2<C64>;
    let mut rho_cur: &nd::Array2<C64>;
    let mut drho: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let iter
        = dt.iter()
        .zip(dt.iter().skip(1))
        .zip(H.axis_iter(nd::Axis(2)).skip(1));
    for ((&dtk, &dtkp1), Hk) in iter {
        rho_old = rho.get(len - 2).unwrap();
        rho_cur = rho.last().unwrap();
        drho
            = (-C64::i() * commutator(&Hk, rho_cur) + lindbladian(Y, rho_cur))
            * (dtk + dtkp1);
        rho_new = rho_old + drho;
        N = trace(&rho_new);
        rho.push(rho_new / N);
        len += 1;
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("lindblad_evolve_t: array stacking error")
}

/// Numerically integrate the Lindblad equation using fourth-order Runge-Kutta
/// for a time-dependent Hamiltonian.
///
/// Requires `H` to be in units of angular frequency. The third index of `H`
/// should correspond to time.
pub fn lindblad_evolve_rk4(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    let dt = array_diff(t);
    let mut rho: Vec<nd::Array2<C64>> = Vec::with_capacity(t.len());
    rho.push(rho0.clone());
    let mut rho_old: &nd::Array2<C64>;
    let mut r1: nd::Array2<C64>;
    let mut r2: nd::Array2<C64>;
    let mut r3: nd::Array2<C64>;
    let mut r4: nd::Array2<C64>;
    let mut rho_mid: nd::Array2<C64>;
    let mut rho_new: nd::Array2<C64>;
    let mut N: C64;
    let rhs
        = |H: nd::ArrayView2<C64>, r: &nd::Array2<C64>| {
            -C64::i() * commutator(&H, r) + lindbladian(Y, r)
        };
    let iter
        = dt.iter().step_by(2)
        .zip(dt.iter().skip(1).step_by(2))
        .zip(
            H.axis_iter(nd::Axis(2)).step_by(2)
            .zip(H.axis_iter(nd::Axis(2)).skip(1).step_by(2))
            .zip(H.axis_iter(nd::Axis(2)).skip(2).step_by(2))
        );
    for ((&dtk, &dtkp1), ((Hk, Hkp1), Hkp2)) in iter {
        rho_old = rho.last().unwrap();
        r1 = rhs(Hk, rho_old);
        r2 = rhs(Hkp1, &(rho_old + &r1 * dtk));
        r3 = rhs(Hkp1, &(rho_old + &r2 * dtk));
        r4 = rhs(Hkp2, &(rho_old + &r3 * (dtk + dtkp1)));
        rho_new
            = rho_old
            + (r1 + r2 * 2.0 + r3 * 2.0 + r4) * ((dtk + dtkp1) / 12.0);
        N = trace(&rho_new);
        rho_new /= N;
        rho_mid = (rho_old + &rho_new) * 0.5;
        rho.push(rho_mid);
        rho.push(rho_new);
    }
    if rho.len() < t.len() {
        rho.push(rho.last().unwrap().clone());
    }
    stack_arrays(nd::Axis(2), &rho)
        .expect("lindblad_evolve_rk4: array stacking error")
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

