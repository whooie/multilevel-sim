///! Evolution functions for the Schrödinger equation.

use super::*;

/// Compute the Schrödinger coherent evolution of the initial state `psi0` for
/// Hamiltonian `H`.
///
/// Note: `psi0` should be the initial state at time `t = 0`, not necessarily
/// any of the elements of `t`.
pub fn eigen(
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
    let iter = t.iter().zip(psi.axis_iter_mut(nd::Axis(1)));
    for (&tk, psik) in iter {
        V.dot(&(&c * &E.mapv(|e| (-C64::i() * e * tk).exp())))
            .move_into(psik);
        }
    psi
}

/// Compute the Schrödinger coherent evolution of an initial state for a
/// time-independent Hamiltonian.
///
/// Fails if a time-independent Hamiltonian cannot be built or the initial state
/// description is invalid.
pub fn eigen_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let h = hbuilder.build_static()?;
    Some(evolve(&psi0, &h, t))
}

/// Compute the Schrödinger coherent evolution of an initial state `psi0` for
/// time-dependent Hamiltonian `H` by diagonalizing at each time step.
pub fn eigen_t(
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
///
/// Fails if the initial state description is invalid.
pub fn eigen_t_with<'a, P, H, S>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
    S: PartialEq + 'a,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let h = hbuilder.build(t);
    Some(evolve_t(&psi0, &h, t))
}
fn rhs(h: &nd::Array2<C64>, psi: &nd::Array1<C64>)
    -> nd::Array1<C64>
{
    -C64::i() * h.dot(psi)
}

fn rhs_view(h: nd::ArrayView2<C64>, psi: &nd::Array1<C64>)
    -> nd::Array1<C64>
{
    -C64::i() * h.dot(psi)
}

/// Numerically integrate the Schrödinger equation for a time-independent
/// Hamiltonian.
pub fn evolve(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    do_evolve(psi0, H, rhs, t)
}

/// Numerically integrate the Schrödinger equation for a time-independent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_with<'a, S, P, H>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    S: PartialEq + 'a,
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    Some(do_evolve(&psi0, &H, rhs, t))
}

/// Numerically integrate the Schrödinger equation for a time-dependent
/// Hamiltonian.
pub fn evolve_t(
    psi0: &nd::Array1<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
{
    do_evolve_t(psi0, H, rhs_view, t)
}

/// Numerically integrate the Schrödinger equation for a time-dependent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid.
pub fn evolve_t_with<'a, S, P, H>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    S: PartialEq + 'a,
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build(t);
    Some(do_evolve_t(&psi0, &H, rhs_view, t))
}

/// Numerically intergrate the Schrödinger equation for a time-dependent
/// Hamiltonian given by a function.
pub fn evolve_fn<H>(
    psi0: &nd::Array1<C64>,
    H: H,
    t: &nd::Array1<f64>,
) -> nd::Array2<C64>
where H: Fn(f64) -> nd::Array2<C64>
{
    do_evolve_fn(psi0, H, rhs, t)
}

/// Numerically integrate the Schrödinger equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`].
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_with<'a, S, P, H>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array2<C64>>
where
    S: PartialEq + 'a,
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    Some(do_evolve_fn(&psi0, H, rhs, t))
}

/// Numerically integrate the Schrödinger equation for a time-independent
/// Hamiltonian with reduced integration output.
pub fn evolve_reduced<X, T>(
    psi0: &nd::Array1<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where X: Fn(&nd::Array1<C64>) -> T
{
    do_evolve_reduced(psi0, H, rhs, t, x)
}

/// Numerically integrate the Schrödinger equation for a time-independent
/// Hamiltonian with reduced integration output.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_reduced_with<'a, S, P, H, X, T>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: PartialEq + 'a,
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
    X: Fn(&nd::Array1<C64>) -> T,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    Some(do_evolve_reduced(&psi0, &H, rhs, t, x))
}

/// Numerically integrate the Schrödinger equation for a time-dependent
/// Hamiltonian given by a function with reduced integration output.
pub fn evolve_fn_reduced<H, X, T>(
    psi0: &nd::Array1<C64>,
    H: H,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    H: Fn(f64) -> nd::Array2<C64>,
    X: Fn(&nd::Array1<C64>) -> T,
{
    do_evolve_fn_reduced(psi0, H, rhs, t, x)
}

/// Numerically integrate the Schrödinger equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`] with reduced
/// integration output.
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_reduced_with<'a, S, P, H, X, T>(
    psi0: P,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: PartialEq + 'a,
    P: Into<Pure<'a, S>>,
    H: HBuild<'a, S>,
    X: Fn(&nd::Array1<C64>) -> T,
{
    let psi0 = psi0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    Some(do_evolve_fn_reduced(&psi0, H, rhs, t, x))
}

