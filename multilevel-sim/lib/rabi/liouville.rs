//! Evolution functions for the Liouville equation.

use super::*;

fn rhs(h: &nd::Array2<C64>, rho: &nd::Array2<C64>)
    -> nd::Array2<C64>
{
    -C64::i() * commutator(h, rho)
}

fn rhs_view(h: nd::ArrayView2<C64>, rho: &nd::Array2<C64>)
    -> nd::Array2<C64>
{
    -C64::i() * commutator(&h, rho)
}

/// Numerically integrate the Liouville equation for a time-independent
/// Hamiltonian.
pub fn evolve(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    do_evolve(rho0, H, rhs, t)
}

/// Numerically integrate the Liouville equation for a time-independent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_with<'a, S, D, H>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    Some(do_evolve(&rho0, &H, rhs, t))
}

/// Numerically integrate the Liouville equation using fourth-order for a
/// time-dependent Hamiltonian.
pub fn evolve_t(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    do_evolve_t(rho0, H, rhs_view, t)
}

/// Numerically integrate the Liouville equation for a time-dependent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid.
pub fn evolve_t_with<'a, S, D, H>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build(t);
    Some(do_evolve_t(&rho0, &H, rhs_view, t))
}

/// Numerically intergrate the Liouville equation for a time-dependent
/// Hamiltonian given by a function.
pub fn evolve_fn<H>(
    rho0: &nd::Array2<C64>,
    H: H,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
where H: Fn(f64) -> nd::Array2<C64>
{
    do_evolve_fn(rho0, H, rhs, t)
}

/// Numerically integrate the Liouville equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`].
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_with<'a, S, D, H>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    Some(do_evolve_fn(&rho0, H, rhs, t))
}

/// Numerically integrate the Liouville equation for a time-independent
/// Hamiltonian with reduced integration output.
pub fn evolve_reduced<X, T>(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where X: Fn(&nd::Array2<C64>) -> T
{
    do_evolve_reduced(rho0, H, rhs, t, x)
}

/// Numerically integrate the Liouville equation for a time-independent
/// Hamiltonian with reduced integration output.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_reduced_with<'a, S, D, H, X, T>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    Some(do_evolve_reduced(&rho0, &H, rhs, t, x))
}

/// Numerically integrate the Liouville equation for a time-dependent
/// Hamiltonian given by a function with reduced integration output.
pub fn evolve_fn_reduced<H, X, T>(
    rho0: &nd::Array2<C64>,
    H: H,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    H: Fn(f64) -> nd::Array2<C64>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    do_evolve_fn_reduced(rho0, H, rhs, t, x)
}

/// Numerically integrate the Liouville equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`] with reduced
/// integration output.
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_reduced_with<'a, S, D, H, X, T>(
    rho0: D,
    hbuilder: &'a H,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    Some(do_evolve_fn_reduced(&rho0, H, rhs, t, x))
}

