//! Evolution functions for the Lindblad equation.

use super::*;

fn rhs(
    h: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    -C64::i() * commutator(h, rho) + lindbladian(Y, rho)
}

fn rhs_view(
    h: nd::ArrayView2<C64>,
    Y: &nd::Array2<f64>,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
{
    -C64::i() * commutator(&h, rho) + lindbladian(Y, rho)
}

fn rhs_op<'a, S, L>(
    h: &nd::Array2<C64>,
    l: &'a L,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
where L: LOp<'a, S>
{
    -C64::i() * commutator(h, rho) + l.op(rho)
}

fn rhs_view_op<'a, S, L>(
    h: nd::ArrayView2<C64>,
    l: &'a L,
    rho: &nd::Array2<C64>,
) -> nd::Array2<C64>
where L: LOp<'a, S>
{
    -C64::i() * commutator(&h, rho) + l.op(rho)
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian.
///
/// See also [`lindbladian`] for info on the decay coupling matrix `Y`.
pub fn evolve(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    do_evolve(rho0, H, |h, rho| rhs(h, Y, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian.
pub fn evolve_op<'a, S, L>(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    L: &'a L,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
where L: LOp<'a, S>
{
    do_evolve(rho0, H, |h, rho| rhs_op(h, L, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_with<'a, S, D, H, L>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    Some(do_evolve(&rho0, &H, |h, rho| rhs_op(h, loperator, rho), t))
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian.
///
/// See also [`lindbladian`] for info on the decay coupling matrix `Y`.
pub fn evolve_t(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
{
    do_evolve_t(rho0, H, |h, rho| rhs_view(h, Y, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian.
pub fn evolve_t_op<'a, S, L>(
    rho0: &nd::Array2<C64>,
    H: &nd::Array3<C64>,
    L: &'a L,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
where L: LOp<'a, S>
{
    do_evolve_t(rho0, H, |h, rho| rhs_view_op(h, L, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian.
///
/// Fails if the initial state description is invalid.
pub fn evolve_t_with<'a, S, D, H, L>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build(t);
    let res = do_evolve_t(
        &rho0, &H, |h, rho| rhs_view_op(h, loperator, rho), t);
    Some(res)
}

/// Numerically intergrate the Lindblad equation for a time-dependent
/// Hamiltonian given by a function.
///
/// See also [`lindbladian`] for info on the decay coupling matrix `Y`.
pub fn evolve_fn<H>(
    rho0: &nd::Array2<C64>,
    H: H,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
where H: Fn(f64) -> nd::Array2<C64>
{
    do_evolve_fn(rho0, H, |h, rho| rhs(h, Y, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian given by a function.
pub fn evolve_fn_op<'a, S, H, L>(
    rho0: &nd::Array2<C64>,
    H: H,
    L: &'a L,
    t: &nd::Array1<f64>,
) -> nd::Array3<C64>
where
    H: Fn(f64) -> nd::Array2<C64>,
    L: LOp<'a, S>,
{
    do_evolve_fn(rho0, H, |h, rho| rhs_op(h, L, rho), t)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`].
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_with<'a, S, D, H, L>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
) -> Option<nd::Array3<C64>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    Some(do_evolve_fn(&rho0, H, |h, rho| rhs_op(h, loperator, rho), t))
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian with reduced integration output.
///
/// See also [`lindbladian`] for info on the decay coupling matrix `Y`.
pub fn evolve_reduced<X, T>(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where X: Fn(&nd::Array2<C64>) -> T
{
    do_evolve_reduced(rho0, H, |h, rho| rhs(h, Y, rho), t, x)
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian with reduced integration output.
pub fn evolve_reduced_op<'a, S, L, X, T>(
    rho0: &nd::Array2<C64>,
    H: &nd::Array2<C64>,
    L: &'a L,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    L: LOp<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    do_evolve_reduced(rho0, H, |h, rho| rhs_op(h, L, rho), t, x)
}

/// Numerically integrate the Lindblad equation for a time-independent
/// Hamiltonian with reduced integration output.
///
/// Fails if the initial state description is invalid or a time-independent
/// Hamiltonian cannot be built.
pub fn evolve_reduced_with<'a, S, D, H, L, X, T>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = hbuilder.build_static()?;
    let res = do_evolve_reduced(
        &rho0, &H, |h, rho| rhs_op(h, loperator, rho), t, x);
    Some(res)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian given by a function with reduced integration output.
///
/// See also [`lindbladian`] for info on the decay coupling matrix `Y`.
pub fn evolve_fn_reduced<H, X, T>(
    rho0: &nd::Array2<C64>,
    H: H,
    Y: &nd::Array2<f64>,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    H: Fn(f64) -> nd::Array2<C64>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    do_evolve_fn_reduced(rho0, H, |h, rho| rhs(h, Y, rho), t, x)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian given by a function with reduced integration output.
pub fn evolve_fn_reduced_op<'a, S, H, L, X, T>(
    rho0: &nd::Array2<C64>,
    H: H,
    L: &'a L,
    t: &nd::Array1<f64>,
    x: X,
) -> Vec<T>
where
    H: Fn(f64) -> nd::Array2<C64>,
    L: LOp<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    do_evolve_fn_reduced(rho0, H, |h, rho| rhs_op(h, L, rho), t, x)
}

/// Numerically integrate the Lindblad equation for a time-dependent
/// Hamiltonian with the functional interface of [`HBuild`] with reduced
/// integration output.
///
/// Fails if the initial state description is invalid.
pub fn evolve_fn_reduced_with<'a, S, D, H, L, X, T>(
    rho0: D,
    hbuilder: &'a H,
    loperator: &'a L,
    t: &nd::Array1<f64>,
    x: X,
) -> Option<Vec<T>>
where
    S: Clone + PartialEq + 'a,
    D: Into<Density<'a, S>>,
    H: HBuild<'a, S>,
    L: LOp<'a, S>,
    X: Fn(&nd::Array2<C64>) -> T,
{
    let rho0 = rho0.into().into_array(hbuilder.get_basis())?;
    let H = |t| hbuilder.build_at(t);
    let res = do_evolve_fn_reduced(
        &rho0, H, |h, rho| rhs_op(h, loperator, rho), t, x);
    Some(res)
}

