//! Definitions to describe states, bases, and combinations thereof.

use std::{ hash::Hash, ops::{ Deref, DerefMut } };
use ndarray as nd;
use indexmap::IndexMap;
use itertools::Itertools;
use num_complex::Complex64 as C64;
use num_traits::{ Zero, One };
use crate::{
    spin::{ Spin }
};

/* States *********************************************************************/

/// A single basis state.
pub trait BasisState: Clone + PartialEq + Eq + Hash + std::fmt::Debug {
    /// Return `true` if two states can be coupled by an electric dipole
    /// transition.
    ///
    /// This method should be reflextive in its inputs.
    fn couples_to(&self, other: &Self) -> bool;
}

/// Extends [`BasisState`] to include spin state properties.
pub trait SpinState: BasisState {
    /// Get the [`Spin`] of the basis state.
    fn spin(&self) -> Spin;
}

/// Extends [`BasisState`] to include Rydberg properties.
pub trait RydbergState: BasisState {
    /// Return `true` if `self` is a Rydberg state.
    fn is_rydberg(&self) -> bool;

    /// Return the C6 coefficient in units of angular frequency times length^6
    /// for a Rydberg interaction with another state.
    fn c6_with(&self, other: &Self) -> Option<f64>;
}

/// Extends [`BasisState`] to include motional Fock state properties in a magic
/// trap.
pub trait TrappedMagic: BasisState {
    /// Trap frequency, in units of angular frequency, for all states.
    const TRAP_FREQ: f64;
}

/// Extends [`BasisState`] to include motional Fock state properties in a
/// non-magic trap.
pub trait Trapped: BasisState {
    /// Trap frequency, in units of angular frequency, for a particular state.
    fn trap_freq(&self) -> f64;
}

/// Combination of an atomic state `S` with a motional Fock state index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fock<S>(S, usize)
where S: BasisState;

impl<S> BasisState for Fock<S>
where S: BasisState
{
    fn couples_to(&self, other: &Self) -> bool {
        self.0.couples_to(&other.0)
    }
}

impl<S> SpinState for Fock<S>
where S: SpinState
{
    fn spin(&self) -> Spin { self.0.spin() }
}

impl<S> Fock<S>
where S: BasisState
{
    /// Return the atomic state.
    pub fn atomic_state(&self) -> &S { &self.0 }

    /// Return `true` if two states are in the same atomic state.
    pub fn same_atomic(&self, other: &Self) -> bool { self.0 == other.0 }

    /// Return the Fock state index.
    pub fn fock_index(&self) -> usize { self.1 }
}

impl<S> Fock<S>
where S: TrappedMagic {
    /// Trap frequency, in units of angular frequency, for all states.
    const TRAP_FREQ: f64 = S::TRAP_FREQ;
}

impl<S> Fock<S>
where S: Trapped {
    /// Return the trap frequency, in units of angular frequency, of a
    /// particular state.
    ///
    /// This will give equal trap frequencies for equal atomic states.
    pub fn trap_freq(&self) -> f64 { self.0.trap_freq() }
}

impl<S> From<(S, usize)> for Fock<S>
where S: BasisState
{
    fn from(sn: (S, usize)) -> Self {
        let (s, n) = sn;
        Self(s, n)
    }
}

/// Extends [`BasisState`] to include spontaneous decay properties.
pub trait SpontaneousDecay: BasisState {
    /// Get the rate of spontaneous decay to another state in units of angular
    /// frequency.
    fn decay_rate(&self, other: &Self) -> Option<f64>;
}

/// Compute the outer product of two state vectors.
pub fn outer_prod(a: &nd::Array1<C64>, b: &nd::Array1<C64>)
    -> nd::Array2<C64>
{
    let na = a.len();
    let nb = b.len();
    nd::Array2::from_shape_vec(
        (na, nb),
        a.iter().cartesian_product(b)
            .map(|(ai, bj)| *ai * bj.conj())
            .collect(),
    )
    .unwrap()
}

/* Bases **********************************************************************/

/// A collection of unique [`BasisState`]s with associated energies in units of
/// angular frequency.
///
/// This collection is backed by a single [`IndexMap`], which can be accessed
/// via [`AsRef`], [`AsMut`], [`Deref`] and [`DerefMut`].
#[derive(Clone, Debug)]
pub struct Basis<S>
where S: BasisState
{
    energies: IndexMap<S, f64>,
}

impl<S> AsRef<IndexMap<S, f64>> for Basis<S>
where S: BasisState
{
    fn as_ref(&self) -> &IndexMap<S, f64> { &self.energies }
}

impl<S> AsMut<IndexMap<S, f64>> for Basis<S>
where S: BasisState
{
    fn as_mut(&mut self) -> &mut IndexMap<S, f64> { &mut self.energies }
}

impl<S> Deref for Basis<S>
where S: BasisState
{
    type Target = IndexMap<S, f64>;

    fn deref(&self) -> &Self::Target { &self.energies }
}

impl<S> DerefMut for Basis<S>
where S: BasisState
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.energies }
}

impl<S> Default for Basis<S>
where S: BasisState
{
    fn default() -> Self { Self { energies: IndexMap::default() } }
}

impl<S> FromIterator<(S, f64)> for Basis<S>
where S: BasisState
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (S, f64)>
    {
        Self { energies: iter.into_iter().collect() }
    }
}

impl<S> Basis<S>
where S: BasisState
{
    /// Create a new, empty basis.
    pub fn new() -> Self { Self::default() }

    /// Get the energy in units of angular frequency of a particular basis
    /// state.
    pub fn get_energy(&self, state: &S) -> Option<f64> {
        self.energies.get(state).copied()
    }

    /// Get the energy in units of angular frequency of a particular basis
    /// state.
    pub fn get_energy_mut(&mut self, state: &S) -> Option<&mut f64> {
        self.energies.get_mut(state)
    }

    /// Get the energy in units of angular frequency of a particular basis state
    /// by index.
    pub fn get_energy_index(&self, index: usize) -> Option<f64> {
        self.energies.get_index(index).map(|(_, e)| e).copied()
    }

    /// Get the energy in units of angular frequency of a particular basis state
    /// by index.
    pub fn get_energy_index_mut(&mut self, index: usize) -> Option<&mut f64> {
        self.energies.get_index_mut(index).map(|(_, e)| e)
    }

    /// Get an array representation of a particular basis state.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector(&self, state: &S) -> Option<nd::Array1<C64>> {
        self.energies.get_index_of(state)
            .map(|k| {
                let n = self.energies.len();
                (0..n).map(|j| if j == k { C64::one() } else { C64::zero() })
                    .collect()
            })
    }

    /// Get an array representation of a particular basis state by index.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector_index(&self, index: usize) -> Option<nd::Array1<C64>> {
        let n = self.energies.len();
        (index < n).then(|| {
            (0..n).map(|j| if j == index { C64::one() } else { C64::zero() })
                .collect()
        })
    }

    /// Get an array representation of a linear combination of basis states,
    /// with weights determined by a weighting function.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector_weighted<F>(&self, weights: F) -> nd::Array1<C64>
    where F: Fn(&S, usize, f64) -> C64
    {
        self.energies.iter().enumerate()
            .map(|(index, (state, energy))| weights(state, index, *energy))
            .collect()
    }

    /// Get an array representation of the density matrix for a particular basis
    /// state.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density(&self, state: &S) -> Option<nd::Array2<C64>> {
        self.get_vector(state)
            .map(|diag| nd::Array2::from_diag(&diag))
    }

    /// Get an array representation of the density matrix for a particular basis
    /// state by index.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density_index(&self, index: usize) -> Option<nd::Array2<C64>> {
        self.get_vector_index(index)
            .map(|diag| nd::Array2::from_diag(&diag))
    }

    /// Get an array representation of the density matrix for a linear
    /// combination of basis states with weights determined by a weighting
    /// function.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density_weighted<F>(&self, weights: F) -> nd::Array2<C64>
    where F: Fn(&S, usize, f64) -> C64
    {
        let vector: nd::Array1<C64> = self.get_vector_weighted(weights);
        outer_prod(&vector, &vector)
    }

    /// Get an array representiation of a density matrix for a completely
    /// classical mixture (i.e. a diagonal matrix) of basis states with weights
    /// determined by a weighting function.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density_diag<F>(&self, weights: F) -> nd::Array2<C64>
    where F: Fn(&S, usize, f64) -> f64
    {
        let diag: nd::Array1<C64>
            = self.energies.iter().enumerate()
            .map(|(index, (state, energy))| {
                C64::from(weights(state, index, *energy))
            })
            .collect();
        nd::Array2::from_diag(&diag)
    }

    /// Create a new [`ProdBasis`] from the Kronecker product of `self` on the
    /// left with another basis on the right.
    pub fn kron_with(&self, rhs: &Self) -> ProdBasis<S> {
        self.energies.iter()
            .flat_map(|(s1, e1)| {
                rhs.energies.iter()
                    .map(|(s2, e2)| (vec![s1.clone(), s2.clone()], *e1 + *e2))
            })
            .collect()
    }

    /// Create a new [`ProdBasis`] from the Kronecker product of `self` on the
    /// left with a `ProdBasis` on the right.
    pub fn kron_with_prod(&self, rhs: &ProdBasis<S>) -> ProdBasis<S> {
        self.energies.iter()
            .flat_map(|(s1, e1)| {
                rhs.energies.iter()
                    .map(move |(ss2, e2)| {
                        let prod_state: Vec<S>
                            = [s1].into_iter().cloned()
                            .chain(ss2.iter().cloned())
                            .collect();
                        (prod_state, e1 + e2)
                    })
            })
            .collect()
    }

    /// Calculate the time dependence of the states' phases (integral of state
    /// frequencies) over an array of time coordinates.
    pub fn gen_time_dep(&self, time: &nd::Array1<f64>)
        -> IndexMap<S, nd::Array1<f64>>
    {
        self.energies.iter()
            .map(|(s, e)| (s.clone(), *e * time))
            .collect()
    }
}

/* Product-state basis ********************************************************/

/// A collection of unique compositions of [`BasisState`]s with associated
/// energies in units of angular frequency.
///
/// This collection is backed by a single [`IndexMap`], which can be accessed
/// via [`AsRef`], [`AsMut`], [`Deref`] and [`DerefMut`].
#[derive(Clone, Debug)]
pub struct ProdBasis<S>
where S: BasisState
{
    energies: IndexMap<Vec<S>, f64>,
}

impl<S> AsRef<IndexMap<Vec<S>, f64>> for ProdBasis<S>
where S: BasisState
{
    fn as_ref(&self) -> &IndexMap<Vec<S>, f64> { &self.energies }
}

impl<S> AsMut<IndexMap<Vec<S>, f64>> for ProdBasis<S>
where S: BasisState
{
    fn as_mut(&mut self) -> &mut IndexMap<Vec<S>, f64> { &mut self.energies }
}

impl<S> Deref for ProdBasis<S>
where S: BasisState
{
    type Target = IndexMap<Vec<S>, f64>;

    fn deref(&self) -> &Self::Target { &self.energies }
}

impl<S> DerefMut for ProdBasis<S>
where S: BasisState
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.energies }
}

impl<S> Default for ProdBasis<S>
where S: BasisState
{
    fn default() -> Self { Self { energies: IndexMap::default() } }
}

impl<S> FromIterator<(Vec<S>, f64)> for ProdBasis<S>
where S: BasisState
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (Vec<S>, f64)>
    {
        Self { energies: iter.into_iter().collect() }
    }
}

impl<S> ProdBasis<S>
where S: BasisState
{
    /// Create a new, empty product basis.
    pub fn new() -> Self { Self::default() }

    /// Get the energy in units of angular frequency of a particular basis
    /// state.
    pub fn get_energy(&self, state: &[S]) -> Option<f64> {
        self.energies.get(state).copied()
    }

    /// Get the energy in units of angular frequency of a particular basis
    /// state.
    pub fn get_energy_mut(&mut self, state: &[S]) -> Option<&mut f64> {
        self.energies.get_mut(state)
    }

    /// Get the energy in units of angular frequency of a particular basis state
    /// by index.
    pub fn get_energy_index(&self, index: usize) -> Option<f64> {
        self.energies.get_index(index).map(|(_, e)| e).copied()
    }

    /// Get the energy in units of angular frequency of a particular basis state
    /// by index.
    pub fn get_energy_index_mut(&mut self, index: usize) -> Option<&mut f64> {
        self.energies.get_index_mut(index).map(|(_, e)| e)
    }

    /// Get an array representation of a particular product state.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector(&self, state: &[S]) -> Option<nd::Array1<C64>> {
        self.energies.get_index_of(state)
            .map(|k| {
                let n = self.energies.len();
                (0..n).map(|j| if j == k { C64::one() } else { C64::zero() })
                    .collect()
            })
    }

    /// Get an array representation of a particular product state by index.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector_index(&self, index: usize) -> Option<nd::Array1<C64>> {
        let n = self.energies.len();
        (index < n).then(|| {
            (0..n).map(|j| if j == index { C64::one() } else { C64::zero() })
                .collect()
        })
    }

    /// Get an array representation of a linear combination of product states,
    /// with weights determined by a weighting function.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_vector_weighted<F>(&self, weights: F) -> nd::Array1<C64>
    where F: Fn(&[S], usize, f64) -> C64
    {
        self.energies.iter().enumerate()
            .map(|(index, (state, energy))| weights(state, index, *energy))
            .collect()
    }

    /// Get an array representation of the density matrix for a particular
    /// product state.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density(&self, state: &[S]) -> Option<nd::Array2<C64>> {
        self.get_vector(state)
            .map(|diag| nd::Array2::from_diag(&diag))
    }

    /// Get an array representation of the density matrix for a particular
    /// product state by index.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density_index(&self, index: usize) -> Option<nd::Array2<C64>> {
        self.get_vector_index(index)
            .map(|diag| nd::Array2::from_diag(&diag))
    }

    /// Get an array representation of the density matrix for a linear
    /// combination of product states with weights determined by a weighting
    /// function.
    ///
    /// The array is sized to match the number of states currently in `self`.
    pub fn get_density_weighted<F>(&self, weights: F) -> nd::Array2<C64>
    where F: Fn(&[S], usize, f64) -> C64
    {
        let vector: nd::Array1<C64> = self.get_vector_weighted(weights);
        outer_prod(&vector, &vector)
    }

    /// Get an array representation of a density matrix for a completely
    /// classical mixture (i.e. a diagonal matrix) of basis states with weights
    /// determined by a weighting function.
    ///
    /// the array is sized to match the number of states currently in `self`.
    pub fn get_density_diag<F>(&self, weights: F) -> nd::Array2<C64>
    where F: Fn(&[S], usize, f64) -> f64
    {
        let diag: nd::Array1<C64>
            = self.energies.iter().enumerate()
            .map(|(index, (state, energy))| {
                C64::from(weights(state, index, *energy))
            })
            .collect();
        nd::Array2::from_diag(&diag)
    }

    /// Create a new `ProdBasis` from the Kronecker product of `self` on the
    /// left with another `ProdBasis` on the right.
    pub fn kron_with(&self, rhs: &Self) -> Self {
        self.energies.iter()
            .flat_map(|(ss1, e1)| {
                rhs.energies.iter()
                    .map(|(ss2, e2)| {
                        let prod_state: Vec<S>
                            = [ss1.to_owned(), ss2.to_owned()].concat();
                        (prod_state, *e1 + *e2)
                    })
            })
            .collect()
    }

    /// Create a new `ProdBasis` from the Kronecker product of `self` on the
    /// left with a [`Basis`] on the right.
    pub fn kron_with_single(&self, rhs: &Basis<S>) -> Self {
        self.energies.iter()
            .flat_map(|(ss1, e1)| {
                rhs.energies.iter()
                    .map(|(s2, e2)| {
                        let mut prod_state = ss1.to_owned();
                        prod_state.push(s2.clone());
                        (prod_state, *e1 + *e2)
                    })
            })
            .collect()
    }

    /// Create a new `ProdBasis` from the Kronecker product of several
    /// [`Bases`][Basis].
    pub fn from_kron<'a, I>(bases: I) -> Self
    where
        I: IntoIterator<Item = &'a Basis<S>>,
        S: BasisState + 'a
    {
        bases.into_iter()
            .map(|basis| basis.iter())
            .multi_cartesian_product()
            .map(|s_e: Vec<(&S, &f64)>| {
                let prod_state: Vec<S>
                    = s_e.iter().map(|se| se.0.clone()).collect();
                let energy: f64 = s_e.iter().map(|se| *se.1).sum();
                (prod_state, energy)
            })
            .collect()
    }

    /// Calculate the time dependence of the states' phases (integral of state
    /// frequencies) over an array of time coordinates.
    pub fn gen_time_dep(&self, time: &nd::Array1<f64>)
        -> IndexMap<Vec<S>, nd::Array1<f64>>
    {
        self.energies.iter()
            .map(|(s, e)| (s.clone(), *e * time))
            .collect()
    }
}

/// Generate a function that calculates non-linear Breit-Rabi energy shifts due
/// to a static magnetic field.
///
/// It is assumed that the function is characterized by five real parameters
/// `A0, ..., A4`, giving the shift as
/// ```text
/// A0 + A1 * B + A2 * sqrt(1 + A3 * B + A4 * B^2)
/// ```
/// The returned function has the signature
/// ```ignore
/// fn $name(B: f64) -> f64
/// ```
#[macro_export]
macro_rules! zm_br_fn {
    (
        $name:ident : { $A0:expr, $A1:expr, $A2:expr, $A3:expr, $A4:expr $(,)? }
    ) => {
        fn $name(B: f64) -> f64 {
            $A0 + $A1 * B + $A2 * (1.0 + $A3 * B + $A4 * B.powi(2)).sqrt()
        }
    }
}

