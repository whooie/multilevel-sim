//! Hamiltonian builders for various systems.

use std::{
    marker::PhantomData,
    ops::{ Deref, DerefMut },
};
use itertools::Itertools;
use ndarray::{ self as nd, s };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    hilbert::{
        Basis,
        BasisState,
        Cavity,
        CavityCoupling,
        PhotonLadder,
        RydbergState,
        SpinState,
        StateIter,
    },
};

pub mod generic;
pub use generic::{ HBuilder, HParams };

pub mod rydberg;
pub use rydberg::{ HBuilderRydberg, HRydbergParams, RydbergCoupling };

pub mod magic_trap;
pub use magic_trap::{ HBuilderMagicTrap, HMagicTrapParams, MotionalParams, FockCutoff };

pub mod cavity;
pub use cavity::{ HBuilderCavity, HCavityParams };

pub mod cavity_rydberg;
pub use cavity_rydberg::{ HBuilderCavityRydberg, HCavityRydbergParams };

pub mod transverse_ising;
pub use transverse_ising::{ HBuilderTransverseIsing, HTransverseIsingParams };

/// Basic requirements for any Hamiltonian builder.
pub trait HBuild<'a, S> {
    /// Initialization data type.
    type Params;

    /// Basis type containing state energies.
    type Basis: PartialEq + StateIter<'a, State = S>;

    /// Initialize `self`.
    fn new_builder(params: Self::Params) -> Self;

    /// Build a time-independent Hamiltonian matrix, if possible.
    fn build_static(&self) -> Option<nd::Array2<C64>>;

    /// Build a the Hamiltonian matrix at a given time.
    fn build_at(&self, t: f64) -> nd::Array2<C64>;

    /// Build the Hamiltonian array.
    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64>;

    /// Return a reference to the basis.
    fn get_basis(&self) -> &Self::Basis;
}

/// Aggregator for [`HBuild`] implementors for easy initialization and overlay
/// of time-independent Hamiltonians.
///
/// Time-independent counterpart to [`SequenceBuilder`].
pub struct OverlayBuilder<'a, S, H>(Vec<H>, PhantomData<&'a S>)
where H: HBuild<'a, S>;

impl<'a, S, H> AsRef<Vec<H>> for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn as_ref(&self) -> &Vec<H> { &self.0 }
}

impl<'a, S, H> AsMut<Vec<H>> for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn as_mut(&mut self) -> &mut Vec<H> { &mut self.0 }
}

impl<'a, S, H> Deref for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    type Target = Vec<H>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a, S, H> DerefMut for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<'a, S, H> From<Vec<H>> for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn from(builders: Vec<H>) -> Self { Self(builders, PhantomData) }
}

impl<'a, S, H> FromIterator<H::Params> for OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = H::Params>
    {
        Self(iter.into_iter().map(H::new_builder).collect(), PhantomData)
    }
}

impl<'a, S, H> OverlayBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    /// Create a new `OverlayBuilder` from pre-constructed [`HBuild`]s.
    pub fn from_builders<I>(builders: I) -> Self
    where I: IntoIterator<Item = H>
    {
        Self(builders.into_iter().collect(), PhantomData)
    }

    /// Return `Some(true`) if all builders' bases are equal; `None` if no
    /// builders are present.
    pub fn equal_bases(&self) -> Option<bool> {
        self.0.first()
            .map(|b| b.get_basis())
            .map(|comp| self.0.iter().all(|b| b.get_basis() == comp))
    }

    /// Return `Some` if at least one builder is present and all builders' bases
    /// are equal.
    pub fn get_basis(&self) -> Option<&H::Basis> {
        if let Some(true) = self.equal_bases() {
            self.0.first().map(|builder| builder.get_basis())
        } else {
            None
        }
    }

    /// Return the basis of the `n`-th builder.
    pub fn get_basis_at(&self, n: usize) -> Option<&H::Basis> {
        self.0.get(n).map(|builder| builder.get_basis())
    }

    /// Build the total Hamiltonian as the overlay of all builders' generated
    /// Hamiltonians if all builders are able to generate time-independent
    /// Hamiltonians.
    pub fn build(&self) -> Option<nd::Array2<C64>> {
        fn superpose(A: &mut nd::Array2<C64>, B: &nd::Array2<C64>) {
            // arrays are guaranteed to have the same shape and state energies
            // due to an `equal_bases` check
            *A += B;
            A.diag_mut().iter_mut().for_each(|e| { *e /= 2.0; });
        }

        if matches!(self.equal_bases(), None | Some(false)) { return None; }
        let mut gen
            = self.0.iter().map(|builder| builder.build_static());
        let mut H //     v guaranteed by `equal_bases` check
            = gen.next().unwrap()?;
        for maybe_h in gen {
            if let Some(h) = maybe_h {
                superpose(&mut H, &h);
            } else {
                return None;
            }
        }
        Some(H)
    }

    /// Diagonalize the total overlay of all builders' generated Hamiltonians,
    /// if possible.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        match self.build()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the total overlay of all builders' generated Hamiltonians
    /// and return a ground state of the system, if possible.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// ground state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        let (E, V) = self.diagonalize()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }
}

trait ModifyableCavityCoupling<const N: usize, const P: usize, S>
where S: BasisState
{
    fn f_coupling(&self, s1: &S, s2: &S) -> Option<PhotonLadder>;

    fn get_basis(&self) -> &Basis<Cavity<N, P, S>>;
}

impl<'a, const N: usize, const P: usize, S> ModifyableCavityCoupling<N, P, S>
    for HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    fn f_coupling(&self, s1: &S, s2: &S) -> Option<PhotonLadder> {
        self.do_f_coupling(s1, s2)
    }

    fn get_basis(&self) -> &Basis<Cavity<N, P, S>> {
        self.basis()
    }
}

impl<'a, const N: usize, const P: usize, S> ModifyableCavityCoupling<N, P, S>
    for HBuilderCavityRydberg<'a, N, P, S>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    fn f_coupling(&self, s1: &S, s2: &S) -> Option<PhotonLadder> {
        self.do_f_coupling(s1, s2)
    }

    fn get_basis(&self) -> &Basis<Cavity<N, P, S>> {
        self.basis()
    }
}

fn superpose_cavity<const N: usize, const P: usize, S, H>(
    A: &mut nd::Array2<C64>,
    builder_A: &H,
    B: &nd::Array2<C64>,
    builder_B: &H,
)
where
    S: SpinState + CavityCoupling<P>,
    H: ModifyableCavityCoupling<N, P, S>,
{
    use PhotonLadder::*;
    // arrays are guaranteed to have the same shape and state energies
    // due to an `equal_bases` check
    *A += B;
    A.diag_mut().iter_mut().for_each(|e| { *e /= 2.0; });

    let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
        = HashSet::default();
    let mut ss1: &[S; N];
    let mut ss2: &[S; N];
    let mut nn1: &[usize; P];
    let mut nn2: &[usize; P];
    let iter
        = A.iter_mut()
        .zip(
            builder_A.get_basis().keys()
            .cartesian_product(builder_A.get_basis().keys())
        );
    for (a, (sn1, sn2)) in iter {
        // if visited.contains(&(sn1, sn2)) { continue; }
        ss1 = sn1.atomic_states();
        ss2 = sn2.atomic_states();
        nn1 = sn1.photons();
        nn2 = sn2.photons();

        if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
            let (s1, s2)
                = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
            match builder_A.f_coupling(s1, s2)
                .zip(builder_B.f_coupling(s1, s2))
            {
                Some((Emit(m_A, g_A), Emit(m_B, g_B)))
                    if m_A == m_B && g_A == g_B
                => {
                    if let Some(gn)
                        = nn1.get(m_B)
                        .zip(nn2.get(m_B))
                        .and_then(|(&n1, &n2)| {
                            (n1 + 1 == n2)
                                .then_some(g_B * (n2 as f64).sqrt())
                        })
                    {
                        *a -= gn;
                    }
                },
                Some((Absorb(m_A, g_A), Absorb(m_B, g_B)))
                    if m_A == m_B && g_A == g_B
                => {
                    if let Some(gn)
                        = nn1.get(m_B)
                        .zip(nn2.get(m_B))
                        .and_then(|(&n1, &n2)| {
                            (n1 == n2 + 1)
                                .then_some(g_B * (n1 as f64).sqrt())
                        })
                    {
                        *a -= gn;
                    }
                },
                _ => { },
            }
        }
        visited.insert((sn2, sn1));
    }
}

impl<'a, const N: usize, const P: usize, S>
    OverlayBuilder<'a, Cavity<N, P, S>, HBuilderCavity<'a, N, P, S>>
where S: SpinState + CavityCoupling<P>
{
    /// Like [`Self::build`], but preserving off-diagonal elements due to cavity
    /// mode couplings in addition to state energies.
    pub fn build_cavity(&self) -> Option<nd::Array2<C64>> {
        if !self.equal_bases()? { return None; }
        let mut gen
            = self.0.iter().map(|builder| {
                builder.build_static().map(|h| (h, builder))
            });
        let (mut H, mut builder) = gen.next()??;
        for maybe_h in gen {
            if let Some((h, builder_h)) = maybe_h {
                superpose_cavity(&mut H, builder, &h, builder_h);
                builder = builder_h;
            } else {
                return None;
            }
        }
        Some(H)
    }

    /// Like [`Self::diagonalize`], but preserving off-diagonal elements due to
    /// cavity mode couplings in addition to state energies in the generated
    /// Hamiltonian.
    pub fn diagonalize_cavity(&self)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    {
        match self.build_cavity()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Like [`Self::ground_state`], but preserving off-diagonal elements due to
    /// cavity mode couplings in addition to state energies in the generated
    /// Hamiltonian.
    pub fn ground_state_cavity(&self) -> Option<(f64, nd::Array1<C64>)> {
        let (E, V) = self.diagonalize_cavity()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }
}

impl<'a, const N: usize, const P: usize, S>
    OverlayBuilder<'a, Cavity<N, P, S>, HBuilderCavityRydberg<'a, N, P, S>>
where S: SpinState + RydbergState + CavityCoupling<P>
{
    /// Like [`Self::build`], but preserving off-diagonal elements due to cavity
    /// mode couplings in addition to state energies.
    pub fn build_cavity_rydberg(&self) -> Option<nd::Array2<C64>> {
        if !self.equal_bases()? { return None; }
        let mut gen
            = self.0.iter().map(|builder| {
                builder.build_static().map(|h| (h, builder))
            });
        let (mut H, mut builder) = gen.next()??;
        for maybe_h in gen {
            if let Some((h, builder_h)) = maybe_h {
                superpose_cavity(&mut H, builder, &h, builder_h);
                builder = builder_h;
            } else {
                return None;
            }
        }
        Some(H)
    }

    /// Like [`Self::diagonalize`], but preserving off-diagonal elements due to
    /// cavity mode couplings in addition to state energies in the generated
    /// Hamiltonian.
    pub fn diagonalize_cavity_rydberg(&self)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    {
        match self.build_cavity_rydberg()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Like [`Self::ground_state`], but preserving off-diagonal elements due to
    /// cavity mode couplings in addition to state energies in the generated
    /// Hamiltonian.
    pub fn ground_state_cavity_rydberg(&self) -> Option<(f64, nd::Array1<C64>)>
    {
        let (E, V) = self.diagonalize_cavity_rydberg()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }
}

/// Aggregator for [`HBuild`] implementors for easy initialization and
/// composition of time-dependent Hamiltonians.
pub struct SequenceBuilder<'a, S, H>(Vec<H>, PhantomData<&'a S>)
where H: HBuild<'a, S>;

impl<'a, S, H> AsRef<Vec<H>> for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn as_ref(&self) -> &Vec<H> { &self.0 }
}

impl<'a, S, H> AsMut<Vec<H>> for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn as_mut(&mut self) -> &mut Vec<H> { &mut self.0 }
}

impl<'a, S, H> Deref for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    type Target = Vec<H>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a, S, H> DerefMut for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<'a, S, H> From<Vec<H>> for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn from(builders: Vec<H>) -> Self { Self(builders, PhantomData) }
}

impl<'a, S, H> FromIterator<H::Params> for SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = H::Params>
    {
        Self(iter.into_iter().map(H::new_builder).collect(), PhantomData)
    }
}

impl<'a, S, H> SequenceBuilder<'a, S, H>
where H: HBuild<'a, S>
{
    /// Create a new `SequenceBuilder` from pre-constructed [`HBuild`]s.
    pub fn from_builders<I>(builders: I) -> Self
    where I: IntoIterator<Item = H>
    {
        Self(builders.into_iter().collect(), PhantomData)
    }

    /// Return `Some(true`) if all builders' bases are equal; `None` if no
    /// builders are present.
    pub fn equal_bases(&self) -> Option<bool> {
        self.0.first()
            .map(|b| b.get_basis())
            .map(|comp| self.0.iter().all(|b| b.get_basis() == comp))
    }

    /// Return `Some` if at least one builder is present and all builders' bases
    /// are equal.
    pub fn get_basis(&self) -> Option<&H::Basis> {
        if let Some(true) = self.equal_bases() {
            self.0.first().map(|builder| builder.get_basis())
        } else {
            None
        }
    }

    /// Return the basis of the `n`-th builder.
    pub fn get_basis_at(&self, n: usize) -> Option<&H::Basis> {
        self.0.get(n).map(|builder| builder.get_basis())
    }

    fn build_at(&self, t: f64) -> Option<nd::Array2<C64>> {
        if self.equal_bases().is_some_and(|b| !b) { return None; }
        self.0.iter()
            .fold(
                None,
                |acc: Option<nd::Array2<C64>>, builder: &H| {
                    if let Some(h) = acc {
                        Some(h + builder.build_at(t))
                    } else {
                        Some(builder.build_at(t))
                    }
                }
            )
    }

    /// Build the total Hamiltonian as the sum of all builders' generated
    /// Hamiltonians.
    pub fn build(&self, time: &nd::Array1<f64>) -> Option<nd::Array3<C64>> {
        if self.equal_bases().is_some_and(|b| !b) { return None; }
        self.0.iter()
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

