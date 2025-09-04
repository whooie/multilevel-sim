//! A pair of atoms coupled to motional Fock bases and to each other with a
//! motion-dependent Rydberg interaction.

#![allow(unused_imports, dead_code)]

use std::f64::consts::TAU;
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rand::{ prelude as rnd, Rng };
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::{
            HBuild,
            magic_trap::{ HBuilderMagicTrap, FockCutoff, MotionalParams },
        },
        DriveParams,
        PolarizationParams,
        multiatom_kron,
        transition_cg,
    },
    hilbert::{
        Basis,
        BasisState,
        Fock,
        RydbergShift,
        SpinState,
        TrappedMagic,
        outer_prod,
    },
    rabi::StateNorm,
};

pub type FockPair<S> = (Fock<S>, Fock<S>);

/// Hamiltonian builder for a driven two-(identical) atom system including effects from
/// motion in a magic, harmonic trap as well as motion-dependent Rydberg
/// interactions.
///
/// Useful points of comparison are [`HBuilderMagicTrap`] and
/// [`HBuilderRydberg`][crate::dynamics::hamiltonians::HBuilderRydberg] -- this
/// builder combines features of both by relying on a `Fock<S>: RydbergShift`
/// bound (as opposed to merely `S: RydbergShift`). This allows the Rydberg
/// interaction to have full resolution and articulation in the motional Fock
/// basis via [`RydbergShift::shift_with`].
#[derive(Clone, Debug)]
pub struct HBuilderMotionalRydberg<'a, S>
where
    S: SpinState + TrappedMagic,
    Fock<S>: RydbergShift,
{
    pub(crate) single: HBuilderMagicTrap<'a, S>,
    pub(crate) basis: Basis<FockPair<S>>,
}

impl<'a, S> HBuilderMotionalRydberg<'a, S>
where
    S: SpinState + TrappedMagic,
    Fock<S>: RydbergShift,
{
    pub(crate) const HBAR: f64 = HBuilderMagicTrap::<'a, S>::HBAR;
    pub(crate) const KB: f64 = HBuilderMagicTrap::<'a, S>::KB;
    pub(crate) const TRAP_FREQ: f64 = HBuilderMagicTrap::<'a, S>::TRAP_FREQ;

    /// Create a new `HBuilderMotionalRydberg` from a set of magic-trapping
    /// parameters.
    ///
    /// Internally, this creates an [`HBuilderMagicTrap`] and modifies state
    /// energies according to the `Fock<S>: RydbergShift` impl. See
    /// [`HBuilderMagicTrap::new`] for information on units and motional basis
    /// parameterization.
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        params: MotionalParams,
    ) -> Self
    {
        Self::from_site(
            HBuilderMagicTrap::new(atom_basis, drive, polarization, params))
    }

    /// Create a new `HBuilderMotionalRydberg`, given an existing builder for a
    /// single atom.
    ///
    /// The driven dynamics are assumed to be identical for the pair of atoms
    /// this builder models.
    pub fn from_site(single_site: HBuilderMagicTrap<'a, S>) -> Self {
        let basis: Basis<FockPair<S>> =
            single_site.basis().iter()
            .flat_map(|(sn0, e0)| {
                single_site.basis().iter()
                .map(|(sn1, e1)| {
                    let sn01 = (sn0.clone(), sn1.clone());
                    let e01 = *e0 + *e1 + sn0.shift_with(sn1).unwrap_or(0.0);
                    (sn01, e01)
                })
            })
            .collect();
        Self { single: single_site, basis }
    }

    /// Return a reference to the single-site atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.single.atom_basis() }

    /// Return a reference to the single-site atom-motional Fock basis.
    pub fn atom_motion_basis(&self) -> &Basis<Fock<S>> { self.single.basis() }

    /// Return a reference to the full two-atom [`FockPair`] basis.
    pub fn basis(&self) -> &Basis<FockPair<S>> { &self.basis }

    /// Return the atomic mass in original units.
    pub fn mass(&self) -> f64 { self.single.mass() }

    /// Return the laser wavelength in original units.
    pub fn wavelength(&self) -> f64 { self.single.wavelength() }

    /// Return the maximum Fock state index.
    pub fn nmax(&self) -> usize { self.single.nmax() }

    /// Return the Lamb-Dicke parameters.
    pub fn lamb_dicke(&self) -> f64 { self.single.lamb_dicke() }

    /// Return the drive parameters.
    pub fn drive(&self) -> &DriveParams<'_> { &self.single.drive }

    /// Return the polarization parameters.
    pub fn polarization(&self) -> &PolarizationParams {
        &self.single.polarization
    }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        let single_H: nd::Array2<C64> = self.single.gen_static()?;
        let mut H: nd::Array2<C64> =
            multiatom_kron([single_H.view(), single_H.view()]);
        self.basis.values()
            .zip(H.diag_mut())
            .for_each(|(e, h)| { *h = C64::from(*e); });
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        match self.gen_static()?.eigh_into(UPLO::Lower) {
            Ok((e, v)) => Some((e, v)),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// ground state is returned.
    pub fn ground_state(&self) -> Option<(f64, nd::Array1<C64>)> {
        let (e, v) = self.diagonalize()?;
        let e: f64 = e[0];
        let v: nd::Array1<C64> = v.slice(nd::s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        let single_H: nd::Array2<C64> = self.single.gen_at(t);
        let mut H: nd::Array2<C64> =
            multiatom_kron([single_H.view(), single_H.view()]);
        let mut shift0: f64;
        let mut shift1: f64;
        let mut shift_phase: C64;
        // above the diagonal
        let iter =
            self.basis.keys().enumerate()
            .flat_map(|(j, sn0)| {
                self.basis.keys().enumerate().take(j)
                .map(move |(i, sn1)| ((j, sn0), (i, sn1)))
            });
        for ((j, sn0), (i, sn1)) in iter {
            if H[[i, j]].norm() < 1e-15 { continue; }
            shift0 = sn0.0.shift_with(&sn0.1).unwrap_or(0.0);
            shift1 = sn1.0.shift_with(&sn1.1).unwrap_or(0.0);
            if (shift1 - shift0).abs() < 1e-15 { continue; }
            shift_phase = C64::cis((shift1 - shift0) * t);
            H[[i, j]] *= shift_phase;
            H[[j, i]] *= shift_phase.conj();
        }
        H
    }

    /// compute the full time-dependent Hamiltonian as a 3D array, with the last
    /// axis corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let single_H: nd::Array3<C64> = self.single.gen(time);
        let time_slices: Vec<nd::Array2<C64>> =
            single_H.axis_iter(nd::Axis(2))
            .map(|single_t| multiatom_kron([single_t, single_t]))
            .collect();
        let mut H: nd::Array3<C64> =
            nd::stack(
                nd::Axis(2),
                &time_slices.iter().map(|Ht| Ht.view()).collect::<Vec<_>>(),
            )
            .unwrap();
        let mut shift0: f64;
        let mut shift1: f64;
        let mut dshift: f64;
        let iter =
            self.basis.keys().enumerate()
            .flat_map(|(j, sn0)| {
                self.basis.keys().enumerate().take(j)
                .map(move |(i, sn1)| ((j, sn0), (i, sn1)))
            });
        for ((j, sn0), (i, sn1)) in iter {
            shift0 = sn0.0.shift_with(&sn0.1).unwrap_or(0.0);
            shift1 = sn1.0.shift_with(&sn1.1).unwrap_or(0.0);
            dshift = shift1 - shift0;
            if dshift.abs() < 1e-15 { continue; }
            H.slice_mut(nd::s![i, j, ..]).iter_mut()
                .zip(time.iter())
                .for_each(|(ht, t)| { *ht *= C64::cis(dshift * *t); });
            H.slice_mut(nd::s![j, i, ..]).iter_mut()
                .zip(time.iter())
                .for_each(|(ht, t)| { *ht *= C64::cis(-dshift * *t); });
        }
        H
    }
}

impl<'a, S> From<HBuilderMagicTrap<'a, S>> for HBuilderMotionalRydberg<'a, S>
where
    S: SpinState + TrappedMagic,
    Fock<S>: RydbergShift,
{
    fn from(magic_trap: HBuilderMagicTrap<'a, S>) -> Self {
        Self::from_site(magic_trap)
    }
}

/// Initialization data for [`HBuilderMotionalRydberg`].
pub enum HMotionalRydbergParams<'a, S>
where
    S: SpinState + TrappedMagic,
    Fock<S>: RydbergShift,
{
    New {
        basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        motion: MotionalParams,
    },
    FromSite {
        site: HBuilderMagicTrap<'a, S>,
    },
}

impl<'a, S> HBuild<'a, FockPair<S>> for HBuilderMotionalRydberg<'a, S>
where
    S: SpinState + TrappedMagic,
    Fock<S>: RydbergShift,
{
    type Params = HMotionalRydbergParams<'a, S>;
    type Basis = Basis<FockPair<S>>;

    fn new_builder(params: Self::Params) -> Self {
        use HMotionalRydbergParams::*;
        match params {
            New { basis, drive, polarization, motion } =>
                Self::new(basis, drive, polarization, motion),
            FromSite { site } =>
                Self::from_site(site),
        }
    }

    fn build_static(&self) -> Option<nd::Array2<C64>> {
        self.gen_static()
    }

    fn build_at(&self, t: f64) -> nd::Array2<C64> {
        self.gen_at(t)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis() }
}

