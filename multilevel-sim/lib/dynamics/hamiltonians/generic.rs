//! A generic multilevel system.
//!
//! See also [`lindbladians::generic`][super::super::lindbladians::generic].

use itertools::Itertools;
use ndarray::{ self as nd, s };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::HBuild,
        DriveParams,
        PolarizationParams,
        transition_cg,
    },
    hilbert::{ Basis, SpinState },
};

/// Hamiltonian builder for a driven single-atom system.
#[derive(Clone, Debug)]
pub struct HBuilder<'a, S>
where S: SpinState
{
    pub(crate) basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
}

impl<'a, S> HBuilder<'a, S>
where S: SpinState
{
    /// Create a new `HBuilder`.
    pub fn new(
        basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
    ) -> Self
    {
        Self { basis, drive, polarization }
    }

    /// Get a reference to the basis.
    pub fn basis(&self) -> &Basis<S> { self.basis }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        let DriveParams::Constant { frequency, strength, phase }
            = &self.drive else { return None; };
        let mut H: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis.values().map(|e| C64::from(*e))
                    .collect::<nd::Array1<C64>>()
            );
        let mut visited: HashSet<(&S, &S)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: C64;
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, s1), (i, s2)) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            drive = 0.5 * drive_weight * C64::from_polar(*strength, *phase);

            H[[i, j]] = drive;
            H[[j, i]] = drive.conj();
            H[[i.max(j); 2]] -= frequency;
            visited.insert((s1, s2));
        }
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        match self.gen_static()?.eigh_into(UPLO::Lower) {
            Ok((E, V)) => Some((E, V)),
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
        let (E, V) = self.diagonalize()?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        let n = self.basis.len();
        let mut H: nd::Array2<C64> = nd::Array2::zeros((n, n));
        let (drive_phase, drive_strength) = self.drive.gen_at(t);
        if drive_strength <= 1e-15 { return H; }
        let state_phases = self.basis.phases_at(t);
        let mut visited: HashSet<(&S, &S)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: C64;
        let iter
            = state_phases.iter().enumerate()
            .cartesian_product(state_phases.iter().enumerate());
        for ((j, (s1, ph1)), (i, (s2, ph2))) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            drive
                = 0.5 * drive_weight * drive_strength
                * (C64::i() * (drive_phase - (*ph2 - *ph1))).exp();
            H[[i, j]] += drive;
            H[[j, i]] += drive.conj();
            visited.insert((s1, s2));
        }
        H
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let (drive_phase, drive_strength) = self.drive.gen_time_dep(time);
        if drive_strength.iter().map(|Wk| Wk.abs()).sum::<f64>() < 1e-15 {
            return H;
        }
        let state_phases = self.basis.gen_time_dep(time);
        let mut visited: HashSet<(&S, &S)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut drive: nd::Array1<C64>;
        let mut drive_conj: nd::Array1<C64>;
        let iter
            = state_phases.iter().enumerate()
            .cartesian_product(state_phases.iter().enumerate());
        for ((j, (s1, ph1)), (i, (s2, ph2))) in iter {
            if visited.contains(&(s2, s1)) || !s1.couples_to(s2) { continue; }
            maybe_drive_weight
                = self.polarization.drive_weight(s1.spin(), s2.spin())
                .and_then(|pol| {
                    transition_cg(s1.spin(), s2.spin()).map(|cg| pol * cg)
                });
            drive_weight
                = if let Some(weight) = maybe_drive_weight {
                    weight
                } else {
                    continue;
                };
            drive
                = ph1.iter().zip(ph2).zip(&drive_phase).zip(&drive_strength)
                .map(|(((ph1k, ph2k), phk), Wk)| {
                    *Wk * 0.5 * drive_weight
                        * (C64::i() * (*phk - (*ph2k - *ph1k))).exp()
                })
                .collect();
            drive_conj = drive.mapv(|a| a.conj());

            drive.move_into(H.slice_mut(s![i, j, ..]));
            drive_conj.move_into(H.slice_mut(s![j, i, ..]));
            visited.insert((s1, s2));
        }
        H
    }
}

/// Initialization data for [`HBuilder`].
#[derive(Clone, Debug)]
pub struct HParams<'a, S>
where S: SpinState
{
    pub basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
}

impl<'a, S> HBuild<'a, S> for HBuilder<'a, S>
where S: SpinState
{
    type Params = HParams<'a, S>;
    type Basis = Basis<S>;

    fn new_builder(params: Self::Params) -> Self {
        let HParams { basis, drive, polarization } = params;
        Self::new(basis, drive, polarization)
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

