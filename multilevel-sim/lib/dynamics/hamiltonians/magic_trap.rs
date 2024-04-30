//! A single atom coupled to a motional Fock basis, with assumed uniform
//! trapping conditions for all states.

use std::f64::consts::TAU;
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use ndarray_linalg::{ EighInto, InverseInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rand::{ prelude as rnd, Rng };
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::HBuild,
        DriveParams,
        PolarizationParams,
        transition_cg,
    },
    hilbert::{
        Basis,
        BasisState,
        Fock,
        SpinState,
        TrappedMagic,
        outer_prod,
    },
    rabi::StateNorm,
};

/// Determines the maximum Fock state index included in the simulation.
#[derive(Copy, Clone, Debug)]
pub enum FockCutoff {
    /// Set the maximum Fock state as that for which the Boltzmann distribution
    /// crosses this threshold.
    Boltzmann(f64),
    /// Set the maximum Fock state index explicitly.
    NMax(usize),
}

/// Parameters for [`HBuilderMagicTrap`].
#[derive(Copy, Clone, Debug)]
pub struct MotionalParams {
    /// Atomic mass (see [`HBuilderMagicTrap`] for info on units)
    pub mass: f64,
    /// Wavelength of the drive in meters, assumed constant over the range of
    /// possible drive frequencies.
    pub wavelength: f64,
    /// Initial atom temperature (see [`HBuilderMagicTrap`] for info on
    /// units).
    pub temperature: f64,
    /// Boltzmann distribution cutoff to determine how many motional states are
    /// included in the model.
    pub fock_cutoff: Option<FockCutoff>,
}

/// Hamiltonian builder for a driven single-atom system including effects from
/// motion in a magic, harmonic trap.
#[derive(Clone, Debug)]
pub struct HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    pub(crate) atom_basis: &'a Basis<S>,
    pub(crate) basis: Basis<Fock<S>>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub(crate) mass: f64,
    pub(crate) wavelength: f64,
    pub(crate) temperature: f64,
    pub(crate) nmax: usize,
    pub(crate) x: f64,
    pub(crate) Z: f64,
    pub(crate) lamb_dicke: f64,
}

impl<'a, S> HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    const HBAR: f64 = 1.0545718001391127e-34; // J / Hz / 2π
    const KB: f64 = 1.38064852e-23; // J / K

    /// Trap frequency, in units of angular frequency, for all states.
    const TRAP_FREQ: f64 = S::TRAP_FREQ;

    /// Create a new `HBuilderMagicTrap`.
    ///
    /// The maximum Fock state included in the model, if not [set
    /// explicitly][FockCutoff::NMax], will be that for which the associated
    /// probability in a thermal distribution at `temperature` is at or below
    /// `boltzmann_threshold` (default 10^-6). The driving wavelength is assumed
    /// to be approximately constant over the range of possible driving
    /// frequencies.
    ///
    /// # Units
    /// To keep the ratio ħω/kT invariant, temperature should be provided in
    /// units that are commensurate with those used for frequency/energy: For
    /// units where `10^n ħ = 1`, the temperature should be provided in units
    /// where `10^n = 1` (i.e. units of `10^n K`). To likewise keep the
    /// Lamb-Dicke parameter invariant, the atomic mass should be provided
    /// in units where `10^-n = 1` (i.e. units of `10^-n kg`). The wavelength
    /// should be in meters.
    ///
    /// For example, when working with frequency/energy described in MHz, `n =
    /// 6`, so temperature should be provided in units of 1 MK (`1 MK = 10^6
    /// K`) and the atomic mass should be provided in units of 1 mg (`1 mg =
    /// 10^-6 kg`).
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        params: MotionalParams,
    ) -> Self
    {
        let MotionalParams {
            mass,
            wavelength,
            temperature,
            fock_cutoff,
        } = params;
        let x: f64
            = Self::HBAR * Self::TRAP_FREQ / (Self::KB * temperature);
        let Z: f64 = (2.0 * (x / 2.0).sinh()).recip();
        let nmax: f64
            = match fock_cutoff.unwrap_or(FockCutoff::Boltzmann(1e-6)) {
                FockCutoff::Boltzmann(p) => {
                    (-(Z * p).ln() / x - 0.5).ceil()
                },
                FockCutoff::NMax(nmax) => {
                    nmax as f64
                },
            };
        if nmax > usize::MAX as f64 {
            panic!(
                "HBuilderMagicTrap::new: maximum Fock index overflows usize"
            );
        }
        if atom_basis.len() as f64 * nmax > usize::MAX as f64 {
            panic!(
                "HBuilderMagicTrap::new: maximum atom*Fock index overflows \
                usize"
            );
        }
        let nmax = nmax as usize;
        let basis: Basis<Fock<S>>
            = atom_basis.iter()
            .flat_map(|(s, e)| {
                (0..=nmax)
                    .map(|n| {
                        (
                            Fock::from((s.clone(), n)),
                            *e + Self::TRAP_FREQ * (n as f64 + 0.5),
                        )
                    })
            })
            .collect();
        let lamb_dicke: f64
            = TAU / wavelength
            * (Self::HBAR / 2.0 / mass / Self::TRAP_FREQ).sqrt();
        Self {
            atom_basis,
            basis,
            drive,
            polarization,
            mass,
            wavelength,
            temperature,
            nmax,
            x,
            Z,
            lamb_dicke,
        }
    }

    /// Return a reference to the atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.atom_basis }

    /// Return a reference to the full Fock basis.
    pub fn basis(&self) -> &Basis<Fock<S>> { &self.basis }

    /// Return the atomic mass in original units.
    pub fn mass(&self) -> f64 { self.mass }

    /// Return the laser wavelength in original units.
    pub fn wavelength(&self) -> f64 { self.wavelength }

    /// Return the atomic temperature in original units.
    pub fn temperature(&self) -> f64 { self.temperature }

    /// Return the maximum Fock state index.
    pub fn nmax(&self) -> usize { self.nmax }

    /// Return the Lamb-Dicke parameter.
    pub fn lamb_dicke(&self) -> f64 { self.lamb_dicke }

    /// Generate a thermal state vector following a Boltzmann distribution over
    /// the motional states of a single atomic state.
    ///
    /// The resulting state is given random phases on all motional states,
    /// sampled uniformly over [0, 2π).
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the Boltzmann distribution
    /// cutoff is not sufficiently low, this can cause the average Fock index to
    /// disagree with theory.
    pub fn thermal_state_vector(&self, atomic_state: &S)
        -> Option<nd::Array1<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_state() == atomic_state) {
            return None;
        }
        let mut rng = rnd::thread_rng();
        let mut psi: nd::Array1<C64>
            = self.basis.keys()
            .map(|s| {
                if s.atomic_state() == atomic_state {
                    C64::from_polar(
                        (
                            (-self.x * (s.fock_index() as f64 + 0.5)).exp()
                            / self.Z
                        ).sqrt(),
                        TAU * rng.gen::<f64>(),
                    )
                } else {
                    C64::zero()
                }
            })
            .collect();
        let N: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= N;
        Some(psi)
    }

    /// Generate a thermal state density matrix following a Boltzmann
    /// distribution over the motional states of a single atomic state.
    ///
    /// The resulting state has no off-diagonal elements.
    ///
    /// **Note**: the returned matrix is renormalized such that its trace is
    /// equal to 1; in cases where the Boltzmann distribution cutoff is not
    /// sufficiently low, this can cause the average Fock index to disagree with
    /// theory.
    pub fn thermal_state_density(&self, atomic_state: &S)
        -> Option<nd::Array2<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_state() == atomic_state) {
            return None;
        }
        let mut rho: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis.keys()
                    .map(|s| {
                        if s.atomic_state() == atomic_state {
                            C64::from(
                                (-self.x * (s.fock_index() as f64 + 0.5)).exp()
                                / self.Z
                            )
                        } else {
                            C64::zero()
                        }
                    })
                    .collect::<nd::Array1<C64>>()
            );
        let N: C64 = rho.diag().iter().sum::<C64>();
        rho /= N;
        Some(rho)
    }

    /// Generate a thermal state density matrix with coherence only in the
    /// atomic subspace. The motional subspace is taken to follow a (decohered)
    /// Boltzmann distribution.
    ///
    /// **Note**: the returned matrix is renormalized such that its trace is
    /// equal to 1; in cases where the Boltzmann distribution cutoff is not
    /// sufficiently low, this can cause the average Fock index to disagree with
    /// theory.
    ///
    /// *Panics* if the length of `atomic_state` disagrees with the size of the
    /// atomic basis.
    pub fn thermal_density_atomic<F>(&self, amps: F) -> nd::Array2<C64>
    where F: Fn(&S, usize, f64) -> C64
    {
        let mut atom_state: nd::Array1<C64>
            = self.atom_basis.get_vector_weighted(amps);
        let N: C64 = atom_state.norm();
        atom_state /= N;
        let atom_density: nd::Array2<C64>
            = outer_prod(&atom_state, &atom_state);

        let mut motional_density: nd::Array2<C64>
            = nd::Array2::from_diag(
                &(0..=self.nmax)
                    .map(|n| {
                        C64::from(
                            (-self.x * (n as f64 + 0.5)).exp() / self.Z
                        )
                    })
                    .collect::<nd::Array1<C64>>()
            );
        let N: C64 = motional_density.diag().iter().sum::<C64>();
        motional_density /= N;

        kron(&atom_density, &motional_density)
    }

    fn make_x(&self) -> nd::Array2<C64> {
        let n_x = self.nmax + 1;
        let mut x_op: nd::Array2<C64> = nd::Array2::zeros((n_x, n_x));
        let coeff: f64 = self.wavelength * self.lamb_dicke / TAU;
        x_op // a
            .slice_mut(s![..n_x - 1, 1..n_x])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(coeff * (n as f64 + 1.0).sqrt());
            });
        x_op // a^\dagger
            .slice_mut(s![1..n_x, ..n_x - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(coeff * (n as f64 + 1.0).sqrt());
            });
        x_op
    }

    /// Compute the *x* (position) operator over the entire atom-motion basis.
    pub fn gen_x(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_x())
    }

    fn make_p(&self) -> nd::Array2<C64> {
        let n_p = self.nmax + 1;
        let mut p_op: nd::Array2<C64> = nd::Array2::zeros((n_p, n_p));
        let coeff: C64
            = C64::i() * self.wavelength / TAU
            * self.mass * Self::TRAP_FREQ
            * self.lamb_dicke;
        p_op // a
            .slice_mut(s![..n_p - 1, 1..n_p])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = -coeff * (n as f64 + 1.0).sqrt();
            });
        p_op // a^\dagger
            .slice_mut(s![1..n_p, ..n_p - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem =  coeff * (n as f64 + 1.0).sqrt();
            });
        p_op
    }

    /// Compute the *p* (momentum) operator over the entire atom-motion basis.
    pub fn gen_p(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_p())
    }

    fn make_exp_ikx(&self) -> nd::Array2<C64> {
        let n_kx = self.nmax + 1;
        let mut kx_op: nd::Array2<C64> = nd::Array2::zeros((n_kx, n_kx));
        kx_op // a
            .slice_mut(s![..n_kx - 1, 1..n_kx])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(self.lamb_dicke * (n as f64 + 1.0).sqrt());
            });
        kx_op // a^\dagger
            .slice_mut(s![1..n_kx, ..n_kx - 1])
            .diag_mut()
            .indexed_iter_mut()
            .for_each(|(n, elem)| {
                *elem = C64::from(self.lamb_dicke * (n as f64 + 1.0).sqrt());
            });
        let (evals, evects): (nd::Array1<f64>, nd::Array2<C64>)
            = kx_op.eigh_into(UPLO::Lower)
            .expect("HBuilderMagicTrap::make_exp_ikx: error diagonalizing");
        let L = nd::Array2::from_diag(
            &evals.mapv(|lk| C64::from_polar(1.0, lk)));
        let V = evects.clone();
        let U = evects.inv_into()
            .expect("HBuilderMagicTrap::make_exp_ikx: error inverting");
        V.dot(&L).dot(&U)
    }

    /// Compute the <i>e<sup>ikx</sup></i> (laser phase) operator over the
    /// entire atom-motion basis.
    pub fn gen_eikx(&self) -> nd::Array2<C64> {
        kron(&nd::Array2::eye(self.atom_basis.len()), &self.make_exp_ikx())
    }

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
                &self.basis().values().map(|e| C64::from(*e))
                    .collect::<nd::Array1<C64>>()
            );
        let eikx = self.make_exp_ikx();
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::default();
        let mut maybe_drive_weight: Option<C64>;
        let mut drive_weight: C64;
        let mut eikx_weight: C64;
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
            eikx_weight = eikx[[s2.fock_index(), s1.fock_index()]].conj();
            drive
                = 0.5 * drive_weight * eikx_weight
                * C64::from_polar(*strength, *phase);

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
        if drive_strength.abs() < 1e-15 { return H; }
        let state_phases = self.basis.phases_at(t);
        let eikx = self.make_exp_ikx();
        let mut eikx_weight: C64;
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::default();
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
            eikx_weight = eikx[[s2.fock_index(), s1.fock_index()]].conj();
            drive
                = 0.5 * drive_weight * eikx_weight * drive_strength
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
        let eikx = self.make_exp_ikx();
        let mut eikx_weight: C64;
        let mut visited: HashSet<(&Fock<S>, &Fock<S>)> = HashSet::default();
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
            eikx_weight = eikx[[s2.fock_index(), s1.fock_index()]].conj();
            drive
                = ph1.iter().zip(ph2).zip(&drive_phase).zip(&drive_strength)
                .map(|(((ph1k, ph2k), phk), Wk)| {
                    *Wk * 0.5 * drive_weight * eikx_weight
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

/// Initialization data for [`HBuilderMagicTrap`].
#[derive(Clone, Debug)]
pub struct HMagicTrapParams<'a, S>
where S: SpinState + TrappedMagic
{
    pub basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub motion: MotionalParams,
}

impl<'a, S> HBuild<'a, Fock<S>> for HBuilderMagicTrap<'a, S>
where S: SpinState + TrappedMagic
{
    type Params = HMagicTrapParams<'a, S>;
    type Basis = Basis<Fock<S>>;

    fn new_builder(params: Self::Params) -> Self {
        let HMagicTrapParams { basis, drive, polarization, motion }
            = params;
        Self::new(basis, drive, polarization, motion)
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

