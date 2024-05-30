//! A collection of atoms uniformly coupled to a number of optical cavity modes.
//!
//! See also [`lindbladians::cavity`][super::super::lindbladians::cavity].

use std::{ fmt, rc::Rc };
use itertools::Itertools;
use ndarray::{ self as nd, s, linalg::kron };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::{ HBuild, HBuilder },
        DriveParams,
        PolarizationParams,
        multiatom_kron,
    },
    hilbert::{ Basis, Cavity, CavityCoupling, PhotonLadder, SpinState },
};

/// Hamiltonian builder for a collectively driven `N`-site linear array of atoms
/// coupled to `P` cavity modes.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct HBuilderCavity<'a, const N: usize, const P: usize, S>
where S: SpinState + CavityCoupling<P>
{
    pub(crate) atom_basis: &'a Basis<S>,
    pub(crate) basis: Basis<Cavity<N, P, S>>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub(crate) nmax: [usize; P],
    pub(crate) f_coupling: Option<Rc<dyn Fn(&S, &S) -> Option<PhotonLadder> + 'a>>,
}

impl<'a, const N: usize, const P: usize, S> fmt::Debug
    for HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBuilderCavity {{ \
            atom_basis: {:?}, \
            basis: {:?}, \
            drive: {:?}, \
            polarization: {:?}, \
            nmax: {:?}, \
            f_coupling: ",
            self.atom_basis,
            self.basis,
            self.drive,
            self.polarization,
            self.nmax,
        )?;
        if self.f_coupling.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, const N: usize, const P: usize, S> HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    fn def_coupling(s1: &S, s2: &S) -> Option<PhotonLadder> {
        s1.coupling(s2)
    }

    pub(crate) fn do_f_coupling(&self, s1: &S, s2: &S) -> Option<PhotonLadder> {
        if let Some(f) = &self.f_coupling {
            f(s1, s2)
        } else {
            Self::def_coupling(s1, s2)
        }
    }

    /// Create a new `HBuilderCavity`.
    pub fn new(
        atom_basis: &'a Basis<S>,
        drive: DriveParams<'a>,
        polarization: PolarizationParams,
        nmax: [usize; P],
    ) -> Self
    {
        let atom_iter
            = (0..N).map(|_| atom_basis.iter()).multi_cartesian_product();
        let cavity_iter
            = nmax.iter().map(|p| 0..=*p).multi_cartesian_product();
        let basis: Basis<Cavity<N, P, S>>
            = atom_iter.cartesian_product(cavity_iter)
            .map(|(ss, nn)| {
                let atoms: [S; N]
                    = ss.iter()
                    .map(|(s, _)| (*s).clone())
                    .collect::<Vec<S>>()
                    .try_into()
                    .unwrap();
                let atom_energy: f64
                    = ss.iter()
                    .map(|(_, e)| *e)
                    .sum();
                let photons: [usize; P] = nn.try_into().unwrap();
                let photon_energy: f64
                    = photons.iter()
                    .zip(&S::MODE_SPACING)
                    .map(|(n, e)| (*n as f64) * *e)
                    .sum();
                ((atoms, photons).into(), atom_energy + photon_energy)
            })
            .collect();
        Self {
            atom_basis,
            basis,
            drive,
            polarization,
            nmax,
            f_coupling: None,
        }
    }

    /// Use a provided cavity coupling function instead of the
    /// [`CavityCoupling`] implementation.
    pub fn with_g<F>(mut self, f_coupling: F) -> Self
    where F: Fn(&S, &S) -> Option<PhotonLadder> + 'a
    {
        self.f_coupling = Some(Rc::new(f_coupling));
        self
    }

    /// Return a reference to the atomic basis.
    pub fn atom_basis(&self) -> &Basis<S> { self.atom_basis }

    /// Return a reference to the full atom-cavity basis.
    pub fn basis(&self) -> &Basis<Cavity<N, P, S>> { &self.basis }

    /// Return the maximum cavity mode numbers for each mode.
    pub fn nmax(&self) -> &[usize; P] { &self.nmax }

    /// Generate the state vector for coherent states over all cavity modes for
    /// a single state of the atomic array.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_vector(
        &self,
        atomic_states: &[S; N],
        alpha: &[C64; P],
    ) -> Option<nd::Array1<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_states() == atomic_states) {
            return None;
        }
        let pref: Vec<C64>
            = alpha.iter().map(|a| (-0.5 * a * a.conj()).exp()).collect();
        let mut psi: nd::Array1<C64>
            = self.basis.keys()
            .map(|s| {
                if s.atomic_states() == atomic_states {
                    s.photons().iter()
                        .zip(&pref)
                        .zip(alpha)
                        .map(|((n, p), a)| {
                            let n = *n as i32;
                            let fact_n: f64
                                = (1..=n).map(f64::from).product();
                            p / fact_n.sqrt() * a.powi(n)
                        })
                        .product()
                } else {
                    C64::zero()
                }
            })
            .collect();
        let norm: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= norm;
        Some(psi)
    }

    /// Generate the state vector for coherent states over all cavity modes for
    /// an arbitrary admixture of atomic array states.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_atomic<F>(&self, atom_amps: F, alpha: &[C64; P])
        -> nd::Array1<C64>
    where F: Fn(&[S; N], usize, f64) -> C64
    {
        let pref: Vec<C64>
            = alpha.iter().map(|a| (-0.5 * a * a.conj()).exp()).collect();
        let mut psi: nd::Array1<C64>
            = self.basis.iter().enumerate()
            .map(|(index, (sn, energy))| {
                let atom = atom_amps(sn.atomic_states(), index, *energy);
                let photon: C64
                    = sn.photons().iter()
                    .zip(&pref)
                    .zip(alpha)
                    .map(|((n, p), a)| {
                        let n = *n as i32;
                        let fact_n: f64
                            = (1..=n).map(f64::from).product();
                        p / fact_n.sqrt() * a.powi(n)
                    })
                    .product();
                atom * photon
            })
            .collect();
        let norm: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= norm;
        psi
    }

    /// Compute a time-independent Hamiltonian if `self.drive` is
    /// [`DriveParams::Constant`].
    ///
    /// The returned Hamiltonian is in the frame of the drive in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        if let Some(f) = &self.f_coupling {
            self.gen_static_with(f.as_ref())
        } else {
            self.gen_static_with(Self::def_coupling)
        }
    }

    /// Like [`Self::gen_static`], but using a provided cavity coupling function
    /// rather than that of the [`CavityCoupling`] implementation.
    pub fn gen_static_with<F>(&self, f_coupling: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let H_site: nd::Array2<C64>
            = HBuilder::new(
                self.atom_basis,
                self.drive.clone(),
                self.polarization,
            )
            .gen_static()?;
        let photon_eye: nd::Array2<C64>
            = nd::Array2::eye(self.nmax.iter().map(|&n| n + 1).product());
        let mut H: nd::Array2<C64>
            = kron(
                &multiatom_kron((0..N).map(|_| H_site.view())),
                &photon_eye,
            );
        H.diag_mut().iter_mut()
            .zip(self.basis.values())
            .for_each(|(h, e)| { *h = (*e).into(); });
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut ss1: &[S; N];
        let mut ss2: &[S; N];
        let mut nn1: &[usize; P];
        let mut nn2: &[usize; P];
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, sn1), (i, sn2)) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            nn1 = sn1.photons();
            nn2 = sn2.photons();

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match f_coupling(s1, s2) {
                    None => { visited.insert((sn2, sn1)); }
                    Some(PhotonLadder::Emit(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                // only Jaynes-Cummings
                                // (n1 + 1 == n2).then_some(g * (n2 as f64).sqrt())

                                // with anti-Jaynes-Cummings
                                (n1.abs_diff(n2) == 1)
                                    .then_some(g * (n1.max(n2) as f64).sqrt())
                            })
                        {
                            H[[i, j]] += gn;
                            H[[j, i]] += gn;
                            visited.insert((sn2, sn1));
                        }
                    },
                    Some(PhotonLadder::Absorb(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                // only Jaynes-Cummings
                                // (n1 == n2 + 1).then_some(g * (n1 as f64).sqrt())

                                // with anti-Jaynes-Cummings
                                (n1.abs_diff(n2) == 1)
                                    .then_some(g * (n1.max(n2) as f64).sqrt())
                            })
                        {
                            H[[i, j]] += gn;
                            H[[j, i]] += gn;
                            visited.insert((sn2, sn1));
                        }
                    },
                }
            }
        }
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        if let Some(f) = &self.f_coupling {
            self.diagonalize_with(f.as_ref())
        } else {
            self.diagonalize_with(Self::def_coupling)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn diagonalize_with<F>(&self, f_coupling: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        match self.gen_static_with(f_coupling)?.eigh_into(UPLO::Lower) {
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
        if let Some(f) = &self.f_coupling {
            self.ground_state_with(f.as_ref())
        } else {
            self.ground_state_with(Self::def_coupling)
        }
    }

    /// Like [`Self::ground_state`], but using a provided cavity coupling
    /// function rather than that of the [`CavityCoupling`] implementation.
    pub fn ground_state_with<F>(&self, f_coupling: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let (E, V) = self.diagonalize_with(f_coupling)?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        if let Some(f) = &self.f_coupling {
            self.gen_at_with(t, f.as_ref())
        } else {
            self.gen_at_with(t, Self::def_coupling)
        }
    }

    /// Like [`Self::gen_at`], but using a produced cavity coupling function
    /// rather than that of the [`CavityCoupling`] implementation.
    pub fn gen_at_with<F>(&self, t: f64, f_coupling: F) -> nd::Array2<C64>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let H_site: nd::Array2<C64>
            = HBuilder::new(
                self.atom_basis,
                self.drive.clone(),
                self.polarization,
            )
            .gen_at(t);
        let photon_eye: nd::Array2<C64>
            = nd::Array2::eye(self.nmax.iter().map(|&n| n + 1).product());
        let mut H: nd::Array2<C64>
            = kron(&multiatom_kron((0..N).map(|_| H_site.view())), &photon_eye);
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut ss1: &[S; N];
        let mut ss2: &[S; N];
        let mut nn1: &[usize; P];
        let mut nn2: &[usize; P];
        let mut coupling: C64;
        let iter
            = self.basis.iter().enumerate()
            .cartesian_product(self.basis.iter().enumerate());
        for ((j, (sn1, e1)), (i, (sn2, e2))) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            nn1 = sn1.photons();
            nn2 = sn2.photons();

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match f_coupling(s1, s2) {
                    None => {
                        visited.insert((sn2, sn1));
                        continue;
                    },
                    Some(PhotonLadder::Emit(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 + 1 == n2).then_some(g * (n2 as f64).sqrt())
                            })
                        {
                            coupling = C64::from_polar(gn, (*e2 - *e1) * t);
                            H[[i, j]] += coupling;
                            H[[j, i]] += coupling.conj();
                            visited.insert((sn2, sn1));
                        }
                    },
                    Some(PhotonLadder::Absorb(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 == n2 + 1).then_some(g * (n1 as f64).sqrt())
                            })
                        {
                            coupling = C64::from_polar(gn, (*e2 - *e1) * t);
                            H[[i, j]] += coupling;
                            H[[j, i]] += coupling.conj();
                            visited.insert((sn2, sn1));
                        }
                    },
                }
            }
        }
        H
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        if let Some(f) = &self.f_coupling {
            self.gen_with(time, f.as_ref())
        } else {
            self.gen_with(time, Self::def_coupling)
        }
    }

    /// Like [`Self::gen`], but using a provided cavity coupling function rather
    /// than that of the [`CavityCoupling`] implementation.
    pub fn gen_with<F>(&self, time: &nd::Array1<f64>, f_coupling: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<PhotonLadder>
    {
        let H_site: nd::Array3<C64>
            = HBuilder::new(
                self.atom_basis,
                self.drive.clone(),
                self.polarization,
            )
            .gen(time);
        let photon_eye: nd::Array2<C64>
            = nd::Array2::eye(self.nmax.iter().map(|&n| n + 1).product());
        let H: Vec<nd::Array2<C64>>
            = (0..time.len())
            .map(|k| {
                kron(
                    &multiatom_kron(
                        (0..N).map(|_| H_site.slice(s![.., .., k]))),
                    &photon_eye,
                )
            })
            .collect();
        let mut H: nd::Array3<C64>
            = nd::stack(
                nd::Axis(2),
                &H.iter().map(|Ht| Ht.view()).collect::<Vec<_>>()
            ).unwrap();
        let mut visited: HashSet<(&Cavity<N, P, S>, &Cavity<N, P, S>)>
            = HashSet::default();
        let mut ss1: &[S; N];
        let mut ss2: &[S; N];
        let mut nn1: &[usize; P];
        let mut nn2: &[usize; P];
        let mut freq: f64;
        let mut coupling: nd::Array1<C64>;
        let mut coupling_conj: nd::Array1<C64>;
        let iter
            = self.basis.iter().enumerate()
            .cartesian_product(self.basis.iter().enumerate());
        for ((j, (sn1, e1)), (i, (sn2, e2))) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            nn1 = sn1.photons();
            nn2 = sn2.photons();

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                coupling = nd::Array1::zeros(time.len());
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match f_coupling(s1, s2) {
                    None => { visited.insert((sn2, sn1)); },
                    Some(PhotonLadder::Emit(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 + 1 == n2).then_some(g * (n2 as f64).sqrt())
                            })
                        {
                            freq = *e2 - *e1;
                            coupling.iter_mut()
                                .zip(time)
                                .for_each(|(c, t)| {
                                    *c = C64::from_polar(gn, freq * t);
                                });
                            coupling_conj = coupling.mapv(|a| a.conj());
                            H.slice_mut(s![i, j, ..]).iter_mut()
                                .zip(&coupling)
                                .for_each(|(Hijk, couplingk)| {
                                    *Hijk += *couplingk;
                                });
                            H.slice_mut(s![j, i, ..]).iter_mut()
                                .zip(coupling_conj)
                                .for_each(|(Hjik, couplingk)| {
                                    *Hjik += couplingk;
                                });
                            visited.insert((sn2, sn1));
                        }
                    },
                    Some(PhotonLadder::Absorb(m, g)) => {
                        if let Some(gn)
                            = nn1.get(m)
                            .zip(nn2.get(m))
                            .and_then(|(&n1, &n2)| {
                                (n1 == n2 + 1).then_some(g * (n1 as f64).sqrt())
                            })
                        {
                            freq = *e2 - *e1;
                            coupling.iter_mut()
                                .zip(time)
                                .for_each(|(c, t)| {
                                    *c = C64::from_polar(gn, freq * t);
                                });
                            coupling_conj = coupling.mapv(|a| a.conj());
                            H.slice_mut(s![i, j, ..]).iter_mut()
                                .zip(&coupling)
                                .for_each(|(Hijk, couplingk)| {
                                    *Hijk += *couplingk;
                                });
                            H.slice_mut(s![j, i, ..]).iter_mut()
                                .zip(coupling_conj)
                                .for_each(|(Hjik, couplingk)| {
                                    *Hjik += couplingk;
                                });
                            visited.insert((sn2, sn1));
                        }
                    },
                }
            }
        }
        H
    }
}

/// Initialization data for [`HBuilderCavity`].
#[derive(Clone, Debug)]
pub struct HCavityParams<'a, const P: usize, S>
where S: SpinState + CavityCoupling<P>
{
    pub atom_basis: &'a Basis<S>,
    pub drive: DriveParams<'a>,
    pub polarization: PolarizationParams,
    pub nmax: [usize; P],
}

impl<'a, const N: usize, const P: usize, S> HBuild<'a, Cavity<N, P, S>>
    for HBuilderCavity<'a, N, P, S>
where S: SpinState + CavityCoupling<P>
{
    type Params = HCavityParams<'a, P, S>;
    type Basis = Basis<Cavity<N, P, S>>;

    fn new_builder(params: Self::Params) -> Self {
        let HCavityParams { atom_basis, drive, polarization, nmax }
            = params;
        Self::new(atom_basis, drive, polarization, nmax)
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

