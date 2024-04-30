//! A linear spin chain with a σ<sub>*z*</sub>σ<sub>*z*</sub> Rydberg-like
//! interaction, coupled to a single optical cavity mode.
//!
//! See also [`lindbladians::transverse_ising`][super::super::lindbladians::transverse_ising].

use itertools::Itertools;
use ndarray::{ self as nd, s };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use num_traits::Zero;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::hamiltonians::HBuild,
    hilbert::{ Basis, Cavity, HSpin },
};

/// Specialized Hamiltonian builder for a `N`-site Rydberg spin chain coupled to
/// a single optical cavity mode.
///
/// Degrees of freedom and couplings in this model are limited to an assumed
/// form, so trait bounds are relaxed. This builder produces the (1D) quantum
/// transverse field Ising model (QTFIM) with the Hamiltonian
/// ```math
/// \begin{align*}
///     H_\text{QTFIM}
///         &= \frac{\omega_z}{2} \sum_n \sigma_n^z
///         \\
///         &+ \omega_a a^\dagger a
///         \\
///         &- J_z \sum_n \sigma_n^z \sigma_{n + 1}^z
///         \\
///         &+ \frac{g}{2 \sqrt{N}} (a + a^\dagger) \sum_n \sigma_n^x
/// \end{align*}
/// ```
/// where the first line gives the interaction-free collective state energies,
/// the second line nearest-neighbor Rydberg interactions, and the third line
/// atom-cavity couplings. Parameters `\omega_z`, `\omega_a`, `J_z`, and `g` are
/// tunable.
#[derive(Clone, Debug)]
pub struct HBuilderTransverseIsing<const N: usize> {
    pub(crate) basis: Basis<Cavity<N, 1, HSpin>>,
    pub(crate) omega_z: f64,
    pub(crate) omega_a: f64,
    pub(crate) j_z: f64,
    pub(crate) g: f64,
    pub(crate) nmax: usize,
}

impl<const N: usize> HBuilderTransverseIsing<N> {
    /// Create a new `HBuilderTransverseIsing`.
    pub fn new(
        omega_z: f64,
        omega_a: f64,
        j_z: f64,
        g: f64,
        nmax: usize,
    ) -> Self
    {
        let basis: Basis<Cavity<N, 1, HSpin>>
            = (0..N).map(|_| [HSpin::Dn, HSpin::Up]).multi_cartesian_product()
            .cartesian_product(0..=nmax)
            .map(|(ss, n)| {
                let spin_energy: f64
                    = ss.iter()
                    .map(|s| s.sz() * omega_z / 0.5)
                    .sum();
                let spin_shift: f64
                    = ss.iter()
                    .zip(ss.iter().skip(1))
                    .map(|(sn, snp1)| -j_z * sn.sz() * snp1.sz())
                    .sum();
                let spins: [HSpin; N] = ss.try_into().unwrap();
                let photon_energy: f64 = (n as f64) * omega_a;
                let total_energy = spin_energy + spin_shift + photon_energy;
                ((spins, [n]).into(), total_energy)
            })
            .collect();
        Self {
            basis,
            omega_z,
            omega_a,
            j_z,
            g,
            nmax,
        }
    }

    /// Return a reference to the full spin-cavity basis.
    pub fn basis(&self) -> &Basis<Cavity<N, 1, HSpin>> { &self.basis }

    /// Return all model parameters.
    pub fn params(&self) -> HTransverseIsingParams {
        HTransverseIsingParams {
            omega_z: self.omega_z,
            omega_a: self.omega_a,
            j_z: self.j_z,
            g: self.g,
            nmax: self.nmax,
        }
    }

    /// Generate the state vector for coherent states over all cavity modes for
    /// a single state of the spin chain.
    ///
    /// **Note**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_vector(
        &self,
        spin_states: &[HSpin; N],
        alpha: C64,
    ) -> Option<nd::Array1<C64>>
    {
        if !self.basis.keys().any(|s| s.atomic_states() == spin_states) {
            return None;
        }
        let pref: C64 = (-0.5 * alpha * alpha.conj()).exp();
        let mut psi: nd::Array1<C64>
            = self.basis.keys()
            .map(|s| {
                if s.atomic_states() == spin_states {
                    let n = s.photons()[0] as i32;
                    let fact_n: f64 = (1..=n).map(f64::from).product();
                    pref / fact_n.sqrt() * alpha.powi(n)
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
    /// an arbitrary admixture of spin chain states.
    ///
    /// **Node**: the returned state is renormalized such that its inner product
    /// with itself is equal to 1; in cases where the maximum photon numbers for
    /// the cavity modes are not sufficiently high, this can cause average
    /// photon numbers to disagree with theory.
    pub fn coherent_state_spin<F>(&self, spin_amps: F, alpha: C64)
        -> nd::Array1<C64>
    where F: Fn(&[HSpin; N], usize, f64) -> C64
    {
        let pref: C64 = (-0.5 * alpha * alpha.conj()).exp();
        let mut psi: nd::Array1<C64>
            = self.basis.iter().enumerate()
            .map(|(index, (sn, energy))| {
                let spin = spin_amps(sn.atomic_states(), index, *energy);
                let n = sn.photons()[0] as i32;
                let fact_n: f64 = (1..=n).map(f64::from).product();
                let photon: C64 = pref / fact_n.sqrt() * alpha.powi(n);
                spin * photon
            })
            .collect();
        let norm: C64 = psi.iter().map(|a| a * a.conj()).sum::<C64>().sqrt();
        psi /= norm;
        psi
    }

    /// Compute a time-independent Hamiltonian.
    pub fn gen_static(&self) -> nd::Array2<C64> {
        let mut H: nd::Array2<C64>
            = nd::Array2::from_diag(
                &self.basis.values().map(|e| C64::from(*e))
                    .collect::<nd::Array1<C64>>()
            );
        let pref: f64 = 0.5 / (N as f64).sqrt();
        let mut visited: HashSet<(&Cavity<N, 1, HSpin>, &Cavity<N, 1, HSpin>)>
            = HashSet::default();
        let mut ss1: &[HSpin; N];
        let mut ss2: &[HSpin; N];
        let mut n1: usize;
        let mut n2: usize;
        let mut offd: f64;
        let iter
            = self.basis.keys().enumerate()
            .cartesian_product(self.basis.keys().enumerate());
        for ((j, sn1), (i, sn2)) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            n1 = sn1.photons()[0];
            n2 = sn2.photons()[0];

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match (*s1, *s2) {
                    (HSpin::Up, HSpin::Dn) if n1.abs_diff(n2) == 1 => {
                        offd = pref * self.g * (n1.max(n2) as f64).sqrt();
                        H[[i, j]] += offd;
                        H[[j, i]] += offd;
                        visited.insert((sn2, sn1));
                    },
                    (HSpin::Dn, HSpin::Up) if n1.abs_diff(n2) == 1 => {
                        offd = pref * self.g * (n1.max(n2) as f64).sqrt();
                        H[[i, j]] += offd;
                        H[[j, i]] += offd;
                        visited.insert((sn2, sn1));
                    },
                    _ => { visited.insert((sn2, sn1)); },
                }
            }
        }
        H
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> (nd::Array1<f64>, nd::Array2<C64>) {
        match self.gen_static().eigh_into(UPLO::Lower) {
            Ok((E, V)) => (E, V),
            Err(err) => panic!("unexpected diagonalization error: {}", err),
        }
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian and return a ground state of the system.
    ///
    /// Note that, in general, there may be more than one state that minimizes
    /// the energy of the system; this method offers no guarantees about which
    /// ground state is returned.
    pub fn ground_state(&self) -> (f64, nd::Array1<C64>) {
        let (E, V) = self.diagonalize();
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        (e, v)
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        let n = self.basis.len();
        let mut H: nd::Array2<C64> = nd::Array2::zeros((n, n));
        let pref: f64 = 0.5 / (N as f64).sqrt();
        let mut visited: HashSet<(&Cavity<N, 1, HSpin>, &Cavity<N, 1, HSpin>)>
            = HashSet::default();
        let mut ss1: &[HSpin; N];
        let mut ss2: &[HSpin; N];
        let mut n1: usize;
        let mut n2: usize;
        let mut offd_amp: f64;
        let mut offd_freq: f64;
        let mut offd: C64;
        let iter
            = self.basis.iter().enumerate()
            .cartesian_product(self.basis.iter().enumerate());
        for ((j, (sn1, &e1)), (i, (sn2, &e2))) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            n1 = sn1.photons()[0];
            n2 = sn2.photons()[0];

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match (*s1, *s2) {
                    (HSpin::Up, HSpin::Dn) if n1.abs_diff(n2) == 1 => {
                        offd_amp = pref * self.g * (n1.max(n2) as f64).sqrt();
                        offd_freq = e2 - e1;
                        offd = C64::from_polar(offd_amp, offd_freq * t);
                        H[[i, j]] += offd;
                        H[[j, i]] += offd.conj();
                        visited.insert((sn2, sn1));
                    },
                    (HSpin::Dn, HSpin::Up) if n1.abs_diff(n2) == 1 => {
                        offd_amp = pref * self.g * (n1.max(n2) as f64).sqrt();
                        offd_freq = e2 - e1;
                        offd = C64::from_polar(offd_amp, offd_freq * t);
                        H[[i, j]] += offd;
                        H[[j, i]] += offd.conj();
                        visited.insert((sn2, sn1));
                    },
                    _ => { visited.insert((sn2, sn1)); },
                }
            }
        }
        H
    }

    /// Compute a time-dependent Hamiltonian.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        let n = self.basis.len();
        let nt = time.len();
        let mut H: nd::Array3<C64> = nd::Array3::zeros((n, n, nt));
        let pref: f64 = 0.5 / (N as f64).sqrt();
        let mut visited: HashSet<(&Cavity<N, 1, HSpin>, &Cavity<N, 1, HSpin>)>
            = HashSet::default();
        let mut ss1: &[HSpin; N];
        let mut ss2: &[HSpin; N];
        let mut n1: usize;
        let mut n2: usize;
        let mut offd_amp: f64;
        let mut offd_freq: f64;
        let mut offd: nd::Array1<C64>;
        let mut offd_conj: nd::Array1<C64>;
        let iter
            = self.basis.iter().enumerate()
            .cartesian_product(self.basis.iter().enumerate());
        for ((j, (sn1, &e1)), (i, (sn2, &e2))) in iter {
            if visited.contains(&(sn1, sn2)) { continue; }
            ss1 = sn1.atomic_states();
            ss2 = sn2.atomic_states();
            n1 = sn1.photons()[0];
            n2 = sn2.photons()[0];

            if ss1.iter().zip(ss2).filter(|(s1, s2)| s1 != s2).count() == 1 {
                let (s1, s2)
                    = ss1.iter().zip(ss2).find(|(s1, s2)| s1 != s2).unwrap();
                match (*s1, *s2) {
                    (HSpin::Up, HSpin::Dn) if n1.abs_diff(n2) == 1 => {
                        offd_amp = pref * self.g * (n1.max(n2) as f64).sqrt();
                        offd_freq = e2 - e1;
                        offd
                            = time.mapv(|t| {
                                C64::from_polar(offd_amp, offd_freq * t)
                            });
                        offd_conj = offd.mapv(|a| a.conj());
                        offd.move_into(H.slice_mut(s![i, j, ..]));
                        offd_conj.move_into(H.slice_mut(s![j, i, ..]));
                        visited.insert((sn2, sn1));
                    },
                    (HSpin::Dn, HSpin::Up) if n1.abs_diff(n2) == 1 => {
                        offd_amp = pref * self.g * (n1.max(n2) as f64).sqrt();
                        offd_freq = e2 - e1;
                        offd
                            = time.mapv(|t| {
                                C64::from_polar(offd_amp, offd_freq * t)
                            });
                        offd_conj = offd.mapv(|a| a.conj());
                        offd.move_into(H.slice_mut(s![i, j, ..]));
                        offd_conj.move_into(H.slice_mut(s![j, i, ..]));
                        visited.insert((sn2, sn1));
                    },
                    _ => { visited.insert((sn2, sn1)); },
                }
            }
        }
        H
    }
}

/// Initialization data for [`HBuilderTransverseIsing`].
#[derive(Copy, Clone, Debug)]
pub struct HTransverseIsingParams {
    pub omega_z: f64,
    pub omega_a: f64,
    pub j_z: f64,
    pub g: f64,
    pub nmax: usize,
}

impl<const N: usize> HBuild<'_, Cavity<N, 1, HSpin>>
    for HBuilderTransverseIsing<N>
{
    type Params = HTransverseIsingParams;
    type Basis = Basis<Cavity<N, 1, HSpin>>;

    fn new_builder(params: Self::Params) -> Self {
        let HTransverseIsingParams {
            omega_z,
            omega_a,
            j_z,
            g,
            nmax,
        } = params;
        Self::new(omega_z, omega_a, j_z, g, nmax)
    }

    fn build_static(&self) -> Option<nd::Array2<C64>> {
        Some(self.gen_static())
    }

    fn build_at(&self, t: f64) -> nd::Array2<C64> {
        self.gen_at(t)
    }

    fn build(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        self.gen(time)
    }

    fn get_basis(&self) -> &Self::Basis { self.basis() }
}

