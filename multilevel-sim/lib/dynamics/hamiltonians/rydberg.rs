//! A number of atoms experiencing a Rydberg interaction for a subset of their
//! accessible states.

use std::{
    fmt,
    rc::Rc,
};
use itertools::Itertools;
use ndarray::{ self as nd, s };
use ndarray_linalg::{ EighInto, UPLO };
use num_complex::Complex64 as C64;
use rustc_hash::FxHashSet as HashSet;
use crate::{
    dynamics::{
        hamiltonians::{ HBuild, HBuilder },
        multiatom_kron,
    },
    hilbert::{ Basis, ProdBasis, RydbergState, SpinState },
};

/// Connectivity between atoms.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RydbergCoupling {
    /// All-to-all coupling, where every atom is taken to be a fixed distance
    /// from all others.
    AllToAll(f64),
    /// Arrangement in a 1D chain with fixed spacing.
    Chain(f64),
}

impl RydbergCoupling {
    /// Compute the total Rydberg shift for a single multi-atom state.
    pub fn compute_shift<S>(&self, state: &[S]) -> f64
    where S: RydbergState
    {
        self.compute_shift_with(state, |s1, s2| s1.c6_with(s2))
    }

    /// Compute the total Rydberg shift for a single multi-atom state using a
    /// provided C6 function rather than that of the [`RydbergState`]
    /// implementation.
    pub fn compute_shift_with<S, F>(&self, state: &[S], f_c6: F) -> f64
    where F: Fn(&S, &S) -> Option<f64>
    {
        match self {
            Self::AllToAll(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| f_c6(s1, s2))
                            .map(|c6| c6 / r.powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
            Self::Chain(r) => {
                state.iter().enumerate()
                    .cartesian_product(state.iter().enumerate())
                    .filter_map(|((i, s1), (j, s2))| {
                        (i != j).then_some(())
                            .and_then(|_| f_c6(s1, s2))
                            .map(|c6| c6 / (*r * (j as f64 - i as f64)).powi(6))
                    })
                    .sum::<f64>() / 2.0
            },
        }
    }
}

/// Hamiltonian builder for a driven multi-atom system including ~1/r^6 Rydberg
/// interactions.
#[derive(Clone)]
pub struct HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    pub(crate) sites: Vec<HBuilder<'a, S>>,
    pub(crate) prod_basis: ProdBasis<S>,
    pub coupling: RydbergCoupling,
    pub(crate) f_c6: Option<Rc<dyn Fn(&S, &S) -> Option<f64> + 'a>>,
}

impl<'a, S> fmt::Debug for HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HBuilderRydberg {{ \
            sites: {:?}, \
            prod_basis: {:?}, \
            coupling: {:?}, \
            f_c6: ",
            self.sites,
            self.prod_basis,
            self.coupling,
        )?;
        if self.f_c6.is_some() {
            write!(f, "Some(...)")?;
        } else {
            write!(f, "None")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<'a, S> HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    fn def_c6(s1: &S, s2: &S) -> Option<f64> { s1.c6_with(s2) }

    /// Create a new `HBuilderRydberg` where the drives for each atom are
    /// specified individually.
    pub fn new<I>(
        sites: I,
        coupling: RydbergCoupling,
    ) -> Self
    where I: IntoIterator<Item = HBuilder<'a, S>>
    {
        let sites: Vec<HBuilder<S>> = sites.into_iter().collect();
        let mut prod_basis
            = ProdBasis::from_kron(
                sites.iter().map(|builder| builder.basis()));
        prod_basis.iter_mut()
            .for_each(|(ss, e)| *e += coupling.compute_shift(ss));
        Self { sites, prod_basis, coupling, f_c6: None }
    }

    /// Create a new `HBuilderRydberg` where all atoms are driven globally.
    pub fn new_nsites(
        hbuilder: HBuilder<'a, S>,
        nsites: usize,
        coupling: RydbergCoupling,
    ) -> Self
    {
        let sites: Vec<HBuilder<S>>
            = (0..nsites).map(|_| hbuilder.clone()).collect();
        let mut prod_basis
            = ProdBasis::from_kron(
                sites.iter().map(|builder| builder.basis()));
        prod_basis.iter_mut()
            .for_each(|(ss, e)| *e += coupling.compute_shift(ss));
        Self { sites, prod_basis, coupling, f_c6: None }
    }

    /// Use a provided C6 function instead of the [`RydbergState`]
    /// implementation.
    pub fn with_c6<F>(self, f_c6: F) -> Self
    where F: Fn(&S, &S) -> Option<f64> + 'a
    {
        let new_f_c6 = Rc::new(f_c6);
        let Self {
            sites,
            mut prod_basis,
            coupling,
            f_c6: old_f_c6,
        } = self;
        if let Some(f) = &old_f_c6 {
            prod_basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= coupling.compute_shift_with(ss, f.as_ref());
                    *e += coupling.compute_shift_with(ss, new_f_c6.as_ref());
                });
        } else {
            prod_basis.iter_mut()
                .for_each(|(ss, e)| {
                    *e -= coupling.compute_shift(ss);
                    *e += coupling.compute_shift_with(ss, new_f_c6.as_ref());
                });
        }
        Self { sites, prod_basis, coupling, f_c6: Some(new_f_c6) }
    }

    /// Return a reference to the [`HBuilder`] for site `index`.
    pub fn hbuilder(&self, index: usize) -> Option<&HBuilder<S>> {
        self.sites.get(index)
    }

    /// Return a reference to the basis for site `index`.
    pub fn basis(&self, index: usize) -> Option<&Basis<S>> {
        self.sites.get(index).map(|builder| builder.basis())
    }

    /// Compute and return a reference to a [`ProdBasis`] holding all n-atom
    /// states and energies including Rydberg shifts.
    pub fn prod_basis(&self) -> &ProdBasis<S> { &self.prod_basis }

    /// Compute a time-independent Hamiltonian if all drives are
    /// [`DriveParams::Constant`][crate::dynamics::DriveParams::Constant].
    ///
    /// The returned Hamiltonian is in the frame of the drive(s) in the rotating
    /// wave approximation.
    pub fn gen_static(&self) -> Option<nd::Array2<C64>> {
        if let Some(f) = &self.f_c6 {
            self.gen_static_with(f.as_ref())
        } else {
            self.gen_static()
        }
    }

    /// Like [`Self::gen_static`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn gen_static_with<F>(&self, f_c6: F) -> Option<nd::Array2<C64>>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let sites_H: Vec<nd::Array2<C64>>
            = self.sites.iter()
            .map(|site| site.gen_static())
            .collect::<Option<Vec<_>>>()?;
        let mut H: nd::Array2<C64>
            = multiatom_kron(sites_H.iter().map(|h| h.view()));
        self.prod_basis.keys()
            .zip(H.diag_mut())
            .for_each(|(ss, e)| {
                *e += self.coupling.compute_shift_with(ss, &f_c6);
            });
        Some(H)
    }

    /// Diagonalize the [time-independent representation][Self::gen_static] of
    /// the Hamiltonian.
    pub fn diagonalize(&self) -> Option<(nd::Array1<f64>, nd::Array2<C64>)> {
        if let Some(f) = &self.f_c6 {
            self.diagonalize_with(f.as_ref())
        } else {
            self.diagonalize_with(Self::def_c6)
        }
    }

    /// Like [`Self::diagonalize`], but using a provided C6 function rather than
    /// that of the [`RydbergState`] implementation.
    pub fn diagonalize_with<F>(&self, f_c6: F)
        -> Option<(nd::Array1<f64>, nd::Array2<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        match self.gen_static_with(f_c6)?.eigh_into(UPLO::Lower) {
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
        if let Some(f) = &self.f_c6 {
            self.ground_state_with(f.as_ref())
        } else {
            self.ground_state_with(Self::def_c6)
        }
    }

    /// Like [`Self::ground_state`], but using a provided C6 function rather
    /// than that of the [`RydbergState`] implementation.
    pub fn ground_state_with<F>(&self, f_c6: F)
        -> Option<(f64, nd::Array1<C64>)>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let (E, V) = self.diagonalize_with(f_c6)?;
        let e: f64 = E[0];
        let v: nd::Array1<C64> = V.slice(s![.., 0]).to_owned();
        Some((e, v))
    }

    /// Compute the time-dependent Hamiltonian at a given time as a 2D array.
    pub fn gen_at(&self, t: f64) -> nd::Array2<C64> {
        if let Some(f) = &self.f_c6 {
            self.gen_at_with(t, f.as_ref())
        } else {
            self.gen_at_with(t, Self::def_c6)
        }
    }

    /// Like [`Self::gen_at`], but using a provided C6 function rather than that
    /// of the [`RydbergState`] implementation.
    pub fn gen_at_with<F>(&self, t: f64, f_c6: F) -> nd::Array2<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let sites_H: Vec<nd::Array2<C64>>
            = self.sites.iter().map(|site| site.gen_at(t)).collect();
        let mut H: nd::Array2<C64>
            = multiatom_kron(sites_H.iter().map(|h| h.view()));
        let mut visited: HashSet<(&Vec<S>, &Vec<S>)> = HashSet::default();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: C64;
        let iter
            = self.prod_basis.keys().enumerate()
            .cartesian_product(self.prod_basis.keys().enumerate());
        for ((j, ss1), (i, ss2)) in iter {
            if visited.contains(&(ss2, ss1)) { continue; }
            shift1 = self.coupling.compute_shift_with(ss1, &f_c6);
            shift2 = self.coupling.compute_shift_with(ss2, &f_c6);
            if (shift2 - shift1).abs() < 1e-15 { continue; }
            shift_phase = (C64::i() * (shift2 - shift1) * t).exp();
            H[[i, j]] *= shift_phase;
            H[[j, i]] *= shift_phase.conj();
            visited.insert((ss1, ss2));
        }
        H
    }

    /// Compute the time-dependent Hamiltonian as a 3D array, with the last axis
    /// corresponding to time.
    pub fn gen(&self, time: &nd::Array1<f64>) -> nd::Array3<C64> {
        if let Some(f) = &self.f_c6 {
            self.gen_with(time, f.as_ref())
        } else {
            self.gen_with(time, Self::def_c6)
        }
    }

    /// Like [`Self::gen`], but using a provided C6 function rather than that of
    /// the [`RydbergState`] implementation.
    pub fn gen_with<F>(&self, time: &nd::Array1<f64>, f_c6: F)
        -> nd::Array3<C64>
    where F: Fn(&S, &S) -> Option<f64>
    {
        let sites_H: Vec<nd::Array3<C64>>
            = self.sites.iter().map(|site| site.gen(time)).collect();
        let H: Vec<nd::Array2<C64>>
            = (0..time.len())
            .map(|k| {
                multiatom_kron(
                    sites_H.iter().map(|H| H.slice(s![.., .., k])))
            })
            .collect();
        let mut H: nd::Array3<C64>
            = nd::stack(
                nd::Axis(2),
                &H.iter().map(|Ht| Ht.view()).collect::<Vec<_>>()
            ).unwrap();
        let mut visited: HashSet<(&Vec<S>, &Vec<S>)> = HashSet::default();
        let mut shift1: f64;
        let mut shift2: f64;
        let mut shift_phase: nd::Array1<C64>;
        let mut shift_phase_conj: nd::Array1<C64>;
        let iter
            = self.prod_basis.keys().enumerate()
            .cartesian_product(self.prod_basis.keys().enumerate());
        for ((j, ss1), (i, ss2)) in iter {
            if visited.contains(&(ss2, ss1)) { continue; }
            shift1 = self.coupling.compute_shift_with(ss1, &f_c6);
            shift2 = self.coupling.compute_shift_with(ss2, &f_c6);
            if (shift2 - shift1).abs() < 1e-15 { continue; }
            shift_phase
                = time.mapv(|t| (C64::i() * (shift2 - shift1) * t).exp());
            shift_phase_conj = shift_phase.mapv(|a| a.conj());

            H.slice_mut(s![i, j, ..]).iter_mut()
                .zip(shift_phase)
                .for_each(|(Hijk, shiftk)| *Hijk *= shiftk);
            H.slice_mut(s![j, i, ..]).iter_mut()
                .zip(shift_phase_conj)
                .for_each(|(Hjik, shiftk)| *Hjik *= shiftk);
            visited.insert((ss1, ss2));
        }
        H
    }
}

/// Initialization data for [`HBuilderRydberg`].
#[derive(Clone, Debug)]
pub struct HRydbergParams<'a, I, S>
where
    I: IntoIterator<Item = HBuilder<'a, S>>,
    S: SpinState + RydbergState + 'a,
{
    pub sites: I,
    pub coupling: RydbergCoupling,
}

impl<'a, S> HBuild<'a, Vec<S>> for HBuilderRydberg<'a, S>
where S: SpinState + RydbergState
{
    type Params = HRydbergParams<'a, Vec<HBuilder<'a, S>>, S>;
    type Basis = ProdBasis<S>;

    fn new_builder(params: Self::Params) -> Self {
        let HRydbergParams { sites, coupling } = params;
        Self::new(sites, coupling)
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

    fn get_basis(&self) -> &Self::Basis { self.prod_basis() }
}

