#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]
#![allow(static_mut_refs)]
#![allow(clippy::approx_constant)]

use std::{
    f64::consts::{ PI, TAU },
    fmt::{ Display, LowerExp },
    ops::{ Add, AddAssign, Sub, SubAssign, Mul, MulAssign },
    path:: PathBuf,
    rc::Rc,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    mkdir,
    write_npz,
    hilbert::{ Basis, TrappedMagic, Fock },
    dynamics::*,
    rabi::*,
};
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use lib::systems::quoct::*;

macro_rules! paramv {
    (
        {
            { $th0:expr, $ph0:expr $(,)? },
            { $th1:expr, $ph1:expr $(,)? },
            { $th2:expr, $ph2:expr $(,)? } $(,)?
        }
    ) => {
        ParamV {
            u0: PulseParamV { th: $th0, ph: $ph0 },
            u1: PulseParamV { th: $th1, ph: $ph1 },
            u2: PulseParamV { th: $th2, ph: $ph2 },
        }
    }
}

const M: f64 = 2.8384644058191703e-25; // kg
const WL: f64 = 578e-9; // m
const NMAX: usize = 2;

// const RABI_FREQ: f64 = 0.01e-3; // MHz
// const RABI_FREQ: f64 = 0.03e-3; // MHz
// const RABI_FREQ: f64 = 0.05e-3; // MHz
// const RABI_FREQ: f64 = 0.1e-3; // MHz
const RABI_FREQ: f64 = 0.3e-3; // MHz
// const RABI_FREQ: f64 = 0.5e-3; // MHz
// const RABI_FREQ: f64 = 0.7e-3; // MHz
// const RABI_FREQ: f64 = 1.0e-3; // MHz
// const RABI_FREQ: f64 = 1.2e-3; // MHz
// const RABI_FREQ: f64 = 3.0e-3; // MHz
// const RABI_FREQ: f64 = 5.0e-3; // MHz
// const RABI_FREQ: f64 = 10.0e-3; // MHz
// const RABI_FREQ: f64 = 30.0e-3; // MHz
// const RABI_FREQ: f64 = 50.0e-3; // MHz
const NSCAN: usize = 5;
const MAXITERS: usize = 100;
// const MAXITERS: usize = 500;
// const MAXITERS: usize = 1000000;

// const DO_SCAN: bool = true;
const DO_SCAN: bool = false;
const INIT: ParamV<f64> =
    /* test */
    // paramv!(
    //     {{     6.086836,     0.392699, },
    //      {     3.239767,     0.392699, },
    //      {     1.816233,     0.392699, }}
    // );
    paramv!(
        {{1.036 * PI, 0.138 * PI, },
         {1.620 * PI, 0.417 * PI, },
         {1.036 * PI, 0.695 * PI, }}
    );
    // paramv!(
    //     {{   3.22774e0,   1.05058e0, },
    //      {   5.06378e0,   1.30860e0, },
    //      {   3.22765e0,   1.56781e0, }}
    // );

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PulseParamSel { Th, Ph }
use PulseParamSel::*;

#[derive(Copy, Clone, Debug, PartialEq)]
struct PulseParamV<T> {
    th: T, // pulse nutation angle
           // (not accounting for reduced Rabi frequency from using a sideband)
    ph: T, // pulse phase
}

impl PulseParamV<f64> {
    fn zeros() -> Self { Self { th: 0.0, ph: 0.0 } }

    fn ones() -> Self { Self { th: 1.0, ph: 1.0 } }

    fn ax(ax: PulseParamSel) -> Self {
        let mut ret = Self::zeros();
        match ax {
            Th => { ret.th = 1.0; },
            Ph => { ret.ph = 1.0; },
        }
        ret
    }

    fn dot(self, rhs: Self) -> f64 { self.th * rhs.th + self.ph * rhs.ph }

    fn step(mut self, ax: PulseParamSel, step: f64) -> Self {
        match ax {
            Th => { self.th += step; },
            Ph => { self.ph += step; },
        }
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PulseSel { U0, U1, U2 }
use PulseSel::*;

#[derive(Copy, Clone, Debug, PartialEq)]
struct ParamV<T> {
    u0: PulseParamV<T>,
    u1: PulseParamV<T>,
    u2: PulseParamV<T>,
}

impl ParamV<f64> {
    fn zeros() -> Self {
        Self {
            u0: PulseParamV::zeros(),
            u1: PulseParamV::zeros(),
            u2: PulseParamV::zeros(),
        }
    }

    fn ones() -> Self {
        Self {
            u0: PulseParamV::ones(),
            u1: PulseParamV::ones(),
            u2: PulseParamV::ones(),
        }
    }

    fn ax(pulse: PulseSel, param: PulseParamSel) -> Self {
        let mut u0: PulseParamV<f64>;
        let mut u1: PulseParamV<f64>;
        let mut u2: PulseParamV<f64>;
        match pulse {
            U0 => {
                u0 = PulseParamV::ax(param);
                u1 = PulseParamV::zeros();
                u2 = PulseParamV::zeros();
            },
            U1 => {
                u0 = PulseParamV::zeros();
                u1 = PulseParamV::ax(param);
                u2 = PulseParamV::zeros();
            },
            U2 => {
                u0 = PulseParamV::zeros();
                u1 = PulseParamV::zeros();
                u2 = PulseParamV::ax(param);
            },
        }
        Self { u0, u1, u2 }
    }

    fn dot(self, rhs: Self) -> f64 {
        self.u0.dot(rhs.u0)
        + self.u1.dot(rhs.u1)
        + self.u2.dot(rhs.u2)
    }

    fn step(mut self, pulse: PulseSel, param: PulseParamSel, step: f64)
        -> Self
    {
        match pulse {
            U0 => { self.u0 = self.u0.step(param, step); },
            U1 => { self.u1 = self.u1.step(param, step); },
            U2 => { self.u2 = self.u2.step(param, step); },
        }
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct PhaseV {
    e: f64,
    n: f64,
    m: f64,
}

impl PhaseV {
    fn zeros() -> Self {
        Self { e: 0.0, n: 0.0, m: 0.0 }
    }

    fn ones() -> Self {
        Self { e: 1.0, n: 1.0, m: 1.0 }
    }

    fn dot(self, rhs: Self) -> f64 {
        self.e * rhs.e
        + self.n * rhs.n
        + self.m * rhs.m
    }
}

macro_rules! impl_pairwise_op {
    (
        $t:ty,
        $trait:ident,
        $trait_fn:ident,
        $trait_assign:ident,
        $trait_assign_fn:ident,
        $op_assign:tt,
        $( $field:ident ),*
    ) => {
        impl $trait_assign for $t {
            fn $trait_assign_fn(&mut self, rhs: Self) {
                $( self.$field $op_assign rhs.$field; )*
            }
        }

        impl $trait for $t {
            type Output = Self;

            fn $trait_fn(mut self, rhs: Self) -> Self::Output {
                self $op_assign rhs;
                self
            }
        }
    }
}
impl_pairwise_op!(PulseParamV<f64>, Add, add, AddAssign, add_assign, +=, th, ph);
impl_pairwise_op!(PulseParamV<f64>, Sub, sub, SubAssign, sub_assign, -=, th, ph);
impl_pairwise_op!(ParamV<f64>, Add, add, AddAssign, add_assign, +=, u0, u1, u2);
impl_pairwise_op!(ParamV<f64>, Sub, sub, SubAssign, sub_assign, -=, u0, u1, u2);
impl_pairwise_op!(PhaseV, Add, add, AddAssign, add_assign, +=, e, n, m);
impl_pairwise_op!(PhaseV, Sub, sub, SubAssign, sub_assign, -=, e, n, m);

macro_rules! impl_scalar_mul {
    ( $t:ty, $( $field:ident ),* ) => {
        impl MulAssign<f64> for $t {
            fn mul_assign(&mut self, rhs: f64) {
                $( self.$field *= rhs; )*
            }
        }

        impl Mul<f64> for $t {
            type Output = Self;

            fn mul(mut self, rhs: f64) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl Mul<$t> for f64 {
            type Output = $t;

            fn mul(self, mut rhs: $t) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    }
}
impl_scalar_mul!(PulseParamV<f64>, th, ph);
impl_scalar_mul!(ParamV<f64>, u0, u1, u2);
impl_scalar_mul!(PhaseV, e, n, m);

macro_rules! impl_displaylike {
    ( $t:ty, $trait:ident, $( $field:ident ),* ) => {
        impl $trait for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{{ ")?;
                $( $trait::fmt(&self.$field, f)?; write!(f, ", ")?; )*
                write!(f, "}}")?;
                Ok(())
            }
        }
    }
}
impl_displaylike!(PulseParamV<f64>, Display, th, ph);
impl_displaylike!(PulseParamV<f64>, LowerExp, th, ph);
impl_displaylike!(PhaseV, Display, e, n, m);
impl_displaylike!(PhaseV, LowerExp, e, n, m);

macro_rules! impl_displaylike_paramv {
    ( $trait:ident ) => {
        impl $trait for ParamV<f64> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{{")?;
                $trait::fmt(&self.u0, f)?;
                write!(f, ",\n ")?;
                $trait::fmt(&self.u1, f)?;
                write!(f, ",\n ")?;
                $trait::fmt(&self.u2, f)?;
                write!(f, "}}")?;
                Ok(())
            }
        }
    }
}
impl_displaylike_paramv!(Display);
impl_displaylike_paramv!(LowerExp);

fn make_unitary(params: ParamV<f64>) -> nd::Array2<C64> {
    let basis: Basis<State> =
        [
            (State::G0, 0.0),
            (State::G1, 0.0),
            (State::E0, 0.0),
            (State::E1, 0.0),
        ]
        .into_iter()
        .collect();

    // let rabi_freq: f64 = 0.01e-3; // MHz
    // let rabi_freq: f64 = 0.03e-3; // MHz
    // let rabi_freq: f64 = 0.05e-3; // MHz
    // let rabi_freq: f64 = 0.1e-3; // MHz
    // let rabi_freq: f64 = 0.3e-3; // MHz
    // let rabi_freq: f64 = 0.5e-3; // MHz
    // let rabi_freq: f64 = 0.7e-3; // MHz
    // let rabi_freq: f64 = 1.0e-3; // MHz
    // let rabi_freq: f64 = 3.0e-3; // MHz
    // let rabi_freq: f64 = 5.0e-3; // MHz
    // let rabi_freq: f64 = 10.0e-3; // MHz
    // let rabi_freq: f64 = 30.0e-3; // MHz
    // let rabi_freq: f64 = 50.0e-3; // MHz

    let rabi_freq: f64 = RABI_FREQ;

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(NMAX)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E1).unwrap()
        - basis.get_energy(&State::G0).unwrap()
        - State::TRAP_FREQ;
    // √(3/2) from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = rabi_freq * 1.5_f64.sqrt() * 1.060 * 2.9;

    let tau0: f64 = 2.0 * params.u0.th / TAU / rabi_freq;
    let tau1: f64 = 2.0 * params.u1.th / TAU / rabi_freq;
    let tau2: f64 = 2.0 * params.u2.th / TAU / rabi_freq;

    let t0 = 0.0;
    let t1 = t0 + tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;

    // pulse 0
    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.u0.ph,
    };
    let h0 = HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    // pulse 1
    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.u1.ph,
    };
    let h1 = HBuilderMagicTrap::new(&basis, drive1, polarization, motion);

    // pulse 2
    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.u2.ph,
    };
    let h2 = HBuilderMagicTrap::new(&basis, drive2, polarization, motion);

    let dt = (TAU / State::TRAP_FREQ).min(1.0 / rabi_freq) / 10.0;
    let nsteps = (t3 / dt).ceil() as usize;
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t3, nsteps + 1);
    let H: nd::Array3<C64> = h0.gen(&time) + h1.gen(&time) + h2.gen(&time);

    const ATOM_STATES: &[State] = &[State::G0, State::G1, State::E0, State::E1];
    let uni_size: usize = ATOM_STATES.len() * (NMAX + 1);
    let mut uni: nd::Array2<C64> = nd::Array2::zeros((uni_size, uni_size));
    for (k, mut col) in uni.columns_mut().into_iter().enumerate() {
        let init: nd::Array1<C64> =
            (0..uni_size)
            .map(|j| C64::from(if j == k { 1.0 } else { 0.0 }))
            .collect();
        let psi: nd::Array2<C64> = schrodinger::evolve_t(&init, &H, &time);
        col.assign(&psi.slice(nd::s![.., nsteps]));
    }
    uni
}

#[derive(Clone, Debug, PartialEq)]
struct TargetDiags {
    k: usize,
    elems: [C64; 8],
}

impl TargetDiags {
    fn new(phases: PhaseV) -> Self {
        let elems =
            [
                C64::cis(0.0                           ),
                C64::cis(                      phases.m),
                C64::cis(           phases.n           ),
                C64::cis(           phases.n + phases.m),
                C64::cis(phases.e                      ),
                C64::cis(phases.e            + phases.m),
                C64::cis(phases.e + phases.n           ),
                C64::cis(phases.e + phases.n + phases.m),
            ];
        Self { k: 7, elems }
    }
}

impl Iterator for TargetDiags {
    type Item = [C64; 8];

    fn next(&mut self) -> Option<Self::Item> {
        if !(0..8).contains(&self.k) { return None; }
        // if self.k > 0 { self.elems[self.k - 1] *= -1.0; }
        self.elems[self.k] *= -1.0;
        self.k += 1;
        Some(self.elems)
    }
}

impl ExactSizeIterator for TargetDiags {
    fn len(&self) -> usize { 8 - self.k }
}

impl std::iter::FusedIterator for TargetDiags { }

fn fidelity_phase(uni: &nd::Array2<C64>, phases: PhaseV) -> f64 {
    let diag = uni.diag();
    TargetDiags::new(phases)
        .map(|targ| {
            diag.iter().enumerate()
            .filter_map(|(k, u)| (k % (NMAX + 1) < 2).then_some(u))
            .zip(targ)
            .map(|(u, t)| *u * t)
            .sum::<C64>()
            .norm()
        })
        .max_by(|l, r| l.total_cmp(r))
        .unwrap()
        / 8.0
}

fn fidelity_phase_grad(uni: &nd::Array2<C64>, phases: PhaseV) -> (f64, PhaseV) {
    let diag = uni.diag();
    let (targ, tr, fid): ([C64; 8], C64, f64) =
        TargetDiags::new(phases)
        .map(|targ| {
            let tr: C64 =
                diag.iter().enumerate()
                .filter_map(|(k, u)| (k % (NMAX + 1) < 2).then_some(u))
                .zip(targ)
                .map(|(u, t)| *u * t)
                .sum();
            let fid = tr.norm() / 8.0;
            (targ, tr, fid)
        })
        .max_by(|(_, _, l), (_, _, r)| l.total_cmp(r))
        .unwrap();
    let grad_e =
        -(
            (
                targ[4] * diag[6]
                + targ[5] * diag[7]
                + targ[6] * diag[9]
                + targ[7] * diag[10]
            )
            * tr.conj()
        ).im / 64.0 / fid;
    let grad_n =
        -(
            (
                targ[2] * diag[3]
                + targ[3] * diag[4]
                + targ[6] * diag[9]
                + targ[7] * diag[10]
            )
            * tr.conj()
        ).im / 64.0 / fid;
    let grad_m =
        -(
            (
                targ[1] * diag[1]
                + targ[3] * diag[4]
                + targ[5] * diag[7]
                + targ[7] * diag[10]
            )
            * tr.conj()
        ).im / 64.0 / fid;
    (fid, PhaseV { e: grad_e, n: grad_n, m: grad_m })
}

#[allow(unused_variables)]
fn fidelity(params: ParamV<f64>) -> (f64, PhaseV) {
    const GAMMA: f64 = 0.1;
    let uni = make_unitary(params);
    let scanvals: nd::Array1<f64> = nd::Array1::linspace(0.0, TAU, 121);
    let mut phases = PhaseV::zeros();
    let mut max_fid = f64::NEG_INFINITY;
    let scan =
        scanvals.iter()
        .cartesian_product(scanvals.iter())
        .cartesian_product(scanvals.iter())
        .map(|((e, n), m)| PhaseV { e: *e, n: *n, m: *m });
    for test in scan {
        let fid = fidelity_phase(&uni, test);
        if fid > max_fid {
            max_fid = fid;
            phases = test;
        }
    }
    for k in 0..1_000_000_usize {
        let (fid, grad) = fidelity_phase_grad(&uni, phases);
        if grad.dot(grad).sqrt() < 1e-3 {
            // println!("\n{k} : {fid} {phases} {grad}");
            return (fid, phases);
        }
        phases += GAMMA * grad;
    }
    panic!("fidelity calculation did not converge!");
    // let fid = fidelity_phase(&uni, phases);
    // (fid, phases)
}

// macro_rules! grad_component {
//     (
//         $grad:ident,
//         $cur:ident,
//         $step:ident,
//         $pulse:ident,
//         $param:ident,
//         $pulse_sel:expr,
//         $param_sel:expr
//     ) => {
//         {
//             let pcur = $cur.step($pulse_sel, $param_sel,  $step.$pulse.$param);
//             let mcur = $cur.step($pulse_sel, $param_sel, -$step.$pulse.$param);
//             let pstep = fidelity(pcur).0;
//             let mstep = fidelity(mcur).0;
//             $grad.$pulse.$param = (pstep - mstep) / (2.0 * $step.$pulse.$param);
//         }
//     }
// }
//
// fn compute_grad(cur: ParamV<f64>, step: ParamV<f64>) -> ParamV<f64> {
//     let mut grad = ParamV::zeros();
//     grad_component!(grad, cur, step, u0, th, U0, Th);
//     grad_component!(grad, cur, step, u0, ph, U0, Ph);
//     grad_component!(grad, cur, step, u1, th, U1, Th);
//     grad_component!(grad, cur, step, u1, ph, U1, Ph);
//     grad_component!(grad, cur, step, u2, th, U2, Th);
//     grad_component!(grad, cur, step, u2, ph, U2, Ph);
//     grad
// }

fn compute_grad(cur: ParamV<f64>, step: ParamV<f64>) -> ParamV<f64> {
    let steps: Vec<f64> =
        [
            cur.step(U0, Th,  step.u0.th),
            cur.step(U0, Th, -step.u0.th),
            cur.step(U0, Ph,  step.u0.ph),
            cur.step(U0, Ph, -step.u0.ph),
            cur.step(U1, Th,  step.u1.th),
            cur.step(U1, Th, -step.u1.th),
            cur.step(U1, Ph,  step.u1.ph),
            cur.step(U1, Ph, -step.u1.ph),
            cur.step(U2, Th,  step.u2.th),
            cur.step(U2, Th, -step.u2.th),
            cur.step(U2, Ph,  step.u2.ph),
            cur.step(U2, Ph, -step.u2.ph),
        ]
        .into_par_iter()
        .map(|params| fidelity(params).0)
        .collect();
    ParamV {
        u0: PulseParamV {
            th: (steps[ 0] - steps[ 1]) / (2.0 * step.u0.th),
            ph: (steps[ 2] - steps[ 3]) / (2.0 * step.u0.ph),
        },
        u1: PulseParamV {
            th: (steps[ 4] - steps[ 5]) / (2.0 * step.u1.th),
            ph: (steps[ 6] - steps[ 7]) / (2.0 * step.u1.ph),
        },
        u2: PulseParamV {
            th: (steps[ 8] - steps[ 9]) / (2.0 * step.u2.th),
            ph: (steps[10] - steps[11]) / (2.0 * step.u2.ph),
        },
    }
}

fn learning_param(
    last: ParamV<f64>,
    last_grad: ParamV<f64>,
    cur: ParamV<f64>,
    cur_grad: ParamV<f64>,
) -> f64
{
    let diff = cur - last;
    let diff_grad = cur_grad - last_grad;
    diff.dot(diff_grad).abs() / diff_grad.dot(diff_grad)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum GradResult {
    Converged(ParamV<f64>),
    NotConverged(ParamV<f64>),
}

#[allow(unused_assignments, unused_variables, unused_mut)]
fn grad_ascent(
    init: ParamV<f64>,
    step: ParamV<f64>,
    init_learning_param: f64,
    eps: f64,
    maxiters: usize,
) -> GradResult
{
    let z = (maxiters as f64).log10().floor() as usize + 1;
    let mut last = init;
    let mut last_grad = compute_grad(init, step);
    let mut cur = init;
    let mut cur_grad = last_grad;
    let mut gamma = init_learning_param;
    let mut tau0: f64 = 2.0 * cur.u0.th / TAU / RABI_FREQ;
    let mut tau1: f64 = 2.0 * cur.u1.th / TAU / RABI_FREQ;
    let mut tau2: f64 = 2.0 * cur.u2.th / TAU / RABI_FREQ;

    eprintln!("\n\n\n\n\n\n\n\n\n");
    for k in 0..maxiters {
        let fid = fidelity(cur).0;
        if (1.0 - fid).abs() < eps {
            return GradResult::Converged(cur);
        }
        eprint!("\r\x1b[10A");
        eprintln!("{:w$} / {} : F = {:.9}, γ = {:.3e}  ",
            k, maxiters, fid, gamma, w=z);
        eprintln!("grad:\n{:+12.5e}", cur_grad);
        eprintln!("cur:\n{:+12.5e}", cur);
        eprintln!("time : {:.3} ms  ", (tau0 + tau1 + tau2) / 1000.0);
        cur += gamma * cur_grad;
        // cur.u0.th =
        //     if cur.u0.th < 0.0 { cur.u0.th.rem_euclid(TAU) } else { cur.u0.th };
        // cur.u1.th =
        //     if cur.u1.th < 0.0 { cur.u1.th.rem_euclid(TAU) } else { cur.u1.th };
        // cur.u2.th =
        //     if cur.u2.th < 0.0 { cur.u2.th.rem_euclid(TAU) } else { cur.u2.th };
        cur_grad = compute_grad(cur, step);
        gamma = learning_param(last, last_grad, cur, cur_grad);
        if gamma.is_nan() { gamma = init_learning_param; }
        last = cur;
        last_grad = cur_grad;
        tau0 = 2.0 * cur.u0.th / TAU / RABI_FREQ;
        tau1 = 2.0 * cur.u1.th / TAU / RABI_FREQ;
        tau2 = 2.0 * cur.u2.th / TAU / RABI_FREQ;
    }
    GradResult::NotConverged(cur)
}

fn neighborhood(scan: (f64, f64, usize)) -> nd::Array1<f64> {
    nd::Array1::linspace(scan.0, scan.1, scan.2)
}

fn ad_hoc_scan(scans: ParamV<(f64, f64, usize)>) -> ParamV<f64> {
    let vth0 = neighborhood(scans.u0.th);
    let vph0 = neighborhood(scans.u0.ph);
    let vth1 = neighborhood(scans.u1.th);
    let vph1 = neighborhood(scans.u1.ph);
    let vth2 = neighborhood(scans.u2.th);
    let vph2 = neighborhood(scans.u2.ph);
    let check: Vec<ParamV<f64>> =
        [
            vth0.as_slice().unwrap(),
            vph0.as_slice().unwrap(),
            vth1.as_slice().unwrap(),
            vph1.as_slice().unwrap(),
            vth2.as_slice().unwrap(),
            vph2.as_slice().unwrap(),
        ]
        .into_iter()
        .multi_cartesian_product()
        .map(|v| {
            let u0 = PulseParamV { th: *v[0], ph: *v[1] };
            let u1 = PulseParamV { th: *v[2], ph: *v[3] };
            let u2 = PulseParamV { th: *v[4], ph: *v[5] };
            ParamV { u0, u1, u2 }
        })
        .collect();

    let tot: usize =
        scans.u0.th.2 * scans.u0.ph.2
        * scans.u1.th.2 * scans.u1.ph.2
        * scans.u2.th.2 * scans.u2.ph.2;
    let z = (tot as f64).log10().floor() as usize + 1;
    static mut COUNTER: usize = 0;
    eprint!("\r {:w$} / {:w$} ", 0, tot, w=z);
    let max =
        check.into_par_iter()
        .map(|params| {
            let fid = fidelity(params).0;
            unsafe {
                COUNTER += 1;
                eprint!("\r {:w$} / {:w$} ", COUNTER, tot, w=z);
            }
            (params, fid)
        })
        .max_by(|l, r| l.1.total_cmp(&r.1))
        .unwrap();
    eprintln!();
    max.0
}

fn do_scan() {
    // let scans =
    //     ParamV {
    //         u0: PulseParamV {
    //             th: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //             ph: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //         },
    //         u1: PulseParamV {
    //             th: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //             ph: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //         },
    //         u2: PulseParamV {
    //             th: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //             ph: (PI / 16.0, 31.0 * PI / 16.0, NSCAN),
    //         },
    //     };

    let scans =
        ParamV {
            u0: PulseParamV {
                th: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
                ph: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
            },
            u1: PulseParamV {
                th: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
                ph: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
            },
            u2: PulseParamV {
                th: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
                ph: (PI / 8.0, 31.0 * PI / 16.0, NSCAN),
            },
        };

    // let scan_center =
    //     ParamV {
    //         u0: PulseParamV { th: 2.31281e0, ph:  3.44345e-2 },
    //         u1: PulseParamV { th: 2.24769e0, ph: -1.31174e-5 },
    //         u2: PulseParamV { th: 1.98230e0, ph:  3.72207e-2 },
    //     };
    // let scans =
    //     ParamV {
    //         u0: PulseParamV {
    //             th: ( 0.9 * scan_center.u0.th, 1.1 * scan_center.u0.th, 4),
    //             ph: ( 0.9 * scan_center.u0.ph, 1.1 * scan_center.u0.ph, 4),
    //         },
    //         u1: PulseParamV {
    //             th: ( 0.9 * scan_center.u1.th, 1.1 * scan_center.u1.th, 4),
    //             ph: ( 0.9 * scan_center.u1.ph, 1.1 * scan_center.u1.ph, 4),
    //         },
    //         u2: PulseParamV {
    //             th: ( 0.9 * scan_center.u2.th, 1.1 * scan_center.u2.th, 4),
    //             ph: ( 0.9 * scan_center.u2.ph, 1.1 * scan_center.u2.ph, 4),
    //         },
    //     };
    // let (fid0, _) = fidelity(scan_center);
    // println!("center: {:.9}", fid0);
    // println!();

    let scan_max = ad_hoc_scan(scans);
    let uni = make_unitary(scan_max);
    let (fid, phases) = fidelity(scan_max);
    println!("{:12.6}", scan_max);
    // println!("{:+.3}", uni);
    // println!("{:+.3}", uni_diag_phase(&uni, phases));
    println!("{:.9}", fid);
    println!("{:+12.5e}", phases);
}

fn do_grad_ascent() {
    let init =
        // ParamV {
        //     u0: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: PI / 2.0,
        //     },
        //     u1: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: 0.0,
        //     },
        //     u2: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: -PI / 2.0,
        //     },
        // };
        // ParamV {
        //     u0: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: 0.0,
        //     },
        //     u1: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: 0.0,
        //     },
        //     u2: PulseParamV {
        //         th: TAU / 3.0,
        //         ph: 0.0,
        //     },
        // };
        // ParamV {
        //     u0: PulseParamV {
        //         th: 5.89049,
        //         ph: 4.97419,
        //     },
        //     u1: PulseParamV {
        //         th: 5.89049,
        //         ph: 4.97419,
        //     },
        //     u2: PulseParamV {
        //         th: 3.92699e-1,
        //         ph: 2.22529,
        //     },
        // };
        INIT;
    let init_fid = fidelity(init).0;

    println!("{:+12.5e}", init);
    // println!("{:+.2}", make_unitary(init));
    println!("{:.9}", init_fid);
    let step = 1e-8 * ParamV::ones();
    let init_learning_param: f64 = 1.0;
    let eps: f64 = 1e-3;
    // let maxiters: usize = 100_000_000;
    let maxiters: usize = MAXITERS;
    let params =
        match grad_ascent(init, step, init_learning_param, eps, maxiters) {
            GradResult::Converged(params) => {
                println!("converged");
                params
            },
            GradResult::NotConverged(params) => {
                println!("not converged");
                params
            },
        };

    let uni = make_unitary(params);
    let (fid, phases) = fidelity(params);
    println!("{:12.6}", params);
    // println!("{:+.3}", uni);
    // println!("{:+.3}", uni_diag_phase(&uni, phases));
    println!("{:.9} {:+12.5e}", fid, phases);
    let tau0: f64 = 2.0 * params.u0.th / TAU / RABI_FREQ;
    let tau1: f64 = 2.0 * params.u1.th / TAU / RABI_FREQ;
    let tau2: f64 = 2.0 * params.u2.th / TAU / RABI_FREQ;
    println!("{:.6} ms", (tau0 + tau1 + tau2) / 1000.0);
}

fn uni_diag_phase(uni: &nd::Array2<C64>, phases: PhaseV) -> nd::Array1<C64> {
    let z =
        [
            C64::cis(0.0                           ),
            C64::cis(                      phases.m),
            C64::cis(           phases.n           ),
            C64::cis(           phases.n + phases.m),
            C64::cis(phases.e                      ),
            C64::cis(phases.e            + phases.m),
            C64::cis(phases.e + phases.n           ),
            C64::cis(phases.e + phases.n + phases.m),
        ];
    let diag = uni.diag();
    nd::array![
        z[0] * diag[0],
        z[1] * diag[1],
        z[2] * diag[3],
        z[3] * diag[4],
        z[4] * diag[6],
        z[5] * diag[7],
        z[6] * diag[9],
        z[7] * diag[10],
    ]
}

fn res() {
    let params =
        ParamV {
            u0: PulseParamV {
                th: 6.195317,
                ph: 0.322143,
            },
            u1: PulseParamV {
                th: 3.348247,
                ph: 0.384575,
            },
            u2: PulseParamV {
                th: 1.924716,
                ph: 0.471378,
            },
        };
    let uni = make_unitary(params);
    let (fid, phases) = fidelity(params);
    // let phases = PhaseV {
    //     e: 6.17531,
    //     n: 6.17531,
    //     m: 6.24016,
    // };
    println!("{:+12.6}", params);
    println!("{:+.3}", uni);
    println!("{:+.3}", uni_diag_phase(&uni, phases));
    println!("{:.9} {:+12.5e}", fid, phases);
    let tau0: f64 = 2.0 * params.u0.th / TAU / RABI_FREQ;
    let tau1: f64 = 2.0 * params.u1.th / TAU / RABI_FREQ;
    let tau2: f64 = 2.0 * params.u2.th / TAU / RABI_FREQ;
    println!("{:.6} ms", (tau0 + tau1 + tau2) / 1000.0);
}

fn main() {
    // if DO_SCAN {
    //     do_scan();
    // } else {
    //     do_grad_ascent();
    // }
    // do_scan();
    do_grad_ascent();
    // res();
}

