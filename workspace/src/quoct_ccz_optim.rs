#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

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

const M: f64 = 2.8384644058191703e-25; // kg
const WL: f64 = 578e-9; // m
const NMAX: usize = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PulseParamSel { Tau, Rabi, Det, Phase, HVMix, HVPhase }
use PulseParamSel::*;

#[derive(Copy, Clone, Debug, PartialEq)]
struct PulseParamV {
    tau: f64, // pulse time; μs
    rabi: f64, // Rabi frequency; μs⁻¹
    det: f64, // pulse detuning; μs⁻¹
    phase: f64, // pulse phase
    hv_mix: f64, // H-V mixing angle
    hv_phase: f64, // H-V relative phase
}

impl PulseParamV {
    fn zeros() -> Self {
        Self {
            tau: 0.0,
            rabi: 0.0,
            det: 0.0,
            phase: 0.0,
            hv_mix: 0.0,
            hv_phase: 0.0,
        }
    }

    fn ones() -> Self {
        Self {
            tau: 1.0,
            rabi: 1.0,
            det: 1.0,
            phase: 1.0,
            hv_mix: 1.0,
            hv_phase: 1.0,
        }
    }

    fn ax(ax: PulseParamSel) -> Self {
        let mut ret = Self::zeros();
        match ax {
            Tau => { ret.tau = 1.0; },
            Rabi => { ret.rabi = 1.0; },
            Det => { ret.det = 1.0; },
            Phase => { ret.phase = 1.0; },
            HVMix => { ret.hv_mix = 1.0; },
            HVPhase => { ret.hv_phase = 1.0; },
        }
        ret
    }

    fn dot(self, rhs: Self) -> f64 {
        self.tau * rhs.tau
        + self.rabi * rhs.rabi
        + self.det * rhs.det
        + self.phase * rhs.phase
        + self.hv_mix * rhs.hv_mix
        + self.hv_phase * rhs.hv_phase
    }

    fn step(mut self, ax: PulseParamSel, step: f64) -> Self {
        match ax {
            Tau => { self.tau += step; },
            Rabi => { self.rabi += step; },
            Det => { self.det += step; },
            Phase => { self.phase += step; },
            HVMix => { self.hv_mix += step; },
            HVPhase => { self.hv_phase += step; },
        }
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PulseSel { U0, U1, U2 }
use PulseSel::*;

#[derive(Copy, Clone, Debug, PartialEq)]
struct ParamV {
    u0: PulseParamV,
    u1: PulseParamV,
    u2: PulseParamV,
}

impl ParamV {
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
        let mut u0: PulseParamV;
        let mut u1: PulseParamV;
        let mut u2: PulseParamV;
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
        $t:ident,
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
impl_pairwise_op!(PulseParamV, Add, add, AddAssign, add_assign, +=, tau, rabi, det, phase, hv_mix, hv_phase);
impl_pairwise_op!(PulseParamV, Sub, sub, SubAssign, sub_assign, -=, tau, rabi, det, phase, hv_mix, hv_phase);
impl_pairwise_op!(ParamV, Add, add, AddAssign, add_assign, +=, u0, u1, u2);
impl_pairwise_op!(ParamV, Sub, sub, SubAssign, sub_assign, -=, u0, u1, u2);
impl_pairwise_op!(PhaseV, Add, add, AddAssign, add_assign, +=, e, n, m);
impl_pairwise_op!(PhaseV, Sub, sub, SubAssign, sub_assign, -=, e, n, m);

macro_rules! impl_scalar_mul {
    ( $t:ident, $( $field:ident ),* ) => {
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
impl_scalar_mul!(PulseParamV, tau, rabi, det, phase, hv_mix, hv_phase);
impl_scalar_mul!(ParamV, u0, u1, u2);
impl_scalar_mul!(PhaseV, e, n, m);

macro_rules! impl_displaylike {
    ( $t:ident, $trait:ident, $( $field:ident ),* ) => {
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
impl_displaylike!(PulseParamV, Display, tau, rabi, det, phase, hv_mix, hv_phase);
impl_displaylike!(PulseParamV, LowerExp, tau, rabi, det, phase, hv_mix, hv_phase);
impl_displaylike!(PhaseV, Display, e, n, m);
impl_displaylike!(PhaseV, LowerExp, e, n, m);

macro_rules! impl_displaylike_paramv {
    ( $trait:ident ) => {
        impl $trait for ParamV {
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

fn do_sim(params: ParamV, init: &nd::Array1<C64>) -> nd::Array1<C64> {
    let basis: Basis<State> =
        [
            (State::G0, 0.0),
            (State::G1, 0.0),
            (State::E0, 0.0),
            (State::E1, 0.0),
        ]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(NMAX)),
    };

    let t0 = 0.0;
    let t1 = t0 + params.u0.tau;
    let t2 = t1 + params.u1.tau;
    let t3 = t2 + params.u2.tau;

    // pulse 0
    let pol0 = PolarizationParams::Poincare {
        alpha: params.u0.hv_mix,
        beta: params.u0.hv_phase,
        theta: PI / 2.0,
    };
    let freq0 = params.u0.det;
    let rabi0 = params.u0.rabi;
    let tau0 = params.u0.tau;
    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| freq0),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { rabi0 } else { 0.0 }
        }),
        phase: params.u0.phase,
    };
    let h0 = HBuilderMagicTrap::new(&basis, drive0, pol0, motion);

    // pulse 1
    let pol1 = PolarizationParams::Poincare {
        alpha: params.u1.hv_mix,
        beta: params.u1.hv_phase,
        theta: PI / 2.0,
    };
    let freq1 = params.u1.det;
    let rabi1 = params.u1.rabi;
    let tau1 = params.u1.tau;
    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| freq1),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { rabi1 } else { 0.0 }
        }),
        phase: params.u1.phase,
    };
    let h1 = HBuilderMagicTrap::new(&basis, drive1, pol1, motion);

    // pulse 2
    let pol2 = PolarizationParams::Poincare {
        alpha: params.u2.hv_mix,
        beta: params.u2.hv_phase,
        theta: PI / 2.0,
    };
    let freq2 = params.u2.det;
    let rabi2 = params.u2.rabi;
    let tau2 = params.u2.tau;
    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|_| freq2),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { rabi2 } else { 0.0 }
        }),
        phase: params.u2.phase,
    };
    let h2 = HBuilderMagicTrap::new(&basis, drive2, pol2, motion);

    let max_rabi_freq = rabi0.max(rabi1).max(rabi2);
    let dt = (TAU / State::TRAP_FREQ).min(TAU / max_rabi_freq) / 10.0;
    let nsteps = (t3 / dt).ceil() as usize;
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t3, nsteps + 1);
    let H: nd::Array3<C64> = h0.gen(&time) + h1.gen(&time) + h2.gen(&time);
    let psi: nd::Array2<C64> = schrodinger::evolve_t(init, &H, &time);
    psi.slice(nd::s![.., nsteps]).to_owned()
}

fn make_unitary(params: ParamV) -> nd::Array2<C64> {
    const ATOM_STATES: &[State] = &[State::G0, State::G1, State::E0, State::E1];
    let uni_size: usize = ATOM_STATES.len() * (NMAX + 1);
    let mut uni: nd::Array2<C64> = nd::Array2::zeros((uni_size, uni_size));
    for (k, col) in uni.columns_mut().into_iter().enumerate() {
        let init: nd::Array1<C64> =
            (0..uni_size)
            .map(|j| C64::from(if j == k { 1.0 } else { 0.0 }))
            .collect();
        do_sim(params, &init).move_into(col);
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
                C64::from(1.0),
                C64::cis(phases.n),
                C64::cis(phases.e),
                C64::cis(phases.e + phases.n),
                C64::cis(phases.m),
                C64::cis(phases.n + phases.m),
                C64::cis(phases.e + phases.m),
                C64::cis(phases.e + phases.n + phases.m),
            ];
        Self { k: 0, elems }
    }
}

impl Iterator for TargetDiags {
    type Item = [C64; 8];

    fn next(&mut self) -> Option<Self::Item> {
        if !(0..8).contains(&self.k) { return None; }
        if self.k > 0 { self.elems[self.k - 1] *= -1.0; }
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
            targ.into_iter().zip(&diag)
            .map(|(t, u)| t * *u)
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
                targ.into_iter().zip(&diag)
                .map(|(t, u)| t * *u)
                .sum();
            let fid = tr.norm() / 8.0;
            (targ, tr, fid)
        })
        .max_by(|(_, _, l), (_, _, r)| l.total_cmp(r))
        .unwrap();
    let grad_e =
        -(
            (targ[2] * diag[2] + targ[3] * diag[3] + targ[6] * diag[6] + targ[7] * diag[7])
            * tr.conj()
        ).im / 64.0 / fid;
    let grad_n =
        -(
            (targ[1] * diag[1] + targ[3] * diag[3] + targ[5] * diag[5] + targ[7] * diag[7])
            * tr.conj()
        ).im / 64.0 / fid;
    let grad_m =
        -(
            (targ[4] * diag[4] + targ[5] * diag[5] + targ[6] * diag[6] + targ[7] * diag[7])
            * tr.conj()
        ).im / 64.0 / fid;
    (fid, PhaseV { e: grad_e, n: grad_n, m: grad_m })
}

#[allow(unused_variables)]
fn fidelity(params: ParamV) -> (f64, PhaseV) {
    const GAMMA: f64 = 0.1;
    let uni = make_unitary(params);
    let scanvals: nd::Array1<f64> = nd::Array1::linspace(0.0, TAU, 51);
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
}

macro_rules! grad_component {
    (
        $grad:ident,
        $cur:ident,
        $step:ident,
        $pulse:ident,
        $param:ident,
        $pulse_sel:expr,
        $param_sel:expr
    ) => {
        {
            let pcur = $cur.step($pulse_sel, $param_sel,  $step.$pulse.$param);
            let mcur = $cur.step($pulse_sel, $param_sel, -$step.$pulse.$param);
            let pstep = fidelity(pcur).0;
            let mstep = fidelity(mcur).0;
            $grad.$pulse.$param = (pstep - mstep) / (2.0 * $step.$pulse.$param);
        }
    }
}

fn compute_grad(cur: ParamV, step: ParamV) -> ParamV {
    let mut grad = ParamV::zeros();
    grad_component!(grad, cur, step, u0, tau,      U0, Tau    );
    grad_component!(grad, cur, step, u0, rabi,     U0, Rabi   );
    grad_component!(grad, cur, step, u0, det,      U0, Det    );
    grad_component!(grad, cur, step, u0, phase,    U0, Phase  );
    grad_component!(grad, cur, step, u0, hv_mix,   U0, HVMix  );
    grad_component!(grad, cur, step, u0, hv_phase, U0, HVPhase);
    grad_component!(grad, cur, step, u1, tau,      U1, Tau    );
    grad_component!(grad, cur, step, u1, rabi,     U1, Rabi   );
    grad_component!(grad, cur, step, u1, det,      U1, Det    );
    grad_component!(grad, cur, step, u1, phase,    U1, Phase  );
    grad_component!(grad, cur, step, u1, hv_mix,   U1, HVMix  );
    grad_component!(grad, cur, step, u1, hv_phase, U1, HVPhase);
    grad_component!(grad, cur, step, u2, tau,      U2, Tau    );
    grad_component!(grad, cur, step, u2, rabi,     U2, Rabi   );
    grad_component!(grad, cur, step, u2, det,      U2, Det    );
    grad_component!(grad, cur, step, u2, phase,    U2, Phase  );
    grad_component!(grad, cur, step, u2, hv_mix,   U2, HVMix  );
    grad_component!(grad, cur, step, u2, hv_phase, U2, HVPhase);
    grad
}

fn learning_param(
    last: ParamV,
    last_grad: ParamV,
    cur: ParamV,
    cur_grad: ParamV,
) -> f64
{
    let diff = cur - last;
    let diff_grad = cur_grad - last_grad;
    diff.dot(diff_grad).abs() / diff_grad.dot(diff_grad)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum GradResult {
    Converged(ParamV),
    NotConverged(ParamV),
}

#[allow(unused_assignments, unused_variables, unused_mut)]
fn grad_ascent(
    init: ParamV,
    step: ParamV,
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

    eprintln!("\n\n\n\n\n\n\n\n");
    for k in 0..maxiters {
        let fid = fidelity(cur).0;
        if (1.0 - fid).abs() < eps {
            return GradResult::Converged(cur);
        }
        eprint!("\r\x1b[9A");
        eprintln!("{:w$} / {} : F = {:.9}, γ = {:.3e}",
            k, maxiters, fid, gamma, w=z);
        eprintln!("grad:\n{:+12.5e}", cur_grad);
        eprintln!("cur:\n{:+12.5e}", cur);
        cur += gamma * cur_grad;
        cur_grad = compute_grad(cur, step);
        gamma = learning_param(last, last_grad, cur, cur_grad);
        if gamma.is_nan() { gamma = init_learning_param; }
        last = cur;
        last_grad = cur_grad;
    }
    GradResult::NotConverged(cur)
}

fn main() {
    let init =
        ParamV {
            u0: PulseParamV {
                tau: 1000.0,
                rabi: 10e-3 * 3.0_f64.sqrt() * 1.060,
                det: 0.0,
                phase: PI / 2.0,
                hv_mix: 0.0,
                hv_phase: 0.0,
            },
            u1: PulseParamV {
                tau: 1000.0,
                rabi: 10e-3 * 3.0_f64.sqrt() * 2.9,
                det: -State::TRAP_FREQ,
                phase: PI / 2.0,
                hv_mix: 0.0,
                hv_phase: 0.0,
            },
            u2: PulseParamV {
                tau: 1000.0,
                rabi: 10e-3 * 3.0_f64.sqrt() * 2.9,
                det: State::TRAP_FREQ,
                phase: PI / 2.0,
                hv_mix: 0.0,
                hv_phase: 0.0,
            },
        };
    let init_fid = fidelity(init).0;

    println!("{:+12.5e}", init);
    // println!("{:+.2}", make_unitary(init));
    println!("{:.9}", init_fid);
    let step = 1e-6 * ParamV::ones();
    let init_learning_param: f64 = 0.001;
    let eps: f64 = 1e-3;
    let maxiters: usize = 100_000_000;
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
    println!("{:+12.5e}", params);
    println!("{:+.3}", uni);
    println!("{:.9}", fid);
    println!("{:+12.5e}", phases);
}

