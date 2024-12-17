#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU },
    path::PathBuf,
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
use lib::systems::motion_simple::*;

const M: f64 = 2.8384644058191703e-25; // kg
const WL: f64 = 578e-9; // m
const NMAX: usize = 2;
const RABI_FREQ: f64 = 0.1e-3; // MHz

#[derive(Copy, Clone, Debug, PartialEq)]
struct Params {
    th0: f64,
    ph0: f64,
    th1: f64,
    ph1: f64,
    th2: f64,
    ph2: f64,
}

impl Params {
    fn step_th0(mut self, step: f64) -> Self {
        self.th0 += step;
        self
    }

    fn step_ph0(mut self, step: f64) -> Self {
        self.ph0 += step;
        self
    }

    fn step_th1(mut self, step: f64) -> Self {
        self.th1 += step;
        self
    }

    fn step_ph1(mut self, step: f64) -> Self {
        self.ph1 += step;
        self
    }

    fn step_th2(mut self, step: f64) -> Self {
        self.th2 += step;
        self
    }

    fn step_ph2(mut self, step: f64) -> Self {
        self.ph2 += step;
        self
    }

    fn step(mut self, gamma: f64, step: Self) -> Self {
        self.th0 += gamma * step.th0;
        self.ph0 += gamma * step.ph0;
        self.th1 += gamma * step.th1;
        self.ph1 += gamma * step.ph1;
        self.th2 += gamma * step.th2;
        self.ph2 += gamma * step.ph2;
        self
    }

    fn diff(mut self, rhs: Self) -> Self {
        self.th0 -= rhs.th0;
        self.ph0 -= rhs.ph0;
        self.th1 -= rhs.th1;
        self.ph1 -= rhs.ph1;
        self.th2 -= rhs.th2;
        self.ph2 -= rhs.ph2;
        self
    }

    fn dot(self, other: Self) -> f64 {
        self.th0 * other.th0
        + self.ph0 * other.ph0
        + self.th1 * other.th1
        + self.ph1 * other.ph1
        + self.th2 * other.th2
        + self.ph2 * other.ph2
    }
}

impl std::fmt::Display for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        self.th0.fmt(f)?;
        write!(f, ", ")?;
        self.ph0.fmt(f)?;
        write!(f, ", ")?;
        self.th1.fmt(f)?;
        write!(f, ", ")?;
        self.ph1.fmt(f)?;
        write!(f, ", ")?;
        self.th2.fmt(f)?;
        write!(f, ", ")?;
        self.ph2.fmt(f)?;
        write!(f, " }}")?;
        Ok(())
    }
}

impl std::fmt::LowerExp for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        self.th0.fmt(f)?;
        write!(f, ", ")?;
        self.ph0.fmt(f)?;
        write!(f, ", ")?;
        self.th1.fmt(f)?;
        write!(f, ", ")?;
        self.ph1.fmt(f)?;
        write!(f, ", ")?;
        self.th2.fmt(f)?;
        write!(f, ", ")?;
        self.ph2.fmt(f)?;
        write!(f, " }}")?;
        Ok(())
    }
}

fn do_sim(params: Params, init: &nd::Array1<C64>) -> nd::Array1<C64> {
    let basis: Basis<State> =
        [(State::G, 0.0), (State::E, 0.0)]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(2)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E).unwrap()
        - basis.get_energy(&State::G).unwrap()
        // - State::TRAP_FREQ;
        + State::TRAP_FREQ;
    // √3 from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = RABI_FREQ * 3.0_f64.sqrt() * 1.060 * 2.9;

    let tau0: f64 = 2.0 * params.th0 / TAU / RABI_FREQ;
    let tau1: f64 = 2.0 * params.th1 / TAU / RABI_FREQ;
    let tau2: f64 = 2.0 * params.th2 / TAU / RABI_FREQ;

    let t0 = 0.0;
    let t1 = t0 + tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;

    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.ph0,
    };
    let hbuilder0 =
        HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.ph1,
    };
    let hbuilder1 =
        HBuilderMagicTrap::new(&basis, drive1, polarization, motion);

    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: params.ph2,
    };
    let hbuilder2 =
        HBuilderMagicTrap::new(&basis, drive2, polarization, motion);

    let dt = (TAU / State::TRAP_FREQ).min(1.0 / RABI_FREQ) / 10.0;
    let nsteps = (t3 / dt).ceil() as usize;
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t3, nsteps + 1);
    let H: nd::Array3<C64> =
        hbuilder0.gen(&time)
        + hbuilder1.gen(&time)
        + hbuilder2.gen(&time);
    let psi: nd::Array2<C64> = schrodinger::evolve_t(init, &H, &time);
    psi.slice(nd::s![.., nsteps]).to_owned()
}

fn make_unitary(params: Params) -> nd::Array2<C64> {
    const ATOM_STATES: &[State] = &[State::G, State::E];
    let uni_size: usize = ATOM_STATES.len() * (NMAX + 1);
    let mut uni: nd::Array2<C64> = nd::Array2::zeros((uni_size, uni_size));
    for (k, col) in uni.columns_mut().into_iter().enumerate() {
        let init: nd::Array1<C64> =
            (0..uni_size)
            .map(|j| if j == k { C64::from(1.0) } else { C64::from(0.0) })
            .collect();
        do_sim(params, &init).move_into(col);
    }
    uni
}

fn fidelity_phase(uni: &nd::Array2<C64>, ph_e: f64, ph_m: f64) -> f64 {
    /* e<->m swap */
    // don't care about what happens to any of the n=2 states
    // let aph = C64::cis(ph_e);
    // let bph = C64::cis(ph_m);
    // let abph = C64::cis(ph_e + ph_m);
    // let tr =
    //     uni[[0, 0]]
    //     + bph * uni[[1, 3]]
    //     + aph * uni[[3, 1]]
    //     + abph * uni[[4, 4]];
    // tr.norm() / 4.0

    /* e-m shelve */
    // don't care about what happens to the g,2 state
    let aph = C64::cis(ph_e);
    let bph = C64::cis(ph_m);
    let abph = C64::cis(ph_e + ph_m);
    let a2bph = C64::cis(ph_e + 2.0 * ph_m);
    let tr =
        uni[[0, 0]]
        + bph * uni[[1, 5]]
        + a2bph * uni[[5, 1]]
        + aph * uni[[3, 3]]
        + abph * uni[[4, 4]];
    tr.norm() / 5.0
}

fn fidelity_phase_grad(uni: &nd::Array2<C64>, ph_e: f64, ph_m: f64) -> (f64, f64) {
    /* e<->m swap */
    // let aph = C64::cis(ph_e);
    // let bph = C64::cis(ph_m);
    // let abph = C64::cis(ph_e + ph_m);
    // let tr =
    //     uni[[0, 0]]
    //     + bph * uni[[1, 3]]
    //     + aph * uni[[3, 1]]
    //     + abph * uni[[4, 4]];
    // let fid = tr.norm() / 4.0;
    // let grad_ph_e =
    //     -((aph * uni[[3, 1]] + abph * uni[[4, 4]]) * tr.conj()).im / 16.0 / fid;
    // let grad_ph_m =
    //     -((bph * uni[[1, 3]] + abph * uni[[4, 4]]) * tr.conj()).im / 16.0 / fid;
    // (grad_ph_e, grad_ph_m)

    /* e-m shelve */
    let aph = C64::cis(ph_e);
    let bph = C64::cis(ph_m);
    let abph = C64::cis(ph_e + ph_m);
    let a2bph = C64::cis(ph_e + 2.0 * ph_m);
    let tr =
        uni[[0, 0]]
        + bph * uni[[1, 5]]
        + a2bph * uni[[5, 1]]
        + aph * uni[[3, 3]]
        + abph * uni[[4, 4]];
    let fid = tr.norm() / 5.0;
    let grad_ph_e =
        -(
            (a2bph * uni[[5, 1]] + aph * uni[[3, 3]] + abph * uni[[4, 4]])
            * tr.conj()
        ).im / 25.0 / fid;
    let grad_ph_m =
        -(
            (bph * uni[[1, 5]] + 2.0 * a2bph * uni[[5, 1]] + abph * uni[[4, 4]])
            * tr.conj()
        ).im / 25.0 / fid;
    (grad_ph_e, grad_ph_m)
}

#[allow(unused_variables)]
fn fidelity(params: Params) -> (f64, f64, f64) {
    const GAMMA: f64 = 0.1;
    let uni = make_unitary(params);
    let scanvals: nd::Array1<f64> = nd::Array1::linspace(0.0, TAU, 121);
    let mut a: f64 = 0.0;
    let mut b: f64 = 0.0;
    let mut max_fid = f64::NEG_INFINITY;
    let scan = scanvals.iter().cartesian_product(scanvals.iter());
    for (&a_test, &b_test) in scan {
        let fid = fidelity_phase(&uni, a_test, b_test);
        if fid > max_fid {
            max_fid = fid;
            a = a_test;
            b = b_test;
        }
    }
    for k in 0..1000000_usize {
        let fid = fidelity_phase(&uni, a, b);
        let (grad_a, grad_b) = fidelity_phase_grad(&uni, a, b);
        if (grad_a * grad_a + grad_b * grad_b).sqrt() < 1e-3 {
            // println!("\n{k} : {fid} {a} {b} {grad_a} {grad_b}");
            return (fid, a, b);
        }
        a += GAMMA * grad_a;
        b += GAMMA * grad_b;
    }
    panic!("fidelity calculation did not converge!");
}

fn compute_grad(cur: Params, step: Params) -> Params {
    let mut grad =
        Params {
            th0: 0.0, ph0: 0.0,
            th1: 0.0, ph1: 0.0,
            th2: 0.0, ph2: 0.0,
        };
    let th0p = fidelity(cur.step_th0( step.th0)).0;
    let th0m = fidelity(cur.step_th0(-step.th0)).0;
    grad.th0 = (th0p - th0m) / (2.0 * step.th0);
    let ph0p = fidelity(cur.step_ph0( step.ph0)).0;
    let ph0m = fidelity(cur.step_ph0(-step.ph0)).0;
    grad.ph0 = (ph0p - ph0m) / (2.0 * step.ph0);
    let th1p = fidelity(cur.step_th1( step.th1)).0;
    let th1m = fidelity(cur.step_th1(-step.th1)).0;
    grad.th1 = (th1p - th1m) / (2.0 * step.th1);
    let ph1p = fidelity(cur.step_ph1( step.ph1)).0;
    let ph1m = fidelity(cur.step_ph1(-step.ph1)).0;
    grad.ph1 = (ph1p - ph1m) / (2.0 * step.ph1);
    let th2p = fidelity(cur.step_th2( step.th2)).0;
    let th2m = fidelity(cur.step_th2(-step.th2)).0;
    grad.th2 = (th2p - th2m) / (2.0 * step.th2);
    let ph2p = fidelity(cur.step_ph2( step.ph2)).0;
    let ph2m = fidelity(cur.step_ph2(-step.ph2)).0;
    grad.ph2 = (ph2p - ph2m) / (2.0 * step.ph2);
    grad
}

fn learning_param(
    last: Params,
    last_grad: Params,
    cur: Params,
    cur_grad: Params,
) -> f64
{
    let diff = cur.diff(last);
    let diff_grad = cur_grad.diff(last_grad);
    diff.dot(diff_grad).abs() / diff_grad.dot(diff_grad)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum GradResult {
    Converged(Params),
    NotConverged(Params),
}

#[allow(unused_assignments, unused_variables, unused_mut)]
fn grad_ascent(
    init: Params,
    step: Params,
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

    eprintln!("\n\n");
    for k in 0..maxiters {
        let fid = fidelity(cur).0;
        if (1.0 - fid).abs() < eps {
            return GradResult::Converged(cur);
        }
        eprint!("\r\x1b[3A");
        eprintln!("  {:w$} / {} : F = {:.9}, γ = {:.3e}", k, maxiters, fid, gamma, w=z);
        eprintln!("  grad : {:+12.5e}", cur_grad);
        eprintln!("   cur : {:+12.5e}", cur);
        cur = cur.step(gamma, cur_grad);
        cur_grad = compute_grad(cur, step);
        // gamma = learning_param(last, last_grad, cur, cur_grad).max(init_learning_param);
        gamma = learning_param(last, last_grad, cur, cur_grad);
        if gamma.is_nan() { gamma = init_learning_param; }
        last = cur;
        last_grad = cur_grad;
    }
    GradResult::NotConverged(cur)
}

fn init_scan(npoints: usize) -> (Params, f64) {
    const MIN: f64 = PI / 8.0;
    const MAX: f64 = 15.0 * PI / 8.0;
    let vals: nd::Array1<f64> = nd::Array1::linspace(MIN, MAX, npoints);
    let check: Vec<Params> =
        (0..6)
        .map(|_| vals.iter())
        .multi_cartesian_product()
        .map(|p| {
            Params {
                th0: *p[0], ph0: *p[1],
                th1: *p[2], ph1: *p[3],
                th2: *p[4], ph2: *p[5],
            }
        })
        .collect();
    static mut COUNTER: usize = 0;
    eprint!("\r  0 ");
    let max =
        check.into_par_iter()
        .map(|params| {
            let fid = fidelity(params).0;
            unsafe {
                COUNTER += 1;
                eprint!("\r  {} ", COUNTER);
            }
            (params, fid)
        })
        .max_by(|l, r| l.1.total_cmp(&r.1))
        .unwrap();
    eprintln!();
    max
}

fn neighborhood(center: f64, margin: f64, npoints: usize) -> nd::Array1<f64> {
    nd::Array1::linspace(
        (1.0 - margin) * center, (1.0 + margin) * center, npoints)
}

fn ad_hoc_scan(center: Params, margin: f64, npoints: usize) -> (Params, f64) {
    let vals_th0 = neighborhood(center.th0, margin, npoints);
    let vals_ph0 = neighborhood(center.ph0, margin, npoints);
    let vals_th1 = neighborhood(center.th1, margin, npoints);
    let vals_ph1 = neighborhood(center.ph1, margin, npoints);
    let vals_th2 = neighborhood(center.th2, margin, npoints);
    let vals_ph2 = neighborhood(center.ph2, margin, npoints);
    let check: Vec<Params> =
        vals_th0.iter()
        .cartesian_product(&vals_ph0)
        .cartesian_product(&vals_th1)
        .cartesian_product(&vals_ph1)
        .cartesian_product(&vals_th2)
        .cartesian_product(&vals_ph2)
        .map(|(((((th0, ph0), th1), ph1), th2), ph2)| {
            Params {
                th0: *th0, ph0: *ph0,
                th1: *th1, ph1: *ph1,
                th2: *th2, ph2: *ph2,
            }
        })
        .collect();
    static mut COUNTER: usize = 0;
    eprint!("\r  0 ");
    let max =
        check.into_par_iter()
        .map(|params| {
            let fid = fidelity(params).0;
            unsafe {
                COUNTER += 1;
                eprint!("\r  {} ", COUNTER);
            }
            (params, fid)
        })
        .max_by(|l, r| l.1.total_cmp(&r.1))
        .unwrap();
    eprintln!();
    max
}

fn main() {
    // let (init, init_fid) = init_scan(7);

    // let init =
    //     Params {
    //         th0: 1.05 * 1.740e0, ph0: 1.05 * 5.629e-1,
    //         th1: 1.05 * 1.740e0, ph1: 1.05 * 4.915e0,
    //         th2: 1.05 * 2.985e0, ph2: 1.05 * 3.314e0,
    //     };
    // let init_fid = fidelity(init).0;

    // println!("{:.5e}", init);
    // println!("{:.9}", init_fid);
    // let step =
    //     Params {
    //         th0: 1e-6, ph0: 1e-6,
    //         th1: 1e-6, ph1: 1e-6,
    //         th2: 1e-6, ph2: 1e-6,
    //     };
    // let init_learning_param: f64 = 0.001;
    // let eps: f64 = 1e-3;
    // let maxiters: usize = 100_000_000;
    // let params =
    //     match grad_ascent(init, step, init_learning_param, eps, maxiters) {
    //         GradResult::Converged(params) => {
    //             println!("converged");
    //             params
    //         },
    //         GradResult::NotConverged(params) => {
    //             println!("not converged");
    //             params
    //         },
    //     };

    let ad_hoc =
        Params {
            th0: 1.364469853e0, ph0: 5.772125233e0,
            th1: 4.972480000e0, ph1: 3.868397600e0,
            th2: 3.985500533e-1, ph2: 1.439294750e0,
        };
    let (params, _) = ad_hoc_scan(ad_hoc, 0.005, 7);

    let uni = make_unitary(params);
    let fid = fidelity(params);
    println!("{:.9e}", params);
    println!("{:+.3}", uni);
    println!("{:.6?}", fid);
}

