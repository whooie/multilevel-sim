#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU },
    path::PathBuf,
    rc::Rc,
};
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

const M: f64 = 2.8384644058191703e-25; // kg, 171Yb
const WL: f64 = 578e-9; // m

fn corpse_angles(theta: f64) -> [f64; 3] {
    let th = theta / 2.0;
    let asth = (th.sin() / 2.0).asin();
    let th1 = TAU + th - asth;
    let th2 = TAU - 2.0 * asth;
    let th3 = th - asth;
    [th1 * 180.0 / PI, th2 * 180.0 / PI, th3 * 180.0 / PI]
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Output {
    ndiff: f64,
    fidelity: f64,
    phase: f64,
}

fn do_mpp(
    rabi_freq: f64,
    init_state: Option<Fock<State>>,
    outdir: Option<PathBuf>,
) -> Output
{
    let angles = corpse_angles(PI / 2.0);
    // let rabi_freq: f64 = 800.0e-3; // MHz

    // let outdir = PathBuf::from("output");
    // mkdir!(outdir);

    let basis: Basis<State> =
        [
            (State::G0, 0.0),
            (State::G1, 0.0),
            (State::E0, 5000.0),
            (State::E1, 0.0),
        ]
        .into_iter()
        .collect();
    let nmax: usize = 20;

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(nmax)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E1).unwrap()
        - basis.get_energy(&State::G1).unwrap();
    let strength: f64 = rabi_freq * 3.0_f64.sqrt();

    let tau0: f64 = angles[0] / 360.0 / rabi_freq;
    let tau1: f64 = angles[1] / 360.0 / rabi_freq;
    let tau2: f64 = angles[2] / 360.0 / rabi_freq;

    let t0 = 0.0;
    let t1 = t0 + tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;
    let t4 = t3 + tau0;
    let t5 = t4 + tau1;
    let t6 = t5 + tau2;

    let drive00 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder00 =
        HBuilderMagicTrap::new(&basis, drive00, polarization, motion);

    let drive01 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PI,
    };
    let hbuilder01 =
        HBuilderMagicTrap::new(&basis, drive01, polarization, motion);

    let drive02 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder02 =
        HBuilderMagicTrap::new(&basis, drive02, polarization, motion);

    let drive10 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t3..t4).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder10 =
        HBuilderMagicTrap::new(&basis, drive10, polarization, motion);

    let drive11 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t4..t5).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PI,
    };
    let hbuilder11 =
        HBuilderMagicTrap::new(&basis, drive11, polarization, motion);

    let drive12 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t5..t6).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder12 =
        HBuilderMagicTrap::new(&basis, drive12, polarization, motion);
    
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t6, 10000);
    let H: nd::Array3<C64> =
        hbuilder00.gen(&time)
        + hbuilder01.gen(&time)
        + hbuilder02.gen(&time)
        + hbuilder10.gen(&time)
        + hbuilder11.gen(&time)
        + hbuilder12.gen(&time);

    let psi0: nd::Array1<C64> =
        if let Some(s0) = init_state {
            hbuilder00.basis().get_vector(&s0).unwrap()
        } else {
            hbuilder00.basis().get_vector(&(State::G1, 0).into()).unwrap()
                / 6.0_f64.sqrt()
            + hbuilder00.basis().get_vector(&(State::G1, 1).into()).unwrap()
                / 3.0_f64.sqrt()
            + hbuilder00.basis().get_vector(&(State::G0, 0).into()).unwrap()
                / 6.0_f64.sqrt()
            + hbuilder00.basis().get_vector(&(State::G0, 1).into()).unwrap()
                / 3.0_f64.sqrt()
        };
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    if let Some(dir) = outdir {
        let x = hbuilder00.gen_x();
        let p = hbuilder00.gen_p();
        let psi_conj = psi.mapv(|a| a.conj());
        let x_avg: nd::Array1<C64> =
            psi_conj.axis_iter(nd::Axis(1)).zip(psi.axis_iter(nd::Axis(1)))
            .map(|(ps_, ps)| ps_.dot(&x).dot(&ps))
            .collect();
        let p_avg: nd::Array1<C64> =
            psi_conj.axis_iter(nd::Axis(1)).zip(psi.axis_iter(nd::Axis(1)))
            .map(|(ps_, ps)| ps_.dot(&p).dot(&ps))
            .collect();
        write_npz!(
            dir.join("quoct_mpp.npz"),
            arrays: {
                "time" => &time,
                "psi" => &psi,
                "tmark" => &nd::array![t0, t1, t2, t3, t4, t5, t6],
                "x_avg" => &x_avg,
                "p_avg" => &p_avg,
            }
        );
    }

    let psi_last = psi.slice(nd::s![.., time.len() - 1]);

    let n0: f64 =
        hbuilder00.basis().keys()
        .zip(psi0.iter())
        .map(|(state, ak)| state.1 as f64 * ak.norm_sqr())
        .sum();
    let nlast: f64 =
        hbuilder00.basis().keys()
        .zip(psi_last.iter())
        .map(|(state, ak)| state.1 as f64 * ak.norm_sqr())
        .sum();

    let u_exp_atom: nd::Array2<C64> =
        nd::array![
            [ C64::from(1.0), C64::from(0.0), C64::from(0.0), C64::from(0.0) ],
            [ C64::from(0.0), C64::from(0.0), C64::from(0.0), C64::from(1.0) ],
            [ C64::from(0.0), C64::from(0.0), C64::from(1.0), C64::from(0.0) ],
            [ C64::from(0.0), C64::from(1.0), C64::from(0.0), C64::from(0.0) ],
        ];
    let u_exp: nd::Array2<C64> =
        nd::linalg::kron(&u_exp_atom, &nd::Array2::eye(nmax + 1));
    let psi_last_exp = u_exp.dot(&psi0);
    let dot: C64 =
        psi_last.iter()
        .zip(psi_last_exp.iter())
        .map(|(ak, ak_exp)| ak_exp.conj() * *ak)
        .sum();

    let phase: f64 = dot.im.atan2(dot.re);
    println!("{}", phase / std::f64::consts::PI * 180.0);

    Output { ndiff: nlast - n0, fidelity: dot.norm_sqr(), phase }
}

fn rabi_scan() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()
        .unwrap();

    let outdir = PathBuf::from("output");
    let nmax: usize = 2;
    let rabi_range: nd::Array1<f64> =
        nd::Array1::linspace(600.0e-3, 1600.0e-3, 101);

    let args: Vec<(usize, f64)> =
        (0..=nmax)
        .flat_map(|m0| rabi_range.iter().map(move |rabi| (m0, *rabi)))
        .collect();
    let len = args.len();
    let progress = std::sync::atomic::AtomicUsize::new(0);
    eprint!("  {} / {} ", 0, len);
    let (ndiff, fidelity): (Vec<f64>, Vec<f64>) =
        args.into_par_iter()
        // args.into_iter()
        .map(|(m0, rabi)| {
            let output = do_mpp(rabi, Some(Fock(State::G1, m0)), None);
            let k = progress.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            eprint!("\r  {} / {} ", k, len);
            (output.ndiff, output.fidelity)
        })
        .unzip();
    eprintln!();
    let ndiff: nd::Array2<f64> =
        nd::Array1::from_vec(ndiff)
        .into_shape((nmax + 1, rabi_range.len()))
        .unwrap();
    let fidelity: nd::Array2<f64> =
        nd::Array1::from_vec(fidelity)
        .into_shape((nmax + 1, rabi_range.len()))
        .unwrap();
    write_npz!(
        outdir.join("quoct_mpp_rabi_scan.npz"),
        arrays: {
            "rabi" => &rabi_range,
            "ndiff" => &ndiff,
            "fidelity" => &fidelity,
        }
    );
}

fn main() {
    let out = do_mpp(
        // 810.0e-3, Some(Fock(State::G1, 1)), Some(PathBuf::from("output")));
        810.0e-3, Some(Fock(State::E1, 0)), Some(PathBuf::from("output")));
    println!("{:?}", out);

    // rabi_scan();
}

