#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU, FRAC_1_SQRT_2 as OVER_RT2 },
    path::PathBuf,
    rc::Rc,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    mkdir,
    write_npz,
    print_flush,
    println_flush,
    hilbert::{ Basis, Fock },
    dynamics::*,
    rabi::*,
    systems::ququart::{ *, State::* },
    utils::FExtremum,
};

const UNIT: f64 = 1e6; // working in MHz
const B: f64 = 120.0; // G
const OMEGA: f64 = 150e-3; // MHz; Kaufman omg paper

const M: f64 = 2.8384644058191703e-25; // kg
const T: f64 = 158e-9; // K; Kaufman omg paper
const WL: f64 = 578e-9; // m

const CORPSE_PI2_DEG: &[f64] = &[384.3, 318.6, 24.3];

#[derive(Clone, Debug)]
struct Output {
    time: nd::Array1<f64>,
    pulse_times: nd::Array1<f64>,
    rho: nd::Array3<C64>,
    basis: Basis<Fock<State>>,
}

fn doit_corpse(rabi_freq: f64, detuning: f64) -> Output {
    let basis: Basis<State>
        = [
            (G0, TAU * zm(C0, B)),
            (G1, TAU * zm(C1, B)),
            (C0, TAU * zm(C0, B)), // assume canceled diff. nuclear splitting
            (C1, TAU * zm(C1, B)), // assume canceled diff. nuclear splitting
            // (N0, TAU * zm(N0, B)),
            // (N1, TAU * zm(N1, B)),
            // (T0, TAU * zm(T0, B)),
            // (T1, TAU * zm(T1, B)),
            // (R0, TAU * zm(R0, B)),
            // (R1, TAU * zm(R1, B)),
            // (R2, TAU * zm(R2, B)),
            // (R3, TAU * zm(R3, B)),
        ]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: UNIT * M,
        wavelength: WL,
        temperature: T / UNIT,
        // fock_cutoff: Some(FockCutoff::Boltzmann(1e-5)),
        fock_cutoff: Some(FockCutoff::NMax(20)),
    };

    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };

    let frequency: f64
        = basis.get_energy(&C0).unwrap() - basis.get_energy(&G0).unwrap()
        + TAU * detuning;
    let strength: f64 = rabi_freq * 3.0_f64.sqrt();

    let tau0: f64 = CORPSE_PI2_DEG[0] / 360.0 / rabi_freq;
    let tau1: f64 = CORPSE_PI2_DEG[1] / 360.0 / rabi_freq;
    let tau2: f64 = CORPSE_PI2_DEG[2] / 360.0 / rabi_freq;
    println!("tau0 = {:.3}; tau1 = {:.3}; tau2 = {:.3}", tau0, tau1, tau2);

    let t0 = 0.0;
    let t1 = tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;
    let t4 = t3 + tau0;
    let t5 = t4 + tau1;
    let t6 = t5 + tau2;
    println!(
        "t1 = {:.3}; \
        t2 = {:.3}; \
        t3 = {:.3}; \
        t4 = {:.3}; \
        t5 = {:.3}; \
        t6 = {:.3}",
        t1, t2, t3, t4, t5, t6
    );

    let drive00 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder00
        = HBuilderMagicTrap::new(&basis, drive00, polarization, motion);

    let drive01 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PI,
    };
    let hbuilder01
        = HBuilderMagicTrap::new(&basis, drive01, polarization, motion);

    let drive02 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder02
        = HBuilderMagicTrap::new(&basis, drive02, polarization, motion);

    let drive10 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t3..t4).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder10
        = HBuilderMagicTrap::new(&basis, drive10, polarization, motion);

    let drive11 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t4..t5).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PI,
    };
    let hbuilder11
        = HBuilderMagicTrap::new(&basis, drive11, polarization, motion);

    let drive12 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t5..t6).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: 0.0,
    };
    let hbuilder12
        = HBuilderMagicTrap::new(&basis, drive12, polarization, motion);

    let time: nd::Array1<f64>
        = nd::Array1::range(
            0.0,
            t6,
            hbuilder00.basis().last().unwrap().1.recip() / 250.0,
        );
    let H: nd::Array3<C64>
        = hbuilder00.gen(&time)
        + hbuilder01.gen(&time)
        + hbuilder02.gen(&time)
        + hbuilder10.gen(&time)
        + hbuilder11.gen(&time)
        + hbuilder12.gen(&time);

    let rho0: nd::Array2<C64>
        = hbuilder00.basis()
        .get_density_weighted(
            |state, _index, _energy| {
                match (state.atomic_state(), state.fock_index()) {
                    // (&G0, 0) => C64::from(1.0),
                    // (&G1, 0) => C64::from(1.0),
                    (&G0, 0) | (&G1, 0) => C64::from(OVER_RT2),
                    _ => C64::from(0.0),
                }
            }
        );
    // let rho0: nd::Array2<C64>
    //     = hbuilder0.thermal_density_atomic(
    //         |state, _index, _energy| {
    //             match state {
    //                 G0 | G1 => C64::from(OVER_RT2),
    //                 _ => C64::from(0.0),
    //             }
    //         }
    //     );
    let rho: nd::Array3<C64> = liouville_evolve_rk4(&rho0, &H, &time);

    Output {
        time,
        pulse_times: nd::array![t0, t1, t2, t3, t4, t5, t6],
        rho,
        basis: hbuilder00.basis().clone(),
    }
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let output = doit_corpse(OMEGA, 0.0);
    write_npz!(
        outdir.join("ququart_clock_corpse.npz"),
        arrays: {
            "rabi_freq" => &nd::array![OMEGA],
            "time" => &output.time,
            "pulse_times" => &output.pulse_times,
            "rho" => &output.rho,
        }
    );

    println!("done");
}

