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
    utils::FExtremum,
};
use lib::systems::ququart::{ *, State::* };

const UNIT: f64 = 1e6; // working in MHz
const B: f64 = 120.0; // G
const OMEGA: f64 = 110e-3; // MHz; Kaufman omg paper

const M: f64 = 2.8384644058191703e-25; // kg
const T: f64 = 158e-9; // K; Kaufman omg paper
const WL: f64 = 578e-9; // m

fn doit_pi2pi2(
    rabi_freq: f64,
    detuning: f64,
    t0: f64,
    t1: f64,
) -> (nd::Array1<f64>, nd::Array3<C64>, Basis<Fock<State>>)
{
    let basis: Basis<State>
        = [
            (G0, TAU * zm(G0, B)),
            (G1, TAU * zm(G1, B)),
            (C0, TAU * zm(C0, B)),
            (C1, TAU * zm(C1, B)),
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

    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| {
            (
                basis.get_energy(&C0).unwrap()
                + basis.get_energy(&C1).unwrap()
            ) / 2.0
            - (
                basis.get_energy(&G0).unwrap()
                + basis.get_energy(&G1).unwrap()
            ) / 2.0
            + TAU * detuning
        }),
        strength: Rc::new(|t: f64| {
            if (0.0..t0).contains(&t) {
                TAU * rabi_freq * 3.0_f64.sqrt()
            } else {
                0.0
            }
        }),
        phase: 0.0,
    };
    let hbuilder0
        = HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| {
            (
                basis.get_energy(&C0).unwrap()
                + basis.get_energy(&C1).unwrap()
            ) / 2.0
            - (
                basis.get_energy(&G0).unwrap()
                + basis.get_energy(&G1).unwrap()
            ) / 2.0
            + TAU * detuning
        }),
        strength: Rc::new(|t: f64| {
            if (t0..t0 + t1).contains(&t) {
                TAU * rabi_freq * 3.0_f64.sqrt()
            } else {
                0.0
            }
        }),
        phase: PI,
    };
    let hbuilder1
        = HBuilderMagicTrap::new(&basis, drive1, polarization, motion);

    let time: nd::Array1<f64>
        = nd::Array1::range(
            0.0,
            t0 + t1,
            hbuilder0.basis().last().unwrap().1.recip() / 250.0,
        );
    let H: nd::Array3<C64> = hbuilder0.gen(&time) + hbuilder1.gen(&time);

    let rho0: nd::Array2<C64>
        = hbuilder0.basis()
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

    (time, rho, hbuilder0.basis().clone())
}

fn pulselen_scan_2d() {
    let mut nstates: usize = 0;
    let t0: nd::Array1<f64> = nd::Array1::linspace(0.15 / OMEGA, 0.35 / OMEGA, 25);
    let t1: nd::Array1<f64> = nd::Array1::linspace(0.65 / OMEGA, 0.85 / OMEGA, 25);
    let pulse_end_probs: nd::Array1<C64>
        = t0.iter().enumerate()
        .cartesian_product(t1.iter().enumerate())
        .flat_map(|((i, &t0i), (j, &t1j))| {
            print_flush!("\r{} {} ", i, j);
            let (time, rho, basis) = doit_pi2pi2(OMEGA, 0.0, t0i, t1j);
            nstates = basis.len();
            let nt = time.len();
            rho.slice(nd::s![.., .., nt - 1]).to_owned()
        })
        .collect();
    println_flush!("");
    let scanned: nd::Array4<C64>
        = pulse_end_probs.into_shape((t0.len(), t1.len(), nstates, nstates))
        .expect("error reshaping");
    write_npz!(
        PathBuf::from("output").join("ququart_clock_pi2pi2_pulselen_scan.npz"),
        arrays: {
            "t0" => &t0,
            "t1" => &t1,
            "scanned" => &scanned,
        }
    );
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    // pulselen_scan_2d();

    // let (time, rho, _) = doit_pi2pi2(OMEGA, 0.0, 1.5 / OMEGA, 0.5 / OMEGA);
    let (time, rho, _) = doit_pi2pi2(OMEGA, 0.0, 2.196970, 6.515152);
    write_npz!(
        outdir.join("ququart_clock_pi2pi2.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
        }
    );

    println!("done");
}

