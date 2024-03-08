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

const UNIT: f64 = 1e6;

const B: f64 = 120.0; // G

// const OMEGA: f64 = 200e-3; // MHz
const OMEGA: f64 = 110e-3; // MHz; Kaufman omg paper

const M: f64 = 2.8384644058191703e-25; // kg
// const T: f64 = 2e-6; // K
const T: f64 = 211e-9; // K; Kaufman omg paper
const WL: f64 = 578e-9; // m

fn compute_zm() {
    let field: nd::Array1<f64> = nd::Array1::linspace(0.0, 150.0, 1000);
    let g0 = field.mapv(|b| zm(G0, b));
    let g1 = field.mapv(|b| zm(G1, b));
    let c0 = field.mapv(|b| zm(C0, b));
    let c1 = field.mapv(|b| zm(C1, b));
    write_npz!(
        PathBuf::from("output").join("ququart_clock_energies.npz"),
        arrays: {
            "B" => &field,
            "G0" => &g0,
            "G1" => &g1,
            "C0" => &c0,
            "C1" => &c1,
        }
    );
}

fn doit_1beam(
    rabi_freq: f64,
    detuning: f64,
    tmax: Option<f64>,
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
        fock_cutoff: Some(FockCutoff::Boltzmann(1e-5)),
    };

    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let drive0 = DriveParams::Constant {
        frequency:
            (
                basis.get_energy(&C0).unwrap()
                + basis.get_energy(&C1).unwrap()
            ) / 2.0
            - (
                basis.get_energy(&G0).unwrap()
                + basis.get_energy(&G1).unwrap()
            ) / 2.0
            + TAU * detuning
            ,
        strength: TAU * rabi_freq,
        phase: 0.0,
    };
    let hbuilder0
        = HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    let time: nd::Array1<f64>
        = nd::Array1::range(
            0.0,
            tmax.unwrap_or(2.0 / rabi_freq),
            hbuilder0.basis().last().unwrap().1.recip() / 250.0,
        );
    let H: nd::Array3<C64> = hbuilder0.gen(&time);

    // let rho0: nd::Array2<C64>
    //     = hbuilder0.basis()
    //     .get_vector_density_weighted(
    //         |state, _index, _energy| {
    //             match (state.atomic_state(), state.fock_index()) {
    //                 // (&G0, 0) => C64::from(1.0),
    //                 // (&G1, 0) => C64::from(1.0),
    //                 (&G0, 0) | (&G1, 0) => C64::from(OVER_RT2),
    //                 _ => C64::from(0.0),
    //             }
    //         }
    //     );
    let rho0: nd::Array2<C64>
        = hbuilder0.thermal_density_atomic(
            |state, _index, _energy| {
                match state {
                    G0 | G1 => C64::from(OVER_RT2),
                    _ => C64::from(0.0),
                }
            }
        );
    let rho: nd::Array3<C64> = liouville_evolve_rk4(&rho0, &H, &time);

    (time, rho, hbuilder0.basis().clone())
}

fn doit_2beam(
    rabi_freq: f64,
    detuning1: f64,
    detuning2: f64,
    tmax: Option<f64>,
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

    // // σ±, two beams
    // let pol_g0c1 = PolarizationParams::Poincare {
    //     alpha: PI / 4.0,
    //     beta: PI / 2.0,
    //     theta: 0.0,
    // };
    // let drive_g0c1 = DriveParams::Constant {
    //     frequency:
    //         basis.get_energy(&C1).unwrap() - basis.get_energy(&G0).unwrap()
    //         + TAU * detuning1
    //         ,
    //     strength: TAU * rabi_freq / (2.0_f64 / 3.0).sqrt(),
    //     phase: 0.0,
    // };
    // let pulse_g0c1
    //     = HBuilderMagicTrap::new(&basis, drive_g0c1, pol_g0c1, motion);
    // let pol_g1c0 = PolarizationParams::Poincare {
    //     alpha: PI / 4.0,
    //     beta: -PI / 2.0,
    //     theta: 0.0,
    // };
    // let drive_g1c0 = DriveParams::Constant {
    //     frequency:
    //         basis.get_energy(&C0).unwrap() - basis.get_energy(&G1).unwrap()
    //         + TAU * detuning2
    //         ,
    //     strength: TAU * rabi_freq / (2.0_f64 / 3.0).sqrt(),
    //     phase: 0.0,
    // };
    // let pulse_g1c0
    //     = HBuilderMagicTrap::new(&basis, drive_g1c0, pol_g1c0, motion);
    // println!("nmax = {}", pulse_g0c1.nmax());
    // // println!("nmax = {}", pulse_g1c0.nmax());

    // ππ, two beams
    let pol_g0c0 = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let drive_g0c0 = DriveParams::Constant {
        frequency:
            basis.get_energy(&C0).unwrap() - basis.get_energy(&G0).unwrap()
            + TAU * detuning1
            ,
        strength: TAU * rabi_freq / (1.0_f64 / 3.0).sqrt(),
        phase: 0.0,
    };
    let pulse_g0c0
        = HBuilderMagicTrap::new(&basis, drive_g0c0, pol_g0c0, motion);
    let pol_g1c1 = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let drive_g1c1 = DriveParams::Constant {
        frequency:
            basis.get_energy(&C1).unwrap() - basis.get_energy(&G1).unwrap()
            + TAU * detuning2
            ,
        strength: TAU * rabi_freq / (1.0_f64 / 3.0).sqrt(),
        phase: 0.0,
    };
    let pulse_g1c1
        = HBuilderMagicTrap::new(&basis, drive_g1c1, pol_g1c1, motion);
    // println!("nmax = {}", pulse_g0c0.nmax());
    // println!("nmax = {}", pulse_g1c1.nmax());

    let time: nd::Array1<f64>
        = nd::Array1::range(
            0.0,
            tmax.unwrap_or(2.0 / rabi_freq),
            // pulse_g0c1.basis().last().unwrap().1.recip() / 250.0,
            pulse_g0c0.basis().last().unwrap().1.recip() / 250.0,
        );
    // println!("nt = {}, dt = {:.3e}", time.len(), time[1] - time[0]);
    let H: nd::Array3<C64>
        // = pulse_g0c1.gen(&time) + pulse_g1c0.gen(&time);
        = pulse_g0c0.gen(&time) + pulse_g1c1.gen(&time);

    let rho0: nd::Array2<C64>
        // = pulse_g0c1.basis()
        = pulse_g0c0.basis()
        .get_density_weighted_pure(
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
    //     = pulse_g0c1.thermal_density_atomic(
    //     // = pulse_g0c0.thermal_density_atomic(
    //         |state, _index, _energy| {
    //             match state {
    //                 G0 | G1 => C64::from(OVER_RT2),
    //                 _ => C64::from(0.0),
    //             }
    //         }
    //     );
    let rho: nd::Array3<C64> = liouville_evolve_rk4(&rho0, &H, &time);

    // (time, rho, pulse_g0c1.basis().clone())
    (time, rho, pulse_g0c0.basis().clone())
}

fn detuning_scan() {
    let detuning: nd::Array1<f64> = nd::Array1::linspace(-0.2, 1.0, 50);
    let pulse_end_state: Vec<nd::Array2<C64>>
        = detuning.iter().enumerate()
        .map(|(i, &det)| {
            print_flush!("\r{} ", i);
            let (time, rho, _) = doit_2beam(OMEGA, det, det, Some(3.0));
            let nt = time.len();
            rho.slice(nd::s![.., .., nt - 1]).to_owned()
        })
        .collect();
    println_flush!("");
    let scanned: nd::Array3<C64> = stack_arrays(nd::Axis(0), &pulse_end_state)
        .expect("error stacking arrays");
    write_npz!(
        PathBuf::from("output").join("ququart_clock_pipi_detscan.npz"),
        arrays: {
            "det" => &detuning,
            "scanned" => &scanned,
        }
    );
}

fn detuning_scan_2d() {
    let mut nstates: usize = 0;
    let detuning1: nd::Array1<f64> = nd::Array1::linspace(-0.2, 0.7, 20);
    let detuning2: nd::Array1<f64> = nd::Array1::linspace(-0.2, 0.7, 20);
    let pulse_end_probs: nd::Array1<C64>
        = detuning1.iter().enumerate()
        .cartesian_product(detuning2.iter().enumerate())
        .flat_map(|((i, &det1), (j, &det2))| {
            print_flush!("\r{} {} ", i, j);
            let (time, rho, basis) = doit_2beam(OMEGA, det1, det2, Some(3.0));
            nstates = basis.len();
            let nt = time.len();
            rho.slice(nd::s![.., .., nt - 1])
                .to_owned()
        })
        .collect();
    println_flush!("");
    let scanned: nd::Array4<C64>
        = pulse_end_probs.into_shape(
            (detuning1.len(), detuning2.len(), nstates, nstates))
        .expect("error reshaping");
    write_npz!(
        PathBuf::from("output").join("ququart_clock_pipi_det2scan.npz"),
        arrays: {
            "det1" => &detuning1,
            "det2" => &detuning2,
            "scanned" => &scanned,
        }
    );
}

fn detuning_pi_scan_2d() {
    let mut nstates: usize = 0;
    let detuning1: nd::Array1<f64> = nd::Array1::linspace(-0.1, 0.1, 25);
    let detuning2: nd::Array1<f64> = nd::Array1::linspace(-0.1, 0.1, 25);
    let pulse_pi_probs: nd::Array1<C64>
        = detuning1.iter().enumerate()
        .cartesian_product(detuning2.iter().enumerate())
        .flat_map(|((i, &det1), (j, &det2))| {
            print_flush!("\r{} {} ", i, j);
            let (time, rho, basis) = doit_2beam(OMEGA, det1, det2, Some(3.0));
            nstates = basis.len();
            let diags: Vec<nd::Array1<C64>>
                = rho.axis_iter(nd::Axis(2))
                .map(|rho_t| rho_t.diag().to_owned())
                .collect();
            let diags: nd::Array2<C64> = stack_arrays(nd::Axis(0), &diags)
                .expect("diag stacking error");
            let C0_selector: nd::Array1<C64>
                = basis.keys()
                .map(|fock| {
                    if fock.atomic_state() == &C0 {
                        1.0.into()
                    } else {
                        0.0.into()
                    }
                })
                .collect();
            let C1_selector: nd::Array1<C64>
                = basis.keys()
                .map(|fock| {
                    if fock.atomic_state() == &C1 {
                        1.0.into()
                    } else {
                        0.0.into()
                    }
                })
                .collect();
            let P_C0: nd::Array1<f64>
                = diags.dot(&C0_selector).mapv(|p| p.re);
            let P_C1: nd::Array1<f64>
                = diags.dot(&C1_selector).mapv(|p| p.re);
            let (k_pi, _) = (P_C0 * P_C1).fmax_idx()
                .expect("couldn't find pi time");
            // println!("{} {}", k_pi, time.len());
            rho.slice(nd::s![.., .., k_pi]).to_owned()
        })
        .collect();
    println_flush!("");
    let scanned: nd::Array4<C64>
        = pulse_pi_probs.into_shape(
            (detuning1.len(), detuning2.len(), nstates, nstates))
        .expect("error reshaping");
    write_npz!(
        PathBuf::from("output").join("ququart_clock_pipi_det2scan.npz"),
        arrays: {
            "det1" => &detuning1,
            "det2" => &detuning2,
            "scanned" => &scanned,
        }
    );
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    // compute_zm();

    // detuning_scan();

    // detuning_scan_2d();

    detuning_pi_scan_2d();

    // let (time, rho, _) = doit_1beam(OMEGA, 0.0, None);
    // let (time, rho, _) = doit_2beam(OMEGA, 0.025, -0.017, Some(3.0));
    // write_npz!(
    //     outdir.join("ququart_clock_pipi.npz"),
    //     arrays: {
    //         "time" => &time,
    //         "rho" => &rho,
    //     }
    // );

    println!("done");
}

