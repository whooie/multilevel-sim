#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU },
    path::PathBuf,
};
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    mkdir,
    write_npz,
    hilbert::Basis,
    dynamics::*,
    rabi::*,
};
use lib::systems::clock_motion::*;

const B: f64 = 120.0; // G
const M: f64 = 2.8384644058191703e-25; // kg
const T: f64 = 2e-6; // K
const WL: f64 = 578e-9; // m

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G0, TAU * zm(State::G0, B)),
            (State::G1, TAU * zm(State::G1, B)),
            (State::C0, TAU * zm(State::C0, B)),
            (State::C1, TAU * zm(State::C1, B)),
        ]
        .into_iter()
        .collect();
    let drive = DriveParams::Constant {
        frequency:
            basis.get_energy(&State::C1).unwrap()
            - basis.get_energy(&State::G0).unwrap()
            ,
        strength: TAU * 0.1e-3,
        phase: 0.0,
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-6 * T,
        fock_cutoff: Some(FockCutoff::Boltzmann(1e-4)),
    };

    let hbuilder = HBuilderMagicTrap::new(&basis, drive, polarization, motion);

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, 20000.0, 3000);
    let H = hbuilder.gen(&time);

    let rho0: nd::Array2<C64>
        = hbuilder.thermal_state_density(&State::G0).unwrap();
    // let rho0: nd::Array2<C64>
    //     = hbuilder.basis().get_density(&(State::G0, 0).into()).unwrap();
    let rho: nd::Array3<C64> = liouville_evolve_rk4(&rho0, &H, &time);
    write_npz!(
        outdir.join("clock_motion.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
        }
    );

    println!("done");
}
