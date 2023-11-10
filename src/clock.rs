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
    systems::clock_motion::*,
};

const B: f64 = 120.0; // G

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
        strength: TAU * 200e-3,
        phase: 0.0,
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };

    let hbuilder = HBuilder::new(&basis, drive, polarization);

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, 10.0, 1000);
    let H = hbuilder.gen(&time);

    // let psi0: nd::Array1<C64>
    //     = hbuilder.basis()
    //     .get_vector(&State::G0).unwrap();
    // let psi: nd::Array2<C64> = schrodinger_evolve_rk4(&psi0, &H, &time);
    // write_npz!(
    //     outdir.join("clock_motion.npz"),
    //     arrays: {
    //         "time" => &time,
    //         "psi" => &psi,
    //     }
    // );

    let rho0: nd::Array2<C64>
        = hbuilder.basis()
        .get_density(&State::G0).unwrap();
    let rho: nd::Array3<C64> = liouville_evolve_rk4(&rho0, &H, &time);
    write_npz!(
        outdir.join("clock.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
        }
    );

    println!("done");
}
