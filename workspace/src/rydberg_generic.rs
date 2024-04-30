#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::PI,
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
use lib::systems::rydberg_generic::*;

const N_ATOMS: usize = 1;

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G0, -1.0 * 2.0 * PI),
            (State::G1, 0.0),
            (State::G2, 1.0 * 2.0 * PI),
            (State::R, 0.0),
        ]
        .into_iter()
        .collect();
    let drive = DriveParams::Constant {
        frequency:
            basis.get_energy(&State::R).unwrap()
            - basis.get_energy(&State::G0).unwrap()
            ,
        strength: 2.0 * PI * 1.0,
        phase: 0.0,
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let rydcoupling = RydbergCoupling::AllToAll(R_SEP);
    let hbuilder = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive, polarization),
        N_ATOMS,
        rydcoupling,
    );
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, 2.0, 1000);
    let H = hbuilder.gen(&time);
    let psi0: nd::Array1<C64>
        = hbuilder.prod_basis()
        .get_vector(&(0..N_ATOMS).map(|_| State::G0).collect::<Vec<State>>())
        .unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    write_npz!(
        outdir.join("rydberg_generic.npz"),
        arrays: {
            "time" => &time,
            "psi" => &psi,
        }
    );
    println!("done");
}
