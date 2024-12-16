#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports)]

use std::{
    f64::consts::{ TAU, PI },
    hash::Hash,
    path::PathBuf,
    rc::Rc,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    c,
    mkdir,
    write_npz,
    hilbert::{ Basis, BasisState, SpinState, Cavity, HSpin },
    dynamics::*,
    rabi::*,
};
use lib::systems::cavity_array::{ *, State::* };


/// Call `print!` and immediately flush.
#[macro_export]
macro_rules! print_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        print!($fmt $(, $val )*);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

/// Call `println!` and immediately flush.
#[macro_export]
macro_rules! println_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        println!($fmt $(, $val, )*);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

const NATOMS: usize = 5;
const NMODES: usize = 1;
const MAX_PHOTONS: usize = 10;
const NMAX: [usize; NMODES] = [MAX_PHOTONS; NMODES];
const ARRAY_SPACING: f64 = 2.0; // μm

fn omega_z_0() -> f64 {
    OMEGA_1 - 0.5 * (OMEGA_DELTA_D - OMEGA_DELTA_E)
    - OMEGA_DRIVE_R * OMEGA_DRIVE_R / 4.0 / DELTA_R
    - 4.0 * J_Z
}

fn g_0() -> f64 {
    (MAX_PHOTONS as f64).sqrt()
    * (G_D + G_E)
    * (OMEGA_DRIVE_D + OMEGA_DRIVE_E)
    / (DELTA_D + DELTA_E)
    / 2.0
}

fn ground_photons(g: f64, wz: f64) -> f64 {
    let k_g = g / g_0();
    let omega_drive_d = OMEGA_DRIVE_D * k_g;
    let omega_drive_e = OMEGA_DRIVE_E * k_g;
    // let g_ =
    //     (MAX_PHOTONS as f64).sqrt()
    //     * (G_D + G_E)
    //     * (omega_drive_d + omega_drive_e)
    //     / (DELTA_D + DELTA_E)
    //     / 2.0;
    // println!("calculated g = {g_}");

    let d_wz = wz - omega_z_0();
    let delta_d = DELTA_D + d_wz;
    let delta_e = DELTA_E - d_wz;
    // let wz_ =
    //     OMEGA_1
    //     - 0.5 * ((OMEGA_D - delta_d) - (OMEGA_E - delta_e))
    //     - OMEGA_DRIVE_R.powi(2) / 4.0 / DELTA_R
    //     - 4.0 * J_Z;
    // println!("calculated wz = {wz_}");

    let atom_basis: Basis<State>
        = [
            (G0, TAU * OMEGA_0),
            (G1, TAU * OMEGA_1),
            (E0, TAU * OMEGA_E),
            (E1, TAU * OMEGA_D),
            (R,  TAU * OMEGA_R),
        ]
        .into_iter()
        .collect();

    let drive1 = DriveParams::Constant {
        frequency: TAU * (OMEGA_D - delta_d),
        strength:
            TAU * omega_drive_d
            / transition_cg(G0.spin(), E1.spin()).unwrap(),
        phase: 0.0,
    };
    let polarization1 = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let hbuilder1: HBuilderCavityRydberg<NATOMS, NMODES, State>
        = HBuilderCavityRydberg::new(
            &atom_basis, drive1, polarization1, NMAX, ARRAY_SPACING);
    // let h1 = hbuilder1.gen_static().unwrap();
    // println!("{:+.3e}", h1.view().split_complex().re);

    let drive2 = DriveParams::Constant {
        frequency: TAU * (OMEGA_E - delta_e),
        strength:
            TAU * omega_drive_e
            / transition_cg(G1.spin(), E0.spin()).unwrap(),
        phase: 0.0,
    };
    let polarization2 = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: -PI / 2.0,
        theta: 0.0,
    };
    let hbuilder2: HBuilderCavityRydberg<NATOMS, NMODES, State>
        = HBuilderCavityRydberg::new(
            &atom_basis, drive2, polarization2, NMAX, ARRAY_SPACING);
    // let h2 = hbuilder1.gen_static().unwrap();
    // println!("{:+.3e}", h2.view().split_complex().re);

    let overlay = OverlayBuilder::from_builders([hbuilder1, hbuilder2]);
    // let h = overlay.build_cavity_rydberg().unwrap();
    // println!("{:+.2e}", h.view().split_complex().re);
    let (_, gs) = overlay.ground_state_cavity_rydberg().unwrap();
    overlay.get_basis().unwrap()
        .keys()
        .zip(&gs)
        .map(|(sn, a)| {
            let n = sn.photons()[0] as f64;
            n * a.norm().powi(2)
        })
        .sum()
}

#[derive(Copy, Clone, Debug)]
struct GS {
    nbar: f64,
    mz: f64,
}

fn ground_state_spinchain(g: f64, wz: f64) -> GS {
    let builder: HBuilderTransverseIsing<NATOMS>
        = HBuilderTransverseIsing::new(wz, OMEGA_A, -J_Z, g, MAX_PHOTONS);
    let (_, gs) = builder.ground_state();
    let nbar
        = builder.basis()
        .keys()
        .zip(&gs)
        .map(|(sn, a)| {
            let n = sn.photons()[0] as f64;
            n * a.norm().powi(2)
        })
        .sum();
    let mz
        = builder.basis()
        .keys()
        .zip(&gs)
        .map(|(sn, a)| {
            let m
                = sn.atomic_states().iter()
                .map(|s| s.sz())
                .sum::<f64>() / NATOMS as f64;
            m * a.norm().powi(2)
        })
        .sum();
    GS { nbar, mz }
}

#[derive(Clone, Debug)]
struct Data {
    g: nd::Array1<f64>,
    wz: nd::Array1<f64>,
    nbar: nd::Array2<f64>,
    mz: nd::Array2<f64>,
}

fn phase_diagram_spinchain() -> Data {
    const GRIDSIZE: usize = 20;
    let g: nd::Array1<f64> = nd::Array1::linspace(0.0, 3.0, GRIDSIZE);
    let wz: nd::Array1<f64> = nd::Array1::linspace(-1.0, 1.0, GRIDSIZE);
    let nbar_mz: nd::Array2<C64>
        = nd::Array2::from_shape_fn(
            (GRIDSIZE, GRIDSIZE),
            |(i, j)| {
                print_flush!("\r  {:3}, {:3} ", i, j);
                let gs = ground_state_spinchain(g[j], wz[i]);
                C64::new(gs.nbar, gs.mz)
            }
        );
    let num_complex::Complex { re: nbar, im: mz }
        = nbar_mz.view().split_complex();
    Data { g, wz, nbar: nbar.to_owned(), mz: mz.to_owned() }
}

#[derive(Clone, Debug)]
struct TimeData {
    time: nd::Array1<f64>,
    nbar: nd::Array1<f64>,
    szbar: nd::Array1<f64>,
}

type SpCav = Cavity<NATOMS, 1, HSpin>;

fn time_evolve_spinchain() -> TimeData {
    const TMAX: f64 = 5.0;
    const NT: usize = 2000;

    let omega_z: f64 = TAU * 0.5 / 2.0;
    let omega_a: f64 = TAU * 1.0 / 2.0;
    let j_z: f64 = TAU * -5.05 / 2.0;
    let g: f64 = TAU * 5.0 / 2.0;

    let gamma: f64 = TAU * 0.00;
    // let gamma: f64 = 0.01;
    // let kappa: f64 = 0.00;
    let kappa: f64 = TAU * 0.01;

    let hbuilder: HBuilderTransverseIsing<NATOMS>
        = HBuilderTransverseIsing::new(omega_z, omega_a, j_z, g, MAX_PHOTONS);
    let loperator: LOperatorTransverseIsing<NATOMS>
        = LOperatorTransverseIsing::new(&hbuilder, gamma, kappa);

    let rho0 = |si: &SpCav, sj: &SpCav| -> C64 {
        if si.atomic_states().iter().all(|s| s == &HSpin::Up)
            && si.photons()[0] == 3
            && sj == si
        {
            1.0.into()
        } else {
            0.0.into()
        }
    };

    let meas = |state: &nd::Array2<C64>| -> (f64, f64) {
        hbuilder.basis().keys()
            .zip(state.diag())
            .map(|(Cavity(ss, [n]), p)| {
                let prob = p.norm();
                (
                    prob * (*n as f64),
                    prob * ss.iter()
                        .map(|sk| sk.sz())
                        .sum::<f64>() / NATOMS as f64,
                )
            })
            .fold((0.0, 0.0), |(n0, sz0), (n, sz)| (n0 + n, sz0 + sz))
    };

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, TMAX, NT);
    let (nbar, szbar): (Vec<f64>, Vec<f64>)
        = lindblad::evolve_reduced_with(
            rho0, &hbuilder, &loperator, &time, meas)
        .unwrap()
        .into_iter()
        .unzip();
    TimeData {
        time,
        nbar: nd::Array::from_vec(nbar),
        szbar: nd::Array::from_vec(szbar),
    }
}

fn main() -> anyhow::Result<()> {
    let outdir = PathBuf::from("output/cavity_array");
    mkdir!(outdir);

    // println!("{}", ground_photons(100.0, 0.0));
    // println!("{}", ground_photons_spinchain(1.0, 0.0));
    // let Data { g, wz, nbar, mz } = phase_diagram_spinchain();
    // write_npz!(
    //     outdir.join("phase_diagram_spinchain.npz"),
    //     arrays: {
    //         "g" => &g,
    //         "wz" => &wz,
    //         "nbar" => &nbar,
    //         "mz" => &mz,
    //     }
    // );

    let TimeData { time, nbar, szbar } = time_evolve_spinchain();
    write_npz!(
        outdir.join("time_evolve_spinchain.npz"),
        arrays: {
            "time" => &time,
            "nbar" => &nbar,
            "szbar" => &szbar,
            "natoms" => &nd::array![NATOMS as u32],
            "nmax" => &nd::array![MAX_PHOTONS as u32],
            // "omega_z" => &nd::array![omega_z],
            // "omega_a" => &nd::array![omega_a],
            // "j_z" => &nd::array![j_z],
            // "g" => &nd::array![g],
            // "gamma" => &nd::array![gamma],
            // "kappa" => &nd::array![kappa],
        }
    );

    println!("done");
    Ok(())
}

