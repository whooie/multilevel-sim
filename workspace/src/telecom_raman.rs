#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU, FRAC_1_SQRT_2 as OVER_RT2 },
    ops::Range,
    path::PathBuf,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    mkdir,
    write_npz,
    print_flush,
    println_flush,
    hilbert::Basis,
    dynamics::*,
    rabi::*,
    utils::FExtremum,
};
use lib::systems::telecom::{ *, State::* };

const UNIT: f64 = 1e6; // working in MHz

#[derive(Clone, Debug)]
struct Output {
    time: nd::Array1<f64>,
    psi: nd::Array2<C64>,
    basis: Basis<State>,
}

fn doit(rabi_freq: f64, detuning: f64, field: f64, pol_angle: f64)
    -> Output
{
    let basis: Basis<State>
        = [
            (C0, TAU * zm(C0, field)),
            (C1, TAU * zm(C1, field)),
            (T0, TAU * zm(T0, field)),
            (T1, TAU * zm(T1, field)),
            (T2, TAU * zm(T2, field)),
            (T3, TAU * zm(T3, field)),
        ]
        .into_iter()
        .collect();
    // println!("{:?}", basis);
    let polarization = PolarizationParams::Poincare {
        alpha: pol_angle,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let drive = DriveParams::Constant {
        frequency:
            basis.get_energy(&T1).unwrap()
            - (
                basis.get_energy(&C0).unwrap()
                + basis.get_energy(&C1).unwrap()
            ) / 2.0
            + TAU * detuning,
        strength: TAU * rabi_freq,
        phase: 0.0,
    };
    let hbuilder = HBuilder::new(&basis, drive, polarization);

    let eff_rabi_est: f64
        = rabi_freq.powi(2) * (2.0 * pol_angle).sin() / detuning.abs() / 8.0;
    let time: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 1.0 / eff_rabi_est, 50000);
    let H: nd::Array3<C64> = hbuilder.gen(&time);
    // println!("{:.3e}", H.slice(nd::s![.., .., 0]).mapv(|a| a.norm()));

    // let splitting_T: f64
    //     = (basis.get_energy(&T1).unwrap() - basis.get_energy(&T2).unwrap()).abs();
    // let energies: nd::Array1<C64>
    //     = nd::array![
    //         0.0_f64.into(),
    //         (basis.get_energy(&C1).unwrap() - basis.get_energy(&C0).unwrap()).into(),
    //         (-TAU * detuning - splitting_T).into(),
    //         (-TAU * detuning).into(),
    //         (-TAU * detuning + splitting_T).into(),
    //         (-TAU * detuning + 2.0 * splitting_T).into(),
    //     ];
    // let diag_energies = nd::Array2::from_diag(&energies);
    // let H = &H.slice(nd::s![.., .., 0]) + &diag_energies;
    // println!("{:.3e}", H);

    let psi0: nd::Array1<C64> = basis.get_vector(&C1).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);
    // let psi: nd::Array2<C64> = eigen_evolve(&psi0, &H, &time);

    drop(hbuilder);
    Output { time, psi, basis }
}

struct ScanOutput {
    field: nd::Array1<f64>,
    pol_angle: nd::Array1<f64>,
    fidelity: nd::Array2<f64>,
}

fn scan(
    rabi_freq: f64,
    detuning: f64,
    field: Range<f64>,
    pol_angle: Range<f64>,
    npoints: usize,
) -> ScanOutput
{
    let field: nd::Array1<f64>
        = nd::Array1::linspace(field.start, field.end, npoints);
    let pol_angle: nd::Array1<f64>
        = nd::Array1::linspace(pol_angle.start, pol_angle.end, npoints);
    let fidelity: nd::Array2<f64>
        = pol_angle.iter().enumerate()
        .cartesian_product(field.iter().enumerate())
        .map(|((i, &pol), (j, &fld))| {
            print_flush!("\r  {:3} {:3} ", i, j);
            let Output { time: _, psi, basis }
                = doit(rabi_freq, detuning, fld, pol);
            let c0: nd::Array1<C64> = basis.get_vector(&C0).unwrap();
            let P_c0: nd::Array1<f64>
                = psi.t().dot(&c0).mapv(|a| a.norm().powi(2));
            P_c0.fmax().unwrap()
        })
        .collect::<nd::Array1<f64>>()
        .into_shape((pol_angle.len(), field.len()))
        .expect("error reshaping");
    println_flush!("");
    ScanOutput { field, pol_angle, fidelity }
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let single_rabi: f64 = 30.0 * 2.0_f64.sqrt(); // MHz
    let detuning: f64 = -300.0; // MHz

    let out: Output
        = doit(single_rabi, detuning, 80.0, 0.4 * PI);
    write_npz!(
        outdir.join("telecom-raman-single.npz"),
        arrays: {
            "time" => &out.time,
            "psi" => &out.psi,
        }
    );

    let scan_out: ScanOutput
        = scan(
            single_rabi, 
            detuning,
            0.0..150.0,
            PI / 12.0..5.0 * PI / 12.0,
            100,
        );
    write_npz!(
        outdir.join("telecom-raman.npz"),
        arrays: {
            "field" => &scan_out.field,
            "pol_angle" => &scan_out.pol_angle,
            "fidelity" => &scan_out.fidelity,
        }
    );

    println!("done");
}

