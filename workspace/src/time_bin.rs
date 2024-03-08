#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ TAU, PI },
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
    hilbert::{ Basis, BasisState },
    dynamics::*,
    rabi::*,
};
use lib::systems::time_bin::{ *, State::* };

const B: f64 = 120.0; // G

#[derive(Clone, Debug)]
struct Data<S>
where S: BasisState
{
    basis: Basis<S>,
    time: nd::Array1<f64>,
    rho: nd::Array3<C64>,
}

fn doit_reg1() -> Data<Early> {
    let basis: Basis<Early>
        = [C0, C1, T0, T1, T2, T3].into_iter()
        .cartesian_product(0..=NMAX)
        .map(|(atom, photon)| (Early(atom, photon), TAU * zm(atom, B)))
        .collect();

    const REG_RABI: f64 = 28.0; // MHz
    const T_REG: f64 = 0.5 / REG_RABI; // μs
    let drive = DriveParams::Variable {
        frequency: Rc::new(|_| TAU * (zm(T3, B) - zm(C1, B))),
        strength: Rc::new(|t: f64| {
            if (0.0..T_REG).contains(&t) {
                TAU * REG_RABI * 2.0_f64.sqrt() // the √2 is for the CG coeff
            } else {
                0.0
            }
        }),
        phase: 0.0,
    };
    let pol = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let pulse = HBuilder::new(&basis, drive, pol);
    let decay = YBuilder::new(&basis);

    const T_TOTAL: f64 = 4.0; // μs
    const N: usize = 5000;
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, T_TOTAL, N);
    let H: nd::Array3<C64> = pulse.gen(&time);
    let Y: nd::Array2<f64> = decay.gen();
    let rho0: nd::Array2<C64>
        // = pulse.basis().get_density(&Early(C1, 0)).unwrap();
        = pulse.basis()
        .get_density_weighted_pure(|state, _, _| {
            match *state {
                Early(C0, 0) | Early(C1, 0) => 0.5_f64.sqrt().into(),
                _ => 0.0.into(),
            }
        });
    let rho: nd::Array3<C64> = lindblad_evolve_rk4(&rho0, &H, &Y, &time);

    drop(pulse);
    drop(decay);
    Data { basis, time, rho }
}

fn doit_raman(reg1: Data<Early>) -> Data<Early> {
    let Data { basis, time, rho } = reg1;
    let t_init: f64 = time[time.len() - 1];
    let rho0: nd::Array2<C64>
        = rho.slice(nd::s![.., .., time.len() - 1]).to_owned();
    drop(time);
    drop(rho);

    const RAMAN_RABI_SINGLE: f64 = 28.0; // MHz
    const RAMAN_DET: f64 = -300.0; // MHz
    let drive = DriveParams::Constant {
        frequency:
            TAU * (zm(T1, B) - (zm(C0, B) + zm(C1, B)) / 2.0 + RAMAN_DET),
        strength: TAU * RAMAN_RABI_SINGLE * 2.0_f64.sqrt(),
        phase: 0.0,
    };
    let pol = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let pulse = HBuilder::new(&basis, drive, pol);
    let decay = YBuilder::new(&basis);

    const T_RAMAN: f64 = 1.246; // μs
    const N: usize = 9000;
    let time: nd::Array1<f64>
        = nd::Array1::linspace(t_init, t_init + T_RAMAN, N);
    let H: nd::Array3<C64> = pulse.gen(&time);
    let Y: nd::Array2<f64> = decay.gen();
    let rho: nd::Array3<C64> = lindblad_evolve_rk4(&rho0, &H, &Y, &time);

    drop(pulse);
    drop(decay);
    Data { basis, time, rho }
}

fn doit_reg2(raman: Data<Early>) -> Data<Late> {
    let Data { basis, time, rho } = raman;
    let t_init: f64 = time[time.len() - 1];
    let rho0: nd::Array2<C64>
        = rho.slice(nd::s![.., .., time.len() - 1]).to_owned();
    drop(time);
    drop(rho);

    let basis: Basis<Late>
        = basis.iter()
        .cartesian_product(0..=NMAX)
        .map(|((&early, &energy), photon)| (Late(early, photon), energy))
        .collect();

    const REG_RABI: f64 = 28.0; // MHz
    const T_REG: f64 = 0.5 / REG_RABI; // μs
    let drive = DriveParams::Variable {
        frequency: Rc::new(|_| TAU * (zm(T3, B) - zm(C1, B))),
        strength: Rc::new(|t: f64| {
            if (t_init..t_init + T_REG).contains(&t) {
                TAU * REG_RABI * 2.0_f64.sqrt()
            } else {
                0.0
            }
        }),
        phase: 0.0,
    };
    let pol = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let pulse = HBuilder::new(&basis, drive, pol);
    let decay = YBuilder::new(&basis);

    const T_TOTAL: f64 = 4.0; // μs
    const N: usize = 5000;
    let time: nd::Array1<f64>
        = nd::Array1::linspace(t_init, t_init + T_TOTAL, N);
    let H: nd::Array3<C64> = pulse.gen(&time);
    let Y: nd::Array2<f64> = decay.gen();
    let late_photon_expand: nd::Array2<C64>
        = nd::Array2::from_diag(
            &(0..=NMAX)
                .map(|n| if n == 0 { c!(1.0) } else { c!(0.0) })
                .collect::<nd::Array1<C64>>()
        );
    let rho0: nd::Array2<C64> = nd::linalg::kron(&rho0, &late_photon_expand);
    let rho: nd::Array3<C64> = lindblad_evolve_rk4(&rho0, &H, &Y, &time);

    drop(pulse);
    drop(decay);
    Data { basis, time, rho }
}

fn main() -> anyhow::Result<()> {
    let outdir = PathBuf::from("output/time_bin");
    mkdir!(outdir);

    println!("reg1");
    let reg1 = doit_reg1();
    write_npz!(
        outdir.join("reg1.npz"),
        arrays: {
            "time" => &reg1.time,
            "rho" => &reg1.rho,
        }
    );
    println!("raman");
    let raman = doit_raman(reg1);
    write_npz!(
        outdir.join("raman.npz"),
        arrays: {
            "time" => &raman.time,
            "rho" => &raman.rho,
        }
    );
    println!("reg2");
    let reg2 = doit_reg2(raman);
    write_npz!(
        outdir.join("reg2.npz"),
        arrays: {
            "time" => &reg2.time,
            "rho" => &reg2.rho,
        }
    );

    println!("done");
    Ok(())
}

