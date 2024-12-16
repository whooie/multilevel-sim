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
use lib::systems::motion_simple::*;

const CORPSE_PI6_DEG: &[f64] = &[367.6, 345.1,  7.6];
const CORPSE_PI4_DEG: &[f64] = &[371.5, 337.9, 11.5];
const CORPSE_PI2_DEG: &[f64] = &[384.3, 318.6, 24.3]; 
const CORPSE_PI_DEG: &[f64]  = &[420.0, 300.0, 60.0];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Extremum { Min, Max }

fn first_extremum<I, T>(elems: I, ext: Extremum) -> Option<(usize, T)>
where
    I: IntoIterator<Item = T>,
    T: PartialOrd,
{
    let mut acc: Option<(usize, T)> = None;
    for (k, elem) in elems.into_iter().enumerate() {
        if let Some((k0, ref val)) = acc {
            match ext {
                Extremum::Min if &elem < val => { acc = Some((k, elem)); },
                Extremum::Min if &elem > val => { return acc; },
                Extremum::Max if &elem > val => { acc = Some((k, elem)); },
                Extremum::Max if &elem < val => { return acc; },
                _ => { continue; },
            }
        } else {
            acc = Some((k, elem));
        }
    }
    acc
}

fn first_min<I, T>(elems: I) -> Option<(usize, T)>
where
    I: IntoIterator<Item = T>,
    T: PartialOrd,
{
    first_extremum(elems, Extremum::Min)
}

fn first_max<I, T>(elems: I) -> Option<(usize, T)>
where
    I: IntoIterator<Item = T>,
    T: PartialOrd,
{
    first_extremum(elems, Extremum::Max)
}

const M: f64 = 2.8384644058191703e-25; // kg
const WL: f64 = 578e-9; // m

fn simple() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G, 0.0), // working in the frame of the transition
            (State::E, 0.0),
        ]
        .into_iter()
        .collect();
    let drive = DriveParams::Constant {
        frequency:
            basis.get_energy(&State::E).unwrap()
            - basis.get_energy(&State::G).unwrap()
            - State::TRAP_FREQ
            ,
        strength: TAU * 1e-3,
        phase: 0.0,
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(2)),
    };

    let hbuilder = HBuilderMagicTrap::new(&basis, drive, polarization, motion);
    println!("Lamb-Dicke = {:.5e}", hbuilder.lamb_dicke());

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, 5000.0, 3000);

    let rho0: nd::Array2<C64> =
        (
            hbuilder.basis().get_density(&(State::G, 1).into()).unwrap()
            + hbuilder.basis().get_density(&(State::E, 1).into()).unwrap()
        ) / 2.0;

    let H = hbuilder.gen(&time);
    let rho: nd::Array3<C64> = liouville::evolve_t(&rho0, &H, &time);

    let j_g1 =
        hbuilder.basis()
        .get_index_of::<Fock<State>>(&(State::G, 1).into())
        .unwrap();
    let k_g1_min =
        first_min(rho.slice(nd::s![j_g1, j_g1, ..]).iter().map(|p| p.re))
        .unwrap().0;
    let j_e1 =
        hbuilder.basis()
        .get_index_of::<Fock<State>>(&(State::E, 1).into())
        .unwrap();
    let k_e1_min =
        first_min(rho.slice(nd::s![j_e1, j_e1, ..]).iter().map(|p| p.re))
        .unwrap().0;

    let t_g1_min = time[k_g1_min];
    let t_e1_min = time[k_e1_min];
    println!("t @ g,1 min = {:.5}", t_g1_min);
    println!("t @ e,1 min = {:.5}", t_e1_min);
    println!("ratio = {:.5}", t_e1_min / t_g1_min);

    write_npz!(
        outdir.join("motion_simple.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
            "tmark" => &nd::array![t_g1_min, t_e1_min],
        }
    );

    println!("done");
}

fn corpse() {
    const CORPSE_PI6_DEG: &[f64] = &[367.6, 345.1,  7.6];
    const CORPSE_PI4_DEG: &[f64] = &[371.5, 337.9, 11.5];
    const CORPSE_PI2_DEG: &[f64] = &[384.3, 318.6, 24.3]; 
    const CORPSE_PI_DEG: &[f64]  = &[420.0, 300.0, 60.0];

    let rabi_freq: f64 = 30e-3; // MHz

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G, 0.0), // working in the frame of the transition
            (State::E, 0.0),
        ]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(2)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E).unwrap()
        - basis.get_energy(&State::G).unwrap()
        - 0.5 * State::TRAP_FREQ;
    let strength: f64 = rabi_freq * 3.0_f64.sqrt();

    let tau0: f64 = CORPSE_PI_DEG[0] / 360.0 / rabi_freq;
    let tau1: f64 = CORPSE_PI_DEG[1] / 360.0 / rabi_freq;
    let tau2: f64 = CORPSE_PI_DEG[2] / 360.0 / rabi_freq;

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

    let rho0: nd::Array2<C64> =
        hbuilder00.basis().get_density(&(State::E, 1).into()).unwrap();
        // (
        //     hbuilder00.basis().get_density(&(State::G, 1).into()).unwrap()
        //     + hbuilder00.basis().get_density(&(State::E, 1).into()).unwrap()
        // ) / 2.0;
    let rho: nd::Array3<C64> = liouville::evolve_t(&rho0, &H, &time);

    write_npz!(
        outdir.join("motion_simple.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
            "tmark" => &nd::array![t0, t1, t2, t3, t4, t5, t6],
        }
    );

    println!("done");
}

#[allow(clippy::excessive_precision)]
fn emswap() {
    const ROT_ANGLES_RAD: &[f64] =
        &[2.2256258753645315, 2.2238315316814696, 4.445200735312436];
    const PHASE_ANGLES_RAD: &[f64] =
        &[4.0718830352918784, 6.4995809932617630, 7.954649435204307];

    let rabi_freq: f64 = 1e-3; // MHz

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G, 0.0), // working in the frame of the transition
            (State::E, 0.0),
        ]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(2)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E).unwrap()
        - basis.get_energy(&State::G).unwrap()
        - State::TRAP_FREQ;
    let strength: f64 = rabi_freq * 3.0_f64.sqrt();

    let tau0: f64 = ROT_ANGLES_RAD[0] / TAU / rabi_freq;
    let tau1: f64 = ROT_ANGLES_RAD[1] / TAU / rabi_freq;
    let tau2: f64 = ROT_ANGLES_RAD[2] / TAU / rabi_freq;

    let t0 = 0.0;
    let t1 = t0 + tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;

    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[0] + PI / 2.0,
    };
    let hbuilder0 =
        HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[1] + PI / 2.0,
    };
    let hbuilder1 =
        HBuilderMagicTrap::new(&basis, drive1, polarization, motion);

    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[2] + PI / 2.0,
    };
    let hbuilder2 =
        HBuilderMagicTrap::new(&basis, drive2, polarization, motion);

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t3, 5000);
    let H: nd::Array3<C64> =
        hbuilder0.gen(&time)
        + hbuilder1.gen(&time)
        + hbuilder2.gen(&time);

    let rho0: nd::Array2<C64> =
        // hbuilder0.basis().get_density(&(State::G, 1).into()).unwrap();
        // hbuilder0.basis().get_density(&(State::E, 0).into()).unwrap();
        hbuilder0.basis().get_density(&(State::E, 1).into()).unwrap();
        // (
        //     hbuilder0.basis().get_density(&(State::G, 1).into()).unwrap()
        //     + hbuilder0.basis().get_density(&(State::E, 0).into()).unwrap()
        // ) / 2.0;
    let rho: nd::Array3<C64> = liouville::evolve_t(&rho0, &H, &time);

    write_npz!(
        outdir.join("motion_simple.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
            "tmark" => &nd::array![t0, t1, t2, t3],
        }
    );

    println!("done");
}

fn main() {
    // simple();
    // corpse();
    emswap();
}
