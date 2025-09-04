#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(unused_imports, unused_variables, unused_mut)]

use std::{
    f64::consts::{ PI, TAU },
    path::PathBuf,
    rc::Rc,
};
use itertools::Itertools;
use ndarray as nd;
use num_complex::Complex64 as C64;
use multilevel_sim::{
    mkdir,
    write_npz,
    hilbert::{ Basis, TrappedMagic, Fock },
    dynamics::*,
    rabi::*,
};
use lib::systems::quoct::*;

const M: f64 = 2.8384644058191703e-25; // kg
const WL: f64 = 578e-9; // m
const NMAX: usize = 2;

#[derive(Clone, Debug, PartialEq)]
struct Data {
    time: nd::Array1<f64>,
    psi: nd::Array2<C64>,
    tmark: nd::Array1<f64>,
}

fn bytearr(bytes: &[u8]) -> nd::Array1<u8> {
    nd::Array1::from_vec((*bytes).into())
}

#[allow(clippy::excessive_precision)]
fn do_sim(init: Option<Fock<State>>) -> Data {
    // const ROT_ANGLES_RAD: &[f64] =
    //     &[
    //         5.86268,
    //         3.65243e-1,
    //         3.65244e-1,
    //     ];
    // const PHASE_ANGLES_RAD: &[f64] =
    //     &[
    //         6.03006,
    //         2.29289e-1,
    //         5.91433,
    //     ];
    // let rabi_freq: f64 = 0.3e-3; // MHz

    // const ROT_ANGLES_RAD: &[f64] =
    //     &[
    //         5.96895,
    //         5.96895,
    //         4.71156e-1,
    //     ];
    // const PHASE_ANGLES_RAD: &[f64] =
    //     &[
    //         5.04866,
    //         5.12653,
    //         1.99848,
    //     ];
    // let rabi_freq: f64 = 0.3e-3; // MHz

    const ROT_ANGLES_RAD: &[f64] =
        &[
            6.195317,
            3.348247,
            1.924716,
        ];
    const PHASE_ANGLES_RAD: &[f64] =
        &[
            0.322143,
            0.384575,
            0.471378,
        ];
    let rabi_freq: f64 = 0.3e-3; // MHz

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let basis: Basis<State>
        = [
            (State::G0, 0.0), // working in the frame of the transition
            (State::G1, 0.0),
            (State::E0, 0.0),
            (State::E1, 0.0),
        ]
        .into_iter()
        .collect();

    let motion = MotionalParams {
        mass: 1e6 * M,
        wavelength: WL,
        temperature: 1e-12,
        fock_cutoff: Some(FockCutoff::NMax(NMAX)),
    };
    // let polarization = PolarizationParams::Poincare {
    //     alpha: PI / 2.0,
    //     beta: 0.0,
    //     theta: PI / 2.0,
    // };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 4.0,
        beta: PI / 2.0,
        theta: 0.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E1).unwrap()
        - basis.get_energy(&State::G0).unwrap()
        - State::TRAP_FREQ;
    // √(3/2) from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = rabi_freq * 1.5_f64.sqrt() * 1.060 * 2.9;

    let tau0: f64 = 2.0 * ROT_ANGLES_RAD[0] / TAU / rabi_freq;
    let tau1: f64 = 2.0 * ROT_ANGLES_RAD[1] / TAU / rabi_freq;
    let tau2: f64 = 2.0 * ROT_ANGLES_RAD[2] / TAU / rabi_freq;

    let t0 = 0.0;
    let t1 = t0 + tau0;
    let t2 = t1 + tau1;
    let t3 = t2 + tau2;

    let drive0 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t0..t1).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[0],
    };
    let hbuilder0 =
        HBuilderMagicTrap::new(&basis, drive0, polarization, motion);

    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t1..t2).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[1],
    };
    let hbuilder1 =
        HBuilderMagicTrap::new(&basis, drive1, polarization, motion);

    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|_| frequency),
        strength: Rc::new(|t: f64| {
            if (t2..t3).contains(&t) { TAU * strength } else { 0.0 }
        }),
        phase: PHASE_ANGLES_RAD[2],
    };
    let hbuilder2 =
        HBuilderMagicTrap::new(&basis, drive2, polarization, motion);

    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, t3, 5000);
    let H: nd::Array3<C64> =
        hbuilder0.gen(&time)
        + hbuilder1.gen(&time)
        + hbuilder2.gen(&time);

    let init: Fock<State> =
        // init.unwrap_or((State::G0, 0).into());
        init.unwrap_or((State::G0, 1).into());
        // init.unwrap_or((State::G0, 2).into());
        // init.unwrap_or((State::G1, 0).into());
        // init.unwrap_or((State::G1, 1).into());
        // init.unwrap_or((State::G1, 2).into());
        // init.unwrap_or((State::E0, 0).into());
        // init.unwrap_or((State::E0, 1).into());
        // init.unwrap_or((State::E0, 2).into());
        // init.unwrap_or((State::E1, 0).into());
        // init.unwrap_or((State::E1, 1).into());
        // init.unwrap_or((State::E1, 2).into());

    let psi0: nd::Array1<C64> = hbuilder0.basis().get_vector(&init).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);
    let tmark: nd::Array1<f64> = nd::array![t0, t1, t2, t3];
    Data { time, psi, tmark }
}

fn make_unitary() -> nd::Array2<C64> {
    const ATOM_STATES: &[State] = &[State::G0, State::G1, State::E0, State::E1];
    let uni_size: usize = ATOM_STATES.len() * (NMAX + 1);
    let mut uni: nd::Array2<C64> = nd::Array2::zeros((uni_size, uni_size));
    let state_iter = ATOM_STATES.iter().copied().cartesian_product(0..=NMAX);
    for (col, init) in uni.columns_mut().into_iter().zip(state_iter) {
        let data = do_sim(Some(init.into()));
        let n = data.time.len();
        data.psi.slice(nd::s![.., n - 1])
            .to_owned()
            .move_into(col);
    }
    uni
}

fn file_out() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    let data_0 = do_sim(Some(Fock(State::G0, 1)));
    let data_1 = do_sim(Some(Fock(State::G0, 2)));

    let psi: nd::Array3<C64> =
        nd::stack!(
            nd::Axis(0),
            data_0.psi,
            data_1.psi,
        );
    let tmark: nd::Array2<f64> =
        nd::stack!(
            nd::Axis(0),
            data_0.tmark,
            data_1.tmark,
        );
    let k_up: nd::Array1<i32> = nd::array![9, 10];
    let s_up: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|110\\rangle$"),
            bytearr(b"$|111\\rangle$"),
        );
    let k_dn: nd::Array1<i32> = nd::array![1, 2];
    let s_dn: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|001\\rangle$"),
            bytearr(b"$|002\\rangle$"),
        );

    write_npz!(
        outdir.join("enmccz.npz"),
        arrays: {
            "time" => &data_1.time,
            "psi" => &psi,
            "tmark" => &tmark,
            "k_up" => &k_up,
            "s_up" => &s_up,
            "k_dn" => &k_dn,
            "s_dn" => &s_dn,
        }
    );
}

fn main() {
    let u = make_unitary();
    println!("{:+.3}", u);
    file_out();
}
