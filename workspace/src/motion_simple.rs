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

const M: f64 = 2.8384644058191703e-25; // kg, 171Yb
const WL: f64 = 578e-9; // m

// const M: f64 = 5.00823449476748e-27; // kg, 3He
// const WL: f64 = 1083e-9 / 2.0; // m

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

fn corpse_angles(theta: f64) -> [f64; 3] {
    let th = theta / 2.0;
    let asth = (th.sin() / 2.0).asin();
    let th1 = TAU + th - asth;
    let th2 = TAU - 2.0 * asth;
    let th3 = th - asth;
    [th1 * 180.0 / PI, th2 * 180.0 / PI, th3 * 180.0 / PI]
}

fn corpse() {
    // const CORPSE_PI6_DEG: [f64; 3] = [367.6, 345.1,   7.6];
    // const CORPSE_PI4_DEG: [f64; 3] = [371.5, 337.9,  11.5];
    // const CORPSE_PI2_DEG: [f64; 3] = [384.3, 318.6,  24.3]; 
    // const CORPSE_PI_DEG:  [f64; 3] = [420.0, 300.0,  60.0];

    let angles = corpse_angles(PI / 2.0);
    eprintln!("{:?}", angles);

    let rabi_freq: f64 = 80e-3; // MHz

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
        fock_cutoff: Some(FockCutoff::NMax(20)),
    };
    let polarization = PolarizationParams::Poincare {
        alpha: PI / 2.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let frequency: f64 =
        basis.get_energy(&State::E).unwrap()
        - basis.get_energy(&State::G).unwrap();
        // - 0.5 * State::TRAP_FREQ;
    let strength: f64 = rabi_freq * 3.0_f64.sqrt();

    let tau0: f64 = angles[0] / 360.0 / rabi_freq;
    let tau1: f64 = angles[1] / 360.0 / rabi_freq;
    let tau2: f64 = angles[2] / 360.0 / rabi_freq;

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

    let psi0: nd::Array1<C64> =
        hbuilder00.basis().get_vector(&(State::G, 0).into()).unwrap()
            / 3.0_f64.sqrt()
        + hbuilder00.basis().get_vector(&(State::E, 1).into()).unwrap()
            / 1.5_f64.sqrt() * C64::i();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);
    for ak in psi.slice(nd::s![.., time.len() - 1]).iter() {
        println!("{:+.6}", ak);
    }
    let x = hbuilder00.gen_x();
    let p = hbuilder00.gen_p();
    let psi_conj = psi.mapv(|a| a.conj());
    let x_avg: nd::Array1<C64> =
        psi_conj.axis_iter(nd::Axis(1)).zip(psi.axis_iter(nd::Axis(1)))
        .map(|(ps_, ps)| ps_.dot(&x).dot(&ps))
        .collect();
    let p_avg: nd::Array1<C64> =
        psi_conj.axis_iter(nd::Axis(1)).zip(psi.axis_iter(nd::Axis(1)))
        .map(|(ps_, ps)| ps_.dot(&p).dot(&ps))
        .collect();

    write_npz!(
        outdir.join("motion_simple.npz"),
        arrays: {
            "time" => &time,
            "rho" => &rho,
            "tmark" => &nd::array![t0, t1, t2, t3, t4, t5, t6],
            "x_avg" => &x_avg,
            "p_avg" => &p_avg,
        }
    );

    println!("done");
}

#[derive(Clone, Debug, PartialEq)]
struct Data {
    time: nd::Array1<f64>,
    psi: nd::Array2<C64>,
    tmark: nd::Array1<f64>,
}

#[allow(clippy::excessive_precision)]
fn do_emswap(init: Option<Fock<State>>, writefile: bool) -> Data {
    const ROT_ANGLES_RAD: &[f64] =
        &[
            4.704729,
            4.704729,
            3.526636,
        ];
    const PHASE_ANGLES_RAD: &[f64] =
        &[
            5.993459,
            3.757841,
            4.974562,
        ];
    let rabi_freq: f64 = 0.8e-3; // MHz
    // fidelity = 0.996, trap freq = 100kHz (171Yb)

    // const ROT_ANGLES_RAD: &[f64] =
    //     &[
    //         6.268731,
    //         0.181896,
    //         0.181896,
    //     ];
    // const PHASE_ANGLES_RAD: &[f64] =
    //     &[
    //         0.494546,
    //         1.650019,
    //         6.023545,
    //     ];
    // let rabi_freq: f64 = 6.0e-3; // MHz
    // // fidelity = 0.996, trap freq = 400kHz (3He)

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
    // √3 from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = rabi_freq * 3.0_f64.sqrt() * 1.060 * 2.9;

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
        // init.unwrap_or((State::G, 0).into());
        init.unwrap_or((State::G, 1).into());
        // init.unwrap_or((State::G, 2).into());
        // init.unwrap_or((State::E, 0).into());
        // init.unwrap_or((State::E, 1).into());
        // init.unwrap_or((State::E, 2).into());

    let psi0: nd::Array1<C64> = hbuilder0.basis().get_vector(&init).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    let tmark: nd::Array1<f64> = nd::array![t0, t1, t2, t3];

    if writefile {
        write_npz!(
            outdir.join("motion_simple.npz"),
            arrays: {
                "time" => &time,
                "psi" => &psi,
                // "rho" => &rho,
                "tmark" => &tmark,
            }
        );
    }

    Data { time, psi, tmark }
}

fn bytearr(bytes: &[u8]) -> nd::Array1<u8> {
    nd::Array1::from_vec((*bytes).into())
}

fn emswap() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    let data_0 = do_emswap(Some(Fock(State::G, 1)), false);
    let data_1 = do_emswap(Some(Fock(State::G, 2)), false);

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
    let k_up: nd::Array1<i32> = nd::array![3, 4];
    let s_up: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|1n0\\rangle$"),
            bytearr(b"$|1n1\\rangle$"),
        );
    let k_dn: nd::Array1<i32> = nd::array![1, 2];
    let s_dn: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|0n1\\rangle$"),
            bytearr(b"$|0n2\\rangle$"),
        );

    write_npz!(
        outdir.join("emswap.npz"),
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

#[allow(clippy::excessive_precision)]
fn do_emshelve(init: Option<Fock<State>>, writefile: bool) -> Data {
    const ROT_ANGLES_RAD: &[f64] =
        &[
            4.012505,
            4.012505,
            2.179505,
        ];
    const PHASE_ANGLES_RAD: &[f64] =
        &[
            1.250536,
            1.935548,
            5.797791,
        ];

    let rabi_freq: f64 = 1.0e-3; // MHz

    // fidelity = 0.995

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
        + State::TRAP_FREQ;
    // √3 from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = rabi_freq * 3.0_f64.sqrt() * 1.060 * 2.9;

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
        // init.unwrap_or((State::G, 0).into());
        init.unwrap_or((State::G, 1).into());
        // init.unwrap_or((State::G, 2).into());
        // init.unwrap_or((State::E, 0).into());
        // init.unwrap_or((State::E, 1).into());
        // init.unwrap_or((State::E, 2).into());

    let psi0: nd::Array1<C64> = hbuilder0.basis().get_vector(&init).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    let tmark: nd::Array1<f64> = nd::array![t0, t1, t2, t3];

    if writefile {
        write_npz!(
            outdir.join("motion_simple.npz"),
            arrays: {
                "time" => &time,
                "psi" => &psi,
                // "rho" => &rho,
                "tmark" => &tmark,
            }
        );
    }

    Data { time, psi, tmark }
}

fn emshelve() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    let data_0 = do_emshelve(Some(Fock(State::G, 1)), false);
    let data_1 = do_emshelve(Some(Fock(State::G, 0)), false);

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
    let k_up: nd::Array1<i32> = nd::array![5, 4];
    let s_up: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|1n2\\rangle$"),
            bytearr(b"$|1n1\\rangle$"),
        );
    let k_dn: nd::Array1<i32> = nd::array![1, 0];
    let s_dn: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|0n1\\rangle$"),
            bytearr(b"$|0n0\\rangle$"),
        );

    write_npz!(
        outdir.join("emshelve.npz"),
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

#[allow(clippy::excessive_precision)]
fn do_emcz(init: Option<Fock<State>>, writefile: bool) -> Data {
    // const ROT_ANGLES_RAD: &[f64] =
    //     &[
    //         4.618265,
    //         4.631898,
    //         4.596400,
    //     ];
    // const PHASE_ANGLES_RAD: &[f64] =
    //     &[
    //         0.318591,
    //         4.689192,
    //         2.767662,
    //     ];
    //
    // let rabi_freq: f64 = 0.5e-3; // MHz

    const ROT_ANGLES_RAD: &[f64] =
        &[
            3.254935,
            5.087935,
            3.254934,
        ];
    const PHASE_ANGLES_RAD: &[f64] =
        &[
            0.434848,
            1.309383,
            2.182770,
        ];

    let rabi_freq: f64 = 1.2e-3; // MHz

    // fidelity = 0.996

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
    // √3 from a CG coefficient
    // 1.060 from a carrier Rabi frequency calibration
    // 2.9 from using a sideband
    let strength: f64 = rabi_freq * 3.0_f64.sqrt() * 1.060 * 2.9;

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
        // init.unwrap_or((State::G, 0).into());
        init.unwrap_or((State::G, 1).into());
        // init.unwrap_or((State::G, 2).into());
        // init.unwrap_or((State::E, 0).into());
        // init.unwrap_or((State::E, 1).into());
        // init.unwrap_or((State::E, 2).into());

    let psi0: nd::Array1<C64> = hbuilder0.basis().get_vector(&init).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    let tmark: nd::Array1<f64> = nd::array![t0, t1, t2, t3];

    if writefile {
        write_npz!(
            outdir.join("motion_simple.npz"),
            arrays: {
                "time" => &time,
                "psi" => &psi,
                // "rho" => &rho,
                "tmark" => &tmark,
            }
        );
    }

    Data { time, psi, tmark }
}

fn emcz() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    let data_0 = do_emcz(Some(Fock(State::G, 1)), false);
    let data_1 = do_emcz(Some(Fock(State::G, 2)), false);

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
    let k_up: nd::Array1<i32> = nd::array![3, 4];
    let s_up: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|1n0\\rangle$"),
            bytearr(b"$|1n1\\rangle$"),
        );
    let k_dn: nd::Array1<i32> = nd::array![1, 2];
    let s_dn: nd::Array2<u8> =
        nd::stack!(
            nd::Axis(0),
            bytearr(b"$|0n1\\rangle$"),
            bytearr(b"$|0n2\\rangle$"),
        );

    write_npz!(
        outdir.join("emcz.npz"),
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
    // simple();
    corpse();

    // do_emswap(None, true);
    // do_emshelve(None, true);
    // do_emcz(None, true);

    // emswap();
    // emshelve();
    // emcz();
}
