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
    hilbert::{ Basis, ProdBasis },
    dynamics::*,
    rabi::*,
    systems::ququart::{ *, State::* },
};

const B: f64 = 120.0; // G

const POL_ALPHA: f64 = 0.0; // horizontal
const POL_BETA: f64 = 0.0; // linear
const POL_THETA: f64 = PI / 2.0; // perpendicular to B-field

const POL_FACTOR: f64 = OVER_RT2;
const RYD_CG: f64 = OVER_RT2;

const OMEGA: f64 = 1.0; // MHz
// const DETUNING: f64 = 0.377 * OMEGA; // MHz
// const DETUNING: f64 = 0.47 * OMEGA; // MHz
// const PHASE_JUMP: f64 = PI * 0.758;

fn do_pulses(
    rabi_freq: f64,
    detuning: f64,
    phase_jump: f64,
    init_state: &[State],
) -> (nd::Array1<f64>, nd::Array2<C64>, ProdBasis<State>)
{
    let basis: Basis<State>
        = [
            // (G0, TAU * zm(G0, B)),
            // (G1, TAU * zm(G1, B)),
            (C0, TAU * zm(C0, B)),
            (C1, TAU * zm(C1, B)),
            // (N0, TAU * zm(N0, B)),
            // (N1, TAU * zm(N1, B)),
            // (T0, TAU * zm(T0, B)),
            // (T1, TAU * zm(T1, B)),
            (R0, TAU * zm(R0, B)),
            (R1, TAU * zm(R1, B)),
            (R2, TAU * zm(R2, B)),
            (R3, TAU * zm(R3, B)),
        ]
        .into_iter()
        .collect();
    let pulse_len: f64
        = (rabi_freq.powi(2) * 2.0 + detuning.powi(2)).sqrt().recip();
    let drive1 = DriveParams::Variable {
        frequency: Rc::new(|t: f64| TAU * (zm(R3, B) - zm(C1, B) - detuning)),
        strength: Rc::new(
            |t: f64| {
                if (0.0..pulse_len).contains(&t) {
                    TAU * rabi_freq / POL_FACTOR / RYD_CG
                } else {
                    0.0
                }
            }
        ),
        phase: 0.0,
    };
    let drive2 = DriveParams::Variable {
        frequency: Rc::new(|t: f64| TAU * (zm(R3, B) - zm(C1, B) - detuning)),
        strength: Rc::new(
            |t: f64| {
                if (pulse_len..2.0 * pulse_len).contains(&t) {
                    TAU * rabi_freq / POL_FACTOR / RYD_CG
                } else {
                    0.0
                }
            }
        ),
        phase: phase_jump,
    };
    let polarization = PolarizationParams::Poincare {
        alpha: 0.0,
        beta: 0.0,
        theta: PI / 2.0,
    };
    let rydcoupling = RydbergCoupling::AllToAll(R_SEP);

    let pulse1 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive1, polarization),
        2,
        rydcoupling,
    );
    let pulse2 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive2, polarization),
        2,
        rydcoupling,
    );
    let time: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 2.0 * pulse_len, 3000);
    let H: nd::Array3<C64> = pulse1.gen(&time) + pulse2.gen(&time);

    let psi0: nd::Array1<C64>
        = pulse1.prod_basis().get_vector(init_state).unwrap();
    let psi: nd::Array2<C64> = schrodinger_evolve_rk4(&psi0, &H, &time);

    (time, psi, basis.kron_with(&basis))
}

fn cz_phase(
    rabi_freq: f64,
    detuning: f64,
    phase_jump: f64,
    init_state: &[State],
) -> f64
{
    let (time, psi, pbasis)
        = do_pulses(rabi_freq, detuning, phase_jump, init_state);
    let nt = time.len();
    let k = pbasis.get_index_of(&init_state.to_owned()).unwrap();
    let ph: f64 = psi[[k, nt - 1]].arg();
    ph
}

fn cz_phases(
    rabi_freq: f64,
    detuning: f64,
    phase_jump: f64,
) -> (f64, f64, f64, f64)
{
    let ph00 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, C0]);
    let ph01 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, C1]);
    let ph10 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, C0]);
    let ph11 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, C1]);
    (ph00, ph01, ph10, ph11)
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    const RABI: f64 = OMEGA;
    const DET: f64 = 0.75 * OMEGA;
    const XI: f64 = 0.708 * PI;

    let (time, psi, _) = do_pulses(RABI, DET, XI, &[C0, C1]);
    write_npz!(
        outdir.join("ququart_cccz.npz"),
        arrays: {
            "time" => &time,
            "psi" => &psi,
        }
    );

    let (ph00, ph01, ph10, ph11) = cz_phases(RABI, DET, XI);
    println!(
        "φ00 = {:.6}\nφ01 = {:.6}\nφ10 = {:.6}\nφ11 = {:.6}",
        ph00, ph01, ph10, ph11,
    );
    println!("2 φ01 - π = {:.6}", 2.0 * ph01 - PI);
    println!("error = {:.6}", 2.0 * ph01 - PI - ph11);

    // let detuning: nd::Array1<f64> = nd::Array1::linspace(0.0, 0.5 * OMEGA, 50);
    // let (ph00, ph01, ph10, ph11): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
    //     = detuning.iter()
    //     .map(|det| cz_phases(OMEGA, *det))
    //     .multiunzip();
    // let ph00 = nd::Array1::from(ph00);
    // let ph01 = nd::Array1::from(ph01);
    // let ph10 = nd::Array1::from(ph10);
    // let ph11 = nd::Array1::from(ph11);
    // write_npz!(
    //     outdir.join("ququart_cccz_phases.npz"),
    //     arrays: {
    //         "det" => &detuning,
    //         "ph00" => &ph00,
    //         "ph01" => &ph01,
    //         "ph10" => &ph10,
    //         "ph11" => &ph11,
    //     }
    // );

    println!("done");
}

