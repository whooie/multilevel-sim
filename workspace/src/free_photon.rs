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
use lib::systems::free_photon::{ *, State::* };

const B: f64 = 120.0; // G

#[derive(Clone, Debug)]
struct Data<S>
where S: BasisState
{
    basis: Basis<S>,
    time: nd::Array1<f64>,
    psi: nd::Array2<C64>,
}

fn doit() -> Data<Photon> {
    let basis: Basis<Photon>
        = [G0, G1, E0, E1].into_iter()
        .cartesian_product(0..=NMAX)
        .map(|(atom, photon)| (Photon(atom, photon), TAU * zm(atom, B)))
        .collect();

    const RABI: f64 = 5.0; // MHz
    const T_PULSE: f64 = 0.44 / RABI; // μs
    let drive = DriveParams::Variable {
        frequency: Rc::new(|_| TAU * (zm(E1, B) - zm(G0, B))),
        strength: Rc::new(|t: f64| {
            if (0.0..T_PULSE).contains(&t) {
                TAU * RABI * 2.0_f64.sqrt()
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

    const T_TOTAL: f64 = 5.0; // μs
    const N: usize = 8000;
    let time: nd::Array1<f64> = nd::Array1::linspace(0.0, T_TOTAL, N);
    let mut H: nd::Array3<C64> = pulse.gen(&time);
    let i: usize = basis.get_index_of(&Photon(G0, 1)).unwrap();
    let j: usize = basis.get_index_of(&Photon(E1, 0)).unwrap();
    H.slice_mut(nd::s![i, j, ..])
        .iter_mut()
        .for_each(|h| { *h += c!(i GAMMA); });
    // H.slice_mut(nd::s![j, i, ..])
    //     .iter_mut()
    //     .for_each(|h| { *h += c!(i GAMMA); });
    // let k: usize = basis.get_index_of(&Photon(G1, 0)).unwrap();
    // H.slice_mut(nd::s![k, k, ..])
    //     .iter_mut()
    //     .for_each(|h| { *h += c!(i GAMMA / 2.0); });
    println!("{:+.3}", H.slice(nd::s![.., .., N - 1]));

    let psi0: nd::Array1<C64>
        = basis.get_vector_weighted(|state, _, _| {
            match *state {
                Photon(G0, 0) | Photon(G1, 0) => 0.5_f64.sqrt().into(),
                _ => 0.0.into()
            }
        });
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    drop(pulse);
    Data { basis, time, psi }
}

fn main() -> anyhow::Result<()> {
    let outdir = PathBuf::from("output/free_photon");
    mkdir!(outdir);

    let Data { basis: _, time, psi } = doit();
    write_npz!(
        outdir.join("data.npz"),
        arrays: {
            "time" => &time,
            "psi" => &psi,
        }
    );

    println!("done");
    Ok(())
}

