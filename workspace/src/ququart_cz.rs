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
    print_flush,
    println_flush,
    hilbert::{ Basis, ProdBasis },
    dynamics::*,
    rabi::*,
};
use lib::systems::ququart::{ *, State::* };

const B: f64 = 120.0; // G

const POL_ALPHA: f64 = 0.0; // horizontal
const POL_BETA: f64 = 0.0; // linear
const POL_THETA: f64 = PI / 2.0; // perpendicular to B-field

const POL_FACTOR: f64 = OVER_RT2;
const RYD_CG: f64 = OVER_RT2;

const OMEGA: f64 = 3.0; // MHz
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
            (G0, TAU * zm(G0, B)),
            (G1, TAU * zm(G1, B)),
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
    let drive11 = DriveParams::Variable {
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
    let drive12 = DriveParams::Variable {
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
    let drive21 = DriveParams::Variable {
        frequency: Rc::new(|t: f64| TAU * (zm(R0, B) - zm(C0, B) - detuning)),
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
    let drive22 = DriveParams::Variable {
        frequency: Rc::new(|t: f64| TAU * (zm(R0, B) - zm(C0, B) - detuning)),
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

    let pulse11 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive11, polarization),
        2,
        rydcoupling,
    );
    let pulse12 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive12, polarization),
        2,
        rydcoupling,
    );
    let pulse21 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive21, polarization),
        2,
        rydcoupling,
    );
    let pulse22 = HBuilderRydberg::new_nsites(
        HBuilder::new(&basis, drive22, polarization),
        2,
        rydcoupling,
    );
    let time: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 2.0 * pulse_len, 6500);
    let H: nd::Array3<C64>
        = pulse11.gen(&time) + pulse12.gen(&time)
        + pulse21.gen(&time) + pulse22.gen(&time);

    let psi0: nd::Array1<C64>
        = pulse11.prod_basis().get_vector(init_state).unwrap();
    let psi: nd::Array2<C64> = schrodinger::evolve_t(&psi0, &H, &time);

    let nt = time.len();
    let psi_ququart: nd::Array1<C64>
        = pulse11.prod_basis().keys()
        .zip(psi.slice(nd::s![.., nt - 1]))
        .filter_map(|(ss, a)| {
            (
                !matches!(ss[0], R0 | R1 | R2 | R3)
                && !matches!(ss[1], R0 | R1 | R2 | R3)
            ).then_some(*a)
        })
        .collect();
    println!("\n{},", format!("{:+.6}", psi_ququart).replace('i', "*sy.I"));

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

#[derive(Copy, Clone, Debug)]
struct Phases {
    ph0000: f64,
    ph0001: f64,
    ph0010: f64,
    ph0011: f64,
    ph0100: f64,
    ph0101: f64,
    ph0110: f64,
    ph0111: f64,
    ph1000: f64,
    ph1001: f64,
    ph1010: f64,
    ph1011: f64,
    ph1100: f64,
    ph1101: f64,
    ph1110: f64,
    ph1111: f64,
}

impl std::fmt::Display for Phases {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Phases {{")?;
        write!(f, "    ph0000: ")?;
        self.ph0000.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0001: ")?;
        self.ph0001.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0010: ")?;
        self.ph0010.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0011: ")?;
        self.ph0011.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0100: ")?;
        self.ph0100.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0101: ")?;
        self.ph0101.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0110: ")?;
        self.ph0110.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph0111: ")?;
        self.ph0111.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1000: ")?;
        self.ph1000.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1001: ")?;
        self.ph1001.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1010: ")?;
        self.ph1010.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1011: ")?;
        self.ph1011.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1100: ")?;
        self.ph1100.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1101: ")?;
        self.ph1101.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1110: ")?;
        self.ph1110.fmt(f)?;
        writeln!(f)?;
        write!(f, "    ph1111: ")?;
        self.ph1111.fmt(f)?;
        writeln!(f)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Phases {
    fn as_pydataline(&self, d_over_w: f64, xi: f64) -> String {
        format!(
            "[ {:.3}, {:.3} * np.pi, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3}, {:+.3} ],",
            d_over_w,
            xi / PI,
            self.ph0000,
            self.ph0001,
            self.ph0010,
            self.ph0011,
            self.ph0100,
            self.ph0101,
            self.ph0110,
            self.ph0111,
            self.ph1000,
            self.ph1001,
            self.ph1010,
            self.ph1011,
            self.ph1100,
            self.ph1101,
            self.ph1110,
            self.ph1111,
        )
    }

    // + -> detuning is > optimum
    // - -> detuning is < optimum
    fn error(&self) -> f64 {
        let ph01: f64
            = (
                self.ph0010 + self.ph0011 + self.ph0110 + self.ph0111
                + self.ph1000 + self.ph1001 + self.ph1100 + self.ph1101
            ) / 8.0;
        let ph11: f64
            = (self.ph1010 + self.ph1011 + self.ph1110 + self.ph1111) / 4.0;
        2.0 * ph01 - PI - ph11
    }
}

fn cz_phases(
    rabi_freq: f64,
    detuning: f64,
    phase_jump: f64,
) -> Phases
{
    println_flush!("");
    print_flush!("\r# 0000 ");
    let ph0000 = cz_phase(rabi_freq, detuning, phase_jump, &[G0, G0]);
    print_flush!("\r# 0001 ");
    let ph0001 = cz_phase(rabi_freq, detuning, phase_jump, &[G0, G1]);
    print_flush!("\r# 0010 ");
    let ph0010 = cz_phase(rabi_freq, detuning, phase_jump, &[G0, C0]);
    print_flush!("\r# 0011 ");
    let ph0011 = cz_phase(rabi_freq, detuning, phase_jump, &[G0, C1]);
    print_flush!("\r# 0100 ");
    let ph0100 = cz_phase(rabi_freq, detuning, phase_jump, &[G1, G0]);
    print_flush!("\r# 0101 ");
    let ph0101 = cz_phase(rabi_freq, detuning, phase_jump, &[G1, G1]);
    print_flush!("\r# 0110 ");
    let ph0110 = cz_phase(rabi_freq, detuning, phase_jump, &[G1, C0]);
    print_flush!("\r# 0111 ");
    let ph0111 = cz_phase(rabi_freq, detuning, phase_jump, &[G1, C1]);
    print_flush!("\r# 1000 ");
    let ph1000 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, G0]);
    print_flush!("\r# 1001 ");
    let ph1001 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, G1]);
    print_flush!("\r# 1010 ");
    let ph1010 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, C0]);
    print_flush!("\r# 1011 ");
    let ph1011 = cz_phase(rabi_freq, detuning, phase_jump, &[C0, C1]);
    print_flush!("\r# 1100 ");
    let ph1100 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, G0]);
    print_flush!("\r# 1101 ");
    let ph1101 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, G1]);
    print_flush!("\r# 1110 ");
    let ph1110 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, C0]);
    print_flush!("\r# 1111 ");
    let ph1111 = cz_phase(rabi_freq, detuning, phase_jump, &[C1, C1]);
    println_flush!("");
    Phases {
        ph0000,
        ph0001,
        ph0010,
        ph0011,
        ph0100,
        ph0101,
        ph0110,
        ph0111,
        ph1000,
        ph1001,
        ph1010,
        ph1011,
        ph1100,
        ph1101,
        ph1110,
        ph1111,
    }
}

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    const RABI: f64 = OMEGA;

    // const DET: f64 = 0.050 * OMEGA; const XI: f64 = 0.959 * PI;
    // const DET: f64 = 0.075 * OMEGA; const XI: f64 = 0.9382 * PI;
    // const DET: f64 = 0.100 * OMEGA; const XI: f64 = 0.9176 * PI;
    // const DET: f64 = 0.125 * OMEGA; const XI: f64 = 0.8975 * PI;
    // const DET: f64 = 0.150 * OMEGA; const XI: f64 = 0.8796 * PI;
    // const DET: f64 = 0.175 * OMEGA; const XI: f64 = 0.8629 * PI;
    // const DET: f64 = 0.200 * OMEGA; const XI: f64 = 0.8447 * PI;
    // const DET: f64 = 0.225 * OMEGA; const XI: f64 = 0.8296 * PI;
    // const DET: f64 = 0.250 * OMEGA; const XI: f64 = 0.8151 * PI;
    // const DET: f64 = 0.275 * OMEGA; const XI: f64 = 0.8007 * PI;
    // const DET: f64 = 0.300 * OMEGA; const XI: f64 = 0.7889 * PI;
    // const DET: f64 = 0.325 * OMEGA; const XI: f64 = 0.7769 * PI;
    // const DET: f64 = 0.350 * OMEGA; const XI: f64 = 0.7657 * PI;
    // const DET: f64 = 0.375 * OMEGA; const XI: f64 = 0.7578 * PI;
    // const DET: f64 = 0.400 * OMEGA; const XI: f64 = 0.7477 * PI;
    // const DET: f64 = 0.425 * OMEGA; const XI: f64 = 0.7397 * PI;
    // const DET: f64 = 0.450 * OMEGA; const XI: f64 = 0.7347 * PI;
    // const DET: f64 = 0.475 * OMEGA; const XI: f64 = 0.7289 * PI;
    // const DET: f64 = 0.500 * OMEGA; const XI: f64 = 0.7224 * PI;

    // // optimal, r_sep = 2.4 μm
    // const DET: f64 = 0.3759 * OMEGA;
    // const XI: f64 = 0.7575 * PI;

    // // optimal, r_sep = 3.6840314986404 μm
    // const DET: f64 = 0.3740 * OMEGA;
    // const XI: f64 = 0.7581 * PI;

    // // optimal, r_sep = 6.25 μm
    // const DET: f64 = 0.363 * OMEGA;
    // const XI: f64 = 0.763 * PI;

    // // optimal, r_sep = 2.35033095 μm
    // const DET: f64 = 0.3760 * OMEGA;
    // const XI: f64 = 0.7574 * PI;

    // // optimal, r_sep = 3.06821169 μm
    // const DET: f64 = 0.3738 * OMEGA;
    // const XI: f64 = 0.7582 * PI;

    // // optimal, r_sep = 4.00536061 μm
    // const DET: f64 = 0.3746 * OMEGA;
    // const XI: f64 = 0.7579 * PI;

    // // optimal, r_sep = 5.22875057 μm
    // const DET: f64 = 0.3524 * OMEGA;
    // const XI: f64 = 0.7644 * PI;

    // optimal, r_sep = 6.82581050 μm
    const DET: f64 = 0.2713 * OMEGA;
    const XI: f64 = 0.8023 * PI;

    let (time, psi, _) = do_pulses(RABI, DET, XI, &[G0, C0]);
    write_npz!(
        outdir.join("ququart_cz.npz"),
        arrays: {
            "time" => &time,
            "psi" => &psi,
        }
    );

    let phases = cz_phases(RABI, DET, XI);
    // println!("{:+.6}", phases);
    // println!("        {}", phases.as_pydataline(DET / RABI, XI));
    // println!("{}", phases.error());

    println!("done");
}

