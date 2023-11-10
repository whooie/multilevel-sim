//! Provides a collection of miscellaneous macros, traits, and functions for
//! general use.

use std::{
    cmp,
    fmt,
    ops::{
        Neg,
        Add,
        AddAssign,
        Sub,
        SubAssign,
        Mul,
        MulAssign,
        Div,
        DivAssign,
        Rem,
        RemAssign,
    },
    str::FromStr,
};
use num_traits::{
    Float,
    Num,
    NumCast,
    One,
    ToPrimitive,
    Zero,
};
use regex::Regex;
use thiserror::Error;

/// Call `print!` and automatically flush.
#[macro_export]
macro_rules! print_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        print!($fmt $(, $val )*);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

/// Call `println!` and automatically flush.
#[macro_export]
macro_rules! println_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        println!($fmt $(, $val, )*);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

/// Call `eprint!` and automatically flush.
#[macro_export]
macro_rules! eprint_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        print!($fmt $(, $val )*);
        std::io::Write::flush(&mut std::io::stderr()).unwrap();
    }
}

/// Call `eprintln!` and automatically flush.
#[macro_export]
macro_rules! eprintln_flush {
    ( $fmt:literal $(, $val:expr )* $(,)?) => {
        println!($fmt $(, $val, )*);
        std::io::Write::flush(&mut std::io::stderr()).unwrap();
    }
}

/// Create a directory, passed as a `PathBuf`, if it doesn't exist.
///
/// Creates all nonexisting parent directories as well.
#[macro_export]
macro_rules! mkdir {
    ( $dir_pathbuf:expr ) => {
        if !$dir_pathbuf.is_dir() {
            println!(":: mkdir -p {}", $dir_pathbuf.to_str().unwrap());
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            std::fs::create_dir_all($dir_pathbuf.as_path())
                .expect(
                    format!(
                        "Couldn't create directory {:?}",
                        $dir_pathbuf.to_str().unwrap()
                    ).as_str()
                );
        }
    }
}

pub extern crate toml;

/// Generate a data struct to read values from a subtable of a TOML-formatted
/// file. This macro can be invoked at most once per scope.
///
/// Each struct member / field to be read is associated with a function through
/// which the value read from the file is passed before being stored in the
/// struct.
///
/// This macro expands into one struct definition and one function definition of
/// the forms:
/// ```ignore
/// #[derive(Clone, Debug)]
/// pub struct Config {
///     // member variables are all `pub`
/// }
///
/// impl Config {
///     pub load<P>(infile: Option<P>) -> Self
///     where P: AsRef<std::path::Path>
///     {
///         // reads from `infile`, defaulting to the $file
///     }
/// }
/// ```
#[macro_export]
macro_rules! config_fn {
    (
        $file:expr, $subtab:expr => {
            $( $key:expr, $def:expr, $intype:ty
                => $field:ident : $outtype:ty = $foo:expr ),+ $(,)?
        }
    ) => {
        #[derive(Clone, Debug)]
        pub struct Config {
            $(
                pub $field: $outtype,
            )+
        }

        impl Config {
            pub fn load<P>(infile: Option<P>) -> Self
            where P: AsRef<std::path::Path>
            {
                let infile_: std::path::PathBuf
                    = infile.unwrap_or(std::path::PathBuf::from($file));
                let table: $crate::utils::toml::Value
                    = std::fs::read_to_string(infile_.clone())
                    .expect(
                        format!("Couldn't read config file {:?}", infile_)
                        .as_str()
                    )
                    .parse::<$crate::utils::toml::Value>()
                    .expect(
                        format!("Couldn't parse config file {:?}", infile_)
                        .as_str()
                    );
                let mut config = Config {
                    $(
                        $field: ($foo)($def),
                    )+
                };
                if let Some(value) = table.get($subtab) {
                    $(
                        if let Some(X) = value.get($key) {
                            let x: $intype
                                = X.clone().try_into()
                                .expect(format!(
                                    "Couldn't coerce type for key {:?}", $key)
                                    .as_str()
                                );
                            config.$field = ($foo)(x);
                        }
                    )+
                }
                config
            }
        }
    }
}

pub extern crate num_complex;

/// Handy macro to create `num_complex::Complex64`s from more natural and
/// succinct syntax.
#[macro_export]
macro_rules! c {
    ( $re:expr )
        => { $crate::utils::num_complex::Complex64::new($re, 0.0) };
    ( i $im:expr )
        => { $crate::utils::num_complex::Complex64::new(0.0, $im) };
    ( e $ph:expr )
        => { $crate::utils::num_complex::Complex64::cis($ph) };
    ( $re:literal + i $im:literal )
        => { $crate::utils::num_complex::Complex64::new($re, $im) };
    ( $re:literal - i $im:literal )
        => { $crate::utils::num_complex::Complex64::new($re, -$im) };
    ( $re:literal + $im:literal i )
        => { $crate::utils::num_complex::Complex64::new($re, $im) };
    ( $re:literal - $im:literal i )
        => { $crate::utils::num_complex::Complex64::new($re, -$im) };
    ( $re:expr, $im:expr )
        => { $crate::utils::num_complex::Complex64::new($re, $im) };
    ( $r:expr, e $ph:expr )
        => { $crate::utils::num_complex::Complex64::from_polar($r, $ph) };
}

pub extern crate itertools;
pub extern crate ndarray;

/// Handles repeated calls with varied inputs to a closure outputting some
/// number of values, storing them in `ndarray::Array`s of appropriate shape.
///
/// Suppose we have some function $`f`$ that is defined over $`N`$ input
/// variables and returns $`M`$ output values. Given $`N`$ arrays of lengths
/// $`n_k`$ ($`k = 0, \dots, N - 1`$) sampling values over the $`N`$ variables,
/// this macro calls $`f`$ on each element of the Cartesian product of all $`N`$
/// arrays and collects the outputs of $`f`$ into $`M`$ arrays whose $`k`$-th
/// axis is associated with the $`k`$-th input array.
///
/// When calling this macro, the function `$caller` should take a single
/// `Vec<usize>` of indices, which should be used in the function body to refer
/// to input values in the sampling arrays, which in turn should be defined
/// beforehand, outside the macro. Output arrays must be returned as `IxDyn`.
///
/// # Example
/// ```
/// # extern crate ndarray;
/// # fn main() {
/// use ndarray::{ Array1, Array3, ArrayD, array };
/// use smorgasbord::{ loop_call };
///
/// let var1: Array1<f64> = array![0.5, 1.0, 2.0];
/// let var2: Array1<i32> = array![-1, 0, 1, 2];
/// let var3: Array1<bool> = array![true, false];
///
/// let mut caller
///     = |Q: Vec<usize>| -> (f64, f64) {
///         let val: f64 = var1[Q[0]].powi(var2[Q[1]]);
///         if var3[Q[2]] {
///             (val, 2.0 * val)
///         } else {
///             (-val, -2.0 * val)
///         }
///     };
/// let (ret1, ret2): (ArrayD<f64>, ArrayD<f64>)
///     = loop_call!(
///         caller => ( ret1: f64, ret2: f64 ),
///         vars: { var1, var2, var3 }
///     );
///
/// let ret1: Array3<f64> = ret1.into_dimensionality().unwrap();
/// let ret2: Array3<f64> = ret2.into_dimensionality().unwrap();
///
/// assert_eq!(
///     ret1,
///     array![
///         [
///             [ 2.00, -2.00 ],
///             [ 1.00, -1.00 ],
///             [ 0.50, -0.50 ],
///             [ 0.25, -0.25 ],
///         ],
///         [
///             [ 1.00, -1.00 ],
///             [ 1.00, -1.00 ],
///             [ 1.00, -1.00 ],
///             [ 1.00, -1.00 ],
///         ],
///         [
///             [ 0.50, -0.50 ],
///             [ 1.00, -1.00 ],
///             [ 2.00, -2.00 ],
///             [ 4.00, -4.00 ],
///         ],
///     ]
/// );
/// assert_eq!(
///     ret2,
///     array![
///         [
///             [ 4.00, -4.00 ],
///             [ 2.00, -2.00 ],
///             [ 1.00, -1.00 ],
///             [ 0.50, -0.50 ],
///         ],
///         [
///             [ 2.00, -2.00 ],
///             [ 2.00, -2.00 ],
///             [ 2.00, -2.00 ],
///             [ 2.00, -2.00 ],
///         ],
///         [
///             [ 1.00, -1.00 ],
///             [ 2.00, -2.00 ],
///             [ 4.00, -4.00 ],
///             [ 8.00, -8.00 ],
///         ],
///     ]
/// );
/// # }
/// ```
#[macro_export]
macro_rules! loop_call {
    (
        $caller:ident => ( $( $rvar:ident: $rtype:ty ),+ $(,)? ),
        vars: { $( $var:ident ),+ $(,)? } $(,)?
    ) => {
        loop_call!(
            $caller => ( $( $rvar: $rtype ),+ ),
            vars: { $( $var ),+ },
            printflag: true,
            lspace: 2
        )
    };
    (
        $caller:ident => ( $( $rvar:ident: $rtype:ty ),+ $(,)? ),
        vars: { $( $var:ident ),+ $(,)? },
        printflag: $printflag:expr $(,)?
    ) => {
        loop_call!(
            $caller => ( $( $rvar: $rtype ),+ ),
            vars: { $( $var ),+ },
            printflag: $printflag,
            lspace: 2
        )
    };
    (
        $caller:ident => ( $( $rvar:ident: $rtype:ty ),+ $(,)? ),
        vars: { $( $var:ident ),+ $(,)? },
        printflag: $printflag:expr,
        lspace: $lspace:expr $(,)?
    ) => {
        {
            let _caller_ = &mut $caller;
            let _Nvals_: Vec<usize> = vec![ $( $var.len() ),+ ];
            let _nvars_: usize = _Nvals_.len();
            let _Z_: Vec<usize>
                = _Nvals_.iter()
                .map(|n| (*n as f64).log10().floor() as usize + 1)
                .collect();
            let _tot_: usize = _Nvals_.iter().product();
            let mut _NN_: Vec<usize>
                = (1.._nvars_).rev()
                .map(|k| _Nvals_[_nvars_ - k.._nvars_].iter().product())
                .collect();
            _NN_.push(1);

            let _mk_outstr_ = |Q: Vec<usize>, last: bool| -> String {
                let mut outstr_items: Vec<String>
                    = Vec::with_capacity(_nvars_ + 3);
                outstr_items.push(" ".repeat($lspace));
                for (k, idx) in Q.iter().enumerate() {
                    outstr_items.push(
                        format!("{0:2$}/{1:2$};  ",
                            idx + !last as usize, _Nvals_[k], _Z_[k]));
                }
                outstr_items.push(
                    format!(
                        "[{:6.2}%] \r",
                        100.0 * (
                            (0.._nvars_)
                                .map(|k| (Q[k] - last as usize) * _NN_[k])
                                .sum::<usize>() as f64
                            + last as usize as f64
                        ) / _tot_ as f64,
                    )
                );
                outstr_items.iter().flat_map(|s| s.chars()).collect()
            };

            let _input_idx_
                = $crate::utils::itertools::Itertools::multi_cartesian_product(
                    _Nvals_.iter().map(|n| 0..*n)
                );
            let mut _outputs_: Vec<( $( $rtype ),+ ,)>
                = Vec::with_capacity(_tot_);
            let _t0_: std::time::Instant = std::time::Instant::now();
            for Q in _input_idx_ {
                if $printflag {
                    print!("{}", _mk_outstr_(Q.clone(), false));
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                _outputs_.push($caller(Q.clone()));
            }
            let _dt_: std::time::Duration = std::time::Instant::now() - _t0_;
            if $printflag {
                println!("{}", _mk_outstr_(_Nvals_.clone(), true));
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                println!(
                    "{}total time elapsed: {:.3} s",
                    " ".repeat($lspace),
                    _dt_.as_secs_f32(),
                );
                println!(
                    "{}average time per call: {:.3} s",
                    " ".repeat($lspace),
                    _dt_.as_secs_f32() / _tot_ as f32,
                );
            }

            let ( $( $rvar ),+ ,): ( $( Vec<$rtype> ),+ ,)
                = $crate::utils::itertools::Itertools::multiunzip(
                    _outputs_.into_iter());
            (
                $(
                    $crate::utils::ndarray::Array::from_vec($rvar)
                        .into_shape(_Nvals_.as_slice())
                        .expect("couldn't reshape")
                ),+
            ,)
        }
    };
}

/// Trait to find the max (or argmax) of a collection of floating-point values
/// (since `f64` and `f32` do not implement `Ord`).
pub trait FExtremum<F> {
    fn fmax(&self) -> Option<F>;

    fn fmin(&self) -> Option<F>;

    fn fmax_idx(&self) -> Option<(usize, F)>;

    fn fmin_idx(&self) -> Option<(usize, F)>;
}

macro_rules! impl_fextremum {
    ( $f:ty ) => {
        fn fmax(&self) -> Option<$f> {
            self.iter()
                .max_by(|l, r| {
                    l.partial_cmp(r)
                        .unwrap_or(std::cmp::Ordering::Greater)
                })
                .map(|v| *v)
        }

        fn fmin(&self) -> Option<$f> {
            self.iter()
                .min_by(|l, r| {
                    l.partial_cmp(r)
                        .unwrap_or(std::cmp::Ordering::Less)
                })
                .map(|v| *v)
        }

        fn fmax_idx(&self) -> Option<(usize, $f)> {
            self.iter().enumerate()
                .max_by(|(_kl, yl), (_kr, yr)| {
                    yl.partial_cmp(yr)
                        .unwrap_or(std::cmp::Ordering::Greater)
                })
                .map(|(k, y)| (k, *y))
        }

        fn fmin_idx(&self) -> Option<(usize, $f)> {
            self.iter().enumerate()
                .min_by(|(_kl, yl), (_kr, yr)| {
                    yl.partial_cmp(yr)
                        .unwrap_or(std::cmp::Ordering::Less)
                })
                .map(|(k, y)| (k, *y))
        }
    }
}

macro_rules! impl_fextremum_simple {
    ( $hasiter:ty, $f:ty ) => {
        impl FExtremum<$f> for $hasiter {
            impl_fextremum!($f);
        }
    }
}

impl_fextremum_simple!(Vec<f64>, f64);
impl_fextremum_simple!(Vec<f32>, f32);
impl_fextremum_simple!([f64], f64);
impl_fextremum_simple!([f32], f32);

// pub fn find_first_max<'a, A: 'a, I>(y: I) -> (usize, A)
// where A: Copy + PartialOrd<A>,
//       I: Iterator<Item=&'a A>,
// {
//     let mut iter = y.peekable();
//     let mut y0: A = **iter.peek().unwrap();
//     let mut k0: usize = 0;
//     for (k, yk) in iter.enumerate() {
//         if yk <= &y0 {
//             break;
//         } else {
//             y0 = *yk;
//             k0 = k;
//         }
//     }
//     (k0, y0);
// }
//
// pub fn find_first_min<'a, A: 'a, I>(y: I) -> (usize, A)
// where A: Copy + PartialOrd<A>,
//       I: Iterator<Item=&'a A>,
// {
//     let mut iter = y.peekable();
//     let mut y0: A = **iter.peek().unwrap();
//     let mut k0: usize = 0;
//     for (k, yk) in iter.enumerate() {
//         if yk >= &y0 {
//             break;
//         } else {
//             y0 = *yk;
//             k0 = k;
//         }
//     }
//     (k0, y0);
// }

/// For use with [`value_str()`].
#[derive(Copy, Clone, Debug)]
pub enum ValueStrSci {
    No,
    Lower,
    Upper,
}

impl ValueStrSci {
    fn is_yes(&self) -> bool { matches!(self, Self::Lower | Self::Upper) }

    fn is_no(&self) -> bool { matches!(self, Self::No) }
}

impl From<bool> for ValueStrSci {
    fn from(b: bool) -> Self { if b { Self::Lower } else { Self::No } }
}

/// Generate a string representation of a real number with an attached
/// uncertainty bound.
///
/// - `trunc`: Whether the number is truncated to its uncertainty (e.g.
///   `0.XX(Y)`) or not (`0.XX +/- 0.0Y`).
/// - `sign`: Whether to include a leading `+`.
/// - `sci`: Whether use scientific notation.
/// - `latex`: Whether to enclose with `$`'s and replace `+/-` with `\pm`.
/// - `dec`: The maximum number of decimal places to include. Note that the
///   exact meaning of this parameter changes depending on `sci`.
///
/// *Panics* if `T.log10()` cannot be represented as an `i32`.
pub fn value_str<T, S>(
    x: T,
    err: T,
    trunc: bool,
    sign: bool,
    sci: S,
    latex: bool,
    dec: Option<usize>,
) -> String
where
    T: num_traits::Float + std::fmt::Display,
    S: Into<ValueStrSci>,
{
    let sci: ValueStrSci = sci.into();
    let ten: T = T::one() + T::one() + T::one() + T::one() + T::one()
        + T::one() + T::one() + T::one() + T::one() + T::one();
    let ord_x: i32
        = x.abs().log10().floor().to_i32()
        .expect("value_str: unrepresentable number");
    let ord_err: Option<i32>
        = (err.is_normal() && err != T::zero())
        .then(|| {
            err.abs().log10().floor().to_i32()
                .expect("value_str: unrepresentable number")
        });
    let (xp, errp, mut z): (T, Option<T>, usize)
        = if sci.is_yes() {
            (
                (x / ten.powi(ord_err.unwrap_or(0))).round()
                    * ten.powi(ord_err.unwrap_or(0) - ord_x),
                ord_err.map(|o| {
                    (err / ten.powi(o)).round() * ten.powi(o - ord_x)
                }),
                cmp::max(ord_x - ord_err.unwrap_or(0), 0) as usize,
            )
        } else {
            (
                (x / ten.powi(ord_err.unwrap_or(0))).round()
                    * ten.powi(ord_err.unwrap_or(0)),
                ord_err.map(|o| {
                    (err / ten.powi(o)).round() * ten.powi(o)
                }),
                cmp::max(-ord_err.unwrap_or(0), 0) as usize,
            )
        };
    z = if let Some(d) = dec { cmp::min(d, z) } else { z };
    let outstr: String
        = if trunc {
            format!(
                "{}({}){}",
                if sign {
                    format!("{:+.w$}", xp, w=z)
                } else {
                    format!("{:.w$}", xp, w=z)
                },
                if let Some(e) = errp {
                    format!("{:.0}", e * ten.powi(z as i32))
                } else {
                    "nan".to_string()
                },
                match sci {
                    ValueStrSci::No => "".to_string(),
                    ValueStrSci::Lower
                        => format!(
                            "e{}{:02}",
                            if ord_x < 0 { "-" } else { "+" },
                            ord_x.abs(),
                        ),
                    ValueStrSci::Upper
                        => format!(
                            "E{}{:02}",
                            if ord_x < 0 { "-" } else { "+" },
                            ord_x.abs(),
                        ),
                },
            )
        } else {
            format!(
                "{}{ex} {} {}{ex}",
                if sign {
                    format!("{:+.w$}", xp, w=z)
                } else {
                    format!("{:.w$}", xp, w=z)
                },
                if latex {
                    r"\pm"
                } else {
                    "+/-"
                },
                if let Some(e) = errp {
                    format!("{:.w$}", e, w=z)
                } else {
                    "nan".to_string()
                },
                ex=match sci {
                    ValueStrSci::No => "".to_string(),
                    ValueStrSci::Lower
                        => format!(
                            "e{}{:02}",
                            if ord_x < 0 { "-" } else { "+" },
                            ord_x.abs(),
                        ),
                    ValueStrSci::Upper
                        => format!(
                            "E{}{:02}",
                            if ord_x < 0 { "-" } else { "+" },
                            ord_x.abs(),
                        ),
                },
            )
        };
    if latex {
        "$".to_string() + &outstr + "$"
    } else {
        outstr
    }
}

/// Macro form of [`value_str`] with optional defaults.
///
/// - `trunc` defaults to `true`
/// - `sign` defaults to `false`
/// - `sci` defaults to `false`
/// - `latex` defaults to `false`
/// - `dec` defaults to `None`
#[macro_export]
macro_rules! value_str {
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, $sci, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, $sci, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, $sci, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        sci: $sci:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, $sci, false, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, false, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, false, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, false, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sign: $sign:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, $sign, false, false, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sci: $sci:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, $sci, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sci: $sci:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, $sci, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sci: $sci:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, $sci, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        sci: $sci:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, $sci, false, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, false, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, false, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, false, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        trunc: $trunc:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, $trunc, false, false, false, None)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, $sci, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, $sci, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        sci: $sci:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, $sci, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        sci: $sci:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, $sci, false, None)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, false, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, false, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, false, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sign: $sign:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, $sign, false, false, None)
    };
    {
        $x:expr,
        $err:expr,
        sci: $sci:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, $sci, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sci: $sci:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, $sci, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        sci: $sci:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, $sci, false, $dec)
    };
    {
        $x:expr,
        $err:expr,
        sci: $sci:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, $sci, false, None)
    };
    {
        $x:expr,
        $err:expr,
        latex: $latex:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, false, $latex, $dec)
    };
    {
        $x:expr,
        $err:expr,
        latex: $latex:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, false, $latex, None)
    };
    {
        $x:expr,
        $err:expr,
        dec: $dec:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, false, false, $dec)
    };
    {
        $x:expr,
        $err:expr $(,)?
    } => {
        $crate::utils::value_str($x, $err, true, false, false, false, None)
    };
}

/// A real number with associated uncertainty. Uncertainties are automatically
/// propagated through arithmetic operations.
///
/// # String formatting
/// This type implements [`std::fmt::Display`] as a shortcut to [`value_str`]
/// along with the usual string padding and alignment options (see also the
/// [`value_str!`][crate::value_str] macro).
///
/// ```
/// # use smorgasbord::utils::ExpVal;
/// let val = ExpVal::new(0.123, 0.001);
///
/// // default formatter is just a call to value_str!
/// assert_eq!(format!("{}", val), "0.123(1)");
/// // alternate display turns truncation off
/// assert_eq!(format!("{:#}", val), "0.123 +/- 0.001");
/// // padding with '$' turns on LaTeX mode
/// assert_eq!(format!("{:$^#}", val), r"$0.123 \pm 0.001$");
/// // regular spaces are used for padding in this case
/// assert_eq!(format!("{:$^#22}", val), r"  $0.123 \pm 0.001$   ");
/// // other characters pad normally
/// assert_eq!(format!("{:_^#22}", val), "___0.123 +/- 0.001____");
/// // precision controls the maximum number of decimal places displayed
/// assert_eq!(format!("{:.1}; {:+.10}", val, val), "0.1(0); +0.123(1)");
/// // scientific notation too!
/// assert_eq!(format!("{:+e}; {:#E}", val, val), "+1.23(1)e-01; 1.23E-01 +/- 0.01E-01");
/// ```
#[derive(Copy, Clone, Debug)]
pub struct ExpVal {
    val: f64,
    err: f64,
}

impl ExpVal {
    pub fn new(val: f64, err: f64) -> Self { Self { val, err: err.abs() } }

    pub fn val(self) -> f64 { self.val }

    pub fn err(self) -> f64 { self.err.abs() }
}

impl Float for ExpVal {
    fn abs(self) -> Self { Self { val: self.val.abs(), err: self.err } }

    fn abs_sub(self, other: Self) -> Self {
        Self {
            val: (self.val - other.val).abs(),
            err: (self.err.powi(2) + other.err.powi(2)).sqrt(),
        }
    }

    fn acos(self) -> Self {
        Self {
            val: self.val.acos(),
            err: self.err / (1.0 - self.val.powi(2)).sqrt(),
        }
    }

    fn acosh(self) -> Self {
        Self {
            val: self.val.acosh(),
            err: self.err / (self.val.powi(2) - 1.0).sqrt(),
        }
    }

    fn asin(self) -> Self {
        Self {
            val: self.val.asin(),
            err: self.err / (1.0 - self.val.powi(2)).sqrt(),
        }
    }

    fn asinh(self) -> Self {
        Self {
            val: self.val.asinh(),
            err: self.err / (self.val.powi(2) + 1.0).sqrt(),
        }
    }

    fn atan(self) -> Self {
        Self {
            val: self.val.atan(),
            err: self.err / (self.val.powi(2) + 1.0),
        }
    }

    fn atan2(self, other: Self) -> Self {
        Self {
            val: self.val.atan2(other.val),
            err: (
                (self.val * other.err).powi(2)
                + (self.err * other.val).powi(2)
            ).sqrt()
            / (self.val.powi(2) + other.val.powi(2)),
        }
    }

    fn atanh(self) -> Self {
        Self {
            val: self.val.atanh(),
            err: self.err / (self.val.powi(2) - 1.0).abs(),
        }
    }

    fn cbrt(self) -> Self {
        Self {
            val: self.val.cbrt(),
            err: self.err / self.val.powi(2).cbrt() / 3.0,
        }
    }

    fn ceil(self) -> Self {
        Self {
            val: self.val.ceil(),
            err: 0.0,
        }
    }

    fn classify(self) -> core::num::FpCategory { self.val.classify() }

    fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            err: self.err * self.val.sin().abs(),
        }
    }

    fn cosh(self) -> Self {
        Self {
            val: self.val.cosh(),
            err: self.err * self.val.sinh().abs(),
        }
    }

    fn exp(self) -> Self {
        let ex: f64 = self.val.exp();
        Self {
            val: ex,
            err: self.err * ex,
        }
    }

    fn exp2(self) -> Self {
        let ex2: f64 = self.val.exp2();
        Self {
            val: ex2,
            err: self.err * std::f64::consts::LN_2 * ex2,
        }
    }

    fn exp_m1(self) -> Self {
        Self {
            val: self.val.exp_m1(),
            err: self.err * self.val.exp(),
        }
    }

    fn floor(self) -> Self {
        Self {
            val: self.val.floor(),
            err: 0.0,
        }
    }

    fn fract(self) -> Self {
        Self {
            val: self.val.fract(),
            err: self.err,
        }
    }

    fn hypot(self, other: Self) -> Self {
        let h: f64 = self.val.hypot(other.val);
        Self {
            val: h,
            err: (
                (self.err * self.val).powi(2)
                + (other.err * other.val).powi(2)
            ).sqrt() / h,
        }
    }

    fn infinity() -> Self {
        Self {
            val: f64::infinity(),
            err: f64::nan(),
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) { self.val.integer_decode() }

    fn is_finite(self) -> bool { self.val.is_finite() }

    fn is_infinite(self) -> bool { self.val.is_infinite() }

    fn is_nan(self) -> bool { self.val.is_nan() }

    fn is_normal(self) -> bool { self.val.is_normal() }

    fn is_sign_negative(self) -> bool { self.val.is_sign_negative() }

    fn is_sign_positive(self) -> bool { self.val.is_sign_positive() }

    fn ln(self) -> Self {
        Self {
            val: self.val.ln(),
            err: self.err / self.val.abs(),
        }
    }

    fn ln_1p(self) -> Self {
        Self {
            val: self.val.ln_1p(),
            err: self.err / (self.val + 1.0).abs(),
        }
    }

    fn log(self, base: Self) -> Self {
        Self {
            val: self.val.log(base.val),
            err: (
                (
                    (self.err / self.val).powi(2)
                    + (base.err * self.val.log(base.val) / base.val).powi(2)
                ) / base.val.ln().powi(2)
            ).sqrt(),
        }
    }

    fn log10(self) -> Self {
        Self {
            val: self.val.log10(),
            err: self.err / std::f64::consts::LN_10 / self.val.abs(),
        }
    }

    fn log2(self) -> Self {
        Self {
            val: self.val.log2(),
            err: self.err / std::f64::consts::LN_2 / self.val.abs(),
        }
    }

    fn max(self, other: Self) -> Self {
        match self.partial_cmp(&other) {
            Some(cmp::Ordering::Greater) => self,
            Some(cmp::Ordering::Equal) => self,
            Some(cmp::Ordering::Less) => other,
            None => match (self.is_normal(), other.is_normal()) {
                (true, true) => self,
                (true, false) => self,
                (false, true) => other,
                (false, false) => self,
            },
        }
    }

    fn max_value() -> Self {
        Self {
            val: f64::max_value(),
            err: 0.0,
        }
    }

    fn min(self, other: Self) -> Self {
        match self.partial_cmp(&other) {
            Some(cmp::Ordering::Greater) => other,
            Some(cmp::Ordering::Equal) => self,
            Some(cmp::Ordering::Less) => self,
            None => match (self.is_normal(), other.is_normal()) {
                (true, true) => self,
                (true, false) => self,
                (false, true) => other,
                (false, false) => self,
            },
        }
    }

    fn min_positive_value() -> Self {
        Self {
            val: f64::min_positive_value(),
            err: 0.0,
        }
    }

    fn min_value() -> Self {
        Self {
            val: f64::min_value(),
            err: 0.0,
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self {
            val: self.val.mul_add(a.val, b.val),
            err: (
                (self.err * a.val).powi(2)
                + (self.val * a.err).powi(2)
                + b.err.powi(2)
            ).sqrt(),
        }
    }

    fn nan() -> Self {
        Self {
            val: f64::nan(),
            err: f64::nan(),
        }
    }

    fn neg_infinity() -> Self {
        Self {
            val: f64::neg_infinity(),
            err: f64::nan(),
        }
    }

    fn neg_zero() -> Self {
        Self {
            val: f64::neg_zero(),
            err: 0.0,
        }
    }

    fn powf(self, n: Self) -> Self {
        Self {
            val: self.val.powf(n.val),
            err: self.val.powf(n.val - 1.0) * (
                (self.err * n.val).powi(2)
                + (n.err * self.val * self.val.ln()).powi(2)
            ).sqrt(),
        }
    }

    fn powi(self, n: i32) -> Self {
        Self {
            val: self.val.powi(n),
            err: self.err * self.val.powi(n) * n.abs() as f64,
        }
    }

    fn recip(self) -> Self {
        Self {
            val: self.val.recip(),
            err: self.err / self.val.powi(2),
        }
    }

    fn round(self) -> Self {
        Self {
            val: self.val.round(),
            err: 0.0,
        }
    }

    fn signum(self) -> Self {
        Self {
            val: self.val.signum(),
            err: 0.0,
        }
    }

    fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            err: self.err * self.val.cos().abs(),
        }
    }

    fn sin_cos(self) -> (Self, Self) { (self.sin(), self.cos()) }

    fn sinh(self) -> Self {
        Self {
            val: self.val.sinh(),
            err: self.err * self.val.cosh(),
        }
    }

    fn sqrt(self) -> Self {
        let sq: f64 = self.val.sqrt();
        Self {
            val: sq,
            err: self.err / sq / 2.0,
        }
    }

    fn tan(self) -> Self {
        Self {
            val: self.val.tan(),
            err: self.err / self.val.cos().powi(2),
        }
    }

    fn tanh(self) -> Self {
        Self {
            val: self.val.tanh(),
            err: self.err / self.val.cosh().powi(2),
        }
    }

    fn trunc(self) -> Self {
        Self {
            val: self.val.trunc(),
            err: 0.0,
        }
    }
}

impl One for ExpVal {
    fn one() -> Self { Self { val: f64::one(), err: 0.0 } }
}

impl Zero for ExpVal {
    fn zero() -> Self { Self { val: f64::zero(), err: 0.0 } }

    fn is_zero(&self) -> bool { self.val.is_zero() }
}

impl PartialEq<ExpVal> for ExpVal {
    fn eq(&self, rhs: &Self) -> bool { self.val == rhs.val }
}

impl PartialOrd<ExpVal> for ExpVal {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&rhs.val)
    }
}

impl From<f64> for ExpVal {
    fn from(f: f64) -> Self { Self { val: f, err: 0.0 } }
}

impl From<(f64, f64)> for ExpVal {
    fn from(f: (f64, f64)) -> Self { Self { val: f.0, err: f.1.abs() } }
}

/// Uncertainties are discarded.
impl ToPrimitive for ExpVal {
    fn to_i64(&self) -> Option<i64> { self.val.to_i64() }

    fn to_u64(&self) -> Option<u64> { self.val.to_u64() }
}

impl NumCast for ExpVal {
    fn from<T>(n: T) -> Option<Self>
    where T: ToPrimitive
    {
        n.to_f64().map(|f| Self { val: f, err: 0.0 })
    }
}

#[derive(Debug, Error)]
pub enum ExpValError {
    #[error("malformed input '{0}'")]
    MalformedInput(String),

    #[error("unmatched '$'")]
    UnmatchedDollar,

    #[error("couldn't parse number: {0}")]
    ParseFloatError(String),
}
pub type ExpValResult<T> = Result<T, ExpValError>;

impl From<std::num::ParseFloatError> for ExpValError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::ParseFloatError(err.to_string())
    }
}

impl From<num_traits::ParseFloatError> for ExpValError {
    fn from(err: num_traits::ParseFloatError) -> Self {
        Self::ParseFloatError(err.to_string())
    }
}

impl Num for ExpVal {
    type FromStrRadixErr = ExpValError;

    fn from_str_radix(s: &str, radix: u32) -> ExpValResult<Self> {
        const ALPHABET: &str = "0123456789abcdefghijklmnopqrstuvwxyz";
        const SIGN: &str = r"[+\-]";
        const NON_NORMAL: &str = r"nan|inf";
        const BASE10_EXP: &str = r"e([+\-]?\d+)";
        (2..=36_u32).contains(&radix).then_some(())
            .ok_or(
                ExpValError::ParseFloatError(
                    "radix must be between 2 and 36".to_string())
            )?;
        let digit: String = format!("[{}]", &ALPHABET[..radix as usize]);
        let number: String
            = format!(r"{nn}|(({d}*\.)?{d}+)", nn=NON_NORMAL, d=digit);
        let (val, err): (f64, f64)
            = if radix == 10 {
                let trunc: String = format!(
                    r"(({s}?)({n}))\(({nn}|{d}+)\)({e})?",
                    s=SIGN, n=number, nn=NON_NORMAL, d=digit, e=BASE10_EXP,
                );
                let pm: String = format!(
                    r"(({s}?)({n}({e})?))[ ]*(\+[/]?-|\\pm)[ ]*({n}({e})?)",
                    s=SIGN, n=number, e=BASE10_EXP,
                );
                let rgx: String = format!(
                    r"^([$])?({t}|{p})([$])?$",
                    t=trunc, p=pm,
                );
                //    ^
                //  1 ([$])?
                //  2 (
                //  3     (
                //  4         ([+\-]?)
                //  5         (nan|inf|(([0123456789]*\.)?[0123456789]+))
                //        )
                //        \(
                //  8     (nan|inf|[0123456789]+)
                //        \)
                //  9     (e([+\-]?\d+))?
                //    |
                // 11     (
                // 12         ([+\-]?)
                // 13         (nan|inf|(([0123456789]*\.)?[0123456789]+)(e([+\-]?\d+))?)
                //        )
                //        [ ]*
                // 18     (\+[/]?-|\\pm)
                //        [ ]*
                // 19     (nan|inf|(([0123456789]*\.)?[0123456789]+)(e([+\-]?\d+))?)
                //    )
                // 24 ([$])?
                //    $
                let pat = Regex::new(&rgx).unwrap();
                if let Some(cap) = pat.captures(&s.to_lowercase()) {
                    if cap.get(1).is_some() ^ cap.get(24).is_some() {
                        return Err(ExpValError::UnmatchedDollar);
                    }
                    if cap[2].contains('(') {
                        let val = f64::from_str(&cap[3])?;
                        let val_str: String = cap[5].replace('.', "");
                        let z: usize
                            = match val_str.as_ref() {
                                "nan" | "inf" => 0,
                                x => x.len() - 1,
                            };
                        let err = f64::from_str(&cap[8])?;
                        let p = f64::from_str(
                            cap.get(10).map(|g| g.as_str()).unwrap_or("0"))?;
                        (val * 10.0.powf(p), err * 10.0.powf(p - z as f64))
                    } else {
                        let val = f64::from_str(&cap[11])?;
                        let err = f64::from_str(&cap[19])?;
                        (val, err)
                    }
                } else {
                    return Err(ExpValError::MalformedInput(s.to_string()));
                }
            } else {
                let trunc: String = format!(
                    r"(({s}?)({n}))\(({nn}|{d}+)\)",
                    s=SIGN, n=number, nn=NON_NORMAL, d=digit,
                );
                let pm: String = format!(
                    r"(({s}?)({n}))[ ]*(\+[/]?-|\\pm)[ ]*({n})",
                    s=SIGN, n=number,
                );
                let rgx: String = format!(
                    r"^([$])?({t}|{p})([$])?$",
                    t=trunc, p=pm,
                );
                //    ^
                //  1 ([$])?
                //  2 (
                //  3     (
                //  4         ([+\-]?)
                //  5         (nan|inf|(([0123456789abcdef]*\.)?[0123456789abcdef]+))
                //        )
                //        \(
                //  8     (nan|inf|[0123456789abcdef]+)
                //        \)
                //    |
                //  9     (
                // 10         ([+\-]?)
                // 11         (nan|inf|(([0123456789abcdef]*\.)?[0123456789abcdef]+))
                //        )
                //        [ ]*
                // 14     (\+[/]?-|\\pm)
                //        [ ]*
                // 15     (nan|inf|(([0123456789abcdef]*\.)?[0123456789abcdef]+))
                //    )
                // 18 ([$])?
                //    $
                let pat = Regex::new(&rgx).unwrap();
                if let Some(cap) = pat.captures(&s.to_lowercase()) {
                    if cap.get(1).is_some() ^ cap.get(18).is_some() {
                        return Err(ExpValError::UnmatchedDollar);
                    }
                    if cap[2].contains('(') {
                        let val = f64::from_str_radix(
                            if cap[4].contains('+') {
                                &cap[5]
                            } else {
                                &cap[3]
                            },
                            radix,
                        )?;
                        let val_str: String = cap[5].replace('.', "");
                        let z: usize
                            = match val_str.as_ref() {
                                "nan" | "inf" => 0,
                                x => x.len() - 1,
                            };
                        let err = f64::from_str_radix(&cap[8], radix)?;
                        (val, err * (radix as f64).powi(-(z as i32)))
                    } else {
                        let val = f64::from_str_radix(
                            if cap[10].contains('+') {
                                &cap[11]
                            } else {
                                &cap[9]
                            },
                            radix,
                        )?;
                        let err = f64::from_str_radix(&cap[15], radix)?;
                        (val, err)
                    }
                } else {
                    return Err(ExpValError::MalformedInput(s.to_string()));
                }
            };
        Ok(ExpVal::new(val, err))
    }
}

impl FromStr for ExpVal {
    type Err = ExpValError;

    fn from_str(s: &str) -> ExpValResult<Self> { Self::from_str_radix(s, 10) }
}

impl Neg for ExpVal {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            val: -self.val,
            err: self.err,
        }
    }
}

impl Add<ExpVal> for ExpVal {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            err: (self.err.powi(2) + rhs.err.powi(2)).sqrt(),
        }
    }
}

impl Add<f64> for ExpVal {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Self {
            val: self.val + rhs,
            err: self.err,
        }
    }
}

impl Add<ExpVal> for f64 {
    type Output = ExpVal;

    fn add(self, rhs: ExpVal) -> ExpVal {
        ExpVal {
            val: self + rhs.val,
            err: rhs.err,
        }
    }
}

impl AddAssign<ExpVal> for ExpVal {
    fn add_assign(&mut self, rhs: ExpVal) {
        self.val += rhs.val;
        self.err = (self.err.powi(2) + rhs.err.powi(2)).sqrt();
    }
}

impl AddAssign<f64> for ExpVal {
    fn add_assign(&mut self, rhs: f64) {
        self.val += rhs;
    }
}

impl Sub<ExpVal> for ExpVal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            err: (self.err.powi(2) + rhs.err.powi(2)).sqrt(),
        }
    }
}

impl Sub<f64> for ExpVal {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Self {
            val: self.val - rhs,
            err: self.err,
        }
    }
}

impl Sub<ExpVal> for f64 {
    type Output = ExpVal;

    fn sub(self, rhs: ExpVal) -> ExpVal {
        ExpVal {
            val: self - rhs.val,
            err: rhs.err,
        }
    }
}

impl SubAssign<ExpVal> for ExpVal {
    fn sub_assign(&mut self, rhs: ExpVal) {
        self.val -= rhs.val;
        self.err = (self.err.powi(2) + rhs.err.powi(2)).sqrt();
    }
}

impl SubAssign<f64> for ExpVal {
    fn sub_assign(&mut self, rhs: f64) {
        self.val -= rhs;
    }
}

impl Mul<ExpVal> for ExpVal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            err: (
                (self.err * rhs.val).powi(2)
                + (self.val * rhs.err).powi(2)
            ).sqrt(),
        }
    }
}

impl Mul<f64> for ExpVal {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            val: self.val * rhs,
            err: self.err * rhs.abs(),
        }
    }
}

impl Mul<ExpVal> for f64 {
    type Output = ExpVal;

    fn mul(self, rhs: ExpVal) -> ExpVal {
        ExpVal {
            val: self * rhs.val,
            err: self.abs() * rhs.err,
        }
    }
}

impl MulAssign<ExpVal> for ExpVal {
    fn mul_assign(&mut self, rhs: ExpVal) {
        self.val *= rhs.val;
        self.err = (
            (self.err * rhs.val).powi(2)
            + (self.val * rhs.err).powi(2)
        ).sqrt();
    }
}

impl MulAssign<f64> for ExpVal {
    fn mul_assign(&mut self, rhs: f64) {
        self.val *= rhs;
        self.err *= rhs.abs();
    }
}

impl Div<ExpVal> for ExpVal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            err: (
                (self.err / rhs.val).powi(2)
                + (rhs.err * self.val / rhs.val.powi(2)).powi(2)
            ).sqrt(),
        }
    }
}

impl Div<f64> for ExpVal {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            val: self.val / rhs,
            err: self.err / rhs.abs(),
        }
    }
}

impl Div<ExpVal> for f64 {
    type Output = ExpVal;

    fn div(self, rhs: ExpVal) -> ExpVal {
        ExpVal {
            val: self / rhs.val,
            err: rhs.err * self.abs() / rhs.val.powi(2),
        }
    }
}

impl DivAssign<ExpVal> for ExpVal {
    fn div_assign(&mut self, rhs: ExpVal) {
        self.val /= rhs.val;
        self.err = (
            (self.err / rhs.val).powi(2)
            + (rhs.err * self.val / rhs.val.powi(2)).powi(2)
        ).sqrt();
    }
}

impl DivAssign<f64> for ExpVal {
    fn div_assign(&mut self, rhs: f64) {
        self.val /= rhs;
        self.err = self.err / rhs.abs();
    }
}

impl Rem<ExpVal> for ExpVal {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self {
            val: self.val.rem_euclid(rhs.val),
            err: (
                self.err.powi(2)
                + (self.val.div_euclid(rhs.val) * rhs.err).powi(2)
            ).sqrt(),
        }
    }
}

impl Rem<f64> for ExpVal {
    type Output = Self;

    fn rem(self, rhs: f64) -> Self {
        Self {
            val: self.val.rem_euclid(rhs),
            err: self.err,
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Rem<ExpVal> for f64 {
    type Output = ExpVal;

    fn rem(self, rhs: ExpVal) -> ExpVal {
        ExpVal {
            val: self.rem_euclid(rhs.val),
            err: self.div_euclid(rhs.val).abs() * rhs.err,
        }
    }
}

impl RemAssign<ExpVal> for ExpVal {
    fn rem_assign(&mut self, rhs: ExpVal) {
        self.val = self.val.rem_euclid(rhs.val);
        self.err = (
            self.err.powi(2)
            + (self.val.div_euclid(rhs.val) * rhs.err).powi(2)
        ).sqrt();
    }
}

impl RemAssign<f64> for ExpVal {
    fn rem_assign(&mut self, rhs: f64) {
        self.val = self.val.rem_euclid(rhs);
    }
}

impl fmt::Display for ExpVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: String
            = value_str(
                self.val,
                self.err,
                !f.alternate(),
                f.sign_plus(),
                ValueStrSci::No,
                f.fill() == '$',
                f.precision(),
            );
        if let (Some(width), Some(align)) = (f.width(), f.align()) {
            let w: usize = s.len();
            let c: String
                = if f.fill() == '$' {
                    ' '.to_string()
                } else {
                    f.fill().to_string()
                };
            if w < width {
                let mut out: String = "".to_string();
                let (fill_l, fill_r): (usize, usize)
                    = match align {
                        fmt::Alignment::Left => (0, width - w),
                        fmt::Alignment::Right => (width - w, 0),
                        fmt::Alignment::Center
                            => (
                                (width - w) / 2,
                                width - w - (width - w) / 2,
                            ),
                    };
                out += &c.repeat(fill_l);
                out += &s;
                out += &c.repeat(fill_r);
                write!(f, "{}", out)?;
            } else {
                write!(f, "{}", s)?;
            }
        } else {
            write!(f, "{}", s)?;
        }
        Ok(())
    }
}

impl fmt::LowerExp for ExpVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: String
            = value_str(
                self.val,
                self.err,
                !f.alternate(),
                f.sign_plus(),
                ValueStrSci::Lower,
                f.fill() == '$',
                f.precision(),
            );
        if let (Some(width), Some(align)) = (f.width(), f.align()) {
            let w: usize = s.len();
            let c: String
                = if f.fill() == '$' {
                    ' '.to_string()
                } else {
                    f.fill().to_string()
                };
            if w < width {
                let mut out: String = "".to_string();
                let (fill_l, fill_r): (usize, usize)
                    = match align {
                        fmt::Alignment::Left => (0, width - w),
                        fmt::Alignment::Right => (width - w, 0),
                        fmt::Alignment::Center
                            => (
                                (width - w) / 2,
                                width - w - (width - w) / 2,
                            ),
                    };
                out += &c.repeat(fill_l);
                out += &s;
                out += &c.repeat(fill_r);
                write!(f, "{}", out)?;
            } else {
                write!(f, "{}", s)?;
            }
        } else {
            write!(f, "{}", s)?;
        }
        Ok(())
    }
}

impl fmt::UpperExp for ExpVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: String
            = value_str(
                self.val,
                self.err,
                !f.alternate(),
                f.sign_plus(),
                ValueStrSci::Upper,
                f.fill() == '$',
                f.precision(),
            );
        if let (Some(width), Some(align)) = (f.width(), f.align()) {
            let w: usize = s.len();
            let c: String
                = if f.fill() == '$' {
                    ' '.to_string()
                } else {
                    f.fill().to_string()
                };
            if w < width {
                let mut out: String = "".to_string();
                let (fill_l, fill_r): (usize, usize)
                    = match align {
                        fmt::Alignment::Left => (0, width - w),
                        fmt::Alignment::Right => (width - w, 0),
                        fmt::Alignment::Center
                            => (
                                (width - w) / 2,
                                width - w - (width - w) / 2,
                            ),
                    };
                out += &c.repeat(fill_l);
                out += &s;
                out += &c.repeat(fill_r);
                write!(f, "{}", out)?;
            } else {
                write!(f, "{}", s)?;
            }
        } else {
            write!(f, "{}", s)?;
        }
        Ok(())
    }
}

