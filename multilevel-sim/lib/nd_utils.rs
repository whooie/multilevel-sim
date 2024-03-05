//! Provides macros for reading and writing `.npz` files and an implementation
//! of the `FExtremum` trait for `ndarray::Array` (see [FExtremum]).

use ndarray::{
    self as nd,
    s,
};
use crate::{
    utils::{
        FExtremum,
    },
};

pub extern crate ndarray_npy;

/// Write arrays to a `.npz` file.
///
/// Expects the file location as an `impl AsRef<Path> + Clone`, with keys as
/// `impl Into<String>` and arrays as `&ndarray::Array`.
#[macro_export]
macro_rules! write_npz {
    (
        $filepath:expr,
        arrays: { $( $key:expr => $arrayref:expr ),+ $(,)? }
    ) => {
        {
            let mut _output_filepath_ = $filepath.clone();
            let mut _output_writer_
                = $crate::nd_utils::ndarray_npy::NpzWriter::new(
                    std::fs::File::create(_output_filepath_.clone())
                        .expect(
                            format!(
                                "Couldn't create npz file '{:?}'",
                                _output_filepath_.clone()
                            ).as_str()
                        )
                );
            $(
                _output_writer_.add_array($key, $arrayref)
                    .expect(
                        format!(
                            "Couldn't add array under key '{}' to file",
                            $key
                        ).as_str()
                    );
            )+
            _output_writer_.finish()
                .expect("Couldn't write out data file");
        }
    }
}

/// Write arrays to a `.npz` file with compression.
///
/// Expects the file location as an `impl AsRef<Path> + Clone`, with keys as
/// `impl Into<String>` and arrays as `&ndarray::Array`.
#[macro_export]
macro_rules! write_npz_compressed {
    (
        $filepath:expr,
        arrays: { $( $key:expr => $arrayref:expr ),+ $(,)? }
    ) => {
        {
            let mut _output_filepath_ = $filepath.clone();
            let mut _output_writer_
                = $crate::nd_utils::ndarray_npy::NpzWriter::new_compressed(
                    std::fs::File::create(_output_filepath_.clone())
                        .expect(
                            format!(
                                "Couldn't create npz file '{:?}'",
                                _output_filepath_.clone()
                            ).as_str()
                        )
                );
        $(
            _output_writer_.add_array($key, $arrayref)
                .expect(
                    format!(
                        "Couldn't add array under key '{}' to file",
                        $key
                    ).as_str()
                );
        )+
        _output_writer_.finish()
            .expect("Couldn't write out data file");
        }
    }
}

/// Read arrays from a `.npz` file, optionally pulling specific keys.
///
/// Expects the file location as an `impl AsRef<Path> + Clone`, with keys as
/// `&str`. Note that keys coming directly from Numpy have an extra `.npy`
/// appended.
#[macro_export]
macro_rules! read_npz {
    ( $filepath:expr ) => {
        {
            let _input_filepath_ = $filepath.clone();
            $crate::nd_utils::ndarray_npy::NpzReader::new(
                    std::fs::File::open(_input_filepath_.clone())
                        .expect(
                            format!(
                                "Couldn't open npz file '{:?}'",
                                _input_filepath_.clone()
                            ).as_str()
                        )
                ).expect(
                    format!(
                        "Couldn't read npz file '{:?}'",
                        _input_filepath_
                    ).as_str()
                )
        }
    };
    (
        $filepath:expr,
        arrays: { $( $key_str:expr ),+ $(,)? }
    ) => {
        {
            let mut _input_reader_ = read_npz!($filepath);
            ( $(
                _input_reader_.by_name($key_str)
                    .expect(
                        format!(
                            "Couldn't retrieve array for key '{:?}'",
                            $key_str
                        ).as_str()
                    )
            ),+ )
        }
    }
}

macro_rules! impl_fextremum_ndarray {
    ( $arrtype:ty, $f:ty ) => {
        impl<D> FExtremum<$f> for $arrtype
        where D: ndarray::Dimension
        {
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
}

impl_fextremum_ndarray!(ndarray::Array<f64, D>, f64);
impl_fextremum_ndarray!(ndarray::Array<f32, D>, f32);
impl_fextremum_ndarray!(ndarray::ArrayView<'_, f64, D>, f64);
impl_fextremum_ndarray!(ndarray::ArrayView<'_, f32, D>, f32);

pub trait RotateIndex {
    fn rotate_l(&self, mid: usize) -> Self;
    fn rotate_r(&self, mid: usize) -> Self;
}

impl<T> RotateIndex for nd::Array1<T>
where T: Clone
{
    fn rotate_l(&self, mid: usize) -> Self {
        let n: usize = self.len();
        nd::concatenate(
            nd::Axis(0),
            &[
                self.slice(s![mid..n]),
                self.slice(s![0..mid]),
            ],
        ).unwrap()
    }

    fn rotate_r(&self, mid: usize) -> Self {
        let n: usize = self.len();
        nd::concatenate(
            nd::Axis(0),
            &[
                self.slice(s![n - mid..n]),
                self.slice(s![0..n - mid]),
            ],
        ).unwrap()
    }
}

pub trait RotateIndexAxis {
    fn rotate_axis_l(&self, axis: usize, mid: usize) -> Self;
    fn rotate_axis_r(&self, axis: usize, mid: usize) -> Self;
}

macro_rules! impl_rotateindexaxis {
    { $array:ty } => {
        impl<T> RotateIndexAxis for $array
        where T: Clone
        {
            fn rotate_axis_l(&self, axis: usize, mid: usize) -> Self {
                let n: usize = self.shape()[axis];
                nd::concatenate(
                    nd::Axis(axis),
                    &[
                        self.slice_axis(
                            nd::Axis(axis),
                            nd::Slice::from(mid..n),
                        ),
                        self.slice_axis(
                            nd::Axis(axis),
                            nd::Slice::from(0..mid),
                        ),
                    ],
                ).unwrap()
            }

            fn rotate_axis_r(&self, axis: usize, mid: usize) -> Self {
                let n: usize = self.shape()[axis];
                nd::concatenate(
                    nd::Axis(axis),
                    &[
                        self.slice_axis(
                            nd::Axis(axis),
                            nd::Slice::from(n - mid..n),
                        ),
                        self.slice_axis(
                            nd::Axis(axis),
                            nd::Slice::from(0..n - mid),
                        ),
                    ],
                ).unwrap()
            }
        }
    }
}
impl_rotateindexaxis!(nd::Array2<T>);
impl_rotateindexaxis!(nd::Array3<T>);
impl_rotateindexaxis!(nd::Array4<T>);
impl_rotateindexaxis!(nd::Array5<T>);
impl_rotateindexaxis!(nd::Array6<T>);
impl_rotateindexaxis!(nd::ArrayD<T>);

