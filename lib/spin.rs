//! Definitions for angular momentum quantum numbers and related quantities.

use std::hash::Hash;
use wigner_symbols::{ ClebschGordan, Wigner3jm, Wigner6j };

/// A single spin-projection quantum number.
///
/// This type is backed by a single `i32` representing the number of halves.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpinProj(i32);

impl SpinProj {
    /// Create a new spin projection.
    pub fn new(m: i32) -> Self { Self(m) }

    /// Raise the projection quantum number by 1.
    ///
    /// This operation will prevent overflowing the underlying `i32` by
    /// saturating to the maximum possible value if needed.
    pub fn raise(&mut self) { self.0 = self.0.saturating_add(2); }

    /// Return a [raised][Self::raise] copy of `self`.
    pub fn raised(self) -> Self { Self(self.0.saturating_add(2)) }

    /// Lower the projection quantum number by 1.
    ///
    /// This operation will prevent underflowing the underlying `i32` by
    /// saturating to the minimum possible value if needed.
    pub fn lower(&mut self) { self.0 = self.0.saturating_sub(2); }

    /// Return a [lowered][Self::lower] copy of `self`.
    pub fn lowered(self) -> Self { Self(self.0.saturating_sub(2)) }

    /// Reflect the projection quantum number across the orthogonal plane, i.e.
    /// apply a minus sign.
    pub fn reflect(&mut self) { self.0 = -self.0 }

    /// Return a [reflected][Self::reflect] copy of `self`.
    pub fn reflected(self) -> Self { Self(-self.0) }

    /// Return `self` as a bare number of halves.
    pub fn halves(self) -> i32 { self.0 }

    /// Return `self` as an `f64`.
    ///
    /// This reflects the "true" value of the projection quantum number; i.e.
    /// there is a relative factor of 2 between this and [`Self::halves`].
    pub fn f(self) -> f64 { f64::from(self.0) / 2.0 }

    /// Create a new spin-projection quantum number from a `f64` value, rounding
    /// to the nearest half-integer.
    pub fn from_f64(f: f64) -> Self { Self((2.0 * f).round() as i32) }
}

impl std::ops::Deref for SpinProj {
    type Target = i32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<M> From<M> for SpinProj
where M: Into<i32>
{
    fn from(m: M) -> Self { Self(m.into()) }
}

impl From<SpinProj> for f64 {
    fn from(m: SpinProj) -> Self { m.f() }
}

/// A single total-spin quantum number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpinTotal(u32);

impl SpinTotal {
    /// Create a new total spin.
    pub fn new(j: u32) -> Self { Self(j) }

    /// Return `self` as a bare number of halves.
    pub fn halves(self) -> u32 { self.0 }

    /// Return `self` as an `f64`.
    ///
    /// This reflects the "true" numerical value of the total-spin quantum
    /// number; i.e. there is a relative factor of 2 between this and
    /// [`Self::halves`].
    pub fn f(self) -> f64 { f64::from(self.0) / 2.0 }

    /// Create a new total-spin quantum number from a `f64` value, rounding
    /// to the nearest half-integer.
    ///
    /// Negative inputs are passed through [`f64::abs`] before rounding.
    pub fn from_f64(f: f64) -> Self { Self((2.0 * f.abs()).round() as u32) }

    /// Return an iterator over available [`Spin`] pairs by ascending projection
    /// number.
    pub fn iter(self) -> SpinProjections {
        SpinProjections { cur: Spin(self, SpinProj(-(self.0 as i32))) }
    }

    /// Return a reversed iterator over available [`Spin`] pairs by descending
    /// projection number.
    pub fn iter_rev(self) -> SpinProjectionsRev {
        SpinProjectionsRev { cur: Spin(self, SpinProj(self.0 as i32)) }
    }
}

impl IntoIterator for SpinTotal {
    type IntoIter = SpinProjections;
    type Item = Spin;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl std::ops::Deref for SpinTotal {
    type Target = u32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<J> From<J> for SpinTotal
where J: Into<u32>
{
    fn from(j: J) -> Self { Self(j.into()) }
}

impl From<SpinTotal> for f64 {
    fn from(j: SpinTotal) -> Self { j.f() }
}

/// A `(total, projection)` spin quantum number pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spin(SpinTotal, SpinProj);

impl Spin {
    /// Create a new spin if the given spin-projection number is valid for the
    /// given total-spin number.
    pub fn new(j: u32, m: i32) -> Option<Self> {
        let j_i64 = i64::from(j);
        let m_i64 = i64::from(m);
        (
            (-j_i64..=j_i64).contains(&m_i64)
            && m_i64.saturating_sub(j_i64) % 2 == 0
        )
        .then_some(Self(SpinTotal(j), SpinProj(m)))
    }

    /// Return the [total-spin][SpinTotal] quantum number.
    pub fn total(self) -> SpinTotal { self.0 }

    /// Return the [spin-projection][SpinProj] quantum number.
    pub fn proj(self) -> SpinProj { self.1 }

    /// Raise the projection quantum number by 1.
    ///
    /// This operation is silently checked so that no change will occur if it
    /// would have caused the projection to exceed the total in magnitude or if
    /// it would have overflowed the underlying `i32` value.
    pub fn raise(&mut self) {
        self.1.0
            = if
                i64::from(self.1.0) <= (i64::from(self.0.0) - 2).max(0)
                && self.1.0 <= i32::MAX - 2
            {
                self.1.0 + 2
            } else {
                self.1.0
            };
    }

    /// Return a [raised][Self::raise] copy of `self`.
    pub fn raised(self) -> Self {
        let mut new = self;
        new.raise();
        new
    }

    /// [Raise][Self::raise] the projection quantum number by 1 and return the
    /// result.
    ///
    /// This operation is silently checked so that no change will occur if it
    /// would have caused the projection to exceed the total in magnitude or if
    /// it would have overflowed the underlying `i32` value.
    pub fn raise_checked(&mut self) -> Option<&mut Self> {
        if
            i64::from(self.1.0) <= (i64::from(self.0.0) - 2).max(0)
            && self.1.0 <= i32::MAX - 2
        {
            self.1.0 += 2;
            Some(self)
        } else {
            None
        }
    }

    /// Return a [raised][Self::raise] copy of `self` if `self` is not already
    /// [stretched][Self::is_stretched_pos].
    pub fn raised_checked(self) -> Option<Self> {
        let mut new = self;
        new.raise_checked().copied()
    }

    /// Lower the projection quantum number by 1.
    ///
    /// This operation is silently checked so that no change will occur if it
    /// would have caused the projection to exceed the total in magnitude or if
    /// it would have underflowed the underlying `i32` value.
    pub fn lower(&mut self) {
        self.1.0
            = if
                i64::from(self.1.0) >= (-i64::from(self.0.0) + 2).min(0)
                && self.1.0 >= i32::MIN + 2
            {
                self.1.0 - 2
            } else {
                self.1.0
            };
    }

    /// Return a [lowered][Self::lower] copy of `self`.
    pub fn lowered(self) -> Self {
        let mut new = self;
        new.lower();
        new
    }

    /// [Lower][Self::lower] the projection quantum number by 1 and return the
    /// result.
    ///
    /// This operation is silently checked so that no change will occur if it
    /// would have caused the projection to exceed the total in magnitude or if
    /// it would have underflowed the underlying `i32` value.
    pub fn lower_checked(&mut self) -> Option<&mut Self> {
        if
            i64::from(self.1.0) >= (-i64::from(self.0.0) + 2).min(0)
            && self.1.0 >= i32::MIN + 2
        {
            self.1.0 -= 2;
            Some(self)
        } else {
            None
        }
    }

    /// Return a [raised][Self::raise] copy of `self` if `self` is not already
    /// [stretched][Self::is_stretched_neg].
    pub fn lowered_checked(self) -> Option<Self> {
        let mut new = self;
        new.lower_checked().copied()
    }

    /// Reflect the projection quantum number across the orthogonal plane, i.e.
    /// apply a minus sign.
    pub fn reflect(&mut self) { self.1.reflect() }

    /// Return a [reflected][Self::reflect] copy of `self`.
    pub fn reflected(self) -> Self { Self(self.0, self.1.reflected()) }

    /// Return `true` if the projection number saturates the upper end of the
    /// range of available spin values.
    pub fn is_stretched_pos(self) -> bool {
        i64::from(self.1.0) >= i64::from(self.0.0)
    }

    /// Return `true` if the projection number saturates the lower end of the
    /// range of availables spin values.
    pub fn is_stretched_neg(self) -> bool {
        i64::from(self.1.0) <= -i64::from(self.0.0)
    }

    /// Return `true` if the projection number saturates the range of available
    /// spin values.
    pub fn is_stretched(self) -> bool {
        let j_i64 = i64::from(self.0.0);
        let m_i64 = i64::from(self.1.0);
        (m_i64 >= j_i64) || (m_i64 <= -j_i64)
    }

    /// Return `self` as a bare pair of halves.
    pub fn halves(self) -> (u32, i32) { (self.0.halves(), self.1.halves()) }

    /// Return `self` as a `(f64, f64)`.
    ///
    /// This reflects the "true" numerical values of the quantum numbers; i.e.
    /// there are factors of two relative to both numbers returned by
    /// [`Self::halves`].
    pub fn f(self) -> (f64, f64) { (self.0.f(), self.1.f()) }

    /// Create new spin quantum numbers from a pair of `f64`s, rounding to the
    /// nearest half-integers according to [`SpinTotal::from_f64`] and
    /// [`SpinProj::from_f64`].
    pub fn from_f64(f: (f64, f64)) -> Self {
        Self(SpinTotal::from_f64(f.0), SpinProj::from_f64(f.1))
    }
}

impl<J, M> From<(J, M)> for Spin
where
    J: Into<SpinTotal>,
    M: Into<SpinProj>,
{
    fn from(jm: (J, M)) -> Self {
        let (j, m) = jm;
        Self::new(j.into().halves(), m.into().halves())
            .expect("Spin::From: invalid spin total-projection combination")
    }
}

impl From<Spin> for (f64, f64) {
    fn from(jm: Spin) -> Self { jm.f() }
}

/// Iterator over spin projection states for a fixed total spin magnitude.
///
/// Projection states are visited in ascending order.
#[derive(Copy, Clone, Debug)]
pub struct SpinProjections {
    cur: Spin,
}

impl Iterator for SpinProjections {
    type Item = Spin;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur.raise_checked().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.cur.0.0 as usize + 1;
        (n, Some(n))
    }
}

impl ExactSizeIterator for SpinProjections { }

/// Reverse iterator over spin projection states for a fixed total spin
/// magnitude.
///
/// Projection states are visited in descending order.
#[derive(Copy, Clone, Debug)]
pub struct SpinProjectionsRev {
    cur: Spin,
}

impl Iterator for SpinProjectionsRev {
    type Item = Spin;

    fn next(&mut self) -> Option<Self::Item> {
        self.cur.lower_checked().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.cur.0.0 as usize + 1;
        (n, Some(n))
    }
}

impl ExactSizeIterator for SpinProjectionsRev { }

/// Calculate the appropriate Clebsch-Gordan coefficient for the angular
/// momentum addition `s1 + s2 = s3`.
pub fn cg<S1, S2, S3>(s1: S1, s2: S2, s3: S3) -> f64
where
    S1: Into<Spin>,
    S2: Into<Spin>,
    S3: Into<Spin>,
{
    let s1 = s1.into();
    let s2 = s2.into();
    let s3 = s3.into();
    ClebschGordan {
        tj1: s1.total().halves() as i32,
        tm1: s1.proj().halves(),
        tj2: s2.total().halves() as i32,
        tm2: s2.proj().halves(),
        tj12: s3.total().halves() as i32,
        tm12: s3.proj().halves(),
    }
    .value()
    .into()
}

/// Calculate the appropriate Wigner 3j symbol for columns (left to right)
/// `s1..s3`.
pub fn w3j<S1, S2, S3>(s1: S1, s2: S2, s3: S3) -> f64
where
    S1: Into<Spin>,
    S2: Into<Spin>,
    S3: Into<Spin>,
{
    let s1 = s1.into();
    let s2 = s2.into();
    let s3 = s3.into();
    Wigner3jm {
        tj1: s1.total().halves() as i32,
        tm1: s1.proj().halves(),
        tj2: s2.total().halves() as i32,
        tm2: s2.proj().halves(),
        tj3: s3.total().halves() as i32,
        tm3: s3.proj().halves(),
    }
    .value()
    .into()
}

/// Calculate the appropriate Wigner 6j symbol for total spins (by row)
/// `j00..j12`.
pub fn w6j<J00, J01, J02, J10, J11, J12>(
    j00: J00,
    j01: J01,
    j02: J02,
    j10: J10,
    j11: J11,
    j12: J12,
) -> f64
where
    J00: Into<SpinTotal>,
    J01: Into<SpinTotal>,
    J02: Into<SpinTotal>,
    J10: Into<SpinTotal>,
    J11: Into<SpinTotal>,
    J12: Into<SpinTotal>,
{
    let j00 = j00.into();
    let j01 = j01.into();
    let j02 = j02.into();
    let j10 = j10.into();
    let j11 = j11.into();
    let j12 = j12.into();
    Wigner6j {
        tj1: j00.halves() as i32,
        tj2: j01.halves() as i32,
        tj3: j02.halves() as i32,
        tj4: j10.halves() as i32,
        tj5: j11.halves() as i32,
        tj6: j12.halves() as i32,
    }
    .value()
    .into()
}

