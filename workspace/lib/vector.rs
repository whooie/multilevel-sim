//! Fixed-dimension vectors.

#![allow(clippy::mem_replace_with_uninit, clippy::uninit_assumed_init)]

use std::{
    mem,
    ops::{
        Add, AddAssign,
        Sub, SubAssign,
        Mul, MulAssign,
        Index, IndexMut,
    },
};
use num_traits::{ Zero, One };

/// A fixed-dimension vector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<const N: usize, T>(pub [T; N]);

impl<const N: usize, T, U> AsRef<U> for Vector<N, T>
where [T; N]: AsRef<U>
{
    fn as_ref(&self) -> &U { self.0.as_ref() }
}

impl<const N: usize, T, U> AsMut<U> for Vector<N, T>
where [T; N]: AsMut<U>
{
    fn as_mut(&mut self) -> &mut U { self.0.as_mut() }
}

impl<const N: usize, T: Zero + Copy> Vector<N, T> {
    /// Create a vector of all zeros.
    pub fn zeros() -> Self { Self([T::zero(); N]) }
}

impl<const N: usize, T: Ones + Copy> Vector<N, T> {
    /// Create a vector of all ones.
    pub fn ones() -> Self { Self([T::ones(); N]) }
}

impl<const N: usize, T> Vector<N, T>
where Self: Dot<Output = T>
{
    /// Return the dot product of two vectors.
    pub fn dot(&self, rhs: &Self) -> T { Dot::dot(self, rhs) }
}

impl<const N: usize, T: Zero + Ones + Copy> Vector<N, T> {
    /// Create a vector with 1 at the `k`-th index and zeros elsewhere.
    ///
    /// This is equivalent to `zeros` if `k ≥ N`.
    pub fn unit(k: usize) -> Self {
        let mut v = Self::zeros();
        if let Some(elem) = v.0.get_mut(k) {
            *elem = T::ones();
        }
        v
    }
}

impl<const N: usize, T> Vector<N, T> {
    /// Return an iterator over references to all elements.
    pub fn iter(&self) -> <&[T; N] as IntoIterator>::IntoIter {
        self.0.iter()
    }

    /// Return an iterator over mutable references to all elements.
    pub fn iter_mut(&mut self) -> <&mut [T; N] as IntoIterator>::IntoIter {
        self.0.iter_mut()
    }

    /// Safe immutable indexing.
    pub fn get(&self, k: usize) -> Option<&T> { self.0.get(k) }

    /// Safe mutable indexing.
    pub fn get_mut(&mut self, k: usize) -> Option<&mut T> { self.0.get_mut(k) }

    /// Call a function on each element, returning results in a new vector.
    pub fn map<U, F>(&self, mut f: F) -> Vector<N, U>
    where F: FnMut(&T) -> U
    {
        let mut new: [U; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        self.0.iter().zip(new.iter_mut())
            .for_each(|(x, y)| { *y = f(x); });
        Vector(new)
    }

    /// Call a function on each element, returning results in a new vector.
    pub fn map_into<U, F>(self, mut f: F) -> Vector<N, U>
    where F: FnMut(T) -> U
    {
        let mut new: [U; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        self.0.into_iter().zip(new.iter_mut())
            .for_each(|(x, y)| { *y = f(x); });
        Vector(new)
    }

    /// Call a function on each element by value, returning results in a new
    /// vector.
    pub fn mapv<U, F>(&self, mut f: F) -> Vector<N, U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        let mut new: [U; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        self.0.iter().zip(new.iter_mut())
            .for_each(|(x, y)| { *y = f(x.clone()); });
        Vector(new)
    }

    /// Call a function on only a single element, returning results in a new
    /// vector.
    ///
    /// This is equivalent to a `clone` if `k ≥ N`.
    pub fn map_at<F>(&self, k: usize, f: F) -> Vector<N, T>
    where
        F: FnOnce(T) -> T,
        T: Clone,
    {
        let mut new = self.clone();
        if let Some(elem) = new.0.get_mut(k) {
            *elem = f(elem.clone());
        }
        new
    }

    /// Call a function on only a single element.
    pub fn map_at_into<F>(mut self, k: usize, f: F) -> Vector<N, T>
    where F: FnOnce(T) -> T
    {
        if let Some(elem) = self.0.get_mut(k) {
            let val = unsafe { mem::replace(elem, mem::zeroed()) };
            *elem = f(val);
        }
        self
    }

    /// Return the results of calling a function on every single element
    /// individually.
    ///
    /// See also [`at_each_iter`][Self::at_each_iter] to compute results lazily.
    pub fn at_each<F>(&self, mut f: F) -> [Vector<N, T>; N]
    where
        T: Clone,
        F: FnMut(T) -> T,
    {
        let mut new: [Vector<N, T>; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        new.iter_mut().enumerate()
            .for_each(|(k, each)| {
                *each = self.clone().map_at_into(k, &mut f);
            });
        new
    }

    /// Like [`at_each`][Self::at_each], but returning results in an iterator.
    ///
    /// This has the advantage of computing results lazily, but lacks the
    /// additional type safety of a fixed-size array.
    pub fn at_each_iter<F>(&self, f: F) -> AtEachIter<N, T, F>
    where
        T: Clone,
        F: FnMut(T) -> T,
    {
        AtEachIter { orig: self.clone(), f, k: 0, k_rev: N - 1 }
    }
}

impl<const N: usize, T> IntoIterator for Vector<N, T> {
    type Item = T;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<const N: usize, T> TryFrom<Vec<T>> for Vector<N, T> {
    type Error = <[T; N] as TryFrom<Vec<T>>>::Error;

    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        let data = <[T; N]>::try_from(vec)?;
        Ok(Self(data))
    }
}

impl<const N: usize, T, I> Index<I> for Vector<N, T>
where [T; N]: Index<I>
{
    type Output = <[T; N] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output { &self.0[index] }
}

impl<const N: usize, T, I> IndexMut<I> for Vector<N, T>
where [T; N]: IndexMut<I>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output { &mut self.0[index] }
}

impl<const N: usize, T: Zero + PartialEq + Copy> Zero for Vector<N, T> {
    fn zero() -> Self { Self([T::zero(); N]) }

    fn is_zero(&self) -> bool {
        let z = T::zero();
        self.0.iter().all(|x| x == &z)
    }
}

/// Like [`One`], but for vectors (i.e. without the extra [`Mul`] bound).
pub trait Ones {
    fn ones() -> Self;
}

impl<T: One> Ones for T {
    fn ones() -> Self { Self::one() }
}

impl<const N: usize, T: Ones + Copy> Ones for Vector<N, T> {
    fn ones() -> Self { Self([T::ones(); N]) }
}

/// Yields the results of calling a function on every single element of a
/// [`Vector`] individually.
#[derive(Copy, Clone, Debug)]
pub struct AtEachIter<const N: usize, T, F> {
    orig: Vector<N, T>,
    f: F,
    k: usize,
    k_rev: usize,
}

impl<const N: usize, T, F> Iterator for AtEachIter<N, T, F>
where
    T: Clone,
    F: FnMut(T) -> T,
{
    type Item = Vector<N, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.k < self.k_rev {
            let mapped = self.orig.clone().map_at_into(self.k, &mut self.f);
            self.k += 1;
            Some(mapped)
        } else {
            None
        }
    }
}

impl<const N: usize, T, F> DoubleEndedIterator for AtEachIter<N, T, F>
where
    T: Clone,
    F: FnMut(T) -> T,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.k < self.k_rev {
            let mapped = self.orig.clone().map_at_into(self.k_rev, &mut self.f);
            if self.k_rev == 0 {
                self.k += 1;
            } else {
                self.k_rev -= 1;
            }
            Some(mapped)
        } else {
            None
        }
    }
}

impl<const N: usize, T, F> ExactSizeIterator for AtEachIter<N, T, F>
where
    T: Clone,
    F: FnMut(T) -> T,
{
    fn len(&self) -> usize {
        if self.k < self.k_rev { self.k_rev - self.k + 1 } else { 0 }
    }
}

impl<const N: usize, T, F> std::iter::FusedIterator for AtEachIter<N, T, F>
where
    T: Clone,
    F: FnMut(T) -> T,
{ }

/// Vector dot product.
pub trait Dot {
    type Output;

    fn dot(&self, rhs: &Self) -> Self::Output;
}

impl<T> Dot for T
where for<'a> &'a T: Mul<&'a T, Output = T>
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self { self * rhs }
}

impl<const N: usize, U, T> Dot for Vector<N, U>
where
    U: Dot<Output = T>,
    T: Dot<Output = T> + Add<T, Output = T> + Zero,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.0.iter().zip(rhs.0.iter())
            .fold(T::zero(), |acc, (l, r)| acc + l.dot(r))
    }
}

impl<const N: usize, T, U> Add<Vector<N, U>> for Vector<N, T>
where T: Add<U, Output = T>
{
    type Output = Self;

    fn add(mut self, rhs: Vector<N, U>) -> Self::Output {
        self.0.iter_mut().zip(rhs.0)
            .for_each(|(l, r)| unsafe {
                let l_val = mem::replace(l, mem::zeroed());
                *l = l_val + r;
            });
        self
    }
}

impl<const N: usize, T, U, V> Add<&Vector<N, U>> for &Vector<N, T>
where for<'a> &'a T: Add<&'a U, Output = V>
{
    type Output = Vector<N, V>;

    fn add(self, rhs: &Vector<N, U>) -> Self::Output {
        let mut data: [V; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        data.iter_mut()
            .zip(self.0.iter().zip(rhs.0.iter()))
            .for_each(|(res, (l, r))| { *res = l + r; });
        Vector(data)
    }
}

impl<const N: usize, T, U> AddAssign<Vector<N, U>> for Vector<N, T>
where T: AddAssign<U>
{
    fn add_assign(&mut self, rhs: Vector<N, U>) {
        self.0.iter_mut().zip(rhs.0)
            .for_each(|(l, r)| { *l += r; });
    }
}

impl<const N: usize, T, U> AddAssign<&Vector<N, U>> for Vector<N, T>
where for<'a> T: AddAssign<&'a U>
{
    fn add_assign(&mut self, rhs: &Vector<N, U>) {
        self.0.iter_mut().zip(rhs.0.iter())
            .for_each(|(l, r)| { *l += r; });
    }
}

impl<const N: usize, T, U> Sub<Vector<N, U>> for Vector<N, T>
where T: Sub<U, Output = T>
{
    type Output = Self;

    fn sub(mut self, rhs: Vector<N, U>) -> Self::Output {
        self.0.iter_mut().zip(rhs.0)
            .for_each(|(l, r)| unsafe {
                let l_val = mem::replace(l, mem::zeroed());
                *l = l_val - r;
            });
        self
    }
}

impl<const N: usize, T, U, V> Sub<&Vector<N, U>> for &Vector<N, T>
where for<'a> &'a T: Sub<&'a U, Output = V>
{
    type Output = Vector<N, V>;

    fn sub(self, rhs: &Vector<N, U>) -> Self::Output {
        let mut data: [V; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        data.iter_mut()
            .zip(self.0.iter().zip(rhs.0.iter()))
            .for_each(|(res, (l, r))| { *res = l - r; });
        Vector(data)
    }
}

impl<const N: usize, T, U> SubAssign<Vector<N, U>> for Vector<N, T>
where T: SubAssign<U>
{
    fn sub_assign(&mut self, rhs: Vector<N, U>) {
        self.0.iter_mut().zip(rhs.0)
            .for_each(|(l, r)| { *l -= r; });
    }
}

impl<const N: usize, T, U> SubAssign<&Vector<N, U>> for Vector<N, T>
where for<'a> T: SubAssign<&'a U>
{
    fn sub_assign(&mut self, rhs: &Vector<N, U>) {
        self.0.iter_mut().zip(rhs.0.iter())
            .for_each(|(l, r)| { *l -= r; });
    }
}

impl<const N: usize, U, T> Mul<T> for Vector<N, U>
where
    T: Clone,
    U: Mul<T, Output = U>,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.0.iter_mut()
            .for_each(|l| unsafe {
                let l_val = mem::replace(l, mem::zeroed());
                *l = l_val * rhs.clone();
            });
        self
    }
}

impl<const N: usize, U, T> MulAssign<T> for Vector<N, U>
where
    T: Clone,
    U: MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0.iter_mut()
            .for_each(|l| { *l *= rhs.clone(); });
    }
}

macro_rules! impl_scalar_mul {
    ( $ty:ty ) => {
        impl<const N: usize, U> Mul<Vector<N, U>> for $ty
        where Self: Clone + Mul<U, Output = U>
        {
            type Output = Vector<N, U>;

            fn mul(self, mut rhs: Vector<N, U>) -> Self::Output {
                rhs.0.iter_mut()
                    .for_each(|r| unsafe {
                        let r_val = mem::replace(r, mem::zeroed());
                        *r = self.clone() * r_val;
                    });
                rhs
            }
        }
    }
}
impl_scalar_mul!(u8);
impl_scalar_mul!(u16);
impl_scalar_mul!(u32);
impl_scalar_mul!(u64);
impl_scalar_mul!(u128);
impl_scalar_mul!(usize);
impl_scalar_mul!(i8);
impl_scalar_mul!(i16);
impl_scalar_mul!(i32);
impl_scalar_mul!(i64);
impl_scalar_mul!(i128);
impl_scalar_mul!(isize);
impl_scalar_mul!(f32);
impl_scalar_mul!(f64);

#[cfg(test)]
mod test {
    use super::Vector;

    #[test]
    fn testo() {
        const N: usize = 10;
        let v: Vector<N, f64> = Vector::zeros();
        let u: Vector<N, f64> = v.map(|x| x + 1.0);
        assert_eq!(u, Vector::<N, f64>::ones());
    }
}

