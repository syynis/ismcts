use self::search::Node;

use super::*;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

pub unsafe trait TranspositionTable<M: MCTS>: Sync + Sized {
    /// **If this function inserts a value, it must return `None`.** Failure to follow
    /// this rule will lead to memory safety violation.
    ///
    /// Attempts to insert a key/value pair.
    ///
    /// If the key is not present, the table *may* insert it. If the table does
    /// not insert it, the table may either return `None` or a reference to another
    /// value existing in the table. (The latter is allowed so that the table doesn't
    /// necessarily need to handle hash collisions, but it will negatively affect the accuracy
    /// of the search.)
    ///
    /// If the key is present, the table may either:
    /// - Leave the table unchanged and return `Some(reference to associated value)`.
    /// - Leave the table unchanged and return `None`.
    ///
    /// The table *may* choose to replace old values.
    /// The table is *not* responsible for dropping values that are replaced.
    fn insert<'a>(&'a self, key: &M::State, value: &'a Node<M>) -> Option<&'a Node<M>>;

    /// Looks up a key.
    ///
    /// If the key is not present, the table *should almost always* return `None`.
    ///
    /// If the key is present, the table *may return either* `None` or a reference
    /// to the associated value.
    fn lookup<'a>(&'a self, key: &M::State) -> Option<&'a Node<M>>;
}

unsafe impl<M: MCTS> TranspositionTable<M> for () {
    fn insert<'a>(&'a self, _: &M::State, _: &'a Node<M>) -> Option<&'a Node<M>> {
        None
    }

    fn lookup<'a>(&'a self, _: &M::State) -> Option<&'a Node<M>> {
        None
    }
}

pub trait TranspositionHash {
    fn hash(&self) -> u64;
}

pub struct ApproxQuadraticProbingHashTable<K: TranspositionHash, V> {
    arr: Box<[Entry16<K, V>]>,
    capacity: usize,
    mask: usize,
    size: AtomicUsize,
}

struct Entry16<K: TranspositionHash, V> {
    k: AtomicU64,
    v: AtomicPtr<V>,
    _marker: std::marker::PhantomData<K>,
}

impl<K: TranspositionHash, V> Default for Entry16<K, V> {
    fn default() -> Self {
        Self {
            k: Default::default(),
            v: Default::default(),
            _marker: Default::default(),
        }
    }
}
impl<K: TranspositionHash, V> Clone for Entry16<K, V> {
    fn clone(&self) -> Self {
        Self {
            k: AtomicU64::new(self.k.load(Ordering::Relaxed)),
            v: AtomicPtr::new(self.v.load(Ordering::Relaxed)),
            _marker: Default::default(),
        }
    }
}

impl<K: TranspositionHash, V> ApproxQuadraticProbingHashTable<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(std::mem::size_of::<Entry16<K, V>>() <= 16);
        assert!(
            capacity.count_ones() == 1,
            "the capacity must be a power of 2"
        );
        let arr = vec![Entry16::default(); capacity].into_boxed_slice();
        let mask = capacity - 1;
        Self {
            arr,
            mask,
            capacity,
            size: AtomicUsize::default(),
        }
    }
    pub fn enough_to_hold(num: usize) -> Self {
        let mut capacity = 1;
        while capacity * 2 < num * 3 {
            capacity <<= 1;
        }
        Self::new(capacity)
    }
}

unsafe impl<K: TranspositionHash, V> Sync for ApproxQuadraticProbingHashTable<K, V> {}
unsafe impl<K: TranspositionHash, V> Send for ApproxQuadraticProbingHashTable<K, V> {}

pub type ApproxTable<M> = ApproxQuadraticProbingHashTable<<M as MCTS>::State, Node<M>>;

fn get_or_write<'a, V>(ptr: &AtomicPtr<V>, v: &'a V) -> Option<&'a V> {
    let result = match ptr.compare_exchange(
        std::ptr::null_mut(),
        v as *const _ as *mut _,
        Ordering::Relaxed,
        Ordering::Relaxed,
    ) {
        Ok(e) => e,
        Err(e) => e,
    };
    convert(result)
}

fn convert<'a, V>(ptr: *const V) -> Option<&'a V> {
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

const PROBE_LIMIT: usize = 16;

unsafe impl<M> TranspositionTable<M> for ApproxTable<M>
where
    M::State: TranspositionHash,
    M: MCTS,
{
    fn insert<'a>(&'a self, key: &M::State, value: &'a Node<M>) -> Option<&'a Node<M>> {
        if self.size.load(Ordering::Relaxed) * 3 > self.capacity * 2 {
            return self.lookup(key);
        }
        let hash = key.hash();
        if hash == 0 {
            return None;
        }
        let mut idx = hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            // SAFETY: posn always smaller or equal than mask which is equal to capacity - 1
            let entry = unsafe { self.arr.get_unchecked(idx) };
            let key_found = entry.k.load(Ordering::Relaxed);
            if key_found == hash {
                let value_here = entry.v.load(Ordering::Relaxed);
                if !value_here.is_null() {
                    return unsafe { Some(&*value_here) };
                }
                return get_or_write(&entry.v, value);
            }
            if key_found == 0 {
                let key_here =
                    match entry
                        .k
                        .compare_exchange(0, hash, Ordering::Relaxed, Ordering::Relaxed)
                    {
                        Ok(k) => k,
                        Err(k) => k,
                    };

                self.size.fetch_add(1, Ordering::Relaxed);
                if key_here == 0 || key_here == hash {
                    return get_or_write(&entry.v, value);
                }
            }
            idx += inc;
            idx &= self.mask;
        }
        None
    }
    fn lookup<'a>(&'a self, key: &M::State) -> Option<&'a Node<M>> {
        let hash = key.hash();
        let mut idx = hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(idx) };
            let key_here = entry.k.load(Ordering::Relaxed);
            if key_here == hash {
                return convert(entry.v.load(Ordering::Relaxed));
            }
            if key_here == 0 {
                return None;
            }
            idx += inc;
            idx &= self.mask;
        }
        None
    }
}
