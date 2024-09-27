use rand::{seq::IteratorRandom, Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::{node, search, Policy, MCTS};

#[derive(Debug, Clone)]
pub struct UCBPolicy;

impl<M: MCTS<Select = Self>> Policy<M> for UCBPolicy {
    type ThreadLocalData = PolicyRng;
    type MoveSelect = ();

    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: search::SearchHandle<M>,
    ) -> (usize, &'a node::MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a node::MoveInfo<M>> + Clone,
    {
        handle
            .thread_data()
            .policy_data
            .select_random(moves)
            .unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct UCTPolicy(pub f64);

#[allow(clippy::cast_precision_loss)]
impl<M: MCTS<Select = Self>> Policy<M> for UCTPolicy {
    type ThreadLocalData = PolicyRng;
    type MoveSelect = ();

    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: search::SearchHandle<M>,
    ) -> (usize, &'a node::MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a node::MoveInfo<M>> + Clone,
    {
        // let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        // let adjusted_total = (total_visits + 1) as f64;
        // let ln_adjusted_total = adjusted_total.ln();
        handle
            .thread_data()
            .policy_data
            .select_by_key(moves, |mov| {
                let sum_rewards = mov.sum_rewards();
                let child_visits = mov.visits();
                let available = mov.availability();
                if child_visits == 0 {
                    f64::INFINITY
                } else {
                    let explore_term =
                        2.0 * ((available as f64 + 1.0).ln() / child_visits as f64).sqrt();
                    let mean_action_value = sum_rewards as f64 / child_visits as f64;
                    self.0 * explore_term + mean_action_value
                }
            })
            .unwrap()
    }
}

#[derive(Clone)]
pub struct PolicyRng {
    rng: XorShiftRng,
}

impl PolicyRng {
    #[must_use]
    pub fn new() -> Self {
        let rng = SeedableRng::seed_from_u64(1337);
        Self { rng }
    }

    pub fn select_random<T, Iter>(&mut self, elts: Iter) -> Option<(usize, T)>
    where
        Iter: Iterator<Item = T> + Clone,
    {
        elts.enumerate().choose(&mut self.rng)
    }

    pub fn select_by_key<T, Iter, KeyFn>(
        &mut self,
        elts: Iter,
        mut key_fn: KeyFn,
    ) -> Option<(usize, T)>
    where
        Iter: Iterator<Item = T>,
        KeyFn: FnMut(&T) -> f64,
    {
        let mut choice = None;
        let mut num_optimal: u32 = 0;
        let mut best_so_far: f64 = f64::NEG_INFINITY;
        for (idx, elt) in elts.enumerate() {
            let score = key_fn(&elt);
            if score > best_so_far {
                choice = Some((idx, elt));
                num_optimal = 1;
                best_so_far = score;
            } else if (score - best_so_far).abs() < 0.001 {
                num_optimal += 1;
                if self.rng.gen_bool(1.0 / num_optimal as f64) {
                    choice = Some((idx, elt));
                }
            }
        }
        choice
    }
}

impl Default for PolicyRng {
    fn default() -> Self {
        Self::new()
    }
}
