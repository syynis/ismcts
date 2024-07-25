use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::{
    search::{self},
    Policy, MCTS,
};

#[derive(Debug, Clone)]
pub struct UCBPolicy;

impl<M: MCTS<Select = Self>> Policy<M> for UCBPolicy {
    type ThreadLocalData = PolicyRng;
    type MoveSelect = ();

    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: search::SearchHandle<M>,
    ) -> (usize, &'a search::MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a search::MoveInfo<M>> + Clone,
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

impl<M: MCTS<Select = Self>> Policy<M> for UCTPolicy {
    type ThreadLocalData = PolicyRng;
    type MoveSelect = ();

    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        mut handle: search::SearchHandle<M>,
    ) -> (usize, &'a search::MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a search::MoveInfo<M>> + Clone,
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        let adjusted_total = (total_visits + 1) as f64;
        let ln_adjusted_total = adjusted_total.ln();
        handle
            .thread_data()
            .policy_data
            .select_by_key(moves, |mov| {
                let sum_rewards = mov.sum_rewards();
                let child_visits = mov.visits();
                if child_visits == 0 {
                    f64::INFINITY
                } else {
                    let explore_term = 2.0 * (ln_adjusted_total / child_visits as f64).sqrt();
                    let mean_action_value = sum_rewards as f64 / child_visits as f64;
                    self.0 * explore_term + mean_action_value
                }
            })
            .unwrap()
    }
}

const RECIPROCAL_TABLE_LEN: usize = 128;

#[derive(Clone, Debug)]
pub struct AlphaGoPolicy {
    exploration_constant: f64,
    reciprocals: Vec<f64>,
}

impl AlphaGoPolicy {
    pub fn new(exploration_constant: f64) -> Self {
        assert!(
            exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant
        );
        let reciprocals = (0..RECIPROCAL_TABLE_LEN)
            .map(|x| if x == 0 { 2.0 } else { 1.0 / x as f64 })
            .collect();
        Self {
            exploration_constant,
            reciprocals,
        }
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }

    fn reciprocal(&self, x: usize) -> f64 {
        if x < RECIPROCAL_TABLE_LEN {
            unsafe { *self.reciprocals.get_unchecked(x) }
        } else {
            1.0 / x as f64
        }
    }
}

// impl<M: MCTS<Select = Self>> Policy<M> for AlphaGoPolicy {
//     type MoveSelect = f64;
//     type ThreadLocalData = PolicyRng;

//     fn choose<'a, MoveIter>(
//         &self,
//         moves: MoveIter,
//         mut handle: SearchHandle<M>,
//     ) -> &'a search::MoveInfo<M>
//     where
//         MoveIter: Iterator<Item = &'a Arc<search::MoveInfo<M>>> + Clone,
//     {
//         let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>() + 1;
//         let sqrt_total_visits = (total_visits as f64).sqrt();
//         let explore_coef = self.exploration_constant * sqrt_total_visits;
//         handle
//             .thread_data()
//             .policy_data
//             .select_by_key(moves, |mov| {
//                 let sum_rewards = mov.sum_rewards() as f64;
//                 let child_visits = mov.visits();
//                 let policy_evaln = *mov.move_select();
//                 (sum_rewards + explore_coef * policy_evaln) * self.reciprocal(child_visits as usize)
//             })
//             .unwrap()
//     }

//     fn validate_evaluations(&self, evalns: &[f64]) {
//         for &x in evalns {
//             assert!(
//                 x >= -1e-6,
//                 "Move evaluation is {} (must be non-negative)",
//                 x
//             );
//         }
//         if !evalns.is_empty() {
//             let evaln_sum: f64 = evalns.iter().sum();
//             assert!(
//                 (evaln_sum - 1.0).abs() < 0.1,
//                 "Sum of evaluations is {} (should sum to 1)",
//                 evaln_sum
//             );
//         }
//     }
// }

#[derive(Clone)]
pub struct PolicyRng {
    rng: XorShiftRng,
}

impl PolicyRng {
    pub fn new() -> Self {
        let rng = SeedableRng::seed_from_u64(1337);
        Self { rng }
    }

    pub fn select_random<T, Iter>(&mut self, elts: Iter) -> Option<(usize, T)>
    where
        Iter: Iterator<Item = T> + Clone,
    {
        let len = elts.clone().count();
        elts.enumerate().nth(self.rng.gen_range(0..len))
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
            } else if score == best_so_far {
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
