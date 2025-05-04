use std::sync::{
    atomic::{AtomicI64, AtomicPtr, AtomicUsize, Ordering},
    RwLock,
};

use itertools::Itertools;

use crate::{search::SearchHandle, Evaluator, Move, StateEval, MCTS};

pub struct Node<M: MCTS> {
    pub moves: RwLock<Vec<MoveInfo<M>>>,
    pub eval: StateEval<M>,
    pub stats: Stats,
}

impl<M: MCTS> Node<M> {
    #[must_use]
    pub fn new(eval: &M::Eval, state: &M::State, handle: Option<SearchHandle<M>>) -> Node<M> {
        Self {
            moves: Vec::new().into(),
            eval: eval.eval_new(state, handle),
            stats: Stats::new(),
        }
    }
}

pub struct MoveInfo<M: MCTS> {
    pub mv: Move<M>,
    pub child: AtomicPtr<Node<M>>,
    pub stats: Stats,
}

impl<M: MCTS> Drop for MoveInfo<M> {
    fn drop(&mut self) {
        let ptr = self.child.load(Ordering::SeqCst);
        if !ptr.is_null() {
            unsafe {
                let x = Box::from_raw(ptr);
                x.moves.write().unwrap().clear();
                drop(x);
            }
        }
    }
}

impl<M: MCTS> MoveInfo<M> {
    #[must_use]
    pub fn new(mv: Move<M>) -> Self {
        Self {
            mv,
            child: AtomicPtr::default(),
            stats: Stats::new(),
        }
    }

    #[must_use]
    pub fn get_move(&self) -> &Move<M> {
        &self.mv
    }

    #[must_use]
    pub fn visits(&self) -> u64 {
        self.stats.visits.load(Ordering::Relaxed) as u64
    }

    #[must_use]
    pub fn availability(&self) -> u64 {
        self.stats.availability_count.load(Ordering::Relaxed) as u64
    }

    #[must_use]
    pub fn sum_rewards(&self) -> i64 {
        self.stats.sum_evaluations.load(Ordering::Relaxed)
    }

    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn computed_stats(&self) -> ComputedStats {
        ComputedStats {
            visits: self.visits(),
            availability_count: self.availability(),
            sum_evaluations: self.sum_rewards(),
            mean_action_value: self.sum_rewards() as f64 / self.visits() as f64,
            availability: 2.0
                * ((1.0 + self.availability() as f64).ln() / self.visits() as f64).sqrt(),
        }
    }

    #[must_use]
    pub fn child(&self) -> Option<NodeHandle<M>> {
        let ptr = self.child.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(NodeHandle { node: &*ptr }) }
        }
    }
}

pub struct Stats {
    visits: AtomicUsize,
    availability_count: AtomicUsize,
    sum_evaluations: AtomicI64,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl Stats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            sum_evaluations: 0.into(),
            availability_count: 0.into(),
            visits: 0.into(),
        }
    }

    pub fn increment_available(&self) {
        self.availability_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn down<M: MCTS>(&self, manager: &M) {
        self.sum_evaluations
            .fetch_sub(manager.virtual_loss(), Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn up<M: MCTS>(&self, manager: &M, eval: i64) {
        let delta = eval + manager.virtual_loss();
        self.sum_evaluations.fetch_add(delta, Ordering::Relaxed);
    }

    pub fn replace(&self, other: &Stats) {
        self.visits
            .store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_evaluations.store(
            other.sum_evaluations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy)]
pub struct NodeHandle<'a, M: 'a + MCTS> {
    pub node: &'a Node<M>,
}

#[allow(clippy::cast_precision_loss)]
impl<M: MCTS> NodeHandle<'_, M> {
    #[must_use]
    pub fn moves(&self) -> Vec<Move<M>> {
        self.node
            .moves
            .read()
            .unwrap()
            .iter()
            .map(|x| x.mv.clone())
            .collect_vec()
    }

    #[must_use]
    pub fn stats(&self) -> Vec<ComputedStats> {
        self.node
            .moves
            .read()
            .unwrap()
            .iter()
            .map(|x| ComputedStats {
                visits: x.visits(),
                availability_count: x.availability(),
                sum_evaluations: x.sum_rewards(),
                mean_action_value: x.sum_rewards() as f64 / x.visits() as f64,
                availability: ((1.0 + x.availability() as f64).ln() / x.visits() as f64).sqrt(),
            })
            .collect_vec()
    }
}

#[derive(Debug)]
pub struct ComputedStats {
    pub visits: u64,
    pub availability_count: u64,
    pub sum_evaluations: i64,
    pub mean_action_value: f64,
    pub availability: f64,
}

impl std::fmt::Display for ComputedStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Visits: {}, Avail: {}, Avail %: {:.3}, Eval: {}, Eval %: {:.3}",
            self.visits,
            self.availability_count,
            self.availability,
            self.sum_evaluations,
            self.mean_action_value
        )
    }
}
