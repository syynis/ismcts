use std::sync::{
    atomic::{AtomicI64, AtomicPtr, AtomicUsize, Ordering},
    RwLock,
};

use itertools::Itertools;

use crate::{search::SearchHandle, Evaluator, Move, StateEval, MCTS};

pub struct MoveInfo<M: MCTS> {
    pub mv: Move<M>,
    pub child: AtomicPtr<Node<M>>,
    pub stats: NodeStats,
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
    pub fn new(mv: Move<M>) -> Self {
        Self {
            mv,
            child: AtomicPtr::default(),
            stats: NodeStats::new(),
        }
    }

    pub fn get_move(&self) -> &Move<M> {
        &self.mv
    }

    pub fn visits(&self) -> u64 {
        self.stats.visits.load(Ordering::Relaxed) as u64
    }

    pub fn availability(&self) -> u64 {
        self.stats.availability_count.load(Ordering::Relaxed) as u64
    }

    pub fn sum_rewards(&self) -> i64 {
        self.stats.sum_evaluations.load(Ordering::Relaxed)
    }

    pub fn child(&self) -> Option<NodeHandle<M>> {
        let ptr = self.child.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(NodeHandle { node: &*ptr }) }
        }
    }
}

pub struct Node<M: MCTS> {
    pub moves: RwLock<Vec<MoveInfo<M>>>,
    pub eval: StateEval<M>,
    pub stats: NodeStats,
}

impl<M: MCTS> Node<M> {
    pub fn new(eval: &M::Eval, state: &M::State, handle: Option<SearchHandle<M>>) -> Node<M> {
        Self {
            moves: Vec::new().into(),
            eval: eval.eval_new(state, handle),
            stats: NodeStats::new(),
        }
    }
}

pub struct NodeStats {
    visits: AtomicUsize,
    availability_count: AtomicUsize,
    sum_evaluations: AtomicI64,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeStats {
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

    pub fn replace(&self, other: &NodeStats) {
        self.visits
            .store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_evaluations.store(
            other.sum_evaluations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}

#[derive(Clone, Copy)]
pub struct NodeHandle<'a, M: 'a + MCTS> {
    pub node: &'a Node<M>,
}

impl<'a, M: MCTS> NodeHandle<'a, M> {
    pub fn moves(&self) -> Vec<Move<M>> {
        self.node
            .moves
            .read()
            .unwrap()
            .iter()
            .map(|x| x.mv.clone())
            .collect_vec()
    }

    pub fn stats(&self) -> Vec<ComputedNodeStats> {
        self.node
            .moves
            .read()
            .unwrap()
            .iter()
            .map(|x| ComputedNodeStats {
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
pub struct ComputedNodeStats {
    pub visits: u64,
    pub availability_count: u64,
    pub sum_evaluations: i64,
    pub mean_action_value: f64,
    pub availability: f64,
}
