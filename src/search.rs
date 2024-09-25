use std::{
    ptr::null_mut,
    sync::{
        atomic::{AtomicI64, AtomicPtr, AtomicUsize, Ordering},
        RwLock,
    },
};

use itertools::Itertools;
use rand::{seq::IteratorRandom, thread_rng};
use smallvec::SmallVec;

use crate::{Evaluator, GameState, Move, Player, Policy, StateEval, ThreadData, MCTS};

pub struct SearchTree<M: MCTS> {
    root: Node<M>,
    root_state: M::State,
    policy: M::Select,
    eval: M::Eval,
    manager: M,

    num_nodes: AtomicUsize,
    expansion_contention_events: AtomicUsize,
}

impl<M: MCTS> SearchTree<M>
where
    Move<M>: std::fmt::Debug,
{
    pub fn display_moves(&self) {
        let inner = self.root.moves.read().unwrap();
        let mut moves: Vec<&MoveInfo<M>> = inner.iter().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mv in moves {
            println!("{:?}", mv.mv);
        }
    }

    pub fn legal_moves(&self) {
        let inner = self.root.moves.read().unwrap();
        let legal = self.root_state.legal_moves();

        let mut moves: Vec<&MoveInfo<M>> = inner
            .iter()
            .filter(|x| legal.clone().into_iter().any(|l| x.mv == l))
            .collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mv in moves {
            println!("{:?}", mv.mv);
        }
    }
}

fn create_node<M: MCTS>(
    eval: &M::Eval,
    _policy: &M::Select,
    state: &M::State,
    handle: Option<SearchHandle<M>>,
) -> Node<M> {
    let eval = eval.state_eval_new(state, handle);
    Node::empty(eval)
}

impl<M: MCTS> SearchTree<M> {
    pub fn new(state: M::State, manager: M, policy: M::Select, eval: M::Eval) -> Self {
        let root = create_node(&eval, &policy, &state, None);
        Self {
            root,
            root_state: state,
            policy,
            eval,
            manager,
            num_nodes: 1.into(),
            expansion_contention_events: 0.into(),
        }
    }

    pub fn reset(self) -> Self {
        Self::new(self.root_state, self.manager, self.policy, self.eval)
    }

    pub fn new_root(self, new_state: M::State) -> Self {
        Self::new(new_state, self.manager, self.policy, self.eval)
    }

    pub fn advance(&mut self, mv: &Move<M>) {
        // advance state
        let mut new_state = self.root_state.clone();
        new_state.make_move(mv);
        self.root_state = new_state;

        let child_idx = {
            let children = self.root.moves.read().unwrap();
            // Find the child corresponding to the move we played
            let idx = children
                .iter()
                .enumerate()
                .find(|(_, x)| x.mv == *mv)
                .map(|(idx, _)| idx)
                .unwrap();
            idx
        };
        let new_root = {
            let mut moves = self.root.moves.write().unwrap();
            moves.remove(child_idx)
        };
        let new_root_ptr = new_root.child.load(Ordering::SeqCst);
        let old_root = std::mem::replace(&mut self.root, unsafe { *Box::from_raw(new_root_ptr) });
        old_root.moves.write().unwrap().clear();
        std::mem::forget(new_root);
    }

    pub fn spec(&self) -> &M {
        &self.manager
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::SeqCst)
    }

    #[inline(never)]
    pub fn playout(&self, tld: &mut ThreadData<M>) -> bool {
        let sentinel = IncreaseSentinel::new(&self.num_nodes);
        if sentinel.num_nodes >= self.manager.node_limit() {
            return false;
        }

        let mut state = self.root_state.clone();
        state.randomize_determination(state.current_player());
        let mut path_indices: SmallVec<usize, 64> = SmallVec::new();
        let mut node_path: SmallVec<(&Node<M>, &Node<M>), 64> = SmallVec::new();
        let mut players: SmallVec<Player<M>, 64> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root;
        loop {
            if path_indices.len() >= self.manager.max_playout_length() {
                break;
            }
            let legal_moves = state.legal_moves();

            // All moves that are legal now but have never been explored yet
            let untried = {
                let node_moves = node.moves.read().unwrap();
                legal_moves
                    .clone()
                    .into_iter()
                    .filter(|lmv| node_moves.is_empty() || !node_moves.iter().any(|c| c.mv == *lmv))
                    .collect_vec()
            };

            // If there are untried moves add one of them to the children
            if !untried.is_empty() {
                let mut children = node.moves.write().unwrap();
                let child_mv = untried[0].clone();
                if !children.iter().any(|c| c.mv == child_mv) {
                    children.push(MoveInfo::new(child_mv));
                }
            }

            let node_moves = node.moves.read().unwrap();

            // Get the children corresponding to all legal moves
            let moves = {
                legal_moves
                    .into_iter()
                    .filter_map(|mv| node_moves.iter().find(|child_mv| child_mv.mv == mv))
                    .collect_vec()
            };

            if moves.is_empty() {
                break;
            }

            // Increment availability count for each legal move we have in the current determinization
            for m in moves.iter() {
                m.stats.increment_available();
            }

            let (child_idx, choice) = self
                .policy
                .choose(moves.iter().cloned(), self.make_handle(node, tld));
            choice.stats.down(&self.manager);
            players.push(state.current_player());
            path_indices.push(child_idx);

            state.make_move(&choice.mv);
            let (new_node, new_did_we_create) = self.descend(&state, choice, node, tld);
            node_path.push((node, new_node));
            node = new_node;
            did_we_create = new_did_we_create;

            node.stats.down(&self.manager);
            if node.stats.visits.load(Ordering::Relaxed)
                <= self.manager.visits_before_expansion() as usize
            {
                break;
            }
        }
        let new_eval = if did_we_create {
            None
        } else {
            Some(
                self.eval
                    .eval_existing(&state, &node.eval, self.make_handle(node, tld)),
            )
        };
        let eval = new_eval.as_ref().unwrap_or(&node.eval);
        let rollout_eval = self.rollout(&mut state, &self.eval, None);
        // self.backpropagation(&path_indices, &node_path, &players, eval);
        self.backpropagation(&path_indices, &node_path, &players, &rollout_eval);
        true
    }

    fn descend<'a, 'b>(
        &'a self,
        state: &M::State,
        choice: &MoveInfo<M>,
        current_node: &'b Node<M>,
        tld: &'b mut ThreadData<M>,
    ) -> (&'a Node<M>, bool) {
        let child = choice.child.load(Ordering::Relaxed) as *const Node<M>;
        if !child.is_null() {
            return unsafe { (&*child, false) };
        }

        let created = create_node(
            &self.eval,
            &self.policy,
            state,
            Some(self.make_handle(current_node, tld)),
        );
        let created = Box::into_raw(Box::new(created));
        let other_child = choice.child.compare_exchange(
            null_mut(),
            created,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        if let Err(other_child) = other_child {
            self.expansion_contention_events
                .fetch_add(1, Ordering::Relaxed);
            unsafe {
                drop(Box::from_raw(created));
                return (&*other_child, false);
            }
        }

        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        unsafe { (&*created, true) }
    }

    fn rollout(
        &self,
        state: &mut M::State,
        eval: &M::Eval,
        rollout_length: Option<usize>,
    ) -> StateEval<M> {
        let rollout_length = rollout_length.unwrap_or(usize::MAX);
        for i in 0..rollout_length {
            if let Some(mv) = state.legal_moves().into_iter().choose(&mut thread_rng()) {
                state.make_move(&mv);
            } else {
                break;
            }
        }
        eval.state_eval_new(state, None)
    }

    fn backpropagation(
        &self,
        path: &[usize],
        nodes: &[(&Node<M>, &Node<M>)],
        players: &[Player<M>],
        eval: &StateEval<M>,
    ) {
        for ((move_info, player), (parent, child)) in
            path.iter().zip(players.iter()).zip(nodes.iter()).rev()
        {
            let eval_value = self.eval.make_relative(eval, player);
            child.stats.up(&self.manager, eval_value);
            parent.moves.read().unwrap()[*move_info]
                .stats
                .replace(&child.stats);
        }
    }

    fn make_handle<'a>(
        &'a self,
        node: &'a Node<M>,
        tld: &'a mut ThreadData<M>,
    ) -> SearchHandle<'a, M> {
        SearchHandle {
            node,
            tld,
            manager: &self.manager,
        }
    }

    pub fn root_state(&self) -> &M::State {
        &self.root_state
    }

    pub fn root(&self) -> NodeHandle<M> {
        NodeHandle { node: &self.root }
    }

    pub fn pv(&self, num_moves: usize) -> Vec<Move<M>> {
        let mut res = Vec::new();
        let mut curr = &self.root;
        let mut curr_state = self.root_state.clone();

        while curr_state.legal_moves().into_iter().count() > 0 && res.len() < num_moves {
            if let Some(choice) = curr
                .moves
                .read()
                .unwrap()
                .iter()
                .filter_map(|mv| {
                    curr_state
                        .legal_moves()
                        .into_iter()
                        .any(|lmv| mv.mv == lmv)
                        .then_some((
                            mv.mv.clone(),
                            mv.visits(),
                            mv.child.load(Ordering::SeqCst) as *const Node<M>,
                        ))
                })
                .max_by_key(|(_, visits, _)| *visits)
            {
                res.push(choice.0.clone());
                curr_state.make_move(&choice.0);
                if choice.2.is_null() {
                    break;
                } else {
                    unsafe {
                        curr = &*choice.2;
                    }
                }
            } else {
                break;
            }
        }
        res
    }

    pub fn print_stats(&self) {
        println!("{} nodes", self.num_nodes.load(Ordering::Relaxed));
        println!(
            "{} e/c events",
            self.expansion_contention_events.load(Ordering::Relaxed)
        );

        for (s, m) in self.root().stats().iter().zip(self.root().moves().iter()) {
            println!("{:?} {:?}", s, m);
        }
    }
}

pub struct MoveInfo<M: MCTS> {
    mv: Move<M>,
    // move_select: MoveEval<M>,
    child: AtomicPtr<Node<M>>,
    stats: NodeStats,
}

pub struct Node<M: MCTS> {
    moves: RwLock<Vec<MoveInfo<M>>>,
    eval: StateEval<M>,
    stats: NodeStats,
}

struct NodeStats {
    visits: AtomicUsize,
    availability_count: AtomicUsize,
    sum_evaluations: AtomicI64,
}

impl<M: MCTS> MoveInfo<M> {
    fn new(mv: Move<M> /*, move_select: MoveEval<M>*/) -> Self {
        Self {
            mv,
            // move_select,
            child: AtomicPtr::default(),
            stats: NodeStats::new(),
        }
    }

    pub fn get_move(&self) -> &Move<M> {
        &self.mv
    }

    // pub fn move_select(&self) -> &MoveEval<M> {
    //     &self.move_select
    // }

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

impl<M: MCTS> Node<M> {
    fn empty(eval: StateEval<M>) -> Self {
        Self {
            moves: Vec::new().into(),
            eval,
            stats: NodeStats::new(),
        }
    }
}

impl NodeStats {
    fn new() -> Self {
        Self {
            sum_evaluations: 0.into(),
            availability_count: 0.into(),
            visits: 0.into(),
        }
    }

    fn increment_available(&self) {
        self.availability_count.fetch_add(1, Ordering::Relaxed);
    }

    fn down<M: MCTS>(&self, manager: &M) {
        self.sum_evaluations
            .fetch_sub(manager.virtual_loss(), Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    fn up<M: MCTS>(&self, manager: &M, eval: i64) {
        let delta = eval + manager.virtual_loss();
        self.sum_evaluations.fetch_add(delta, Ordering::Relaxed);
    }

    fn replace(&self, other: &NodeStats) {
        self.visits
            .store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_evaluations.store(
            other.sum_evaluations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}

pub type MoveInfoHandle<'a, M> = &'a MoveInfo<M>;

pub struct SearchHandle<'a, M: 'a + MCTS> {
    node: &'a Node<M>,
    tld: &'a mut ThreadData<M>,
    manager: &'a M,
}

impl<'a, M: MCTS> SearchHandle<'a, M> {
    pub fn node(&self) -> NodeHandle<'a, M> {
        NodeHandle { node: self.node }
    }

    pub fn thread_data(&mut self) -> &mut ThreadData<M> {
        self.tld
    }

    pub fn mcts(&self) -> &'a M {
        self.manager
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

struct IncreaseSentinel<'a> {
    x: &'a AtomicUsize,
    num_nodes: usize,
}

impl<'a> IncreaseSentinel<'a> {
    fn new(x: &'a AtomicUsize) -> Self {
        let num_nodes = x.fetch_add(1, Ordering::Relaxed);
        Self { x, num_nodes }
    }
}

impl<'a> Drop for IncreaseSentinel<'a> {
    fn drop(&mut self) {
        self.x.fetch_sub(1, Ordering::Relaxed);
    }
}
