use std::{
    ptr::null_mut,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicUsize, Ordering},
        Mutex, RwLock,
    },
};

use itertools::Itertools;
use smallvec::SmallVec;

use crate::{
    Evaluator, GameState, Move, MoveEval, Player, Policy, StateEval, ThreadData,
    TranspositionTable, MCTS,
};

pub struct SearchTree<M: MCTS> {
    root: Node<M>,
    root_state: M::State,
    policy: M::Select,
    eval: M::Eval,
    manager: M,
    table: M::TT,

    num_nodes: AtomicUsize,
    orphaned: Mutex<Vec<Box<Node<M>>>>,
    expansion_contention_events: AtomicUsize,
    tt_hits: AtomicUsize,
    delayed_tt_hits: AtomicUsize,
}

impl<M: MCTS> SearchTree<M>
where
    Move<M>: std::fmt::Debug,
{
    pub fn display_moves(&self) {
        let mut moves: Vec<&MoveInfo<M>> = self.root.moves.iter().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mv in moves {
            println!("{:?}", mv.mv);
        }
    }
}

fn create_node<M: MCTS>(
    eval: &M::Eval,
    policy: &M::Select,
    state: &M::State,
    handle: Option<SearchHandle<M>>,
    determined: bool,
) -> Node<M> {
    let moves = if determined {
        state.legal_moves()
    } else {
        state.all_moves()
    };
    let (move_eval, state_eval) = eval.eval_new(state, &moves, handle);
    policy.validate_evaluations(&move_eval);
    let moves = moves
        .into_iter()
        .zip(move_eval)
        .map(|(m, e)| MoveInfo::new(m, e))
        .collect();
    Node::new(moves, state_eval)
}

fn is_cycle<T>(past: &[&T], current: &T) -> bool {
    past.iter().any(|x| std::ptr::eq(*x, current))
}

impl<M: MCTS> SearchTree<M> {
    pub fn new(
        state: M::State,
        manager: M,
        policy: M::Select,
        eval: M::Eval,
        table: M::TT,
    ) -> Self {
        let root = create_node(&eval, &policy, &state, None, true);
        Self {
            root,
            root_state: state,
            policy,
            eval,
            manager,
            table,
            num_nodes: 1.into(),
            orphaned: Mutex::new(Vec::new()),
            expansion_contention_events: 0.into(),
            tt_hits: 0.into(),
            delayed_tt_hits: 0.into(),
        }
    }

    pub fn reset(self) -> Self {
        Self::new(
            self.root_state,
            self.manager,
            self.policy,
            self.eval,
            self.table,
        )
    }

    pub fn new_root(self, new_state: M::State) -> Self {
        Self::new(new_state, self.manager, self.policy, self.eval, self.table)
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
        let mut path: SmallVec<&MoveInfo<M>, 64> = SmallVec::new();
        let mut node_path: SmallVec<&Node<M>, 64> = SmallVec::new();
        let mut players: SmallVec<Player<M>, 64> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root;
        loop {
            if path.len() >= self.manager.max_playout_length() {
                break;
            }

            let available_moves = state.legal_moves();
            let moves = available_moves
                .into_iter()
                .filter_map(|mv| node.moves.iter().find(|child_mv| child_mv.mv == mv))
                .collect_vec();
            if moves.is_empty() {
                break;
            }

            let choice = self
                .policy
                .choose(moves.iter().cloned(), self.make_handle(node, tld));
            choice.stats.down(&self.manager);
            players.push(state.current_player());
            path.push(choice);

            assert!(path.len() <= self.manager.max_playout_length(),
                "playout length exceeded maximum of {} (maybe the transposition table is creating an infinite loop?)",
                self.manager.max_playout_length());

            state.make_move(&choice.mv);
            let (new_node, new_did_we_create) = self.descend(&state, choice, node, tld);
            node = new_node;
            did_we_create = new_did_we_create;

            if is_cycle(&node_path, node) {
                break;
            }

            node_path.push(node);
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
        self.backpropagation(&path, &node_path, &players, eval);
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

        if let Some(node) = self.table.lookup(state) {
            let child = choice.child.compare_exchange(
                null_mut(),
                node as *const _ as *mut _,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            match child {
                Ok(child) => {
                    self.tt_hits.fetch_add(1, Ordering::Relaxed);
                    return (node, false);
                }
                Err(child) => {
                    return unsafe { (&*child, false) };
                }
            }
        }

        let created = create_node(
            &self.eval,
            &self.policy,
            state,
            Some(self.make_handle(current_node, tld)),
            self.root_state.current_player() == state.current_player(),
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

        if let Some(existing) = self.table.insert(state, unsafe { &*created }) {
            self.delayed_tt_hits.fetch_add(1, Ordering::Relaxed);
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            self.orphaned
                .lock()
                .unwrap()
                .push(unsafe { Box::from_raw(created) });
            return (existing, false);
        }

        choice.owned.store(true, Ordering::Relaxed);
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        unsafe { (&*created, true) }
    }

    fn backpropagation(
        &self,
        path: &[&MoveInfo<M>],
        nodes: &[&Node<M>],
        players: &[Player<M>],
        eval: &StateEval<M>,
    ) {
        for ((move_info, player), node) in path.iter().zip(players.iter()).zip(nodes.iter()).rev() {
            let eval_value = self.eval.make_relativ_player(eval, player);
            node.stats.up(&self.manager, eval_value);
            move_info.stats.replace(&node.stats);
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
                .max_by_key(|(mv, visits, child)| *visits)
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
        println!("{} tt hits", self.tt_hits.load(Ordering::Relaxed));
        println!("{} dtt hits", self.delayed_tt_hits.load(Ordering::Relaxed));
        println!(
            "{} e/c events",
            self.expansion_contention_events.load(Ordering::Relaxed)
        );
        println!("{} orphaned", self.orphaned.lock().unwrap().len());
    }
}

pub struct MoveInfo<M: MCTS> {
    mv: Move<M>,
    move_select: MoveEval<M>,
    child: AtomicPtr<Node<M>>,
    owned: AtomicBool,
    stats: NodeStats,
}

pub struct Node<M: MCTS> {
    moves: Vec<MoveInfo<M>>,
    eval: StateEval<M>,
    stats: NodeStats,
}

struct NodeStats {
    visits: AtomicUsize,
    availability_count: AtomicUsize,
    sum_evaluations: AtomicI64,
}

impl<M: MCTS> MoveInfo<M> {
    fn new(mv: Move<M>, move_select: MoveEval<M>) -> Self {
        Self {
            mv,
            move_select,
            child: AtomicPtr::default(),
            owned: AtomicBool::new(false),
            stats: NodeStats::new(),
        }
    }

    pub fn get_move(&self) -> &Move<M> {
        &self.mv
    }

    pub fn move_select(&self) -> &MoveEval<M> {
        &self.move_select
    }

    pub fn visits(&self) -> u64 {
        self.stats.visits.load(Ordering::Relaxed) as u64
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
        if !self.owned.load(Ordering::SeqCst) {
            return;
        }
        let ptr = self.child.load(Ordering::SeqCst);
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl<M: MCTS> Node<M> {
    fn new(moves: Vec<MoveInfo<M>>, eval: StateEval<M>) -> Self {
        Self {
            moves,
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
    pub fn moves(&self) -> Moves<M> {
        Moves {
            iter: self.node.moves.iter(),
        }
    }
}

#[derive(Clone)]
pub struct Moves<'a, M: 'a + MCTS> {
    iter: std::slice::Iter<'a, MoveInfo<M>>,
}

impl<'a, M: 'a + MCTS> Iterator for Moves<'a, M> {
    type Item = &'a MoveInfo<M>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
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
