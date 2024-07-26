use std::{
    mem::replace,
    ptr::null_mut,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicUsize, Ordering},
        Mutex, RwLock,
    },
};

use itertools::Itertools;
use smallvec::SmallVec;

use crate::{Evaluator, GameState, Move, Player, Policy, StateEval, ThreadData, MCTS};

pub struct SearchTree<M: MCTS> {
    root: Node<M>,
    root_state: M::State,
    policy: M::Select,
    eval: M::Eval,
    manager: M,

    num_nodes: AtomicUsize,
    orphaned: Mutex<Vec<Box<Node<M>>>>,
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
    _determined: bool,
) -> Node<M> {
    // let moves = if determined {
    //     state.legal_moves()
    // } else {
    //     state.all_moves()
    // };
    // let (move_eval, state_eval) = eval.eval_new(state, &moves, handle);
    // policy.validate_evaluations(&move_eval);
    // let moves = moves
    //     .into_iter()
    //     .zip(move_eval)
    //     .map(|(m, e)| MoveInfo::new(m, e))
    //     .collect();
    // Node::new(moves, state_eval)

    let eval = eval.state_eval_new(state, handle);
    Node::empty(eval)
}

impl<M: MCTS> SearchTree<M> {
    pub fn new(state: M::State, manager: M, policy: M::Select, eval: M::Eval) -> Self {
        let root = create_node(&eval, &policy, &state, None, true);
        Self {
            root,
            root_state: state,
            policy,
            eval,
            manager,
            num_nodes: 1.into(),
            orphaned: Mutex::new(Vec::new()),
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

        let child = {
            let children = self.root.moves.read().unwrap();
            // Find the child corresponding to the move we played
            let child = &children.iter().find(|x| x.mv == *mv).unwrap().child;
            // Load raw pointer
            let child_ptr = child.load(Ordering::Relaxed);
            unsafe { std::ptr::read(child_ptr) }
        };

        let old = std::mem::replace(&mut self.root, child);
        self.orphaned.lock().unwrap().push(Box::new(old));
        if self.root.moves.is_poisoned() {
            println!("poison");
        }
        self.root.moves.clear_poison()
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
            let untried = legal_moves
                .clone()
                .into_iter()
                .filter(|lmv| {
                    let n = node.moves.read().unwrap();
                    n.is_empty() || !n.iter().any(|c| c.mv == *lmv)
                })
                .collect_vec();

            // Expand all untried nodes to children
            for u in untried {
                node.moves.write().unwrap().push(MoveInfo::new(u));
            }

            let node_moves = node.moves.read().unwrap();

            // Get the children corresponding to all legal moves
            let moves = legal_moves
                .into_iter()
                .filter_map(|mv| node_moves.iter().find(|child_mv| child_mv.mv == mv))
                .collect_vec();

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
        self.backpropagation(&path_indices, &node_path, &players, eval);
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

        choice.owned.store(true, Ordering::Relaxed);
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        unsafe { (&*created, true) }
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
            let eval_value = self.eval.make_relativ_player(eval, player);
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
        println!("{} orphaned", self.orphaned.lock().unwrap().len());
    }
}

pub struct MoveInfo<M: MCTS> {
    mv: Move<M>,
    // move_select: MoveEval<M>,
    child: AtomicPtr<Node<M>>,
    owned: AtomicBool,
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
            owned: AtomicBool::new(false),
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
    fn empty(eval: StateEval<M>) -> Self {
        Self {
            moves: Vec::new().into(),
            eval,
            stats: NodeStats::new(),
        }
    }

    // fn new(moves: Vec<MoveInfo<M>>, eval: StateEval<M>) -> Self {
    //     Self {
    //         moves: moves.into(),
    //         eval,
    //         stats: NodeStats::new(),
    //     }
    // }
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

    pub fn stats(&self) -> Vec<NonAtomicNodeStats> {
        self.node
            .moves
            .read()
            .unwrap()
            .iter()
            .map(|x| NonAtomicNodeStats {
                visits: x.visits(),
                availability_count: x.availability(),
                sum_evaluations: x.sum_rewards(),
            })
            .collect_vec()
    }
}

#[derive(Debug)]
pub struct NonAtomicNodeStats {
    pub visits: u64,
    pub availability_count: u64,
    pub sum_evaluations: i64,
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
