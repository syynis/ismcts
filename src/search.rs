use std::{
    ptr::null_mut,
    sync::atomic::{AtomicUsize, Ordering},
};

use itertools::Itertools;
use rand::{seq::IteratorRandom, thread_rng};
use smallvec::SmallVec;

use crate::{
    node::{MoveInfo, Node, NodeHandle},
    Evaluator, GameState, Knowledge, Move, Player, Policy, StateEval, ThreadData, MCTS,
};

pub struct Tree<M: MCTS, const N: usize> {
    roots: [Node<M>; N],
    root_state: M::State,
    knowledge: [Knowledge<M>; N],
    policy: M::Select,
    eval: M::Eval,
    manager: M,

    num_nodes: AtomicUsize,
    expansion_contention_events: AtomicUsize,
}

impl<M: MCTS, const N: usize> Tree<M, N> {
    #[must_use]
    pub fn new(state: M::State, manager: M, policy: M::Select, eval: M::Eval) -> Self {
        let knowledge = core::array::from_fn(|i| state.knowledge_from_state(Player::<M>::from(i)));
        Self {
            roots: core::array::from_fn(|_| Node::new(&eval, &state, None)),
            root_state: state,
            knowledge,
            policy,
            eval,
            manager,
            num_nodes: 1.into(),
            expansion_contention_events: 0.into(),
        }
    }

    pub fn advance(&mut self, mv: &Move<M>) {
        // Advance root state by making move and updating knowledge
        let mut new_state = self.root_state.clone();
        for k in &mut self.knowledge {
            new_state.update_knowledge(mv, k);
        }
        new_state.make_move(mv);
        self.root_state = new_state;

        // Advance the root node of each subtree.
        // Also drop branches of the tree that don't belong to the move that was made.
        for root in &mut self.roots {
            // Find the child corresponding to the move we played
            let child_idx = {
                let children = root.moves.read().unwrap();
                children.iter().position(|x| x.mv == *mv).unwrap()
            };
            // Take ownership of data holding node corresponding to move made.
            let new_root = {
                let mut moves = root.moves.write().unwrap();
                moves.remove(child_idx)
            };
            // Load node pointer
            let new_root_ptr = new_root.child.load(Ordering::SeqCst);
            // Replace old root with the new root from loaded pointer
            let old_root = std::mem::replace(root, unsafe { *Box::from_raw(new_root_ptr) });
            // Now drop all hanging branches from tree
            old_root.moves.write().unwrap().clear();
            // Because we take ownership of new_root above and the scope ends here it would run drop
            // this prevents that.
            // TODO There must be a better way to do this
            std::mem::forget(new_root);
        }
    }
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn playout(&self, tld: &mut ThreadData<M>) -> bool {
        let sentinel = IncreaseSentinel::new(&self.num_nodes);
        if sentinel.num_nodes >= self.manager.node_limit() {
            return false;
        }

        let mut state = self.root_state.clone();
        state.randomize_determination(
            state.current_player(),
            &self.knowledge[state.current_player().into()],
        );

        let mut path_indices: [SmallVec<usize, 64>; N] = [const { SmallVec::new() }; N];
        let mut node_path: [SmallVec<(&Node<M>, &Node<M>), 64>; N] = [const { SmallVec::new() }; N];
        let mut players: SmallVec<Player<M>, 64> = SmallVec::new();
        let mut nodes: [&Node<M>; N] = core::array::from_fn(|idx| &self.roots[idx]);

        // Select
        loop {
            let legal_moves = state.legal_moves();
            let to_move = state.current_player();
            let to_move_idx: usize = to_move.into();
            let target_node: &Node<M> = nodes[to_move_idx];

            let no_legal_moves = legal_moves.clone().into_iter().count() == 0;
            if no_legal_moves {
                break;
            }

            // All moves that are legal now but have never been explored yet
            let untried = {
                let node_moves = target_node.moves.read().unwrap();
                legal_moves
                    .clone()
                    .into_iter()
                    .filter(|lmv| node_moves.is_empty() || !node_moves.iter().any(|c| c.mv == *lmv))
                    .collect_vec()
            };
            let any_untried = !untried.is_empty();

            // Select
            let choice_mv = if any_untried {
                let mut node_moves = target_node.moves.write().unwrap();
                let choice = untried.into_iter().choose(&mut thread_rng()).unwrap();
                node_moves.push(MoveInfo::new(choice));
                let choice = node_moves.last().unwrap();
                choice.stats.down(&self.manager);
                choice.mv.clone()
            } else {
                let node_moves = target_node.moves.read().unwrap();
                // Get the children corresponding to all legal moves
                let moves = {
                    legal_moves
                        .clone()
                        .into_iter()
                        .filter_map(|mv| node_moves.iter().find(|child_mv| child_mv.mv == mv))
                        .collect_vec()
                };
                // We know there are no untried moves and there is at least one legal move.
                // This means all legal moves have been expanded once already
                assert!(!moves.is_empty());

                let choice = self
                    .policy
                    .choose(moves.iter().copied(), self.make_handle(target_node, tld))
                    .1;
                choice.stats.down(&self.manager);
                choice.mv.clone()
            };

            for node in nodes {
                if !node
                    .moves
                    .read()
                    .unwrap()
                    .iter()
                    .any(|mv| choice_mv == mv.mv)
                {
                    node.moves
                        .write()
                        .unwrap()
                        .push(MoveInfo::new(choice_mv.clone()));
                }
            }

            players.push(state.current_player());
            state.make_move(&choice_mv);
            let new_nodes = core::array::from_fn(|idx| {
                let node = nodes[idx];
                // Increment availability count for each legal move we have in the current determinization
                {
                    let node_moves = node.moves.read().unwrap();
                    legal_moves
                        .clone()
                        .into_iter()
                        .filter_map(|mv| node_moves.iter().find(|child_mv| child_mv.mv == mv))
                        .for_each(|m| m.stats.increment_available());
                }
                // Expand
                let (new_node, _, choice_idx) = self.descend(&state, &choice_mv, node, tld);
                node_path[idx].push((node, new_node));
                path_indices[idx].push(choice_idx);
                new_node.stats.down(&self.manager);
                new_node
            });
            nodes = new_nodes;
            if any_untried {
                break;
            }
        }

        // Rollout
        let rollout_eval = Self::rollout(&mut state, &self.eval, Some(20));
        // Backprop
        for (idx, _) in nodes.iter().enumerate() {
            self.backpropagation(&path_indices[idx], &node_path[idx], &players, &rollout_eval);
        }
        true
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

    #[must_use]
    fn rollout(
        state: &mut M::State,
        eval: &M::Eval,
        rollout_length: Option<usize>,
    ) -> StateEval<M> {
        let rollout_length = rollout_length.unwrap_or(usize::MAX);
        (0..rollout_length).for_each(|_| {
            if let Some(mv) = state.legal_moves().into_iter().choose(&mut thread_rng()) {
                state.make_move(&mv);
            }
        });
        eval.eval_new(state, None)
    }

    #[must_use]
    fn descend<'a, 'b>(
        &'a self,
        state: &M::State,
        // choice: &MoveInfo<M>,
        choice: &Move<M>,
        current_node: &'b Node<M>,
        tld: &'b mut ThreadData<M>,
    ) -> (&'a Node<M>, bool, usize) {
        let read = &current_node.moves.read().unwrap();
        let (choice, idx) = read
            .iter()
            .enumerate()
            .find_map(|(idx, mv_info)| (mv_info.mv == *choice).then_some((mv_info, idx)))
            .expect("Should exist");
        let child = choice.child.load(Ordering::Relaxed).cast_const();
        if !child.is_null() {
            return unsafe { (&*child, false, idx) };
        }

        let created = Node::new(&self.eval, state, Some(self.make_handle(current_node, tld)));
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
                return (&*other_child, false, idx);
            }
        }

        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        unsafe { (&*created, true, idx) }
    }

    #[must_use]
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

    #[must_use]
    pub fn pv(&self, num_moves: usize) -> Vec<Move<M>> {
        let mut res = Vec::new();
        let mut curr_player: usize = self.root_state.current_player().into();
        let mut curr: [&Node<M>; N] = core::array::from_fn(|i| &self.roots[i]);
        let mut curr_state = self.root_state.clone();

        while curr_state.legal_moves().into_iter().count() > 0 && res.len() < num_moves {
            if let Some(choice) = curr[curr_player]
                .moves
                .read()
                .unwrap()
                .iter()
                .filter_map(|mv| {
                    curr_state
                        .legal_moves()
                        .into_iter()
                        .any(|lmv| mv.mv == lmv)
                        .then_some((mv.mv.clone(), mv.visits()))
                })
                .max_by_key(|(_, visits)| *visits)
                .map(|(mv, _)| mv)
            {
                res.push(choice.clone());
                curr_state.make_move(&choice);
                curr_player = curr_state.current_player().into();
                let new_nodes: [Option<&Node<M>>; N] = core::array::from_fn(|idx| {
                    let node = curr[idx];
                    let read = &node.moves.read().unwrap();
                    let child = read.iter().find(|m| m.mv == choice);
                    let ptr = child.map(|child| child.child.load(Ordering::Relaxed));
                    let next = ptr.map(|ptr| (!ptr.is_null()).then_some(unsafe { &*ptr }));
                    next.flatten()
                });
                if new_nodes.iter().all(std::option::Option::is_some) {
                    let new: [&Node<M>; N] = core::array::from_fn(|idx| new_nodes[idx].unwrap());
                    curr = new;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        res
    }

    #[must_use]
    pub fn spec(&self) -> &M {
        &self.manager
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::SeqCst)
    }

    #[must_use]
    pub fn root_state(&self) -> &M::State {
        &self.root_state
    }

    #[must_use]
    pub fn root(&self) -> NodeHandle<M> {
        NodeHandle {
            node: &self.roots[self.root_state.current_player().into()],
        }
    }

    pub fn display_moves(&self)
    where
        Move<M>: std::fmt::Debug,
    {
        let player_idx = self.root_state.current_player().into();
        let inner = self.roots[player_idx].moves.read().unwrap();
        let mut moves: Vec<&MoveInfo<M>> = inner.iter().collect();
        moves.sort_by_key(|x| x.visits());
        for mv in moves {
            println!("{:?} {}", mv.mv, mv.visits());
        }
    }

    pub fn display_legal_moves(&self)
    where
        Move<M>: std::fmt::Debug,
    {
        let player_idx = self.root_state.current_player().into();
        let inner = self.roots[player_idx].moves.read().unwrap();
        let legal = self.root_state.legal_moves();

        let mut moves: Vec<&MoveInfo<M>> = inner
            .iter()
            .filter(|x| legal.clone().into_iter().any(|l| x.mv == l))
            .collect();
        moves.sort_by_key(|x| x.visits());
        println!("---------------------------------------------------------");
        for mv in moves.iter().rev() {
            print!("Move: {:?}\nStats: {}", mv.mv, mv.computed_stats());
        }
        println!("---------------------------------------------------------");
    }

    pub fn print_stats(&self)
    where
        Move<M>: std::fmt::Debug,
    {
        println!("{} nodes", self.num_nodes.load(Ordering::Relaxed));
        println!(
            "{} e/c events",
            self.expansion_contention_events.load(Ordering::Relaxed)
        );

        for (s, m) in self.root().stats().iter().zip(self.root().moves().iter()) {
            println!("{s:?} {m:?}");
        }
    }

    pub fn print_knowledge(&self)
    where
        Knowledge<M>: std::fmt::Debug,
    {
        for k in &self.knowledge {
            println!("{k:?}");
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct SearchHandle<'a, M: 'a + MCTS> {
    node: &'a Node<M>,
    tld: &'a mut ThreadData<M>,
    manager: &'a M,
}

impl<'a, M: MCTS> SearchHandle<'a, M> {
    #[must_use]
    pub fn node(&self) -> NodeHandle<'a, M> {
        NodeHandle { node: self.node }
    }

    #[must_use]
    pub fn thread_data(&mut self) -> &mut ThreadData<M> {
        self.tld
    }

    #[must_use]
    pub fn mcts(&self) -> &'a M {
        self.manager
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

impl Drop for IncreaseSentinel<'_> {
    fn drop(&mut self) {
        self.x.fetch_sub(1, Ordering::Relaxed);
    }
}
