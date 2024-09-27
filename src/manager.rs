use std::sync::atomic::{AtomicIsize, Ordering};

use crate::{node::ComputedNodeStats, search::SearchTree, GameState, Move, ThreadData, MCTS};

pub struct MCTSManager<M: MCTS> {
    search_tree: SearchTree<M>,
    tld: Option<ThreadData<M>>,
}

impl<M: MCTS> MCTSManager<M>
where
    ThreadData<M>: Default,
{
    pub fn new(state: M::State, manager: M, policy: M::Select, eval: M::Eval) -> Self {
        let search_tree = SearchTree::new(state, manager, policy, eval);

        Self {
            search_tree,
            tld: None,
        }
    }

    pub fn print_stats(&self) {
        self.search_tree.print_stats();
    }

    pub fn playout(&mut self) {
        if self.tld.is_none() {
            self.tld = Some(Default::default())
        }
        self.search_tree.playout(self.tld.as_mut().unwrap());
    }

    pub fn playout_n(&mut self, n: u64) {
        (0..n).for_each(|_| self.playout());
    }

    pub fn playout_n_parallel(&mut self, n: u64, num_threads: usize) {
        if num_threads == 0 {
            return;
        }

        let counter = AtomicIsize::new(n as isize);
        let search_tree = &self.search_tree;
        let _ = crossbeam::scope(|scope| {
            (0..num_threads).for_each(|_| {
                scope.spawn(|_| {
                    let mut tld = Default::default();
                    loop {
                        let count = counter.fetch_sub(1, Ordering::SeqCst);
                        if count <= 0 {
                            break;
                        }
                        search_tree.playout(&mut tld);
                    }
                });
            });
        });
    }

    pub fn tree(&self) -> &SearchTree<M> {
        &self.search_tree
    }

    pub fn pv(&self, num_moves: usize) -> Vec<Move<M>> {
        self.search_tree.pv(num_moves)
    }

    pub fn pv_states(&self, num_moves: usize) -> Vec<(Option<Move<M>>, M::State)> {
        let moves = self.pv(num_moves);
        let mut states = vec![(None, self.search_tree.root_state().clone())];
        for mv in moves {
            let len = states.len() - 1;
            let mut state = states[len].1.clone();
            state.make_move(&mv);
            states[len].0 = Some(mv.clone());
            states.push((None, state));
        }
        states
    }

    pub fn advance(&mut self, mv: Move<M>) {
        self.search_tree.advance(&mv);
    }

    pub fn best_move(&self) -> Option<Move<M>> {
        self.pv(1).first().cloned()
    }

    pub fn moves(&self) -> Vec<Move<M>> {
        self.tree().root().moves()
    }

    pub fn print_root_moves(&self) {
        self.tree().display_moves();
    }

    pub fn stats(&self) -> Vec<ComputedNodeStats> {
        self.tree().root().stats()
    }
}
