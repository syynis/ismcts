#![feature(mapped_lock_guards)]

use search::{MoveInfo, SearchHandle};

pub mod manager;
pub mod policies;
pub mod search;

pub trait MCTS: Sized + Sync {
    type State: GameState + Sync;
    type Eval: Evaluator<Self> + Sync;
    type Select: Policy<Self> + Sync;

    fn virtual_loss(&self) -> i64 {
        0
    }

    fn node_limit(&self) -> usize {
        usize::MAX
    }

    fn visits_before_expansion(&self) -> u64 {
        1
    }

    fn max_playout_length(&self) -> usize {
        1_000
    }

    fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
        children
            .iter()
            .max_by_key(|child| child.visits())
            .expect("Should have at least one child")
    }
}

pub type Move<M> = <<M as MCTS>::State as GameState>::Move;
pub type MoveList<M> = <<M as MCTS>::State as GameState>::MoveList;
pub type MoveEval<M> = <<M as MCTS>::Select as Policy<M>>::MoveSelect;
pub type StateEval<M> = <<M as MCTS>::Eval as Evaluator<M>>::StateEval;
pub type Player<M> = <<M as MCTS>::State as GameState>::Player;
pub type TreePolicyThreadData<M> = <<M as MCTS>::Select as Policy<M>>::ThreadLocalData;

pub trait GameState: Clone {
    type Move: Sync + Send + Clone + PartialEq + std::fmt::Debug;
    type Player: Sync + std::fmt::Debug + PartialEq;
    type MoveList: std::iter::IntoIterator<Item = Self::Move> + Clone;

    fn current_player(&self) -> Self::Player;
    fn legal_moves(&self) -> Self::MoveList;
    fn all_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mv: &Self::Move);
    fn randomize_determination(&mut self, observer: Self::Player);
}

pub trait Evaluator<M: MCTS>: Sync {
    type StateEval: Sync + Send;

    fn state_eval_new(&self, state: &M::State, handle: Option<SearchHandle<M>>) -> Self::StateEval;
    fn eval_new(
        &self,
        state: &M::State,
        moves: &MoveList<M>,
        handle: Option<SearchHandle<M>>,
    ) -> (Vec<MoveEval<M>>, Self::StateEval);
    fn eval_existing(
        &self,
        state: &M::State,
        existing: &Self::StateEval,
        handle: SearchHandle<M>,
    ) -> Self::StateEval;
    fn make_relativ_player(&self, eval: &Self::StateEval, player: &Player<M>) -> i64;
}

pub trait Policy<M: MCTS<Select = Self>>: Sync + Sized {
    type MoveSelect: Sync + Send;
    type ThreadLocalData: Default;

    // fn choose<'a, MoveIter>(&self, moves: MoveIter, handle: SearchHandle<M>) -> &'a MoveInfo<M>
    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        handle: SearchHandle<M>,
    ) -> (usize, &'a MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>> + Clone;
    fn validate_evaluations(&self, _evals: &[Self::MoveSelect]) {}
}

pub struct ThreadData<M: MCTS> {
    pub policy_data: TreePolicyThreadData<M>,
}

impl<M: MCTS> Default for ThreadData<M>
where
    TreePolicyThreadData<M>: Default,
{
    fn default() -> Self {
        Self {
            policy_data: Default::default(),
        }
    }
}
