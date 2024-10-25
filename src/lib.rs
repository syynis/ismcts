#![warn(clippy::pedantic)]
#![allow(clippy::missing_panics_doc, clippy::cast_lossless)]
#![feature(mapped_lock_guards)]

use node::MoveInfo;
use search::SearchHandle;

pub mod manager;
pub mod node;
pub mod policies;
pub mod search;

pub trait MCTS: Sized + Sync {
    type State: GameState + Sync + std::fmt::Debug;
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
pub type StateEval<M> = <<M as MCTS>::Eval as Evaluator<M>>::StateEval;
pub type Player<M> = <<M as MCTS>::State as GameState>::Player;
pub type Knowledge<M> = <<M as MCTS>::State as GameState>::Knowledge;
pub type TreePolicyThreadData<M> = <<M as MCTS>::Select as Policy<M>>::ThreadLocalData;

pub trait GameState: Clone {
    type Move: Sync + Send + Clone + PartialEq + std::fmt::Debug;
    type Player: Sync + std::fmt::Debug + PartialEq + From<usize> + Into<usize>;
    type MoveList: std::iter::IntoIterator<Item = Self::Move> + Clone;
    type Knowledge: Sync + Clone + std::fmt::Debug;

    fn current_player(&self) -> Self::Player;
    fn legal_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mv: &Self::Move);
    fn randomize_determination(&mut self, observer: Self::Player, knowledge: &Self::Knowledge);
    fn update_knowledge(&self, mv: &Self::Move, knowledge: &mut Self::Knowledge);
    fn new_knowledge(&self, observer: Self::Player) -> Self::Knowledge;
    fn knowledge_from_state(&self, observer: Self::Player) -> Self::Knowledge;
}

pub trait Evaluator<M: MCTS>: Sync {
    type StateEval: Sync + Send;

    fn eval_new(&self, state: &M::State, handle: Option<SearchHandle<M>>) -> Self::StateEval;
    fn eval_existing(
        &self,
        state: &M::State,
        existing: &Self::StateEval,
        handle: SearchHandle<M>,
    ) -> Self::StateEval;
    fn make_relative(&self, eval: &Self::StateEval, player: &Player<M>) -> i64;
}

pub trait Policy<M: MCTS<Select = Self>>: Sync + Sized {
    type MoveSelect: Sync + Send;
    type ThreadLocalData: Default;

    fn choose<'a, MoveIter>(
        &self,
        moves: MoveIter,
        handle: SearchHandle<M>,
    ) -> (usize, &'a MoveInfo<M>)
    where
        MoveIter: Iterator<Item = &'a MoveInfo<M>> + Clone;
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
