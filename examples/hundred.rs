use mcts::{manager::MCTSManager, policies::UCTPolicy, *};

#[derive(Clone, Debug)]
struct CountingGame {
    state: i64,
    goal: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Move {
    Add(i64),
    Sub(i64),
    Mul(i64),
    Div(i64),
}

impl GameState for CountingGame {
    type Move = Move;
    type Player = ();
    type MoveList = Vec<Self::Move>;

    fn current_player(&self) -> Self::Player {
        ()
    }

    fn legal_moves(&self) -> Self::MoveList {
        if self.state == self.goal {
            vec![]
        } else {
            vec![
                Move::Add(1),
                Move::Add(3),
                Move::Sub(2),
                Move::Sub(4),
                Move::Add(51),
                Move::Add(153),
                Move::Sub(40),
                Move::Sub(160),
                Move::Mul(-3),
                Move::Mul(7),
                Move::Mul(-13),
                Move::Mul(17),
                Move::Div(-4),
                Move::Div(8),
                Move::Div(-16),
                Move::Div(32),
            ]
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        match *mov {
            Move::Add(x) => self.state += x,
            Move::Sub(x) => self.state -= x,
            Move::Mul(x) => self.state *= x,
            Move::Div(x) => self.state /= x,
        }
    }

    fn randomize_determination(&mut self, _observer: Self::Player) {}
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEval = f64;

    fn state_eval_new(
        &self,
        state: &<MyMCTS as MCTS>::State,
        _handle: Option<search::SearchHandle<MyMCTS>>,
    ) -> Self::StateEval {
        1.0 - ((state.state.abs() as f64 - state.goal as f64).abs() / 1_000_000.0)
    }

    fn eval_new(
        &self,
        state: &<MyMCTS as MCTS>::State,
        moves: &MoveList<MyMCTS>,
        handle: Option<search::SearchHandle<MyMCTS>>,
    ) -> (Vec<MoveEval<MyMCTS>>, Self::StateEval) {
        (vec![(); moves.len()], self.state_eval_new(state, handle))
    }

    fn eval_existing(
        &self,
        _state: &<MyMCTS as MCTS>::State,
        existing: &Self::StateEval,
        _handle: search::SearchHandle<MyMCTS>,
    ) -> Self::StateEval {
        *existing
    }

    fn make_relativ_player(&self, eval: &Self::StateEval, _player: &Player<MyMCTS>) -> i64 {
        *eval as i64
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = CountingGame;
    type Eval = MyEvaluator;
    type Select = UCTPolicy;

    fn virtual_loss(&self) -> i64 {
        1000
    }
}

fn main() {
    let game = CountingGame {
        state: 0,
        goal: 1337,
    };
    let mut mcts = MCTSManager::new(game, MyMCTS, UCTPolicy(1.0), MyEvaluator);
    mcts.playout_n_parallel(5_000_000_0, 8);
    let pv: Vec<_> = mcts
        .pv_states(100)
        .into_iter()
        .map(|(_mv, state)| state)
        .collect();
    println!("Principal variation: {:?}", pv);
    println!("Principal variation length: {:?}", pv.len());
    mcts.print_stats();
}
