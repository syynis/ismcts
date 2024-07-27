extern crate mcts;

use std::{fmt::Display, io};

use enum_map::{Enum, EnumMap};
use itertools::Itertools;
use mcts::{manager::MCTSManager, policies::UCTPolicy, *};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const CARDS: [Card; 5] = [Card::White, Card::Black, Card::Green, Card::Red, Card::Blue];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Enum)]
enum Card {
    White,
    Black,
    Green,
    Red,
    Blue,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Phase {
    Play,
    Respond(Move, Move),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    One,
    Two,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Move {
    Red,
    Black,
    Green,
    Blue,
    Draw,
    Discard(Option<Card>),
    Destroy(Card),
    Revive(Card),
    Counter(Option<Card>),
}

impl Move {
    pub fn card(&self) -> Card {
        match self {
            Move::Draw => Card::White,
            Move::Discard(_) => Card::Black,
            Move::Destroy(_) => Card::Red,
            Move::Counter(_) => Card::Blue,
            Move::Revive(_) => Card::Green,
            Move::Blue => Card::Blue,
            Move::Red => Card::Red,
            Move::Black => Card::Black,
            Move::Green => Card::Green,
        }
    }
}

impl Player {
    fn next(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HandKnowledge {
    enemy_hand: EnumMap<Card, Option<CardKnowledge>>,
    amount_unknown: u8,
}

impl HandKnowledge {
    pub fn new(amount_unknown: u8) -> Self {
        Self {
            enemy_hand: EnumMap::default(),
            amount_unknown,
        }
    }

    pub fn count_known(&self) -> u8 {
        self.enemy_hand
            .values()
            .filter_map(|x| x.as_ref().map(|k| k.amount()))
            .sum::<u8>()
    }

    fn update(&mut self, hand: &EnumMap<Card, u8>) {
        for (card, amount) in hand.iter() {
            self.enemy_hand[card] = Some(CardKnowledge::Exact(*amount));
        }
        self.amount_unknown = 0;
    }

    fn make_atleast(&mut self, card: Card) {
        if let Some(knowledge) = self.enemy_hand[card] {
            self.enemy_hand[card] = Some(CardKnowledge::Atleast(knowledge.amount()));
        }
    }

    fn add(&mut self, card: Card) {
        if let Some(knowledge) = self.enemy_hand[card] {
            match knowledge {
                CardKnowledge::Atleast(x) => CardKnowledge::Atleast(x + 1),
                CardKnowledge::Exact(x) => CardKnowledge::Exact(x + 1),
            };
        } else {
            self.enemy_hand[card] = Some(CardKnowledge::Atleast(1));
        }
    }

    fn remove(&mut self, card: Card) {
        if let Some(knowledge) = self.enemy_hand[card] {
            let res = match knowledge {
                CardKnowledge::Atleast(x) => Some(CardKnowledge::Atleast(x.saturating_sub(1))),
                CardKnowledge::Exact(x) => Some(CardKnowledge::Exact(x - 1)),
            };
            self.enemy_hand[card] = res;
        } else {
            self.amount_unknown = self.amount_unknown.saturating_sub(1);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardKnowledge {
    Atleast(u8),
    Exact(u8),
}

impl CardKnowledge {
    fn amount(&self) -> u8 {
        match self {
            CardKnowledge::Atleast(x) => *x,
            CardKnowledge::Exact(x) => *x,
        }
    }
}

#[derive(Clone)]
struct LandsGame {
    deck: Vec<Card>,
    in_play: [EnumMap<Card, u8>; 2],
    discarded: [EnumMap<Card, u8>; 2],
    phase: Phase,
    hands: [EnumMap<Card, u8>; 2],
    knowledge: [HandKnowledge; 2],
    to_move: Player,
    countered: bool,
}

impl Display for LandsGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Player   {:?}", self.to_move)?;
        writeln!(f, "Phase    {:?}", self.phase)?;
        write!(f, "Play {:?}", self.in_play[0])?;
        writeln!(f, " | Play {:?}", self.in_play[1])?;

        write!(f, "Disc {:?}", self.discarded[0])?;
        writeln!(f, " | Disc {:?}", self.discarded[1])?;

        write!(f, "Hand {:?}", self.hands[0])?;
        writeln!(f, " | Hand {:?}", self.hands[1])?;
        Ok(())
    }
}

impl LandsGame {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut deck = Vec::new();
        deck.push(vec![Card::White; 15]);
        deck.push(vec![Card::Black; 15]);
        deck.push(vec![Card::Red; 15]);
        deck.push(vec![Card::Blue; 15]);
        deck.push(vec![Card::Green; 15]);
        let mut deck = deck.iter().flatten().cloned().collect_vec();
        deck.shuffle(&mut rng);
        assert!(deck.len() == 75);

        let mut hand1 = EnumMap::default();
        let mut hand2 = EnumMap::default();

        (0..5).for_each(|_| {
            if let Some(top) = deck.pop() {
                hand1[top] += 1;
            }
            if let Some(top) = deck.pop() {
                hand2[top] += 1;
            }
        });

        assert!(deck.len() == 65);

        let map: EnumMap<Card, u8> = EnumMap::default();
        Self {
            deck,
            in_play: [map; 2],
            discarded: [map; 2],
            phase: Phase::Play,
            hands: [hand1, hand2],
            knowledge: [HandKnowledge::new(5); 2],
            to_move: Player::One,
            countered: false,
        }
    }

    fn draw(&mut self) {
        self.hands[self.to_move as usize][self.deck.pop().expect("Deck never empty")] += 1;
        CARDS.iter().for_each(|c| {
            self.knowledge[self.opponent() as usize].make_atleast(*c);
        });
        self.knowledge[self.opponent() as usize].amount_unknown += 1;
    }

    fn discard(&mut self, card: Card, player: Player) {
        self.hands[player as usize][card] -= 1;
        self.discarded[player as usize][card] += 1;
        self.knowledge[player.next() as usize].remove(card);
    }

    fn put_in_play(&mut self, card: Card, player: Player) {
        self.hands[player as usize][card] -= 1;
        self.in_play[player as usize][card] += 1;
        self.knowledge[player.next() as usize].remove(card);
    }

    fn put_out_play(&mut self, card: Card, player: Player) {
        self.in_play[player as usize][card] -= 1;
        self.discarded[player as usize][card] += 1;
    }

    fn opponent(&self) -> Player {
        self.to_move.next()
    }

    fn hand(&self, player: Player) -> &EnumMap<Card, u8> {
        &self.hands[player as usize]
    }

    fn won(&self, player: Player) -> bool {
        self.in_play[player as usize].values().any(|v| *v == 5)
            || self.in_play[player as usize].values().all(|v| *v > 0)
    }

    fn randomize_hand(&mut self, player: Player) {
        let hand = &mut self.hands[player as usize];
        let count = hand.values().sum::<u8>();
        // Put back cards into deck
        for (card, amount) in hand.iter() {
            (0..*amount).for_each(|_| self.deck.push(card));
        }
        // Clear hand
        hand.clear();
        self.deck.shuffle(&mut rand::thread_rng());
        // Draw new hand
        (0..count).for_each(|_| {
            hand[self.deck.pop().expect("Not empty")] += 1;
        });
    }

    fn determinize_hand_with_knowledge(&mut self, player: Player) {
        let hand = &mut self.hands[player as usize];
        let count = hand.values().sum::<u8>();
        let knowledge = &self.knowledge[player.next() as usize];
        let knowledge_count = knowledge.count_known();

        assert!(knowledge_count + knowledge.amount_unknown == count);

        let mut unknown = 0;
        // Put back cards into deck
        for ((card, hand_count), knowledge_entry) in
            hand.clone().iter().zip(knowledge.enemy_hand.values())
        {
            let knowledge_amount = knowledge_entry.map_or(0, |e| e.amount());
            let amount = hand_count - knowledge_amount;
            unknown += amount;
            hand[card] = knowledge_amount;
            (0..amount).for_each(|_| self.deck.push(card));
        }

        assert!(unknown == knowledge.amount_unknown);

        self.deck.shuffle(&mut rand::thread_rng());
        // Draw new hand
        (0..unknown).for_each(|_| {
            hand[self.deck.pop().expect("Not empty")] += 1;
        });
    }

    fn print_knowledge(&self) {
        for p in [Player::One, Player::Two].iter() {
            println!("{:?}", self.knowledge[*p as usize]);
        }
    }
}

impl GameState for LandsGame {
    type Move = Move;
    type Player = Player;
    type MoveList = Vec<Self::Move>;

    fn current_player(&self) -> Self::Player {
        self.to_move
    }

    fn legal_moves(&self) -> Self::MoveList {
        let mut moves = Vec::new();
        let hand = self.hand(self.to_move);
        let opp_hand = self.hand(self.opponent());
        let opp_in_play = &self.in_play[self.opponent() as usize];
        let discard = &self.discarded[self.to_move as usize];

        if self.won(self.opponent()) {
            return Vec::new();
        }

        match &self.phase {
            Phase::Play => {
                for card in hand
                    .iter()
                    .filter_map(|(card, count)| (*count > 0).then_some(card))
                {
                    match card {
                        Card::White => moves.push(Move::Draw),
                        Card::Blue => moves.push(Move::Blue),
                        Card::Black => {
                            if opp_hand.values().sum::<u8>() == 0 {
                                moves.push(Move::Black);
                            } else {
                                moves.push(Move::Discard(None));
                            }
                        }
                        Card::Green => {
                            if discard.values().sum::<u8>() == 0 {
                                moves.push(Move::Green);
                            } else {
                                discard
                                    .iter()
                                    .filter_map(|(c, cnt)| (*cnt > 0).then_some(c))
                                    .for_each(|c| moves.push(Move::Revive(c)));
                            }
                        }
                        Card::Red => {
                            if opp_in_play.values().sum::<u8>() == 0 {
                                moves.push(Move::Red);
                            } else {
                                opp_in_play
                                    .iter()
                                    .filter_map(|(c, cnt)| (*cnt > 0).then_some(c))
                                    .for_each(|c| moves.push(Move::Destroy(c)));
                            }
                        }
                    }
                }
            }
            Phase::Respond(first_move, last_move) => {
                if matches!(first_move, Move::Discard(None))
                    && matches!(last_move, Move::Counter(None))
                    && !self.countered
                {
                    if self.hand(self.opponent()).values().sum::<u8>() == 0 {
                        moves.push(Move::Discard(None));
                    } else {
                        self.hands[self.opponent() as usize]
                            .iter()
                            .filter_map(|(c, cnt)| (*cnt > 0).then_some(c))
                            .for_each(|c| moves.push(Move::Discard(Some(c))));
                    }
                } else {
                    let is_blue = |card| -> u8 { matches!(card, Card::Blue) as u8 };
                    let countered = self.countered as u8;

                    moves.push(Move::Counter(None));
                    if hand[Card::Blue] > countered * is_blue(first_move.card())
                        && hand.values().sum::<u8>() > 1
                    {
                        for card in hand.iter().filter_map(|(card, count)| {
                            (*count
                                > is_blue(card) + countered * ((card == first_move.card()) as u8))
                                .then_some(card)
                        }) {
                            moves.push(Move::Counter(Some(card)));
                        }
                    }
                }
            }
        }
        moves
    }

    fn make_move(&mut self, mv: &Self::Move) {
        // println!("{}", self);
        // println!("{:?} can play {:?}", self.to_move, self.legal_moves());
        // println!("{:?} plays {:?}", self.current_player(), mv);
        match self.phase {
            Phase::Play => {
                self.put_in_play(mv.card(), self.to_move);
                self.phase = Phase::Respond(*mv, *mv);
                self.to_move = self.to_move.next();
                self.countered = false;
            }
            Phase::Respond(first_move, _) => {
                if let Move::Counter(Some(card)) = mv {
                    self.discard(*card, self.to_move);
                    self.discard(Card::Blue, self.to_move);
                    self.to_move = self.opponent();
                    self.phase = Phase::Respond(first_move, *mv);
                    self.countered = !self.countered;
                } else {
                    // Make opponent choose discard move
                    if matches!(first_move, Move::Discard(None))
                        && matches!(mv, Move::Counter(None))
                        && !self.countered
                    {
                        self.to_move = self.opponent();
                        self.phase = Phase::Respond(first_move, *mv);
                    } else {
                        // Card was countered
                        if self.countered {
                            self.put_out_play(first_move.card(), self.to_move);
                        } else {
                            let mut first_move = first_move;
                            if let Move::Discard(Some(_)) = mv {
                                self.knowledge[self.to_move as usize]
                                    .update(&self.hands[self.opponent() as usize]);
                                first_move = *mv;
                            } else if !matches!(mv, Move::Discard(None)) {
                                self.to_move = self.opponent();
                            }
                            match first_move {
                                Move::Draw => self.draw(),
                                Move::Discard(Some(c)) => {
                                    self.discard(c, self.opponent());
                                }
                                Move::Destroy(c) => {
                                    self.in_play[self.opponent() as usize][c] -= 1;
                                    self.discarded[self.opponent() as usize][c] += 1;
                                }
                                Move::Revive(c) => {
                                    self.discarded[self.to_move as usize][c] -= 1;
                                    self.hands[self.to_move as usize][c] += 1;
                                    self.knowledge[self.opponent() as usize].add(c);
                                }
                                _ => {}
                            }
                        }
                        self.to_move = self.opponent();
                        self.draw();
                        self.phase = Phase::Play
                    }
                }
            }
        }
    }

    fn randomize_determination(&mut self, observer: Self::Player) {
        self.randomize_hand(observer.next());
    }
}

struct GameEval;

impl Evaluator<AI> for GameEval {
    type StateEval = i64;

    fn state_eval_new(
        &self,
        state: &<AI as MCTS>::State,
        _handle: Option<search::SearchHandle<AI>>,
    ) -> Self::StateEval {
        let won = state.won(state.to_move) as i64 * 100;
        let devotion = *state.in_play[state.to_move as usize]
            .values()
            .max()
            .unwrap() as i64
            - *state.in_play[state.opponent() as usize]
                .values()
                .max()
                .unwrap() as i64;
        let domain = state.in_play[state.to_move as usize]
            .values()
            .map(|v| (*v > 0) as i64)
            .sum::<i64>()
            - state.in_play[state.opponent() as usize]
                .values()
                .map(|v| (*v > 0) as i64)
                .sum::<i64>();
        let card_advantage = (state.in_play[state.to_move as usize].values().sum::<u8>() as i64
            + state.hand(state.to_move).values().sum::<u8>() as i64
            - state.in_play[state.opponent() as usize]
                .values()
                .sum::<u8>() as i64
            - state.hand(state.opponent()).values().sum::<u8>() as i64);
        devotion + domain + card_advantage + won
    }

    fn eval_new(
        &self,
        state: &LandsGame,
        moves: &Vec<Move>,
        handle: Option<search::SearchHandle<AI>>,
    ) -> (Vec<MoveEval<AI>>, Self::StateEval) {
        (vec![(); moves.len()], self.state_eval_new(&state, handle))
    }

    fn eval_existing(
        &self,
        _state: &LandsGame,
        existing: &Self::StateEval,
        _handle: search::SearchHandle<AI>,
    ) -> Self::StateEval {
        *existing
    }

    fn make_relative(&self, eval: &Self::StateEval, player: &mcts::Player<AI>) -> i64 {
        match player {
            Player::One => *eval,
            Player::Two => -*eval,
        }
    }
}

#[derive(Default)]
struct AI;

impl MCTS for AI {
    type State = LandsGame;
    type Eval = GameEval;
    type Select = UCTPolicy;

    fn virtual_loss(&self) -> i64 {
        0
    }
}

fn main() {
    let mut input = String::new();
    let mut mcts = MCTSManager::new(LandsGame::new(1335), AI, UCTPolicy(0.5), GameEval);
    println!("{}", mcts.tree().root_state());

    loop {
        if io::stdin().read_line(&mut input).is_ok() {
            if input == "m\n" {
                mcts.tree().legal_moves();
                println!("{}", mcts.tree().root_state());
            } else if input == "lm\n" {
                println!("{:?}", mcts.tree().root_state().legal_moves());
            } else if input == "pv\n" {
                let pv = mcts.pv(500);
                println!("{:?}", pv);
            } else if input == "bm\n" {
                if let Some(best_move) = mcts.best_move() {
                    println!("Make move {:?}", best_move);
                    mcts = mcts.make_move(best_move);
                }
            } else if input == "adv\n" {
                if let Some(best_move) = mcts.best_move() {
                    println!("Make move {:?}", best_move);
                    mcts.advance(best_move);
                }
            } else if input == "pmm\n" {
                let legal_moves = mcts.tree().root_state().legal_moves();
                if legal_moves.len() == 1 {
                    println!("Make move {:?}", legal_moves[0]);
                    mcts.advance(legal_moves[0]);
                } else {
                    mcts.playout_n_parallel(2_500_000, 8);
                    if let Some(best_move) = mcts.best_move() {
                        println!("Make move {:?}", best_move);
                        mcts.advance(best_move);
                    }
                }
                println!("{}", mcts.tree().root_state());
            } else if let Ok(number) = input.strip_suffix('\n').unwrap().parse::<usize>() {
                let mv = mcts.tree().root_state().legal_moves()[number];
                mcts = mcts.make_move(mv);
            } else if input == "s\n" {
                println!("{}", mcts.tree().root_state());
            } else if input == "stats\n" {
                mcts.print_stats();
            } else if input == "pvs\n" {
                mcts.pv_states(500)
                    .iter()
                    .rev()
                    .skip(1)
                    .rev()
                    .for_each(|s| {
                        println!("Played move {:?}", s.0.unwrap());
                        println!("{}", s.1);
                    });
            } else if input == "p\n" {
                mcts.playout_n_parallel(10_000_000, 8);
                // mcts.playout_n(1);
                println!("playout");
            } else if input == "p1\n" {
                mcts.playout_n(1);
                // mcts.playout_n(1);
                println!("playout");
            } else if input == "k\n" {
                mcts.tree().root_state().print_knowledge();
            } else if input == "q\n" {
                break;
            } else if input == "clear\n" {
                mcts.clear_orphaned();
                println!("cleared");
            } else {
                println!("m for moves, q for quit");
            }
            input.clear();
        }
    }
}
