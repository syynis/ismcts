use std::{fmt::Display, time::Instant};

use enum_map::{Enum, EnumMap};
use ismcts::{manager::Manager, policies::UCTPolicy, *};
use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};

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

impl From<Player> for usize {
    fn from(value: Player) -> Self {
        value as usize
    }
}

impl From<usize> for Player {
    fn from(value: usize) -> Self {
        unsafe { std::mem::transmute(value as u8) }
    }
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
    observer: Player,
    enemy_hand: EnumMap<Card, Option<CardKnowledge>>,
    amount_unknown: u8,
}

impl HandKnowledge {
    pub fn new(observer: Player) -> Self {
        Self {
            observer,
            enemy_hand: EnumMap::default(),
            amount_unknown: 5,
        }
    }

    pub fn count_known(&self) -> u8 {
        self.enemy_hand
            .values()
            .filter_map(|x| x.as_ref().map(|k| k.amount()))
            .sum::<u8>()
    }

    fn update_with_move(&mut self, state: &LandsGame, mv: &Move) {
        let mut player = state.current_player();
        match state.phase {
            Phase::Play => {
                if player != self.observer {
                    self.remove(mv.card());
                }
            }
            Phase::Respond(first_move, _) => {
                if let Move::Counter(Some(card)) = mv {
                    if player != self.observer {
                        self.remove(*card);
                        self.remove(Card::Blue);
                    }
                } else if !(matches!(first_move, Move::Discard(None))
                    && matches!(mv, Move::Counter(None))
                    && !state.countered)
                {
                    if !state.countered {
                        let mut first_move = first_move;
                        if let Move::Discard(Some(_)) = mv {
                            if self.observer == player {
                                self.update_from_hand(&state.hands[player.next() as usize]);
                            }
                            first_move = *mv;
                        } else if !matches!(mv, Move::Discard(None)) {
                            player = player.next();
                        }
                        match first_move {
                            Move::Draw => {
                                if self.observer != player {
                                    self.on_draw();
                                }
                            }
                            Move::Discard(Some(card)) => {
                                if self.observer == player {
                                    self.remove(card);
                                }
                            }
                            Move::Revive(card) => {
                                if self.observer != player {
                                    self.add(card);
                                }
                            }
                            _ => {}
                        }
                    }
                    if self.observer == player {
                        self.on_draw();
                    }
                }
            }
        }
    }

    fn on_draw(&mut self) {
        CARDS.into_iter().for_each(|c| {
            self.make_atleast(c);
        });
        self.amount_unknown += 1;
    }

    fn update_from_hand(&mut self, hand: &EnumMap<Card, u8>) {
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
            let res = match knowledge {
                CardKnowledge::Atleast(x) => CardKnowledge::Atleast(x + 1),
                CardKnowledge::Exact(x) => CardKnowledge::Exact(x + 1),
            };
            self.enemy_hand[card] = Some(res);
        } else {
            self.enemy_hand[card] = Some(CardKnowledge::Atleast(1));
        }
    }

    fn remove(&mut self, card: Card) {
        if let Some(knowledge) = self.enemy_hand[card] {
            let res = match knowledge {
                CardKnowledge::Atleast(0) => {
                    self.amount_unknown = self
                        .amount_unknown
                        .checked_sub(1)
                        .expect("Should never be below 0");
                    CardKnowledge::Atleast(0)
                }
                CardKnowledge::Atleast(x) => CardKnowledge::Atleast(x.saturating_sub(1)),
                CardKnowledge::Exact(x) => CardKnowledge::Exact(x - 1),
            };
            self.enemy_hand[card] = Some(res);
        } else {
            self.amount_unknown = self
                .amount_unknown
                .checked_sub(1)
                .expect("Should never be below 0");
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

#[derive(Debug, Clone)]
struct LandsGame {
    deck: Vec<Card>,
    in_play: [EnumMap<Card, u8>; 2],
    discarded: [EnumMap<Card, u8>; 2],
    phase: Phase,
    hands: [EnumMap<Card, u8>; 2],
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
            to_move: Player::One,
            countered: false,
        }
    }

    fn draw(&mut self) {
        self.hands[self.to_move as usize][self.deck.pop().expect("Deck never empty")] += 1;
    }

    fn discard(&mut self, card: Card, player: Player) {
        self.hands[player as usize][card] -= 1;
        self.discarded[player as usize][card] += 1;
    }

    fn put_in_play(&mut self, card: Card, player: Player) {
        self.hands[player as usize][card] -= 1;
        self.in_play[player as usize][card] += 1;
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

    fn determinize_hand_with_knowledge(&mut self, observer: Player, knowledge: &HandKnowledge) {
        let enemy_hand = &mut self.hands[observer.next() as usize];
        let count = enemy_hand.values().sum::<u8>();
        let knowledge_count = knowledge.count_known();

        assert_eq!(
            knowledge_count + knowledge.amount_unknown,
            count,
            "{knowledge_count} {} {knowledge:?}",
            knowledge.amount_unknown
        );

        let mut unknown = 0;
        // Put back cards into deck
        for ((card, hand_count), knowledge_entry) in
            enemy_hand.clone().iter().zip(knowledge.enemy_hand.values())
        {
            let knowledge_amount = knowledge_entry.map_or(0, |e| e.amount());
            let amount = hand_count - knowledge_amount;
            unknown += amount;
            enemy_hand[card] = knowledge_amount;
            (0..amount).for_each(|_| self.deck.push(card));
        }

        assert_eq!(unknown, knowledge.amount_unknown);

        self.deck.shuffle(&mut rand::thread_rng());
        // Draw new hand
        (0..unknown).for_each(|_| {
            enemy_hand[self.deck.pop().expect("Not empty")] += 1;
        });
    }
}

impl GameState for LandsGame {
    type Move = Move;
    type Player = Player;
    type MoveList = Vec<Self::Move>;
    type Knowledge = HandKnowledge;

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
                self.to_move = self.opponent();
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

    fn randomize_determination(&mut self, observer: Self::Player, knowledge: &Self::Knowledge) {
        self.determinize_hand_with_knowledge(observer, knowledge);
    }

    fn update_knowledge(&self, mv: &Self::Move, knowledge: &mut Self::Knowledge) {
        knowledge.update_with_move(self, mv);
    }

    fn knowledge_from_state(&self, observer: Self::Player) -> Self::Knowledge {
        HandKnowledge::new(observer)
    }
}

struct GameEval;

impl Evaluator<AI> for GameEval {
    type StateEval = i64;

    fn eval_new(
        &self,
        state: &<AI as MCTS>::State,
        _handle: Option<search::SearchHandle<AI>>,
    ) -> Self::StateEval {
        let won = state.won(state.to_move) as i64 * 1000;
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
        let card_advantage = state.in_play[state.to_move as usize].values().sum::<u8>() as i64
            + state.hand(state.to_move).values().sum::<u8>() as i64
            - state.in_play[state.opponent() as usize]
                .values()
                .sum::<u8>() as i64
            - state.hand(state.opponent()).values().sum::<u8>() as i64;
        devotion * 2 + domain * 2 + card_advantage * 3 + won
    }

    fn eval_existing(
        &self,
        _state: &LandsGame,
        existing: &Self::StateEval,
        _handle: search::SearchHandle<AI>,
    ) -> Self::StateEval {
        *existing
    }

    fn make_relative(&self, eval: &Self::StateEval, player: &ismcts::Player<AI>) -> i64 {
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
        1
    }
}

fn main() {
    let mut first_wins = 0;
    let mut second_wins = 0;
    let p = 50_000;
    let t = 8;
    for i in 0..100 {
        let (f, s) = if i & 1 == 0 {
            (17.5, 50.0)
        } else {
            (50.0, 17.5)
        };
        let seed = thread_rng().gen::<u64>();
        let mut mcts: Manager<AI, 2> =
            Manager::new(LandsGame::new(seed), AI, UCTPolicy(f), GameEval);
        let mut mcts2: Manager<AI, 2> =
            Manager::new(LandsGame::new(seed), AI, UCTPolicy(s), GameEval);
        let before = Instant::now();
        mcts.playout_n_parallel(p, t);
        let after = Instant::now();
        println!("time {:?}", (after - before).as_secs_f64());
        mcts2.playout_n_parallel(p, t);
        println!("Starting {i} game");
        loop {
            let current_player = mcts.tree().root_state().current_player();
            let best_move = match current_player {
                Player::One => mcts.best_move(),
                Player::Two => mcts2.best_move(),
            };

            if let Some(best_move) = best_move {
                mcts.advance(&best_move);
                mcts2.advance(&best_move);
                mcts.playout_n_parallel(p, t);
                mcts2.playout_n_parallel(p, t);
            } else {
                if mcts.tree().root_state().won(Player::One) {
                    if i & 1 == 0 {
                        first_wins += 1;
                    } else {
                        second_wins += 1;
                    }
                } else {
                    if i & 1 == 0 {
                        second_wins += 1;
                    } else {
                        first_wins += 1;
                    }
                }
                break;
            }
        }
    }
    println!("{first_wins}, {second_wins}");
}
