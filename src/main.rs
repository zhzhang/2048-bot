use std::io::{self, Write};
use rand::{seq::IteratorRandom, Rng, thread_rng};

type Game = [u8; 16];

#[derive(Copy, Clone)]
enum Move {
    Left,
    Right,
    Up,
    Down
}

static MOVES: [Move; 4] = [Move::Left, Move::Right, Move::Up, Move::Down];

fn insert_random_tile(game: Game) -> Game {
    let open_tiles = get_open_tiles(game);
    // Randomly pick an open tile:
    let mut rng = thread_rng();
    let i = open_tiles.iter().choose(&mut rng).unwrap();
    let value = random_tile_value();
    let mut new_game = game;
    new_game[*i] = value;
    new_game
}

fn get_open_tiles(game: Game) -> Vec<usize> {
    let mut open_tiles: Vec<usize> = Vec::new();
    for i in 0..16 {
        let value = game[i];
        if value == 0 {
            open_tiles.push(i);
        }
    }
    open_tiles
}

fn random_tile_value() -> u8 {
    let mut rng = rand::thread_rng();
    let four: bool = rng.gen_bool(0.1);
    if four {
        2
    } else {
        1
    }
}


fn execute_move(game: Game, m: Move) -> Game{
    // Disallow a move if there's no square shift.
    let mut head: i8 = match m {
        Move::Left => 0,
        Move::Right => 3,
        Move::Up => 0,
        Move::Down => 12
    };
    let step: i8 = match m {
        Move::Left => 1,
        Move::Right => -1,
        Move::Up => 4,
        Move::Down => -4
    };
    let rank_step: i8 = match m {
        Move::Left => 4,
        Move::Right => 4,
        Move::Up => 1,
        Move::Down => 1
    };
    let mut new_game = [0; 16];
    for _ in 0..4 {
        let mut line: Vec<u8> = Vec::new();
        let mut i = head;
        let mut last_value: u8 = 0;
        for _ in 0..4 {
            // println!("i: {}", i);
            let value = game[(i as usize)];
            if value != 0 {
                if value == last_value {
                    line.push(value + 1);
                    last_value = 0;
                } else {
                    if last_value != 0 {
                        line.push(last_value);
                        last_value = 0;
                    }
                    last_value = value;
                }
            }
            // println!("line: {:?}", line);
            i += step;
        }
        if last_value != 0 {
            line.push(last_value);
        }
        let mut i = head;
        for val in line {
            new_game[(i as usize)] = val;
            i += step;
        }
        head += rank_step;
    }
    new_game
}

fn print_game_state(game: Game) {
    for i in 0..4 {
        for j in 0..4 {
            let value = game[i * 4 + j];
            if value == 0 {
                print!("    .");
            } else {
                print!("{:5}", 1 << value);
            }
        }
        println!();
    }
}

fn search(game: Game, depth: u8) -> (u8, Move) {
    if depth == 0 {
        return (score_game(game), Move::Up);
    }
    let mut best_score: u8 = 0;
    let mut best_move: Move = Move::Left;
    for m in MOVES {
        let new_game = execute_move(game, m);
        let open_tiles = get_open_tiles(new_game);
        for i in open_tiles {
            let mut possible_game = new_game;
            possible_game[i] = 1;
            let (score, _) = search(new_game, depth - 1);
            if score > best_score {
                best_score = score;
                best_move = m;
            }
        }
    }
    (best_score, best_move)
}

fn score_game(game: Game) -> u8 {
    let mut score: u8 = 0;
    for i in 0..16 {
        let value = game[i];
        if value != 0 {
            score += value;
        }
    }
    score
}

fn main() {
    let mut game: Game = [0; 16];
    game = insert_random_tile(game);
    game = insert_random_tile(game);
    print_game_state(game);
    loop {
        let (_, best_move) = search(game, 4);
        game = execute_move(game, best_move);
        game = insert_random_tile(game);
        print_game_state(game);
    }
    // loop {
    //     // Wait for input from the user.
    //     let mut input = String::new();
    //     print!("Next move: ");
    //     io::stdout().flush().unwrap();
    //     io::stdin().read_line(&mut input).unwrap();
    //     // Update the game state.
    //     match input.trim() {
    //         "a" => game = execute_move(game, Move::Left),
    //         "w" => game = execute_move(game, Move::Up),
    //         "s" => game = execute_move(game, Move::Down),
    //         "d" => game = execute_move(game, Move::Right),
    //         _ => println!("Invalid move.")
    //     }
    //     // Insert a new tile.
    //     game = insert_random_tile(game);
    //     // Print the game state.
    //     print_game_state(game);
    // }
}