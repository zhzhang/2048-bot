// Implement taken from https://github.com/nneonneo/2048-ai and translated to Rust.
use rand::{seq::IteratorRandom, thread_rng, Rng};
use std::io::{self, Write};
use std::time::{Duration, SystemTime};

#[derive(Copy, Clone)]
enum Move {
    Left,
    Right,
    Up,
    Down,
}

struct Game<'a> {
    left_table: &'a mut [u64; 65536],
    right_table: &'a mut [u64; 65536],
    up_table: &'a mut [u64; 65536],
    down_table: &'a mut [u64; 65536],
}

static MOVES: [Move; 4] = [Move::Left, Move::Right, Move::Up, Move::Down];
static ROW_MASK: u64 = 0xFFFF;
static COL_MASK: u64 = 0x000F000F000F000F;

static NTUPLE_MASKS: [u64; 4] = [
    0xFFF0FFF000000000,
    0x0FF00FF00F000F00,
    0xFFFFFF0000000000,
    0xFF000FFF00F00000,
];

fn random_tile_value() -> u8 {
    let mut rng = rand::thread_rng();
    let four: bool = rng.gen_bool(0.1);
    if four {
        2
    } else {
        1
    }
}

fn insert_random_tile(board: u64) -> u64 {
    let open_tiles = get_open_tiles(board);
    if (open_tiles.len() == 0) {
        return board;
    }
    // Randomly pick an open tile:
    let mut rng = thread_rng();
    let i = open_tiles.iter().choose(&mut rng).unwrap();
    let value = random_tile_value();
    board | ((value as u64) << (4 * i))
}

fn get_open_tiles(board: u64) -> Vec<usize> {
    let mut open_tiles: Vec<usize> = Vec::new();
    for i in 0..16 {
        let value = board >> (4 * i) & 0xF;
        if value == 0 {
            open_tiles.push(i);
        }
    }
    open_tiles
}

fn reverse_line_repr(line_repr: u64) -> u64 {
    // Print bits.
    let mut reversed: u64 = 0;
    ((line_repr >> 12) | ((line_repr >> 4) & 0xF0) | ((line_repr << 4) & 0xF00) | (line_repr << 12))
        & ROW_MASK
}

fn unpack_col(row: u64) -> u64 {
    (row | (row << 12) | (row << 24) | (row << 36)) & COL_MASK
}

fn transpose(board: u64) -> u64 {
    let a1 = board & 0xF0F00F0FF0F00F0F;
    let a2 = board & 0x0000F0F00000F0F0;
    let a3 = board & 0x0F0F00000F0F0000;
    let a = a1 | (a2 << 12) | (a3 >> 12);
    let b1 = a & 0xFF00FF0000FF00FF;
    let b2 = a & 0x00FF00FF00000000;
    let b3 = a & 0x00000000FF00FF00;
    b1 | (b2 >> 24) | (b3 << 24)
}

fn print_game_state(board: u64) {
    for i in 0..16 {
        let value = board >> (4 * i) & 0xF;
        if value == 0 {
            print!("    .");
        } else {
            print!("{:5}", 1 << value);
        }
        if (i + 1) % 4 == 0 {
            println!();
        }
    }
}

impl Game<'_> {
    fn fill_transition_tables(&mut self) {
        for line_repr in 0..65536u64 {
            let mut new_line: Vec<u64> = Vec::new();
            let mut last_value: u64 = 0;
            for position in 0..4 {
                // println!("i: {}", i);
                let value = line_repr >> (4 * position) & 0xF;
                if value != 0 {
                    if value == last_value {
                        new_line.push(value + 1);
                        last_value = 0;
                    } else {
                        if last_value != 0 {
                            new_line.push(last_value);
                            last_value = 0;
                        }
                        last_value = value;
                    }
                }
                // println!("line: {:?}", line);
            }
            if last_value != 0 {
                new_line.push(last_value);
            }
            let mut new_line_repr: u64 = 0;
            for (i, value) in new_line.iter().enumerate() {
                new_line_repr |= (*value as u64) << (4 * i);
            }
            self.left_table[line_repr as usize] = new_line_repr;
            self.right_table[reverse_line_repr(line_repr.into()) as usize] =
                reverse_line_repr(new_line_repr);
            self.up_table[line_repr as usize] = unpack_col(new_line_repr);
            self.down_table[reverse_line_repr(line_repr.into()) as usize] =
                unpack_col(reverse_line_repr(new_line_repr));
        }
    }

    fn execute_move(&self, board: u64, m: Move) -> u64 {
        match m {
            Move::Left => {
                self.left_table[(board & ROW_MASK) as usize]
                    | (self.left_table[(board >> 16 & ROW_MASK) as usize] << 16)
                    | (self.left_table[(board >> 32 & ROW_MASK) as usize] << 32)
                    | (self.left_table[(board >> 48 & ROW_MASK) as usize] << 48)
            }
            Move::Right => {
                self.right_table[(board & ROW_MASK) as usize]
                    | (self.right_table[(board >> 16 & ROW_MASK) as usize] << 16)
                    | (self.right_table[(board >> 32 & ROW_MASK) as usize] << 32)
                    | (self.right_table[(board >> 48 & ROW_MASK) as usize] << 48)
            }
            Move::Up => {
                let tboard = transpose(board);
                self.up_table[(tboard & ROW_MASK) as usize]
                    | (self.up_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
                    | (self.up_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
                    | (self.up_table[(tboard >> 48 & ROW_MASK) as usize] << 12)
            }
            Move::Down => {
                let tboard = transpose(board);
                self.down_table[(tboard & ROW_MASK) as usize]
                    | (self.down_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
                    | (self.down_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
                    | (self.down_table[(tboard >> 48 & ROW_MASK) as usize] << 12)
            }
            _ => board,
        }
    }

    fn search(&self, game: u64, depth: u8) -> (u64, Move) {
        if depth == 0 {
            return (score_game(game), Move::Up);
        }
        let mut best_score: u64 = 0;
        let mut best_move: Move = Move::Left;
        for m in MOVES {
            let new_game = self.execute_move(game, m);
            best_move = m;
        }
        (best_score, best_move)
    }
}

fn score_game(game: u64) -> u64 {
    let mut score: u64 = 0;
    for i in 0..16 {
        let value = game >> (4 * i) & 0xF;
        if value != 0 {
            score += value;
        }
    }
    score
}

fn main() {
    let mut game: Game = Game {
        left_table: &mut [0; 65536],
        right_table: &mut [0; 65536],
        up_table: &mut [0; 65536],
        down_table: &mut [0; 65536],
    };
    let mut start: SystemTime = SystemTime::now();
    let mut count: u32 = 0;
    loop {
        let mut board: u64 = 0;
        game.fill_transition_tables();
        board = insert_random_tile(board);
        board = insert_random_tile(board);
        loop {
            count += 1;
            let (_, best_move) = game.search(board, 5);
            board = game.execute_move(board, best_move);
            board = insert_random_tile(board);
            if (count % 10000 == 0) {
                let elapsed = start.elapsed().unwrap();
                println!("Time: {:?}", elapsed);
                start = SystemTime::now();
            }
        }
    }
    // loop {
    //     // Wait for input from the user.
    //     let mut input = String::new();
    //     print!("Next move: ");
    //     io::stdout().flush().unwrap();
    //     io::stdin().read_line(&mut input).unwrap();
    //     // Update the game state.
    //     match input.trim() {
    //         "a" => board = game.execute_move(board, Move::Left),
    //         "d" => board = game.execute_move(board, Move::Right),
    //         "w" => board = game.execute_move(board, Move::Up),
    //         "s" => board = game.execute_move(board, Move::Down),
    //         _ => println!("Invalid move.")
    //     }
    //     // Insert a new tile.
    //     board= insert_random_tile(board);
    //     // Print the game state.
    //     print_game_state(board);
    // }
}
