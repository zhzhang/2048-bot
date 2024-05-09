// Board implementation taken from https://github.com/nneonneo/2048-ai and translated to Rust.
#[macro_use]
extern crate lazy_static;

use rand::{seq::IteratorRandom, thread_rng, Rng};

use std::iter::zip;
use std::thread;
use std::time::SystemTime;

#[derive(Copy, Clone)]
enum Move {
    Left,
    Right,
    Up,
    Down,
}

const NTUPLE_MASKS_FNS: [fn(u64) -> u64; 8] = [
    ntuple_mask_1,
    ntuple_mask_2,
    ntuple_mask_3,
    ntuple_mask_4,
    ntuple_mask_5,
    ntuple_mask_6,
    ntuple_mask_7,
    ntuple_mask_8,
];

struct Agent {
    score: u64,
    ntuple_values: [Vec<f32>; NTUPLE_MASKS_FNS.len()],
}

struct TransitionTable {
    left_table: Vec<(u64, u64)>,
    right_table: Vec<(u64, u64)>,
    up_table: Vec<(u64, u64)>,
    down_table: Vec<(u64, u64)>,
}

lazy_static! {
    static ref TTABLE: TransitionTable = generate_transition_tables();
}

const MOVES: [Move; 4] = [Move::Left, Move::Right, Move::Up, Move::Down];
const ROW_MASK: u64 = 0xFFFF;
const COL_MASK: u64 = 0x000F000F000F000F;

fn ntuple_mask_1(board: u64) -> u64 {
    ((board >> 36) & 0xFFF) | ((board >> 40) & 0xFFF000)
}

fn ntuple_mask_2(board: u64) -> u64 {
    ((board >> 8) & 0xF)
        | ((board >> 20) & 0xF0)
        | ((board >> 28) & 0xFF00)
        | ((board >> 36) & 0xFF0000)
}

fn ntuple_mask_3(board: u64) -> u64 {
    board >> 40
}

fn ntuple_mask_4(board: u64) -> u64 {
    ((board >> 20) & 0xF) | ((board >> 28) & 0xFFF0) | ((board >> 40) & 0xFF0000)
}

fn ntuple_mask_5(board: u64) -> u64 {
    ((board >> 20) & 0xFF) | ((board >> 32) & 0xF00) | ((board >> 40) & 0xFFF000)
}

fn ntuple_mask_6(board: u64) -> u64 {
    ((board >> 4) & 0xFF)
        | ((board >> 16) & 0xF00)
        | ((board >> 28) & 0xF000)
        | ((board >> 40) & 0xFF0000)
}

fn ntuple_mask_7(board: u64) -> u64 {
    ((board >> 8) & 0xF)
        | ((board >> 20) & 0xFF0)
        | ((board >> 28) & 0xF000)
        | ((board >> 40) & 0xFF0000)
}

fn ntuple_mask_8(board: u64) -> u64 {
    ((board >> 20) & 0xF)
        | ((board >> 32) & 0xF0)
        | ((board >> 36) & 0xF00)
        | ((board >> 40) & 0xFFF000)
}

fn get_ntuples_for_mask_fn(
    board: u64,
    mask_fn: fn(u64) -> u64,
    values: &Vec<f32>,
) -> ([u64; 8], f32) {
    let ntuples = [
        mask_fn(board),
        mask_fn(transpose(board)),
        mask_fn(fliph(board)),
        mask_fn(transpose(fliph(board))),
        mask_fn(flipv(board)),
        mask_fn(transpose(flipv(board))),
        mask_fn(fliph(flipv(board))),
        mask_fn(transpose(fliph(flipv(board)))),
    ];
    let ntuple_values = [
        values[ntuples[0] as usize],
        values[ntuples[1] as usize],
        values[ntuples[2] as usize],
        values[ntuples[3] as usize],
        values[ntuples[4] as usize],
        values[ntuples[5] as usize],
        values[ntuples[6] as usize],
        values[ntuples[7] as usize],
    ];
    (ntuples, ntuple_values.into_iter().sum())
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

fn insert_random_tile(board: u64) -> u64 {
    let open_tiles = get_open_tiles(board);
    if open_tiles.len() == 0 {
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
    ((line_repr >> 12) | ((line_repr >> 4) & 0xF0) | ((line_repr << 4) & 0xF00) | (line_repr << 12))
        & ROW_MASK
}

fn fliph(board: u64) -> u64 {
    ((board & 0x000F000F000F000F) << 12)
        | ((board & 0x00F000F000F000F0) << 4)
        | ((board & 0x0F000F000F000F00) >> 4)
        | ((board & 0xF000F000F000F000) >> 12)
}

fn flipv(board: u64) -> u64 {
    ((board & 0xFFFF000000000000) >> 48)
        | ((board & 0x0000FFFF00000000) >> 16)
        | ((board & 0x00000000FFFF0000) << 16)
        | ((board & 0x000000000000FFFF) << 48)
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

fn generate_transition_tables() -> TransitionTable {
    println!("Generating transition tables.");
    let mut left_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut right_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut up_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut down_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    for line_repr in 0..65536u64 {
        let mut new_line: Vec<u64> = Vec::new();
        let mut last_value: u64 = 0;
        let mut reward: u64 = 0;
        for position in 0..4 {
            // println!("i: {}", i);
            let value = line_repr >> (4 * position) & 0xF;
            if value != 0 {
                if value == last_value {
                    new_line.push(value + 1);
                    reward += 1 << (value + 1);
                    last_value = 0;
                } else {
                    if last_value != 0 {
                        new_line.push(last_value);
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
        left_table[line_repr as usize] = (new_line_repr, reward);
        right_table[reverse_line_repr(line_repr.into()) as usize] =
            (reverse_line_repr(new_line_repr), reward);
        up_table[line_repr as usize] = (unpack_col(new_line_repr), reward);
        down_table[reverse_line_repr(line_repr.into()) as usize] =
            (unpack_col(reverse_line_repr(new_line_repr)), reward);
    }
    TransitionTable {
        left_table,
        right_table,
        up_table,
        down_table,
    }
}

fn execute_move(board: u64, m: Move) -> (u64, u64) {
    match m {
        Move::Left => {
            //     | (self.left_table[(board >> 16 & ROW_MASK) as usize] << 16)
            //     | (self.left_table[(board >> 32 & ROW_MASK) as usize] << 32)
            //     | (self.left_table[(board >> 48 & ROW_MASK) as usize] << 48),
            // self.left_reward_table[(board & ROW_MASK) as usize]
            //     + self.left_reward_table[(board >> 16 & ROW_MASK) as usize]
            //     + self.left_reward_table[(board >> 32 & ROW_MASK) as usize]
            //     + self.left_reward_table[(board >> 48 & ROW_MASK) as usize],
            let (x1, r1) = TTABLE.left_table[(board & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.left_table[(board >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.left_table[(board >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.left_table[(board >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 16) | (x3 << 32) | (x4 << 48), r1 + r2 + r3 + r4)
        }
        Move::Right => {
            // self.right_table[(board & ROW_MASK) as usize]
            //     | (self.right_table[(board >> 16 & ROW_MASK) as usize] << 16)
            //     | (self.right_table[(board >> 32 & ROW_MASK) as usize] << 32)
            //     | (self.right_table[(board >> 48 & ROW_MASK) as usize] << 48),
            // self.right_reward_table[(board & ROW_MASK) as usize]
            //     + self.right_reward_table[(board >> 16 & ROW_MASK) as usize]
            //     + self.right_reward_table[(board >> 32 & ROW_MASK) as usize]
            //     + self.right_reward_table[(board >> 48 & ROW_MASK) as usize],
            let (x1, r1) = TTABLE.right_table[(board & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.right_table[(board >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.right_table[(board >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.right_table[(board >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 16) | (x3 << 32) | (x4 << 48), r1 + r2 + r3 + r4)
        }
        Move::Up => {
            let tboard = transpose(board);
            // (
            //     self.up_table[(tboard & ROW_MASK) as usize]
            //         | (self.up_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
            //         | (self.up_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
            //         | (self.up_table[(tboard >> 48 & ROW_MASK) as usize] << 12),
            //     self.up_reward_table[(tboard & ROW_MASK) as usize]
            //         + self.up_reward_table[(tboard >> 16 & ROW_MASK) as usize]
            //         + self.up_reward_table[(tboard >> 32 & ROW_MASK) as usize]
            //         + self.up_reward_table[(tboard >> 48 & ROW_MASK) as usize],
            // )
            let (x1, r1) = TTABLE.up_table[(tboard & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.up_table[(tboard >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.up_table[(tboard >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.up_table[(tboard >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 4) | (x3 << 8) | (x4 << 12), r1 + r2 + r3 + r4)
        }
        Move::Down => {
            let tboard = transpose(board);
            // (
            //     self.down_table[(tboard & ROW_MASK) as usize]
            //         | (self.down_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
            //         | (self.down_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
            //         | (self.down_table[(tboard >> 48 & ROW_MASK) as usize] << 12),
            //     self.down_reward_table[(tboard & ROW_MASK) as usize]
            //         + self.down_reward_table[(tboard >> 16 & ROW_MASK) as usize]
            //         + self.down_reward_table[(tboard >> 32 & ROW_MASK) as usize]
            //         + self.down_reward_table[(tboard >> 48 & ROW_MASK) as usize],
            // )
            let (x1, r1) = TTABLE.down_table[(tboard & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.down_table[(tboard >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.down_table[(tboard >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.down_table[(tboard >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 4) | (x3 << 8) | (x4 << 12), r1 + r2 + r3 + r4)
        }
    }
}

impl Agent {
    fn get_ntuples(&self, board: u64) -> ([[u64; 8]; 8], f32) {
        let mut output_ntuples = [[0; 8]; 8];
        let mut total_value: f32 = 0.0;
        for (i, (mask_fn, values)) in zip(NTUPLE_MASKS_FNS, &self.ntuple_values).enumerate() {
            let (ntuples, value) = get_ntuples_for_mask_fn(board, mask_fn, values);
            output_ntuples[i] = ntuples;
            total_value += value;
        }
        (output_ntuples, total_value)
    }

    fn do_best_move(&mut self, board: u64) -> (u64, u64) {
        let mut best_afterstate_value: f32 = f32::NEG_INFINITY;
        let mut best_board_afterstate: u64 = 0;
        let mut best_move_reward: u64 = 0;
        for m in MOVES {
            let (board_afterstate, reward) = execute_move(board, m);
            if board_afterstate == board {
                continue;
            }
            let (ntuples, value) = self.get_ntuples(board_afterstate);
            let next_state_value = value + reward as f32;
            if next_state_value > best_afterstate_value {
                best_afterstate_value = next_state_value;
                best_board_afterstate = board_afterstate;
                best_move_reward = reward;
            }
        }
        (best_board_afterstate, best_move_reward)
    }

    fn learn_epoch(&mut self) -> u64 {
        // Create a fresh board.
        let mut board: u64 = 0;
        self.score = 0;
        board = insert_random_tile(board);
        board = insert_random_tile(board);
        // Do the first greedy move to get to the afterstate.
        let (mut board_afterstate, mut move_reward) = self.do_best_move(board);
        let (mut afterstate_ntuples, mut afterstate_value) = self.get_ntuples(board_afterstate);
        self.score += move_reward;
        loop {
            // Transition to next state.
            board = insert_random_tile(board);
            // Greedily pick the best next move.
            (board_afterstate, move_reward) = self.do_best_move(board);
            self.score += move_reward;
            // print_game_state(board);
            // println!("----------");
            // print_game_state(board_afterstate);
            // println!("==========");
            // If the game is over, afterstate values are ???
            if board_afterstate == 0 {
                return board;
            }
            // Get the afterstate ntuples and value.
            let (new_afterstate_ntuples, new_afterstate_value) = self.get_ntuples(board_afterstate);
            // Update afterstate values of the previous afterstate ntuple tables.
            let delta = ((move_reward as f32) + new_afterstate_value) - afterstate_value;
            for (i, ntuple_set) in afterstate_ntuples.iter().enumerate() {
                for ntuple in ntuple_set.iter() {
                    self.ntuple_values[i][*ntuple as usize] += 0.1 * delta / 64.0;
                }
            }
            // Set the current afterstate ntuples to the ntuples of the new best move.
            afterstate_ntuples = new_afterstate_ntuples;
            afterstate_value = new_afterstate_value;
            board = board_afterstate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_BOARD: u64 = 0x0123456789ABCDEF;

    fn board_from_row_values(values: [u8; 16]) -> u64 {
        let mut board: u64 = 0;
        for i in 0..16 {
            board |= (values[i] as u64) << (4 * (15 - i));
        }
        board
    }

    fn ntuple_mask_from_values(values: [u8; 6]) -> u64 {
        let mut mask: u64 = 0;
        for i in 0..6 {
            mask |= (values[i] as u64) << (4 * (5 - i));
        }
        mask
    }

    fn print_ntuple(mask: u64) {
        for i in 0..6 {
            print!("{:5}", ((mask >> (4 * (5 - i))) & 0xF));
        }
        println!();
    }

    #[test]
    fn test_ntuple_masks() {
        let test_values: [(fn(u64) -> u64, [u8; 6]); 8] = [
            (ntuple_mask_1, [0, 1, 2, 4, 5, 6]),
            (ntuple_mask_2, [1, 2, 5, 6, 9, 13]),
            (ntuple_mask_3, [0, 1, 2, 3, 4, 5]),
            (ntuple_mask_4, [0, 1, 5, 6, 7, 10]),
            (ntuple_mask_5, [0, 1, 2, 5, 9, 10]),
            (ntuple_mask_6, [0, 1, 5, 9, 13, 14]),
            (ntuple_mask_7, [0, 1, 5, 8, 9, 13]),
            (ntuple_mask_8, [0, 1, 2, 4, 6, 10]),
        ];
        for (i, (mask_fn, result)) in test_values.into_iter().enumerate() {
            let ntuple = mask_fn(TEST_BOARD);
            let true_ntuple = ntuple_mask_from_values(result);
            if ntuple != true_ntuple {
                println!("Test failed for mask: {}", i + 1);
                print_ntuple(ntuple);
                print_ntuple(true_ntuple);
            }
            assert_eq!(ntuple, true_ntuple);
        }
    }

    #[test]
    fn test_fliph_board() {
        let fliph_board = fliph(TEST_BOARD);
        let true_fliph_board: u64 =
            board_from_row_values([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]);
        assert_eq!(fliph_board, true_fliph_board);
    }

    #[test]
    fn test_flipv_board() {
        let flipv_board = flipv(TEST_BOARD);
        let true_flipv_board: u64 =
            board_from_row_values([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]);
        assert_eq!(flipv_board, true_flipv_board);
    }

    #[test]
    fn test_transpose_board() {
        let transposed_board = transpose(TEST_BOARD);
        let true_transposed_board: u64 =
            board_from_row_values([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]);
        assert_eq!(transposed_board, true_transposed_board);
    }

    #[test]
    fn test_get_ntuples() {
        let ntuple_values_1 = vec![0.0; 16777216];
        let (ntuples1, values1) =
            get_ntuples_for_mask_fn(TEST_BOARD, ntuple_mask_1, &ntuple_values_1);
        let true_ntuples1 = [
            ntuple_mask_from_values([0, 1, 2, 4, 5, 6]),
            ntuple_mask_from_values([0, 4, 8, 1, 5, 9]),
            ntuple_mask_from_values([3, 2, 1, 7, 6, 5]),
            ntuple_mask_from_values([3, 7, 11, 2, 6, 10]),
            ntuple_mask_from_values([12, 13, 14, 8, 9, 10]),
            ntuple_mask_from_values([12, 8, 4, 13, 9, 5]),
            ntuple_mask_from_values([15, 14, 13, 11, 10, 9]),
            ntuple_mask_from_values([15, 11, 7, 14, 10, 6]),
        ];
        for i in 0..8 {
            if ntuples1[i] != true_ntuples1[i] {
                println!("Test failed for ntuple: {}", i + 1);
                print_ntuple(ntuples1[i]);
                print_ntuple(true_ntuples1[i]);
            }
            assert_eq!(ntuples1[i], true_ntuples1[i]);
        }
        assert_eq!(values1, 0.0)
    }

    #[test]
    fn test_reward_tables() {
        for (row, (_, reward)) in TTABLE.left_table.iter().enumerate() {
            let mut actual_reward: u64 = 0;
            let mut last_value: u16 = 0;
            for i in 0..4 {
                let value = (row as u16) >> (4 * i) & 0xF;
                if value != 0 {
                    if value == last_value {
                        actual_reward += 1 << (value + 1);
                        last_value = 0;
                    } else {
                        last_value = value;
                    }
                }
            }
            assert_eq!(*reward, actual_reward);
        }
    }
}

fn main() {
    println!("Starting");
    let mut agent = Agent {
        score: 0,
        ntuple_values: [
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
            vec![0.0; 16777216],
        ],
    };
    let mut start: SystemTime = SystemTime::now();
    let mut count: u32 = 0;
    let mut best_score = 0;
    loop {
        let final_board = agent.learn_epoch();
        count += 1;
        if count % 100 == 0 {
            let elapsed = start.elapsed().unwrap();
            print_game_state(final_board);
            println!(
                "Time: {:?} Games Played: {:?} Best Score: {:?}",
                elapsed, count, best_score
            );
            start = SystemTime::now();
        }
        best_score = best_score.max(agent.score);
        agent.score = 0;
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
