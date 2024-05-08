// Board implementation taken from https://github.com/nneonneo/2048-ai and translated to Rust.
use rand::{seq::IteratorRandom, thread_rng, Rng};

use std::thread;
use std::time::SystemTime;

#[derive(Copy, Clone)]
enum Move {
    Left,
    Right,
    Up,
    Down,
}

struct Game<'a> {
    score: u64,
    left_table: &'a mut [u64; 65536],
    right_table: &'a mut [u64; 65536],
    up_table: &'a mut [u64; 65536],
    down_table: &'a mut [u64; 65536],
    left_reward_table: &'a mut [u64; 65536],
    right_reward_table: &'a mut [u64; 65536],
    up_reward_table: &'a mut [u64; 65536],
    down_reward_table: &'a mut [u64; 65536],
    ntuple_values_1: Box<[f32]>,
    ntuple_values_2: Box<[f32]>,
    ntuple_values_3: Box<[f32]>,
    ntuple_values_4: Box<[f32]>,
    ntuple_values_5: Box<[f32]>,
    ntuple_values_6: Box<[f32]>,
    ntuple_values_7: Box<[f32]>,
    ntuple_values_8: Box<[f32]>,
}

static MOVES: [Move; 4] = [Move::Left, Move::Right, Move::Up, Move::Down];
static ROW_MASK: u64 = 0xFFFF;
static COL_MASK: u64 = 0x000F000F000F000F;

// static NTUPLE_MASKS: [u64; 4] = [
//     0xFFF0FFF000000000,
//     0x0FF00FF00F000F00,
//     0xFFFFFF0000000000,
//     0xFF000FFF00F00000,
// ];

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

fn get_ntuples<F>(board: u64, mask_fn: F, values: &Box<[f32]>) -> ([u64; 8], f32)
where
    F: Fn(u64) -> u64,
{
    let ntuples = [
        mask_fn(board),
        mask_fn(transpose(board)),
        mask_fn(fliph(board)),
        mask_fn(transpose(fliph(board))),
        mask_fn(flipv(board)),
        mask_fn(transpose(flipv(board))),
        mask_fn(fliph(flipv(board))),            // 3+
        mask_fn(transpose(fliph(flipv(board)))), // 3-
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

impl Game<'_> {
    fn fill_transition_tables(&mut self) {
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
            self.left_table[line_repr as usize] = new_line_repr;
            self.left_reward_table[line_repr as usize] = reward;
            self.right_table[reverse_line_repr(line_repr.into()) as usize] =
                reverse_line_repr(new_line_repr);
            self.right_reward_table[reverse_line_repr(line_repr.into()) as usize] = reward;
            self.up_table[line_repr as usize] = unpack_col(new_line_repr);
            self.up_reward_table[line_repr as usize] = reward;
            self.down_table[reverse_line_repr(line_repr.into()) as usize] =
                unpack_col(reverse_line_repr(new_line_repr));
            self.down_reward_table[reverse_line_repr(line_repr.into()) as usize] = reward;
        }
    }

    fn execute_move(&self, board: u64, m: Move) -> (u64, u64) {
        match m {
            Move::Left => (
                self.left_table[(board & ROW_MASK) as usize]
                    | (self.left_table[(board >> 16 & ROW_MASK) as usize] << 16)
                    | (self.left_table[(board >> 32 & ROW_MASK) as usize] << 32)
                    | (self.left_table[(board >> 48 & ROW_MASK) as usize] << 48),
                self.left_reward_table[(board & ROW_MASK) as usize]
                    + self.left_reward_table[(board >> 16 & ROW_MASK) as usize]
                    + self.left_reward_table[(board >> 32 & ROW_MASK) as usize]
                    + self.left_reward_table[(board >> 48 & ROW_MASK) as usize],
            ),
            Move::Right => (
                self.right_table[(board & ROW_MASK) as usize]
                    | (self.right_table[(board >> 16 & ROW_MASK) as usize] << 16)
                    | (self.right_table[(board >> 32 & ROW_MASK) as usize] << 32)
                    | (self.right_table[(board >> 48 & ROW_MASK) as usize] << 48),
                self.right_reward_table[(board & ROW_MASK) as usize]
                    + self.right_reward_table[(board >> 16 & ROW_MASK) as usize]
                    + self.right_reward_table[(board >> 32 & ROW_MASK) as usize]
                    + self.right_reward_table[(board >> 48 & ROW_MASK) as usize],
            ),
            Move::Up => {
                let tboard = transpose(board);
                (
                    self.up_table[(tboard & ROW_MASK) as usize]
                        | (self.up_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
                        | (self.up_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
                        | (self.up_table[(tboard >> 48 & ROW_MASK) as usize] << 12),
                    self.up_reward_table[(tboard & ROW_MASK) as usize]
                        + self.up_reward_table[(tboard >> 16 & ROW_MASK) as usize]
                        + self.up_reward_table[(tboard >> 32 & ROW_MASK) as usize]
                        + self.up_reward_table[(tboard >> 48 & ROW_MASK) as usize],
                )
            }
            Move::Down => {
                let tboard = transpose(board);
                (
                    self.down_table[(tboard & ROW_MASK) as usize]
                        | (self.down_table[(tboard >> 16 & ROW_MASK) as usize] << 4)
                        | (self.down_table[(tboard >> 32 & ROW_MASK) as usize] << 8)
                        | (self.down_table[(tboard >> 48 & ROW_MASK) as usize] << 12),
                    self.down_reward_table[(tboard & ROW_MASK) as usize]
                        + self.down_reward_table[(tboard >> 16 & ROW_MASK) as usize]
                        + self.down_reward_table[(tboard >> 32 & ROW_MASK) as usize]
                        + self.down_reward_table[(tboard >> 48 & ROW_MASK) as usize],
                )
            }
        }
    }

    fn search(&mut self, board: u64) -> (u64, bool) {
        let mut best_value: f32 = f32::NEG_INFINITY;
        let mut best_board: u64 = 0;
        let (ntuples1, values1) = get_ntuples(board, ntuple_mask_1, &self.ntuple_values_1);
        let (ntuples2, values2) = get_ntuples(board, ntuple_mask_2, &self.ntuple_values_2);
        let (ntuples3, values3) = get_ntuples(board, ntuple_mask_3, &self.ntuple_values_3);
        let (ntuples4, values4) = get_ntuples(board, ntuple_mask_4, &self.ntuple_values_4);
        let (ntuples5, values5) = get_ntuples(board, ntuple_mask_5, &self.ntuple_values_5);
        let (ntuples6, values6) = get_ntuples(board, ntuple_mask_6, &self.ntuple_values_6);
        let (ntuples7, values7) = get_ntuples(board, ntuple_mask_7, &self.ntuple_values_7);
        let (ntuples8, values8) = get_ntuples(board, ntuple_mask_8, &self.ntuple_values_8);
        let current_value =
            values1 + values2 + values3 + values4 + values5 + values6 + values7 + values8;
        let mut best_move_reward: u64 = 0;
        for m in MOVES {
            let (mut new_board, reward) = self.execute_move(board, m);
            if new_board == board {
                continue;
            }
            new_board = insert_random_tile(new_board);
            let (_, values1) = get_ntuples(new_board, ntuple_mask_1, &self.ntuple_values_1);
            let (_, values2) = get_ntuples(new_board, ntuple_mask_2, &self.ntuple_values_2);
            let (_, values3) = get_ntuples(new_board, ntuple_mask_3, &self.ntuple_values_3);
            let (_, values4) = get_ntuples(new_board, ntuple_mask_4, &self.ntuple_values_4);
            let (_, values5) = get_ntuples(new_board, ntuple_mask_5, &self.ntuple_values_5);
            let (_, values6) = get_ntuples(new_board, ntuple_mask_6, &self.ntuple_values_6);
            let (_, values7) = get_ntuples(new_board, ntuple_mask_7, &self.ntuple_values_7);
            let (_, values8) = get_ntuples(new_board, ntuple_mask_8, &self.ntuple_values_8);
            let next_state_value = values1
                + values2
                + values3
                + values4
                + values5
                + values6
                + values7
                + values8
                + reward as f32;
            if next_state_value > best_value {
                best_value = next_state_value;
                best_board = new_board;
                best_move_reward = reward;
            }
        }
        if best_board == 0 {
            return (board, true);
        }
        self.score += best_move_reward;
        let delta = best_value - current_value;
        for ntuple in ntuples1.iter() {
            self.ntuple_values_1[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples2.iter() {
            self.ntuple_values_2[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples3.iter() {
            self.ntuple_values_3[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples4.iter() {
            self.ntuple_values_4[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples5.iter() {
            self.ntuple_values_5[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples6.iter() {
            self.ntuple_values_6[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples7.iter() {
            self.ntuple_values_7[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        for ntuple in ntuples8.iter() {
            self.ntuple_values_8[*ntuple as usize] += 0.1 * delta / 64.0;
        }
        (best_board, false)
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
        let ntuple_values_1 = vec![0.0; 16777216].into_boxed_slice();
        let (ntuples1, values1) = get_ntuples(TEST_BOARD, ntuple_mask_1, &ntuple_values_1);
        let true_ntuples1 = [
            ntuple_mask_from_values([0, 1, 2, 4, 5, 6]),
            ntuple_mask_from_values([0, 4, 8, 1, 5, 9]),
            ntuple_mask_from_values([3, 2, 1, 7, 6, 5]),
            ntuple_mask_from_values([3, 7, 11, 2, 6, 10]),
        ];
        for i in 0..4 {
            if ntuples1[i] != true_ntuples1[i] {
                println!("Test failed for ntuple: {}", i + 1);
                print_ntuple(ntuples1[i]);
                print_ntuple(true_ntuples1[i]);
            }
            assert_eq!(ntuples1[i], true_ntuples1[i]);
        }
        assert_eq!(values1, 0.0)
    }
}

fn main() {
    let mut game: Game = Game {
        score: 0,
        left_table: &mut [0; 65536],
        right_table: &mut [0; 65536],
        up_table: &mut [0; 65536],
        down_table: &mut [0; 65536],
        left_reward_table: &mut [0; 65536],
        right_reward_table: &mut [0; 65536],
        up_reward_table: &mut [0; 65536],
        down_reward_table: &mut [0; 65536],
        ntuple_values_1: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_2: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_3: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_4: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_5: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_6: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_7: vec![0.0; 16777216].into_boxed_slice(),
        ntuple_values_8: vec![0.0; 16777216].into_boxed_slice(),
    };
    let mut start: SystemTime = SystemTime::now();
    let mut count: u32 = 0;
    let mut best_score = 0;
    let mut board: u64 = 0;
    game.fill_transition_tables();
    board = insert_random_tile(board);
    board = insert_random_tile(board);
    loop {
        let (new_board, game_over) = game.search(board);
        board = new_board;
        if game_over {
            count += 1;
            if count % 10000 == 0 {
                print_game_state(board);
                let elapsed = start.elapsed().unwrap();
                println!(
                    "Time: {:?} Games Played: {:?} Best Score: {:?}",
                    elapsed, count, best_score
                );
                start = SystemTime::now();
            }
            best_score = best_score.max(game.score);
            board = 0;
            board = insert_random_tile(board);
            board = insert_random_tile(board);
            game.score = 0;
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
