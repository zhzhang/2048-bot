// Board implementation taken from https://github.com/nneonneo/2048-ai and translated to Rust.
#[macro_use]
extern crate lazy_static;

use rand::{seq::IteratorRandom, thread_rng, Rng};

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::SystemTime;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone)]
enum Move {
    Left,
    Right,
    Up,
    Down,
}

impl Move {
    fn as_str(&self) -> &'static str {
        match self {
            Move::Left => "left",
            Move::Right => "right",
            Move::Up => "up",
            Move::Down => "down",
        }
    }
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
    let mut left_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut right_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut up_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    let mut down_table: Vec<(u64, u64)> = vec![(0, 0); 65536];
    for line_repr in 0..65536u64 {
        let mut new_line: Vec<u64> = Vec::new();
        let mut last_value: u64 = 0;
        let mut reward: u64 = 0;
        for position in 0..4 {
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
            let (x1, r1) = TTABLE.left_table[(board & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.left_table[(board >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.left_table[(board >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.left_table[(board >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 16) | (x3 << 32) | (x4 << 48), r1 + r2 + r3 + r4)
        }
        Move::Right => {
            let (x1, r1) = TTABLE.right_table[(board & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.right_table[(board >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.right_table[(board >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.right_table[(board >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 16) | (x3 << 32) | (x4 << 48), r1 + r2 + r3 + r4)
        }
        Move::Up => {
            let tboard = transpose(board);
            let (x1, r1) = TTABLE.up_table[(tboard & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.up_table[(tboard >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.up_table[(tboard >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.up_table[(tboard >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 4) | (x3 << 8) | (x4 << 12), r1 + r2 + r3 + r4)
        }
        Move::Down => {
            let tboard = transpose(board);
            let (x1, r1) = TTABLE.down_table[(tboard & ROW_MASK) as usize];
            let (x2, r2) = TTABLE.down_table[(tboard >> 16 & ROW_MASK) as usize];
            let (x3, r3) = TTABLE.down_table[(tboard >> 32 & ROW_MASK) as usize];
            let (x4, r4) = TTABLE.down_table[(tboard >> 48 & ROW_MASK) as usize];
            (x1 | (x2 << 4) | (x3 << 8) | (x4 << 12), r1 + r2 + r3 + r4)
        }
    }
}

//
// Definition of RL Agent.
//

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

struct Agent {
    // ntuple_values: [Vec<f32>; NTUPLE_MASKS_FNS.len()],
    // Represent ntuples as a raw pointer to a 1D array to allow lock-free access.
    ntuple_raw_ptr: *mut f32,
}

unsafe impl Send for Agent {}
const NTUPLE_LUT_SIZE: usize = 16777216;

impl Agent {
    fn get_ntuple_value(&self, mask_fn_index: usize, ntuple_index: usize) -> f32 {
        unsafe {
            *self
                .ntuple_raw_ptr
                .offset((mask_fn_index * NTUPLE_LUT_SIZE + ntuple_index) as isize)
                as f32
        }
    }

    fn update_ntuple_value(&mut self, mask_fn_index: usize, ntuple_index: usize, update: f32) {
        unsafe {
            *self
                .ntuple_raw_ptr
                .offset((mask_fn_index * NTUPLE_LUT_SIZE + ntuple_index) as isize) += update;
        }
    }

    fn get_ntuples_for_mask_fn(
        &self,
        board: u64,
        mask_fn: fn(u64) -> u64,
        mask_fn_index: usize,
    ) -> ([u64; 8], f32) {
        let ntuples: [u64; 8] = [
            mask_fn(board),
            mask_fn(transpose(board)),
            mask_fn(fliph(board)),
            mask_fn(transpose(fliph(board))),
            mask_fn(flipv(board)),
            mask_fn(transpose(flipv(board))),
            mask_fn(fliph(flipv(board))),
            mask_fn(transpose(fliph(flipv(board)))),
        ];
        let ntuple_values: [f32; 8] = [
            self.get_ntuple_value(mask_fn_index, ntuples[0] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[1] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[2] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[3] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[4] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[5] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[6] as usize),
            self.get_ntuple_value(mask_fn_index, ntuples[7] as usize),
        ];
        (ntuples, ntuple_values.into_iter().sum())
    }

    fn get_ntuples(&self, board: u64) -> ([[u64; 8]; NTUPLE_MASKS_FNS.len()], f32) {
        let mut output_ntuples: [[u64; 8]; NTUPLE_MASKS_FNS.len()] =
            [[0; 8]; NTUPLE_MASKS_FNS.len()];
        let mut total_value: f32 = 0.0;
        for (i, mask_fn) in NTUPLE_MASKS_FNS.iter().enumerate() {
            let (ntuples, value) = self.get_ntuples_for_mask_fn(board, *mask_fn, i);
            output_ntuples[i] = ntuples;
            total_value += value;
        }
        (output_ntuples, total_value)
    }

    fn do_best_move(&mut self, board: u64) -> (u64, u64, [[u64; 8]; NTUPLE_MASKS_FNS.len()], f32) {
        let mut best_afterstate_value: f32 = f32::NEG_INFINITY;
        let mut best_board_afterstate: u64 = 0;
        let mut best_move_reward: u64 = 0;
        let mut best_move_ntuples: [[u64; 8]; NTUPLE_MASKS_FNS.len()] =
            [[0; 8]; NTUPLE_MASKS_FNS.len()];
        for m in MOVES {
            let (board_afterstate, reward) = execute_move(board, m);
            if board_afterstate == board {
                continue;
            }
            let (ntuples, value) = self.get_ntuples(board_afterstate);
            if value + reward as f32 > best_afterstate_value {
                best_afterstate_value = value;
                best_board_afterstate = board_afterstate;
                best_move_reward = reward;
                best_move_ntuples = ntuples;
            }
        }
        (
            best_board_afterstate,
            best_move_reward,
            best_move_ntuples,
            best_afterstate_value,
        )
    }

    fn learn_epoch(&mut self) -> (u64, u64) {
        // Create a fresh board.
        let mut board: u64 = 0;
        let mut score = 0;
        board = insert_random_tile(board);
        board = insert_random_tile(board);
        // Do the first greedy move to get to the afterstate.
        let (mut board_afterstate, move_reward, mut afterstate_ntuples, mut afterstate_value) =
            self.do_best_move(board);
        score += move_reward;
        loop {
            // Transition to next state.
            board = insert_random_tile(board_afterstate);
            // Greedily pick the best next move.
            let (
                new_board_afterstate,
                new_move_reward,
                new_afterstate_ntuples,
                new_afterstate_value,
            ) = self.do_best_move(board);
            // If no move was possible, the game is over.
            if new_board_afterstate == 0 {
                // If the game is over, afterstate value is the score.
                let delta = score as f32 - afterstate_value;
                // println!("{} {}", score, delta);
                for (i, ntuple_set) in afterstate_ntuples.iter().enumerate() {
                    for ntuple in ntuple_set.iter() {
                        self.update_ntuple_value(i, *ntuple as usize, 0.1 * delta / 64.0);
                    }
                }
                return (board, score);
            }
            score += new_move_reward;
            // Update afterstate values of the previous afterstate ntuple tables.
            let delta = ((new_move_reward as f32) + new_afterstate_value) - afterstate_value;
            for (i, ntuple_set) in afterstate_ntuples.iter().enumerate() {
                for ntuple in ntuple_set.iter() {
                    self.update_ntuple_value(i, *ntuple as usize, 0.1 * delta / 64.0);
                }
            }
            // Set the current afterstate ntuples to the ntuples of the new best move.
            afterstate_ntuples = new_afterstate_ntuples;
            afterstate_value = new_afterstate_value;
            board_afterstate = new_board_afterstate;
        }
    }
}

const V_INIT: f32 = 370000.0;
const NTHREADS: usize = 32;
const EPOCH_SIZE: usize = 100000;
const CHECKPOINT_FILE: &str = "ntuple_values.bin";

struct AgentOutput {
    count: u64,
    best_score: u64,
    total_score: u64,
    best_board: u64,
}

fn load_ntuple_values() -> Option<Vec<f32>> {
    match File::open(CHECKPOINT_FILE) {
        Ok(mut file) => {
            let mut buffer = Vec::new();
            if file.read_to_end(&mut buffer).is_ok() {
                match bincode::deserialize::<Vec<f32>>(&buffer) {
                    Ok(values) => {
                        // Verify the size is correct
                        if values.len() == NTUPLE_LUT_SIZE * NTUPLE_MASKS_FNS.len() {
                            println!("Loaded ntuple values from {}", CHECKPOINT_FILE);
                            return Some(values);
                        } else {
                            println!(
                                "Warning: Checkpoint file has incorrect size. Starting fresh."
                            );
                        }
                    }
                    Err(e) => {
                        println!("Warning: Failed to deserialize checkpoint: {}. Starting fresh.", e);
                    }
                }
            }
        }
        Err(_) => {
            println!("No checkpoint file found. Starting fresh.");
        }
    }
    None
}

fn save_ntuple_values(ntuple_values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = bincode::serialize(ntuple_values)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(CHECKPOINT_FILE)?;
    file.write_all(&serialized)?;
    file.sync_all()?;
    Ok(())
}

// Web server types and functions

#[derive(Deserialize)]
struct BoardRequest {
    board: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct MoveResponse {
    best_move: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

// Convert 2D array representation to internal board representation
// The 2D array is expected to be in the form [[row0], [row1], [row2], [row3]]
// where each value is the actual tile value (0, 2, 4, 8, 16, ...)
fn board_from_2d_array(arr: Vec<Vec<u32>>) -> Result<u64, String> {
    if arr.len() != 4 {
        return Err("Board must have 4 rows".to_string());
    }
    
    let mut board: u64 = 0;
    for (row_idx, row) in arr.iter().enumerate() {
        if row.len() != 4 {
            return Err("Each row must have 4 columns".to_string());
        }
        
        for (col_idx, &value) in row.iter().enumerate() {
            let tile_idx = row_idx * 4 + col_idx;
            let encoded_value = if value == 0 {
                0u64
            } else if value.is_power_of_two() && value >= 2 {
                // Convert tile value to log2 representation
                // 2 -> 1, 4 -> 2, 8 -> 3, 16 -> 4, etc.
                value.trailing_zeros() as u64
            } else {
                return Err(format!("Invalid tile value: {}. Must be 0 or a power of 2 >= 2", value));
            };
            
            board |= encoded_value << (4 * tile_idx);
        }
    }
    
    Ok(board)
}

// Find the best move for a given board
fn get_best_move_for_board(board: u64, agent: &Agent) -> Option<Move> {
    let mut best_value: f32 = f32::NEG_INFINITY;
    let mut best_move: Option<Move> = None;
    
    for m in MOVES {
        let (board_afterstate, reward) = execute_move(board, m);
        if board_afterstate == board {
            continue;
        }
        let (_, value) = agent.get_ntuples(board_afterstate);
        let total_value = value + reward as f32;
        if total_value > best_value {
            best_value = total_value;
            best_move = Some(m);
        }
    }
    
    best_move
}

// Shared state for web server
struct AppState {
    ntuple_values: Arc<Vec<f32>>,
}

async fn get_best_move(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<BoardRequest>,
) -> Result<Json<MoveResponse>, ErrorResponse> {
    // Convert 2D array to board representation
    let board = board_from_2d_array(payload.board)
        .map_err(|e| ErrorResponse { error: e })?;
    
    // Create agent with loaded values
    let agent = Agent {
        ntuple_raw_ptr: state.ntuple_values.as_ptr() as *mut f32,
    };
    
    // Get best move
    let best_move = get_best_move_for_board(board, &agent)
        .ok_or_else(|| ErrorResponse {
            error: "No valid moves available".to_string(),
        })?;
    
    Ok(Json(MoveResponse {
        best_move: best_move.as_str().to_string(),
    }))
}

fn run_training() {
    println!("Starting training mode...");
    let mut ntuple_values = load_ntuple_values()
        .unwrap_or_else(|| vec![V_INIT; NTUPLE_LUT_SIZE * NTUPLE_MASKS_FNS.len()]);
    let agent_output = Arc::new(Mutex::new(AgentOutput {
        count: 0,
        best_score: 0,
        total_score: 0,
        best_board: 0,
    }));
    for _ in 0..NTHREADS {
        let agent_output_clone = Arc::clone(&agent_output);
        let mut agent = Agent {
            ntuple_raw_ptr: ntuple_values.as_mut_ptr(),
        };
        thread::spawn(move || loop {
            let (final_board, score) = agent.learn_epoch();
            let mut agent_output = agent_output_clone.lock().unwrap();
            if score > agent_output.best_score {
                agent_output.best_score = score;
                agent_output.best_board = final_board;
            }
            agent_output.total_score += score;
            agent_output.count += 1;
            drop(agent_output);
        });
    }
    let mut last_count = 0;
    let mut start = SystemTime::now();
    loop {
        let mut agent_output = agent_output.lock().unwrap();
        if agent_output.count >= last_count + EPOCH_SIZE as u64 {
            print_game_state(agent_output.best_board);
            println!(
                "Time: {:?} Games Played: {:?} Best Score: {:?} Average Score: {:?}",
                start.elapsed().unwrap(),
                agent_output.count,
                agent_output.best_score,
                agent_output.total_score / (agent_output.count - last_count) as u64
            );
            last_count = agent_output.count;
            agent_output.total_score = 0;
            start = SystemTime::now();
            
            // Save checkpoint after each epoch
            drop(agent_output);
            if let Err(e) = save_ntuple_values(&ntuple_values) {
                eprintln!("Warning: Failed to save checkpoint: {}", e);
            } else {
                println!("Checkpoint saved to {}", CHECKPOINT_FILE);
            }
        } else {
            drop(agent_output);
        }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // Check if server mode is requested
    let server_mode = args.len() > 1 && args[1] == "--server";
    
    if server_mode {
        println!("Starting web server mode on port 2048...");
        
        // Load trained model
        let ntuple_values = load_ntuple_values()
            .unwrap_or_else(|| {
                println!("Warning: No trained model found. Using initial values.");
                vec![V_INIT; NTUPLE_LUT_SIZE * NTUPLE_MASKS_FNS.len()]
            });
        
        let state = Arc::new(AppState {
            ntuple_values: Arc::new(ntuple_values),
        });
        
        // Build router
        let app = Router::new()
            .route("/best-move", post(get_best_move))
            .with_state(state);
        
        // Run server
        let listener = tokio::net::TcpListener::bind("0.0.0.0:2048")
            .await
            .expect("Failed to bind to port 2048");
        
        println!("Server listening on http://0.0.0.0:2048");
        println!("POST to /best-move with JSON body: {{\"board\": [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]]}}");
        
        axum::serve(listener, app)
            .await
            .expect("Server failed");
    } else {
        run_training();
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
        let ntuple_values = vec![V_INIT; NTUPLE_LUT_SIZE * NTUPLE_MASKS_FNS.len()];
        let agent = Agent {
            ntuple_raw_ptr: ntuple_values.as_ptr() as *mut f32,
        };
        let (ntuples1, values1) = agent.get_ntuples_for_mask_fn(TEST_BOARD, ntuple_mask_1, 1);
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
        assert_eq!(values1, 8.0 * V_INIT);
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
