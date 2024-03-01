use rand::Rng;

// Game representation is in row major order.

static VALUE_MASKS: [u64; 16] = [
    0xf, 0xf0, 0xf00, 0xf000,
    0xf0000, 0xf00000, 0xf000000, 0xf0000000,
    0xf00000000, 0xf000000000, 0xf0000000000, 0xf00000000000,
    0xf000000000000, 0xf0000000000000, 0xf00000000000000, 0xf000000000000000
];

static COMPLEMENT_MASKS: [u64; 16] = [
    0xfffffffffffffff0, 0xffffffffffffff0f, 0xfffffffffffff0ff, 0xffffffffffff0fff,
    0xfffffffffff0ffff, 0xffffffffff0fffff, 0xfffffffff0ffffff, 0xffffffff0fffffff,
    0xfffffff0ffffffff, 0xffffff0fffffffff, 0xfffff0ffffffffff, 0xffff0fffffffffff,
    0xfff0ffffffffffff, 0xff0fffffffffffff, 0xf0ffffffffffffff, 0x0fffffffffffffff
];

fn insert_new_tile(game: u64, value: u8, x: u8, y: u8) -> u64 {
    println!("{}", value);
    game
}

fn insert_random_tile(game: u64) -> u64 {
    let open_tiles = get_open_tiles(game);
    let (x, y) = open_tiles[0];
    let value = random_tile_value();
    insert_new_tile(game, value, x, y)
}

fn get_open_tiles(game: u64) -> Vec<(u8, u8)> {
    let mut open_tiles: Vec<(u8, u8)> = Vec::new();
    for i in 0..16 {
        let value = (game >> i) & 0xf;
        if value == 0 {
            open_tiles.push(i);
        }
    }
    open_tiles
}

fn random_tile_value() -> u8 {
    let mut rng = rand::thread_rng();
    let n1: u8 = rng.gen();
    if n1 < 10 {
        1
    } else {
        2
    }
}

fn move_left(game: u64) -> u64 {
    game
}

fn print_game_state(game: u64) {
    for i in 0..4 {
        for j in 0..4 {
            let value = (game >> (4 * (4 * i + j))) & 0xf;
            if value == 0 {
                print!("    .");
            } else {
                print!("{:5}", 1 << value);
            }
        }
        println!();
    }
}

fn main() {
    let game: u64 = 0;
    let game = insert_random_tile(game);
    let game = insert_random_tile(game);
    print_game_state(game);
}