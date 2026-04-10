import torch
import time
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
MOVE_LUT_SIZE = 65536 // 2
NUM_NTUPLE_MASKS = 8
SYMMETRIES_PER_MASK = 8
FEATURES_PER_BOARD = NUM_NTUPLE_MASKS * SYMMETRIES_PER_MASK
NUM_NTUPLE_VALUES = 16777216


@triton.jit
def insert_random_tile(
    boards,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    shifts = 60 - tl.arange(0, 16) * 4
    x = (boards >> shifts) & 0xF
    zeros = (x == 0).to(tl.uint64)
    num_zeros = zeros.sum(1)
    cs = zeros.cumsum(1)
    r1, r2, _, _ = tl.randint4x(seed, offsets)
    r_idx = ((r1 % num_zeros) + 1).expand_dims(1)
    vals = tl.full((BLOCK_SIZE,), 1, dtype=tl.uint64)
    r_val = (r2 % 10) < 1
    vals = vals << r_val
    idx = (cs == r_idx).argmax(1)
    vals = vals << (60 - 4 * idx)
    boards = boards.ravel()
    boards = boards | vals
    return boards


@triton.jit
def insert_random_tile_kernel(
    boards_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Unpack boards into a 2D array of shape (BLOCK_SIZE, 16)
    boards = tl.load(boards_ptr + offsets, mask=offsets < n_elements).expand_dims(1)
    boards = insert_random_tile(boards, seed, BLOCK_SIZE)
    tl.store(boards_ptr + offsets, boards, mask=offsets < n_elements)


def insert_random_tile_wrapper(boards: torch.Tensor) -> torch.Tensor:
    insert_random_tile_kernel[1, 1](
        boards, boards.shape[0], int(time.time() * 1000), BLOCK_SIZE=1024
    )


@triton.jit
def transpose(
    boards,
):
    a1 = boards & 0xF0F00F0FF0F00F0F
    a2 = boards & 0x0000F0F00000F0F0
    a3 = boards & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    output = b1 | (b2 >> 24) | (b3 << 24)
    return output


@triton.jit
def flip_horizontal(
    boards,
):
    a1 = boards & 0x000F000F000F000F
    a2 = boards & 0x00F000F000F000F0
    a3 = boards & 0x0F000F000F000F00
    a4 = boards & 0xF000F000F000F000
    output = (a1 << 12) | (a2 << 4) | (a3 >> 4) | (a4 >> 12)
    return output


@triton.jit
def flip_vertical(
    boards,
):
    a1 = boards & 0xFFFF000000000000
    a2 = boards & 0x0000FFFF00000000
    a3 = boards & 0x00000000FFFF0000
    a4 = boards & 0x000000000000FFFF
    output = (a1 >> 48) | (a2 >> 16) | (a3 << 16) | (a4 << 48)
    return output


@triton.jit
def ntuple_mask_1(board):
    return ((board >> 36) & 0xFFF) | ((board >> 40) & 0xFFF000)


@triton.jit
def ntuple_mask_2(board):
    return (
        ((board >> 8) & 0xF)
        | ((board >> 20) & 0xF0)
        | ((board >> 28) & 0xFF00)
        | ((board >> 36) & 0xFF0000)
    )


@triton.jit
def ntuple_mask_3(board):
    return board >> 40


@triton.jit
def ntuple_mask_4(board):
    return ((board >> 20) & 0xF) | ((board >> 28) & 0xFFF0) | ((board >> 40) & 0xFF0000)


@triton.jit
def ntuple_mask_5(board):
    return ((board >> 20) & 0xFF) | ((board >> 32) & 0xF00) | ((board >> 40) & 0xFFF000)


@triton.jit
def ntuple_mask_6(board):
    return (
        ((board >> 4) & 0xFF)
        | ((board >> 16) & 0xF00)
        | ((board >> 28) & 0xF000)
        | ((board >> 40) & 0xFF0000)
    )


@triton.jit
def ntuple_mask_7(board):
    return (
        ((board >> 8) & 0xF)
        | ((board >> 20) & 0xFF0)
        | ((board >> 28) & 0xF000)
        | ((board >> 40) & 0xFF0000)
    )


@triton.jit
def ntuple_mask_8(board):
    return (
        ((board >> 20) & 0xF)
        | ((board >> 32) & 0xF0)
        | ((board >> 36) & 0xF00)
        | ((board >> 40) & 0xFFF000)
    )


@triton.jit
def get_all_ntuples(
    boards,
    n_boards,
):
    boards_t = transpose(boards)
    boards_h = flip_horizontal(boards)
    boards_ht = transpose(boards_h)
    boards_v = flip_vertical(boards)
    boards_vt = transpose(boards_v)
    boards_hv = flip_horizontal(boards_v)
    boards_hvt = transpose(boards_hv)
    boards1 = tl.join(boards, boards_t)
    boards2 = tl.join(boards_h, boards_ht)
    boards3 = tl.join(boards_v, boards_vt)
    boards4 = tl.join(boards_hv, boards_hvt)
    boards5 = tl.join(boards1, boards2)
    boards6 = tl.join(boards3, boards4)
    all_board_views = tl.join(boards5, boards6).reshape(n_boards, 8)

    nt_1 = ntuple_mask_1(all_board_views)
    nt_2 = ntuple_mask_2(all_board_views)
    nt_3 = ntuple_mask_3(all_board_views)
    nt_4 = ntuple_mask_4(all_board_views)
    nt_5 = ntuple_mask_5(all_board_views)
    nt_6 = ntuple_mask_6(all_board_views)
    nt_7 = ntuple_mask_7(all_board_views)
    nt_8 = ntuple_mask_8(all_board_views)
    nt_p1 = tl.join(nt_1, nt_2)
    nt_p2 = tl.join(nt_3, nt_4)
    nt_p3 = tl.join(nt_5, nt_6)
    nt_p4 = tl.join(nt_7, nt_8)
    nt_p5 = tl.join(nt_p1, nt_p2)
    nt_p6 = tl.join(nt_p3, nt_p4)
    all_ntuples = tl.join(nt_p5, nt_p6).reshape(n_boards, 8, 8)
    return all_ntuples


@triton.jit
def move_left(
    boards,
    move_lut,
    move_rewards,
    BLOCK_SIZE: tl.constexpr,
):
    shifts = 48 - (tl.arange(0, 4) * 16)
    rows = ((boards >> shifts) & 0xFFFF).ravel()
    new_rows = tl.gather(move_lut, rows, axis=0).to(tl.uint64).reshape(BLOCK_SIZE, 4)
    rewards = tl.gather(move_rewards, rows, axis=0).reshape(BLOCK_SIZE, 4).sum(1)
    new_boards = tl.sum(new_rows << shifts, 1)
    return new_boards, rewards


@triton.jit
def do_all_moves(
    boards,
    move_lut,
    move_rewards,
    BLOCK_SIZE: tl.constexpr,
):
    move_left_boards, move_left_rewards = move_left(
        boards, move_lut, move_rewards, BLOCK_SIZE
    )
    boards_r = flip_horizontal(boards)
    move_right_boards, move_right_rewards = move_left(
        boards_r, move_lut, move_rewards, BLOCK_SIZE
    )
    move_right_boards = flip_horizontal(move_right_boards)
    boards_u = transpose(boards)
    move_up_boards, move_up_rewards = move_left(
        boards_u, move_lut, move_rewards, BLOCK_SIZE
    )
    move_up_boards = transpose(move_up_boards)
    boards_d = flip_horizontal(boards_u)
    move_down_boards, move_down_rewards = move_left(
        boards_d, move_lut, move_rewards, BLOCK_SIZE
    )
    move_down_boards = transpose(flip_horizontal(move_down_boards))
    lu_boards = tl.join(move_left_boards, move_up_boards)
    rd_boards = tl.join(move_right_boards, move_down_boards)
    all_boards = tl.join(lu_boards, rd_boards).reshape(BLOCK_SIZE, 4)
    lu_rewards = tl.join(move_left_rewards, move_up_rewards)
    rd_rewards = tl.join(move_right_rewards, move_down_rewards)
    all_rewards = tl.join(lu_rewards, rd_rewards).reshape(BLOCK_SIZE, 4)

    return (
        all_boards,
        all_rewards,
    )


@triton.jit
def do_all_moves_kernel(
    boards_ptr,
    move_lut_ptr,
    move_rewards_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    boards = tl.load(boards_ptr + offsets, mask=offsets < n_elements).expand_dims(1)
    lut_offsets = tl.arange(0, 32768)  # MOVE_LUT_SIZE
    move_lut = tl.load(move_lut_ptr + lut_offsets)
    move_rewards = tl.load(move_rewards_ptr + lut_offsets)
    (all_boards, all_rewards) = do_all_moves(boards, move_lut, move_rewards, BLOCK_SIZE)
    output_offsets = tl.arange(0, BLOCK_SIZE)[:, None] * 4 + tl.arange(0, 4)
    tl.store(
        output_ptr + output_offsets,
        all_boards,
        mask=output_offsets < n_elements * 4,
    )


def do_all_moves_wrapper(
    boards: torch.Tensor, move_lut: torch.Tensor, move_rewards: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((boards.shape[0], 4), device=DEVICE, dtype=torch.uint64)
    do_all_moves_kernel[1, 1](
        boards, move_lut, move_rewards, output, boards.shape[0], BLOCK_SIZE=1024
    )
    return output


@triton.jit
def any_helper(a, b):
    return a | b


@triton.jit
def train_epoch_kernel(
    boards_ptr,
    move_lut_ptr,
    move_rewards_ptr,
    ntuple_values_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    boards = tl.load(boards_ptr + offsets, mask=active).expand_dims(1)
    lut_offsets = tl.arange(0, 32768)  # MOVE_LUT_SIZE
    move_lut = tl.load(move_lut_ptr + lut_offsets)
    move_rewards = tl.load(move_rewards_ptr + lut_offsets)
    zero_scores = tl.zeros((BLOCK_SIZE, 1), dtype=tl.uint32)
    zero_afterstate_values = tl.zeros((BLOCK_SIZE, 4), dtype=tl.float32)
    current_scores = tl.zeros((BLOCK_SIZE, 1), dtype=tl.uint32)
    current_ntuples = get_all_ntuples(boards, BLOCK_SIZE)
    ntuple_offsets = 16777216 * tl.arange(0, 8)[None, None, None, :]

    for i in tl.range(1000, loop_unroll_factor=0):
        (
            all_boards,
            all_rewards,
        ) = do_all_moves(boards, move_lut, move_rewards, BLOCK_SIZE)
        valid_move = all_boards != boards
        any_valid = valid_move.reduce(1, any_helper, keep_dims=True)
        candidate_move_ntuples = get_all_ntuples(all_boards, BLOCK_SIZE * 4).reshape(
            BLOCK_SIZE, 4, 8, 8
        )
        # [batch, move, board_view, ntuple]
        candidate_move_ntuple_values = (
            tl.load(ntuple_values_ptr + candidate_move_ntuples + ntuple_offsets) / 8
        ).sum(2)
        # [batch, move, ntuple]
        candidate_move_afterstate_values = (candidate_move_ntuple_values / 8).sum(
            2
        ) + all_rewards
        # Set the afterstate values to zero if the move was invalid, to prevent argmax from choosing an invalid move.
        candidate_move_afterstate_values = tl.where(
            valid_move, candidate_move_afterstate_values, zero_afterstate_values
        )
        best_move_idx = candidate_move_afterstate_values.argmax(1).expand_dims(1)
        afterstate_values = tl.gather(
            candidate_move_afterstate_values, best_move_idx, axis=1
        )
        afterstate_values = tl.where(any_valid, afterstate_values, current_scores)
        # Only write the ntuple values if the particular game didn't end in the last move and get reset.
        tl.store(
            ntuple_values_ptr + current_ntuples + ntuple_offsets.reshape(1, 1, 8),
            afterstate_values.expand_dims(1),
            mask=any_valid.expand_dims(1),
        )
        boards = tl.gather(all_boards, best_move_idx, axis=1)

        # Prepare the next training step.
        # New scores are the current scores plus the reward of the best move.
        best_move_reward = tl.gather(all_rewards, best_move_idx, axis=1)
        current_scores = current_scores + best_move_reward
        # Reset the board if no valid move was made
        new_games = tl.zeros((BLOCK_SIZE, 1), dtype=tl.uint64)
        insert_random_tile(new_games, i, BLOCK_SIZE)
        insert_random_tile(new_games, i, BLOCK_SIZE)
        insert_random_tile(boards, i, BLOCK_SIZE)
        boards = tl.where(any_valid, boards, new_games)
    tl.store(
        output_ptr + tl.arange(0, BLOCK_SIZE),
        boards.ravel(),
        mask=offsets < n_elements,
    )


@triton.jit
def board_to_repr_kernel(
    board_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE * 16)
    boards = tl.load(board_ptr + offsets, mask=offsets < n_elements * 16).reshape(
        BLOCK_SIZE, 16
    )
    shifts = 64 - (tl.arange(1, 17) * 4)
    boards = boards << shifts
    reprs = tl.sum(boards, 1)
    output_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_offsets, reprs, mask=output_offsets < n_elements)


def board_to_repr(board: torch.Tensor) -> torch.Tensor:
    out = torch.empty(board.shape[0], device=DEVICE, dtype=torch.uint64)
    board_to_repr_kernel[1, 1](board, out, board.shape[0], BLOCK_SIZE=1024)
    return out


@triton.jit
def repr_to_board_kernel(
    repr_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    repr = tl.load(repr_ptr + offsets, mask=mask).reshape(BLOCK_SIZE, 1)
    shifts = 64 - (tl.arange(1, 17) * 4)
    board = (repr >> shifts) & 0xF
    output_offsets = offsets.reshape(BLOCK_SIZE, 1) * 16 + tl.arange(0, 16).reshape(
        1, 16
    )
    tl.store(output_ptr + output_offsets, board, mask=mask.reshape(BLOCK_SIZE, 1))


def repr_to_board(repr: torch.Tensor) -> torch.Tensor:
    out = torch.empty((repr.shape[0], 16), device=DEVICE, dtype=torch.uint64)
    repr_to_board_kernel[(triton.cdiv(repr.shape[0], 1024),)](
        repr, out, repr.shape[0], BLOCK_SIZE=1024
    )
    return out


def print_boards(boards: torch.Tensor):
    if boards.ndim == 1:
        if boards.numel() != 16:
            raise ValueError("Expected a single board with 16 cells.")
        boards = boards.reshape(1, 4, 4)
    elif boards.ndim == 2:
        if boards.shape[1] != 16:
            raise ValueError("Expected boards with shape (n, 16).")
        boards = boards.reshape(-1, 4, 4)
    elif boards.ndim == 3:
        if boards.shape[1:] != (4, 4):
            raise ValueError("Expected boards with shape (n, 4, 4).")
    else:
        raise ValueError("Expected boards with shape (16,), (n, 16), or (n, 4, 4).")

    boards_cpu = boards.detach().to("cpu")
    cell_width = 6

    def format_cell(value: int) -> str:
        return "." if value == 0 else str(1 << value)

    rendered_boards: list[list[str]] = []
    for board in boards_cpu:
        rendered_rows: list[str] = []
        for row in board:
            cells = [format_cell(int(cell)) for cell in row.tolist()]
            rendered_rows.append(" ".join(f"{cell:>{cell_width}}" for cell in cells))
        rendered_boards.append(rendered_rows)

    board_separator = "      "
    for line_idx in range(len(rendered_boards[0])):
        print(
            board_separator.join(
                board_lines[line_idx] for board_lines in rendered_boards
            )
        )


def generate_transition_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    left_rows = torch.empty(MOVE_LUT_SIZE, dtype=torch.uint16)
    left_reward = torch.empty(MOVE_LUT_SIZE, dtype=torch.uint16)

    for line_repr in range(MOVE_LUT_SIZE):
        row = [(line_repr >> (12 - 4 * idx)) & 0xF for idx in range(4)]

        compact = [value for value in row if value != 0]
        merged: list[int] = []
        reward = 0
        cursor = 0
        while cursor < len(compact):
            current = compact[cursor]
            if cursor + 1 < len(compact) and compact[cursor + 1] == current:
                current += 1
                reward += current
                cursor += 2
            else:
                cursor += 1
            merged.append(current)
        if len(merged) > 0 and max(merged) > 15:
            continue

        merged.extend([0] * (4 - len(merged)))
        new_line_repr = 0
        for i, value in enumerate(merged):
            new_line_repr |= value << (12 - 4 * i)
        left_rows[line_repr] = new_line_repr
        left_reward[line_repr] = reward

    return (
        left_rows.to(device=device),
        left_reward.to(device=device),
    )


# boards = board_to_repr(
#     torch.tensor(
#         [[[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 2]]],
#         device=DEVICE,
#         dtype=torch.uint64,
#     )
# )
boards = torch.zeros(128, device=DEVICE, dtype=torch.uint64)
insert_random_tile_wrapper(boards)
insert_random_tile_wrapper(boards)

print_boards(repr_to_board(boards))
print("-" * 100)

move_lut, move_rewards = generate_transition_tables(DEVICE)
ntuple_values = torch.full(
    (8, NUM_NTUPLE_VALUES), 64, device=DEVICE, dtype=torch.float32
)
# all_moves = do_all_moves_wrapper(boards, move_lut, move_rewards)
# print_boards(repr_to_board(all_moves[0, :]))
output = torch.empty((boards.shape[0], 1), device=DEVICE, dtype=torch.int64)
t = time.time()
train_epoch_kernel[1, 1](
    boards,
    move_lut,
    move_rewards,
    ntuple_values,
    output,
    boards.shape[0],
    BLOCK_SIZE=128,
)
print(time.time() - t)
print_boards(repr_to_board(output)[:4, :])
