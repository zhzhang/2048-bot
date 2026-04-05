import torch
import time
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
MOVE_LUT_SIZE = 65536 // 2
NUM_NTUPLE_MASKS = 8
SYMMETRIES_PER_MASK = 8
FEATURES_PER_BOARD = NUM_NTUPLE_MASKS * SYMMETRIES_PER_MASK


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
    tl.store(boards_ptr + offsets, boards, mask=offsets < n_elements)


def insert_random_tile(boards: torch.Tensor) -> torch.Tensor:
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
def emit_all_ntuples(
    boards,
    output_ptr,
    output_offsets,
    output_mask,
):
    board_t = transpose(boards)
    board_h = flip_horizontal(boards)
    board_ht = transpose(board_h)
    board_v = flip_vertical(boards)
    board_vt = transpose(board_v)
    board_hv = flip_horizontal(board_v)
    board_hvt = transpose(board_hv)

    tl.store(output_ptr + output_offsets + 0, ntuple_mask_1(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 1, ntuple_mask_1(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 2, ntuple_mask_1(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 3, ntuple_mask_1(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 4, ntuple_mask_1(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 5, ntuple_mask_1(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 6, ntuple_mask_1(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 7, ntuple_mask_1(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 8, ntuple_mask_2(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 9, ntuple_mask_2(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 10, ntuple_mask_2(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 11, ntuple_mask_2(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 12, ntuple_mask_2(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 13, ntuple_mask_2(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 14, ntuple_mask_2(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 15, ntuple_mask_2(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 16, ntuple_mask_3(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 17, ntuple_mask_3(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 18, ntuple_mask_3(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 19, ntuple_mask_3(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 20, ntuple_mask_3(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 21, ntuple_mask_3(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 22, ntuple_mask_3(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 23, ntuple_mask_3(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 24, ntuple_mask_4(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 25, ntuple_mask_4(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 26, ntuple_mask_4(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 27, ntuple_mask_4(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 28, ntuple_mask_4(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 29, ntuple_mask_4(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 30, ntuple_mask_4(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 31, ntuple_mask_4(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 32, ntuple_mask_5(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 33, ntuple_mask_5(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 34, ntuple_mask_5(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 35, ntuple_mask_5(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 36, ntuple_mask_5(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 37, ntuple_mask_5(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 38, ntuple_mask_5(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 39, ntuple_mask_5(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 40, ntuple_mask_6(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 41, ntuple_mask_6(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 42, ntuple_mask_6(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 43, ntuple_mask_6(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 44, ntuple_mask_6(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 45, ntuple_mask_6(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 46, ntuple_mask_6(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 47, ntuple_mask_6(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 48, ntuple_mask_7(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 49, ntuple_mask_7(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 50, ntuple_mask_7(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 51, ntuple_mask_7(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 52, ntuple_mask_7(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 53, ntuple_mask_7(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 54, ntuple_mask_7(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 55, ntuple_mask_7(board_hvt), mask=output_mask)

    tl.store(output_ptr + output_offsets + 56, ntuple_mask_8(boards), mask=output_mask)
    tl.store(output_ptr + output_offsets + 57, ntuple_mask_8(board_t), mask=output_mask)
    tl.store(output_ptr + output_offsets + 58, ntuple_mask_8(board_h), mask=output_mask)
    tl.store(output_ptr + output_offsets + 59, ntuple_mask_8(board_ht), mask=output_mask)
    tl.store(output_ptr + output_offsets + 60, ntuple_mask_8(board_v), mask=output_mask)
    tl.store(output_ptr + output_offsets + 61, ntuple_mask_8(board_vt), mask=output_mask)
    tl.store(output_ptr + output_offsets + 62, ntuple_mask_8(board_hv), mask=output_mask)
    tl.store(output_ptr + output_offsets + 63, ntuple_mask_8(board_hvt), mask=output_mask)


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
    rewards = tl.gather(move_rewards, rows, axis=0)
    new_boards = tl.sum(new_rows << shifts, 1)
    return new_boards, rewards


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
    output_offsets = tl.arange(0, BLOCK_SIZE) * 4
    tl.store(
        output_ptr + output_offsets,
        move_left_boards,
        mask=output_offsets < n_elements * 4,
    )
    tl.store(
        output_ptr + output_offsets + 1,
        move_right_boards,
        mask=output_offsets < n_elements * 4,
    )
    tl.store(
        output_ptr + output_offsets + 2,
        move_up_boards,
        mask=output_offsets < n_elements * 4,
    )
    tl.store(
        output_ptr + output_offsets + 3,
        move_down_boards,
        mask=output_offsets < n_elements * 4,
    )


def do_all_moves(
    boards: torch.Tensor, move_lut: torch.Tensor, move_rewards: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((boards.shape[0], 4), device=DEVICE, dtype=torch.uint64)
    do_all_moves_kernel[1, 1](
        boards, move_lut, move_rewards, output, boards.shape[0], BLOCK_SIZE=1024
    )
    return output


@triton.jit
def train_epoch_kernel(
    boards_ptr,
    move_lut_ptr,
    move_rewards_ptr,
    ntuples_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    boards = tl.load(boards_ptr + offsets, mask=active)
    output_offsets = offsets * FEATURES_PER_BOARD
    # Output layout is [mask_0_sym_0..7, mask_1_sym_0..7, ..., mask_7_sym_0..7].
    emit_all_ntuples(boards, output_ptr, output_offsets, active)


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
                reward += 1
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


boards = torch.zeros(3, device=DEVICE, dtype=torch.uint64)
insert_random_tile(boards)
insert_random_tile(boards)

print_boards(repr_to_board(boards))
print("-" * 100)

move_lut, move_rewards = generate_transition_tables(DEVICE)
all_moves = do_all_moves(boards, move_lut, move_rewards)
print_boards(repr_to_board(all_moves[1, :]))
