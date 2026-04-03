import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def insert_random_tile_kernel(
    boards_ptr,
    n_elements,
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
    # r = ((tl.randint4x(42, offsets) % num_zeros) + 1).expand_dims(1)
    r1, r2, _, _ = tl.randint4x(42, offsets)
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
    inserted_boards = torch.empty_like(boards)
    insert_random_tile_kernel[1, 1](boards, boards.shape[0], BLOCK_SIZE=1024)
    print(boards)
    return inserted_boards


boards = torch.tensor(
    [
        # [0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 0, 0, 15],
        # [8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0],
        # [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    device=DEVICE,
    dtype=torch.uint64,
)


@triton.jit
def transpose_kernel(
    boards_ptr,  # *Pointer* to first input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    boards = tl.load(boards_ptr + offsets, mask=mask)
    a1 = boards & 0xF0F00F0FF0F00F0F
    a2 = boards & 0x0000F0F00000F0F0
    a3 = boards & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    output = b1 | (b2 >> 24) | (b3 << 24)
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def transpose(boards: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(boards)
    transpose_kernel[1, 1](boards, output, boards.shape[0], BLOCK_SIZE=1024)
    return output


@triton.jit
def flip_horizontal_kernel(
    boards_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    boards = tl.load(boards_ptr + offsets, mask=offsets < n_elements)
    a1 = boards & 0x000F000F000F000F
    a2 = boards & 0x00F000F000F000F0
    a3 = boards & 0x0F000F000F000F00
    a4 = boards & 0xF000F000F000F000
    output = a1 | (a2 << 4) | (a3 << 8) | (a4 << 12)
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)


@triton.jit
def rotate_left_kernel(
    boards_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    boards = tl.load(boards_ptr + offsets, mask=offsets < n_elements)


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


def generate_transition_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    left_rows = torch.empty(65536, dtype=torch.uint16)
    left_reward = torch.empty(65536, dtype=torch.uint32)

    for line_repr in range(65536):
        row = [(line_repr >> (4 * idx)) & 0xF for idx in range(4)]

        compact = [value for value in row if value != 0]
        merged: list[int] = []
        reward = 0
        cursor = 0
        while cursor < len(compact):
            current = compact[cursor]
            if cursor + 1 < len(compact) and compact[cursor + 1] == current:
                current += 1
                reward += 1 << current
                cursor += 2
            else:
                cursor += 1
            merged.append(current)

        merged.extend([0] * (4 - len(merged)))
        new_line_repr = 0
        for i, value in enumerate(merged):
            new_line_repr |= value << (4 * i)
        left_rows[line_repr] = new_line_repr
        left_reward[line_repr] = reward

    return (
        left_rows.to(device=device),
        left_reward.to(device=device),
    )


@triton.jit
def test_rand(
    output_ptr,
):
    a, b, c, d = tl.randint4x(0, tl.arange(0, 16))
    tl.store(output_ptr + tl.arange(0, 32), a)
    tl.store(output_ptr + tl.arange(16, 32), b)
    tl.store(output_ptr + tl.arange(32, 48), c)
    tl.store(output_ptr + tl.arange(48, 64), d)


def randint() -> torch.Tensor:
    output = torch.empty(128, device=DEVICE, dtype=torch.uint32)
    test_rand[1, 1](output)
    return output


print(boards)
board_reprs = board_to_repr(boards)
inserted_boards = insert_random_tile(board_reprs)
print(repr_to_board(board_reprs))
