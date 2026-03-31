import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
def reduce_board(a, b):
    return a << 4 | b


@triton.jit
def board_to_repr_kernel(
    board_ptr,
    output_ptr,
):
    board = tl.load(board_ptr + tl.arange(0, 16))
    repr = tl.reduce(board, 0, reduce_board)
    tl.store(output_ptr, repr)


def print_board(board: torch.Tensor):
    for i in range(16):
        value = board >> (4 * i) & 0xF
        if value == 0:
            print("    .", end="")
        else:
            print(f"{1 << value:5}", end="")
        if (i + 1) % 4 == 0:
            print()


def board_to_repr(board: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(board)
    board_to_repr_kernel[1, 1](board, out)
    return out


torch.manual_seed(0)
board = torch.tensor(
    [0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16],
    device=DEVICE,
    dtype=torch.uint64,
)
print(board_to_repr(board))
