import random
from enum import Enum

import numpy as np


class Move(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class IllegalMove(Exception):
    pass


def init_board():
    board = np.zeros((4, 4), dtype=int)
    add_random_tile(board)
    add_random_tile(board)
    return board


def add_random_tile(board):
    value = 2 if random.random() < 0.9 else 4
    empty = np.transpose((board == 0).nonzero())
    cell_index = random.choice(empty)
    board[cell_index[0], cell_index[1]] = value


def move(board, move: Move):
    new_board = []
    if move == Move.RIGHT:
        board = np.flip(board, 1)
    elif move == Move.UP:
        board = np.transpose(board)
    elif move == Move.DOWN:
        board = np.flip(np.transpose(board), 1)
    for i in range(4):
        new = []
        last_val = 0
        for j in range(4):
            val = board[i, j]
            if val == 0:
                continue
            if len(new) == 0:
                new.append(val)
                last_val = val
            else:
                if val == last_val:
                    new[-1] *= 2
                    last_val = 0
                else:
                    new.append(val)
                    last_val = val
        while len(new) < 4:
            new.append(0)
        new_board.append(new)
    new_board = np.array(new_board)
    if (board == new_board).all():
        raise IllegalMove()
    if move == Move.RIGHT:
        new_board = np.flip(new_board, 1)
    elif move == Move.UP:
        new_board = np.transpose(new_board)
    elif move == Move.DOWN:
        new_board = np.transpose(np.flip(new_board, 1))
    add_random_tile(new_board)
    return new_board


def print_board(board):
    out = ""
    for i in range(4):
        for j in range(4):
            num = str(board[i, j])
            while len(num) < 4:
                num = " " + num
            out += num + "  "
        out += "\n"
    print(out)


class Agent:
    def __init__(self):
        self.board = init_board()
        pass

    def get_next_move():
        pass

    def get_ntuples(self):
        ntuples = [
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
            [[0, 0], [0, 1], [1, 1], [1, 2], [2, 1], [3, 2]],
            [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2]],
            [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3], [2, 2]],
        ]
        pass


if __name__ == "__main__":
    board = init_board()
    print_board(board)
    while True:
        # Wait for user input:
        print("Next move:")
        key = input()
        if key == "w":
            board = move(board, Move.UP)
        elif key == "s":
            board = move(board, Move.DOWN)
        elif key == "a":
            board = move(board, Move.LEFT)
        elif key == "d":
            board = move(board, Move.RIGHT)
        else:
            break
        print_board(board)
