import random
from collections import defaultdict
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
    reward = 0
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
                    reward += new[-1]
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
    return new_board, reward


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
    def __init__(self, alpha=0.1):
        self.reset()
        self.ntuples = [
            [0, 1, 2, 4, 5, 6],
            [1, 2, 5, 6, 9, 13],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 5, 6, 7, 10],
        ]
        self.values = tuple(defaultdict(int) for _ in range(len(self.ntuples)))
        self.alpha = alpha
        self.ntuple_folds = 8 * len(self.ntuples)

    def reset(self):
        self.board = init_board()
        self.score = 0

    def get_next_move(self):
        m = np.random.choice([Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN])
        try:
            new_board, reward = move(self.board, m)
            self.board = new_board
        except IllegalMove:
            return False
        # move_results = []
        # current_value, ntuples = self.get_value()
        # for i, m in enumerate(Move):  # i to break ties
        #     try:
        #         new_board, reward = move(self.board, m)
        #     except IllegalMove:
        #         continue
        #     new_board_value, _ = self.get_value(new_board)
        #     move_results.append((reward + new_board_value, reward, i, new_board))
        # if len(move_results) == 0:
        #     # for i, ntuple in ntuples:
        #     #     self.values[i][ntuple] = float("-inf")
        #     return False
        # new_value, reward, _, new_board = max(move_results)
        # self.score += reward
        # # Update ntuple values
        # for i, ntuple in ntuples:
        #     self.values[i][ntuple] += (
        #         self.alpha * (new_value - current_value) / self.ntuple_folds
        #     )
        # self.board = new_board
        return True

    def get_value(self, board=None):
        value = 0
        ntuples = self.get_ntuples(board)
        for i, ntuple_values in ntuples:
            value += sum(self.values[i][v] for v in ntuple_values)
        return value, ntuples

    def get_ntuples(self, board=None):
        output = []
        for i, nt in enumerate(self.ntuples):
            board = self.board if board is None else board
            for _ in range(4):
                board = np.rot90(board)
                output.append((i, tuple(np.take(board, nt))))
                output.append((i, tuple(np.take(board.T, nt))))
        return output


if __name__ == "__main__":
    agent = Agent()
    import time

    run = 0
    start = time.time()
    while True:
        run += 1
        while agent.get_next_move():
            continue
        if run % 1000 == 0:
            print(f"{time.time() - start} seconds")
            start = time.time()
            print_board(agent.board)
            print(agent.score)
            total_weights = sum(sum(v for v in d.values()) for d in agent.values)
            print(total_weights)
        agent.reset()
    # board = init_board()
    # print_board(board)
    # while True:
    #     # Wait for user input:
    #     print("Next move:")
    #     key = input()
    #     if key == "w":
    #         board = move(board, Move.UP)
    #     elif key == "s":
    #         board = move(board, Move.DOWN)
    #     elif key == "a":
    #         board = move(board, Move.LEFT)
    #     elif key == "d":
    #         board = move(board, Move.RIGHT)
    #     else:
    #         break
    #     print_board(board)
