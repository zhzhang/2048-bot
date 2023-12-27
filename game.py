from copy import deepcopy
from enum import Enum
import random


class Move(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Game:
    def __init__(self):
        self.state = [0 for _ in range(16)]
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        value = 2 if random.random() < 0.9 else 4
        empty = []
        for i, v in enumerate(self.state):
            if v == 0:
                empty.append(i)
        cell_index = random.choice(empty)
        self.state[cell_index] = value

    def move(self, move: Move):
        start, step, next_row = 0, 1, 4
        if move == Move.RIGHT:
            start, step, next_row = 15, -1, -4
        elif move == Move.UP:
            start, step, next_row = 0, 4, 1
        elif move == Move.DOWN:
            start, step, next_row = 15, -4, -1
        for _ in range(4):
            new = []
            last_val = 0
            for index in range(start, start + 4 * step, step):
                val = self.state[index]
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
            for index in range(start, start + 4 * step, step):
                self.state[index] = new.pop(0)
            start += next_row
        self.add_random_tile()

    def __str__(self):
        out = ""
        for i in range(4):
            for j in range(4):
                num = str(self.state[i * 4 + j])
                while len(num) < 4:
                    num = " " + num
                out += num + "  "
            out += "\n"
        return out


class Agent:
    def __init__(self):
        self.game = Game()
        pass

    def get_next_move():
        pass


if __name__ == "__main__":
    game = Game()
    print(game)
    while True:
        # Wait for user input:
        print("Next move:")
        key = input()
        if key == "w":
            game.move(Move.UP)
        elif key == "s":
            game.move(Move.DOWN)
        elif key == "a":
            game.move(Move.LEFT)
        elif key == "d":
            game.move(Move.RIGHT)
        else:
            break
        print(game)
        print(game.state)
