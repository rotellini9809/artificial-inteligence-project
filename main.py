from minesweeper_solver import Board, Game
import time

size = (16, 32)
numBombs = 60
screenSize = (size[1] * 30, size[0] * 30)
count = 0

for i in range(1, 1000):

    board = Board(size, numBombs)
    game = Game(board, screenSize)
    start = time.time()
    temp, end = game.run(True)
    if temp:
        count += 1

    print(temp, end - start)

    print("Risolti:", count, "/", i, "\npercentage:", count / i)