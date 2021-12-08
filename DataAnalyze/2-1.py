import numpy as np


def gen_board(m, n):
    return np.random.randint(low=100, high=1000, size=[m, n])


def _solve(_board, x, y):
    if x < 0 or y < 0:
        return 0
    return max(_solve(_board, x - 1, y), _solve(_board, x, y - 1)) + _board[x][y]


def _solve_with_print(_board, x, y, str):
    if x < 0 or y < 0:
        return 0, ''
    L, Lmsg = _solve_with_print(_board, x - 1, y, str)
    U, Umsg = _solve_with_print(_board, x, y - 1, str)
    if L > U:
        return L + _board[x][y], Lmsg + 'R '
    else:
        return U + _board[x][y], Umsg + 'D '


def solve(_board):
    return _solve(_board, _board.shape[0] - 1, _board.shape[1] - 1)


def solve_with_print(_board):
    return _solve_with_print(_board, _board.shape[0] - 1, _board.shape[1] - 1, '')


board = gen_board(10, 10)
print(board)
#print(solve(board))
print(solve_with_print(board))