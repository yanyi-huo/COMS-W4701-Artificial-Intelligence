import math
import random
from typing import Tuple, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from random import sample

"""
Sudoku board initializer
Credit: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
"""
def generate(n: int, num_clues: int) -> dict:
    # Generate a sudoku problem of order n with "num_clues" cells assigned
    # Return dictionary containing clue cell indices and corresponding values
    # (You do not need to worry about components inside returned dictionary)
    N = range(n)

    rows = [g * n + r for g in sample(N, n) for r in sample(N, n)]
    cols = [g * n + c for g in sample(N, n) for c in sample(N, n)]
    nums = sample(range(1, n**2 + 1), n**2)

    S = np.array(
        [[nums[(n * (r % n) + r // n + c) % (n**2)] for c in cols] for r in rows]
    )
    indices = sample(range(n**4), num_clues)
    values = S.flatten()[indices]

    mask = np.full((n**2, n**4), True)
    mask[:, indices] = False
    i, j = np.unravel_index(indices, (n**2, n**2))

    for c in range(num_clues):
        v = values[c] - 1
        maskv = np.full((n**2, n**2), True)
        maskv[i[c]] = False
        maskv[:, j[c]] = False
        maskv[
            (i[c] // n) * n : (i[c] // n) * n + n, (j[c] // n) * n : (j[c] // n) * n + n
        ] = False
        mask[v] = mask[v] * maskv.flatten()

    return {"n": n, "indices": indices, "values": values, "valid_indices": mask}


def display(problem: dict):
    # Display the initial board with clues filled in (all other cells are 0)
    n = problem["n"]
    empty_board = np.zeros(n**4, dtype=int)
    empty_board[problem["indices"]] = problem["values"]
    print("Sudoku puzzle:\n", np.reshape(empty_board, (n**2, n**2)), "\n")


def initialize(problem: dict) -> npt.NDArray:
    # Returns a random initial sudoku board given problem
    n = problem["n"]
    S = np.zeros(n**4, dtype=int)
    S[problem["indices"]] = problem["values"]

    all_values = list(np.repeat(range(1, n**2 + 1), n**2))
    for v in problem["values"]:
        all_values.remove(v)
    all_values = np.array(all_values)
    np.random.shuffle(all_values)

    indices = [i for i in range(S.size) if i not in problem["indices"]]
    S[indices] = all_values
    S = S.reshape((n**2, n**2))

    return S


def successors(S: npt.NDArray, problem: dict) -> List[npt.NDArray]:
    # Returns list of all successor states of S by swapping two non-clue entries
    mask = problem["valid_indices"]
    indices = [i for i in range(S.size) if i not in problem["indices"]]
    succ = []

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            s = np.copy(S).flatten()
            if s[indices[i]] == s[indices[j]]:
                continue
            if not (
                mask[s[indices[i]] - 1, indices[j]]
                and mask[s[indices[j]] - 1, indices[i]]
            ):
                continue
            s[indices[i]], s[indices[j]] = s[indices[j]], s[indices[i]]
            succ.append(s.reshape(S.shape))

    return succ


"""
WRITE THIS FUNCTION
"""
def num_errors(S: npt.NDArray) -> int:
    # Given a current sudoku board state (2d NumPy array), compute and return total number of errors
    # Count total number of missing numbers from each row, column, and non-overlapping square blocks
    num = 0
    n = len(S)
    k = int(math.sqrt(n))
    # row
    for i in range(0, n):
        number = [x for x in range(1, n+1)]
        for j in range(0, n):
            if S[i][j] in number:
                number.remove(S[i][j])
        num += len(number)
    # column
    for j in range(0, n):
        number = [x for x in range(1, n+1)]
        for i in range(0, n):
            if S[i][j] in number:
                number.remove(S[i][j])
        num += len(number)
    # subgrid
    for i in range(0, n, k):
        for j in range(0, n, k):
            number = [x for x in range(1, n + 1)]
            for x in range(i, i+k):
                for y in range(j, j+k):
                    if S[x][y] in number:
                        number.remove(S[x][y])
            num += len(number)
    return num


"""
WRITE THIS FUNCTION
"""
def hill_climb(
    problem: dict,
    max_sideways: int = 0,
    max_restarts: int = 0
) -> Tuple[npt.NDArray, List[int]]:
    # Given: Sudoku problem and optional max sideways moves and max restarts parameters
    # Return: Board state solution (2d NumPy array), list of errors in each iteration of hill climbing search
    iter = []
    sideways = []
    sample_sideways = [1 for i in range(0,max_sideways)]
    restarts = 0
    current = initialize(problem)
    while True:
        neighbor = successors(current, problem)
        neighbor.sort(key=lambda x: num_errors(x))
        neighbor_error = num_errors(neighbor[0])
        current_error = num_errors(current)
        if neighbor_error > current_error:
            if current_error == 0:
                return current, iter
            current = initialize(problem)
            restarts += 1
            iter.append(num_errors(current))
            if restarts == max_restarts:
                return current, iter
        elif neighbor_error == current_error:
            sideways.append(1)
            if sideways[-1: -1-max_sideways: -1] == sample_sideways:
                current = initialize(problem)
                restarts += 1
                iter.append(num_errors(current))
                if restarts == max_restarts:
                    return current, iter
                continue
            new_current = random.choice((current, neighbor[0]))
            current = new_current
        else:
            current = neighbor[0]
            sideways.append(0)
        iter.append(num_errors(current))



if __name__ == "__main__":
    n = 3
    clues = 40

    problem = generate(n, clues)
    display(problem)
    sol, errors = hill_climb(problem, 10, 10)
    print("Solution:\n", sol)
    plt.plot(errors)
    plt.show()
    '''
    success = 0
    failure = 0
    for i in range(0, 100):
        print(i)
        problem = generate(n, clues)
        sol, errors = hill_climb(problem, 10)
        if num_errors(sol) == 0:
            success += 1
        else:
            failure += 1
    print("success: ", success)
    print(('failure: ', failure))
    '''



