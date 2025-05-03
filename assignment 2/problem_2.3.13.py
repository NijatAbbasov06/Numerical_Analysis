import numpy as np
from numpy.linalg import solve

def hilbert_matrix(n):
    return np.array([[1/(i+j+1) for j in range(n+1)] for i in range(n+1)])

def compute_f(n):
    return np.array([sum(1/(i+j+1) for i in range(n+1)) for j in range(n+1)])


def solve_system(n):
    A = hilbert_matrix(n)
    f = compute_f(n)
    c = solve(A, f)
    print(c)
    return c



n = 10


c = solve_system(n)


print("Computed solution c:")
print(c)
print("\nExact solution:")
print(np.ones(n+1))
print("\nAbsolute error:")
print(c - np.ones(n+1))
print("\nCondition number of A:")
print(np.linalg.cond(hilbert_matrix(n)))

