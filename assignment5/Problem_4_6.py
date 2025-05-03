import numpy as np
from numpy import linalg as LA


def householder_qr(A):
    m, n = A.shape
    H_vec = []
    
    for k in range(min(m-1, n)):
        x = A[k:, k]
        norm_x = np.linalg.norm(x)
        
        if norm_x > 0: 
            sign = -1 if x[0] < 0 else 1
            v = x.copy()
            v[0] += sign * norm_x
            v = v / np.linalg.norm(v)
        
            H = np.zeros(m)
        
            H[k:] = v
            H_vec.append(H)
            A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
    if not H_vec:
        H_vec = [np.zeros(m) for i in range(m)]
    return A, H_vec

def make_q(H_vec):

    m = len(H_vec[0])
    Q = np.eye(m)
    
    for v in reversed(H_vec):
        Q -= 2 * np.outer(v, v @ Q)
    
    return Q



def qr_iteration_shifts(A, tolerance=1e-8, max_iter=1000):
    A_k = A.copy().astype(np.float64)
    n = A.shape[0]
    iterations = 0
    
    while iterations < max_iter:
        sigma = A_k[n-1, n-1]

        R, H_vec = householder_qr((A_k - sigma* np.eye(n)).astype(dtype = np.float64)) 
        Q = make_q(H_vec)
        A_k = R @ Q + sigma * np.eye(n)
        
        if np.all(np.abs(A_k - np.diag(np.diag(A_k))) < tolerance):
            return np.diag(A_k), iter

        iterations += 1
    
    eigvals = np.diag(A_k)
    return eigvals, iterations


A1 = np.array([[1, 1000],
               [0.001, 1]], dtype=np.float64)

B1 = np.array([[1, 1000],
               [0, 1]], dtype=np.float64)

A2 = np.array([[2, 3, 2],
               [10, 3, 4],
               [3, 6, 1]], dtype=np.float64)

def test_matrix(A, name):
    print(f"\nTesting {name}:")
    print("Matrix:")
    print(A)
    
    eigvals, iters = qr_iteration_shifts(A)
    print("\nOur eigenvals:")
    print(f"Eigenvalues: {eigvals}")
    print(f"Iterations needed: {iters}")

    numpy_eigvals = np.linalg.eigvals(A)
    print("\nNumPy eigvals:")
    print(numpy_eigvals)
    

    print("\nAbs difference:")
    diff = np.sort(np.abs(eigvals)) - np.sort(np.abs(numpy_eigvals))
    print(np.abs(diff))

test_matrix(A1, "first")
test_matrix(B1, "second")
test_matrix(A2, "third")