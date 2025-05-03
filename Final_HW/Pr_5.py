import numpy as np
from scipy.linalg import hessenberg
from scipy.linalg import eig
from scipy.linalg import block_diag

n = 6  
h = 1 / n
B = np.diag([4] * (n - 1)) + np.diag([-1] * (n - 2), k=1) + np.diag([-1] * (n - 2), k=-1)

def newton_m(func, jacobian, guess, tol=1e-8, max_iter=100):
    variables = np.array([guess], dtype=float) 
    for iteration in range(max_iter):
        f_val = func(variables[0])
        j_val = jacobian(variables[0])
        delta = f_val/j_val
        variables[0] -= delta

        if abs(delta) < tol:
            return variables, iteration + 1
    return variables, max_iter  



A = np.zeros(((n - 1) ** 2, (n - 1) ** 2))
block_size = n - 1
for i in range(block_size):
    for j in range(block_size):
        idx_row = i * block_size + j
        A[idx_row, idx_row] = B[j, j]  
        if j < block_size - 1:  
            A[idx_row, idx_row + 1] = B[j, j + 1]
            A[idx_row + 1, idx_row] = B[j + 1, j]
        if i < block_size - 1: 
            A[idx_row, idx_row + block_size] = -1
            A[idx_row + block_size, idx_row] = -1
A = A *n**2

def divide_and_conquer_eigenvalues(T):
    n = T.shape[0]
    k = n // 2  
    beta = T[k-1, k]
    u = np.zeros((n, 1))
    u[k-1, 0] = 1 
    u[k, 0] = 1  

    T_adjusted = T - beta * (u @ u.T)
    T1 = T_adjusted[:k, :k]  
    T2 = T_adjusted[k:, k:]
    eig_T1, Q_1 = eig(T1)
    eig_T2, Q_2 = eig(T2)

    Q = block_diag(Q_1,Q_2)
    

    v = Q.T @ u

    eigenvalues_D = np.concatenate([eig_T1, eig_T2]).real



    def f(lmbda):
        denominator = eigenvalues_D - lmbda  
        numerator = v**2        
        term = numerator / denominator 
        return 1 + beta * np.sum(term)  
    
    def fder(lmbda):
        denominator = (eigenvalues_D - lmbda)**2
        numerator = v**2
        term = numerator / denominator
        return -beta * np.sum(term)
    
    eigenvalues = np.zeros(10)

    for i in range(10):
        search_points = np.sort(eigenvalues_D)
        initial_guess = search_points[-i-1] + 1e-10
    
        solution_newton, iterations_newton = newton_m(f, fder, initial_guess)

        eigenvalues[i] = solution_newton

    return np.sort(eigenvalues.real)[::-1]


def compute_eig(n, h):
    myu = np.arange(1, n)
    nyu = np.arange(1, n)
    exact_eigenvalues = []
    for m in myu:
        for v in nyu:
            eig = 4 * h**-2 * (np.sin(m * np.pi * h / 2)**2 + np.sin(v * np.pi * h / 2)**2)
            exact_eigenvalues.append(eig)
    return np.sort(exact_eigenvalues)[::-1] 


T, Q = hessenberg(A, calc_q=True)
T = (T + T.T)/2
T = np.array(T)
principal_submatrix = T[:5, :5]

threshold = 1e-10
principal_submatrix[np.abs(principal_submatrix) < threshold] = 0


print("5x5 Principal Submatrix of T:")
print(principal_submatrix)

eigenvals = divide_and_conquer_eigenvalues(T)


exact_eigenvalues = compute_eig(n, h)

largest_computed = eigenvals[:10]
largest_exact = exact_eigenvalues[:10]

print()
# Display the comparison
print("Comparison:")
print()

print("Computed Eigenvalues: ", largest_computed)
print()
print()
print("Exact Eigenvalues:    ", largest_exact)
print()
print()


