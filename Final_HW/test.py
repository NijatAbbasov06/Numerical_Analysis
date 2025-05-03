import numpy as np
from scipy.linalg import hessenberg
from scipy.linalg import eig
from scipy.linalg import block_diag

n = 6  
h = 1/n

# Matrix construction (unchanged)
B = np.diag([4] * (n-1)) + np.diag([-1] * (n-2), k=1) + np.diag([-1] * (n-2), k=-1)

def newton_method(func, jacobian, guess, tol=1e-8, max_iter=100):
    variables = np.array([guess], dtype=float)
    for iteration in range(max_iter):
        f_val = func(variables[0])
        j_val = jacobian(variables[0])
        if abs(j_val) < 1e-10:  # Avoid division by near-zero
            break
        delta = f_val/j_val
        variables[0] -= delta
        if abs(delta) < tol:
            return variables[0], iteration + 1
    return variables[0], max_iter

# Matrix A construction (unchanged)
A = np.zeros(((n-1)**2, (n-1)**2))
block_size = n-1
for i in range(block_size):
    for j in range(block_size):
        idx_row = i * block_size + j
        A[idx_row, idx_row] = B[j, j]
        if j < block_size - 1:
            A[idx_row, idx_row + 1] = B[j, j+1]
            A[idx_row + 1, idx_row] = B[j+1, j]
        if i < block_size - 1:
            A[idx_row, idx_row + block_size] = -1
            A[idx_row + block_size, idx_row] = -1
A = A * n**2

def divide_and_conquer_eigenvalues(T):
    n = T.shape[0]
    if n <= 2:  # Base case
        return np.linalg.eigvals(T)
        
    k = n // 2
    beta = T[k, k-1]  # Get off-diagonal element
    
    # Split matrix without modifying yet
    T1 = T[:k, :k].copy()
    T2 = T[k:, k:].copy()
    
    # Compute eigenvalues and eigenvectors of subproblems
    eig_T1, Q1 = eig(T1)
    eig_T2, Q2 = eig(T2)
    
    # Create block diagonal matrices
    Q = block_diag(Q1, Q2)
    D = np.concatenate([eig_T1, eig_T2])
    
    # Create vector u
    u = np.zeros(n)
    u[k-1] = u[k] = 1
    
    # Compute v = Q^T * u
    v = Q.T @ u
    
    # Find all eigenvalues using secular equation
    all_eigenvalues = []
    D = np.real(D)  # Ensure D is real
    D_sorted = np.sort(D)
    
    # Function to evaluate secular equation
    def secular_func(lambda_val):
        return 1 + beta * np.sum(v**2 / (D - lambda_val))
    
    def secular_deriv(lambda_val):
        return -beta * np.sum(v**2 / (D - lambda_val)**2)
    
    # Find roots between each pair of diagonal elements and beyond
    for i in range(len(D_sorted)-1):
        # Initial guess between consecutive eigenvalues
        guess = (D_sorted[i] + D_sorted[i+1])/2
        root, _ = newton_method(secular_func, secular_deriv, guess)
        if root > D_sorted[i] and root < D_sorted[i+1]:  # Valid root
            all_eigenvalues.append(root)
    
    # Find largest eigenvalue
    if len(D_sorted) > 0:
        guess = D_sorted[-1] + 1.0
        root, _ = newton_method(secular_func, secular_deriv, guess)
        if root > D_sorted[-1]:  # Valid root
            all_eigenvalues.append(root)
    
    # Find smallest eigenvalue
    if len(D_sorted) > 0:
        guess = D_sorted[0] - 1.0
        root, _ = newton_method(secular_func, secular_deriv, guess)
        if root < D_sorted[0]:  # Valid root
            all_eigenvalues.append(root)
            
    return np.sort(np.array(all_eigenvalues))[::-1]

def compute_exact_eigenvalues(n, h):
    mu = np.arange(1, n)
    nu = np.arange(1, n)
    exact_eigenvalues = []
    for m in mu:
        for v in nu:
            eigenvalue = 4 * h**-2 * (np.sin(m * np.pi * h / 2)**2 + np.sin(v * np.pi * h / 2)**2)
            exact_eigenvalues.append(eigenvalue)
    return np.sort(exact_eigenvalues)[::-1]

# Main execution
T, Q = hessenberg(A, calc_q=True)
T = (T + T.T)/2
T = np.array(T)

# Display principal submatrix
principal_submatrix = T[:5, :5]
threshold = 1e-10
principal_submatrix[np.abs(principal_submatrix) < threshold] = 0
print("5x5 Principal Submatrix of T:")
print(principal_submatrix)

# Compute and compare eigenvalues
eigenvals = divide_and_conquer_eigenvalues(T)
exact_eigenvalues = compute_exact_eigenvalues(n, h)

largest_computed = eigenvals[:10]
largest_exact = exact_eigenvalues[:10]

print("\nComparison:")
print("\nComputed Eigenvalues:", largest_computed)
print("\nExact Eigenvalues:   ", largest_exact)
print("\nMaximum absolute difference:", np.max(np.abs(largest_computed - largest_exact)))