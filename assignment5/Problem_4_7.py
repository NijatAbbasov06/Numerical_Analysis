import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr, norm, eig

def generate_matrix(n):
    B = np.random.rand(n, n)
    Q, _ = qr(B)
    D = np.diag(np.arange(1, n + 1))
    A = Q @ D @ Q.T
    return A

def lanczos_iteration(A, iterat_num):
    n = A.shape[0]
    q_prev = np.zeros(n)
    beta_prev = 0
    q = np.random.rand(n)
    q = q / norm(q)
    alphas = []
    betas = []
    q_vectors = [q]
    
    for k in range(iterat_num):
        u = A @ q
        alpha = q.T @ u
        alphas.append(alpha)
        
        u = u - beta_prev * q_prev - alpha * q
        beta = norm(u)
        
        if beta < 1e-10:
            break
        
        betas.append(beta)
        
        q_prev = q
        q = u / beta
        q_vectors.append(q)
        
        beta_prev = beta

    k_actual = len(alphas)
    T_k = np.diag(alphas[:k_actual]) + np.diag(betas[:k_actual-1], 1) + np.diag(betas[:k_actual-1], -1)
    return T_k, q_vectors

def find_ritz_values(A, num_iterations):
    ritz_values_list = []
    
    for k in range(1, num_iterations + 1):
        T_k, _ = lanczos_iteration(A, k)
        ritz_value = eig(T_k)[0]
        ritz_values_list.append(np.sort(ritz_value))
    
    return ritz_values_list

def plot_ritz(ritz_values_list):
    num_iterations = len(ritz_values_list)
    plt.figure(figsize=(10, 6))
    
    for i in range(num_iterations):
        iteration_num = i + 1
        ritz_values = ritz_values_list[i]
        plt.plot(ritz_values, [iteration_num] * len(ritz_values), 'o') 
    
    plt.ylabel("Iteration Number")
    plt.xlabel("Ritz Values")
    plt.title("Convergence of Ritz Values in Lanczos Algorithm")
    plt.grid(True)
    plt.show()

n = 50
num_iterations = n

A = generate_matrix(n)
ritz_values_list = find_ritz_values(A, num_iterations)
plot_ritz(ritz_values_list)