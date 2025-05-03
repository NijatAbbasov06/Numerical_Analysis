import numpy as np
import time


def gauss_elim_without_pivot(A, b):
    n = A.shape[0]
    for k in range(n-1):
        for i in range(k+1, n):
            m_ik = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= m_ik * A[k, j]
            b[i] -= m_ik * b[k]

    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - sum(A[k, j] * x[j] for j in range(k+1, n))) / A[k, k]
    return x


def gauss_elim_pivot(A, b):
    n = A.shape[0]
    for k in range(n-1):
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[k], b[pivot] = b[pivot], b[k]

        for i in range(k+1, n):
            m_ik = A[i, k] / A[k, k]
            A[i, k:] -= m_ik * A[k, k:]
            b[i] -= m_ik * b[k]

    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - sum(A[k, j] * x[j] for j in range(k+1, n))) / A[k, k]
    return x

def generate_random_system(n):
    A = np.random.rand(n, n)
    x_true = np.random.rand(n)
    b = np.dot(A, x_true)
    return A, b, x_true

n = 50
num_tests = 7
without_pivot_errors = []
without_pivot_relative_errors = []
without_pivot_residuals = []
pivot_errors = []
pivot_relative_errors = []
pivot_residuals = []

for nijat in range(num_tests):
    A, b, x_true = generate_random_system(n)
    A_copy, b_copy = A.copy(), b.copy()
    start_time = time.time()
    x_without_pivot = gauss_elim_without_pivot(A_copy, b_copy)
    end_time = time.time()
    error_without_pivot = np.linalg.norm(x_without_pivot - x_true)
    relative_error_without_pivot = np.linalg.norm(x_without_pivot - x_true) / np.linalg.norm(x_true)
    residual_without_pivot = np.linalg.norm(np.dot(A, x_without_pivot) - b)
    without_pivot_errors.append(error_without_pivot)
    without_pivot_relative_errors.append(relative_error_without_pivot)
    without_pivot_residuals.append(residual_without_pivot)
    A_copy, b_copy = A.copy(), b.copy()
    start_time = time.time()
    x_pivot = gauss_elim_pivot(A_copy, b_copy)
    end_time = time.time()
    error_pivot = np.linalg.norm(x_pivot - x_true)
    relative_error_pivot = np.linalg.norm(x_pivot - x_true) / np.linalg.norm(x_true)
    residual_pivot = np.linalg.norm(np.dot(A, x_pivot) - b)
    pivot_errors.append(error_pivot)
    pivot_relative_errors.append(relative_error_pivot)
    pivot_residuals.append(residual_pivot)


print("\nWithout Pivot:")
print("Error(avg):", np.mean(without_pivot_errors))
print("Relative Error(avg):", np.mean(without_pivot_relative_errors))
print("Residual(avg):", np.mean(without_pivot_residuals))

print("\nPivot:")
print("Error(avg):", np.mean(pivot_errors))
print("Relative Error(avg):", np.mean(pivot_relative_errors))
print("Residual(avg):", np.mean(pivot_residuals))
