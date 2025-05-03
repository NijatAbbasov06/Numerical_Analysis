import numpy as np
import scipy.sparse as sp
import time
from scipy.sparse.linalg import splu
from scipy import linalg


n = 1000

b = np.ones(n)
def my_solve_banded(A, b):
    
    sc = np.zeros((5, len(b)))
    start_time = time.time()
    sc[0, 2:] = A.diagonal(2)
    sc[1, 1:] = A.diagonal(1)
    sc[2, :] = A.diagonal(0)
    sc[3, :-1] = A.diagonal(-1)
    sc[4, :-2] = A.diagonal(-2)
    x = linalg.solve_banded((2, 2), sc, b)
    end_time = time.time()
    return x, end_time - start_time

def expected_accuracy(cond_A):
    epsilon = np.finfo(float).eps
    return -np.log10(cond_A * epsilon)



diagonals = [
    np.ones(n - 2), 
    np.array([-4] * (n - 2) + [-2]), 
    np.array([9] + [6] * (n - 3) + [5, 1]), 
    np.array([-4] * (n - 2) + [-2]), 
    np.ones(n - 2)
]

banded_matrix = sp.diags(diagonals, offsets=np.arange(-2, 3), format='csr')


diagonals_R = [[2] + [1] * (n - 1), np.full(n - 1, -2), np.full(n - 2, 1)]
R_matrix = sp.diags(diagonals_R, offsets=np.arange(0, 3), format='csr')
R = R_matrix.toarray()


C = np.dot(R, R.T)


normal_matrix = banded_matrix.toarray() 
start_time_normal = time.time()
normal_x = np.linalg.solve(normal_matrix, b)
time_normal = time.time() - start_time_normal



x_for_banded, time_banded = my_solve_banded(banded_matrix, b)



print("A = RR^T:", np.allclose(C, normal_matrix))
print("Dense method time:", time_normal, "seconds")
print("Banded method time:", time_banded, "seconds")
print("Solutions are close:", np.allclose(normal_x, x_for_banded))


def solve_for_x(matrix, b_hat):
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b_hat[i] - np.dot(matrix[i, :i], y[:i])) / matrix[i, i]
    return y

def solve_for_y(matrix, y):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(matrix[i, i+1:], x[i+1:])) / matrix[i, i]
    return x

y = solve_for_y(R, b)
x = solve_for_x(R.T, y)



print("To check if the solution from RR^T is equal to x_normal:", np.allclose(x, x_for_banded))
cond_A = np.linalg.cond(normal_matrix)
print("Condition number of A:", cond_A)



def iterative_refinement(A, b_hat, x, max_iter=10):
    r = b_hat - A @ x 
    for _ in range(max_iter):
        delta_x = np.linalg.solve(A, r)  
        x += delta_x                     
        r = b_hat - A @ x                    
    return x, r


error_banded = np.linalg.norm(x_for_banded - normal_x) / np.linalg.norm(normal_x)
error_rrt = np.linalg.norm(x - normal_x) / np.linalg.norm(normal_x)


refined_x_banded, final_residual_banded = iterative_refinement(normal_matrix, b, x_for_banded)
refined_x_rrt, final_residual_rrt = iterative_refinement(normal_matrix, b, x)


error_refined_banded = np.linalg.norm(refined_x_banded - normal_x) / np.linalg.norm(normal_x)
error_refined_rrt = np.linalg.norm(refined_x_rrt - normal_x) / np.linalg.norm(normal_x)

print("Relative error for banded method:", error_banded)
print("Relative error for RR^T method:", error_rrt)
print("Relative error after refinement (banded):", error_refined_banded)
print("Relative error after refinement (RR^T):", error_refined_rrt)

if error_rrt < error_banded:
    print("The RR^T method seems more accurate.")
    less_accurate_method = "banded"
elif error_banded < error_rrt:
    print("The banded method seems more accurate.")
    less_accurate_method = "RR^T"
else:
    print("Both methods have similar accuracy.")
    less_accurate_method = "both"


if less_accurate_method == "banded":
    if error_refined_banded < error_banded:
        print("Iterative refinement improved the accuracy of the banded method.")
    else:
        print("Iterative refinement did not significantly improve the accuracy of the banded method.")
elif less_accurate_method == "RR^T":
    if error_refined_rrt < error_rrt:
        print("Iterative refinement improved the accuracy of the RR^T method.")
    else:
        print("Iterative refinement did not significantly improve the accuracy of the RR^T method.")
else:
    if error_refined_banded < error_banded and error_refined_rrt < error_rrt:
        print("Iterative refinement improved the accuracy of both methods.")
    elif error_refined_banded < error_banded:
        print("Iterative refinement improved the accuracy of the banded method but not the RR^T method.")
    elif error_refined_rrt < error_rrt:
        print("Iterative refinement improved the accuracy of the RR^T method but not the banded method.")
    else:
        print("Iterative refinement did not significantly improve the accuracy of either method.")

print("expected accuracy is", expected_accuracy(cond_A))