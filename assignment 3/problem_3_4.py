import numpy as np


A = np.array([[0.16, 0.10],
              [0.17, 0.11],
              [2.02, 1.29]])

b_a = np.array([0.26, 0.28, 3.31])


x_a, residuals_a, rank_a, s_a = np.linalg.lstsq(A, b_a, rcond=None)

print("Solution for part (a):", x_a)


b_b = np.array([0.27, 0.25, 3.33])


x_b, residuals_b, rank_b, s_b = np.linalg.lstsq(A, b_b, rcond=None)

print("Solution for part (b):", x_b)


print("\nDifference between the solutions:", x_a - x_b)
condition_number = np.linalg.cond(A)
print("Condition number of A", condition_number )