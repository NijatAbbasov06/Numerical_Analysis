import numpy as np
from numpy.linalg import eig, cond, norm

A = np.array([[1, 1000], [0.001, 1]])
B = np.array([[1, 1000], [0, 1]])

eigenval_A, right_eigenvec_A = eig(A)
_, left_eigenvec_A = eig(A.T)

print("Eigenvalues of A:", eigenval_A)
print("Eigenvectors of A:\n", right_eigenvec_A)

cond_num_eigenvec_A = cond(right_eigenvec_A)
print("\nCondition number of eigenvector matrix of A:", cond_num_eigenvec_A)

abs_cond_num = [
    (norm(left_eigenvec_A[:, i]) * norm(right_eigenvec_A[:, i])) / abs(np.dot(left_eigenvec_A[:, i].conj().T, right_eigenvec_A[:, i]))
    for i in range(len(eigenval_A))
]
print("Absolute condition numbers for each eigenvalue of A:", abs_cond_num)

eigenval_B, eigenvec_B = eig(B)
print("\nEigenvalues of B:", eigenval_B)

print("\nComparison of Eigenvalues of A and B:")
for i in range(len(eigenval_A)):
    print("Eigenvalue of A:", eigenval_A[i], ", Eigenvalue of B:", eigenval_B[i])