import numpy as np

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
    
    return A, H_vec

def make_q(H_vec):
    m = len(H_vec[0])
    Q = np.eye(m)
    
    for v in reversed(H_vec):
        Q -= 2 * np.outer(v, v @ Q)
    
    return Q


if __name__ == "__main__":

    m, n = 2, 2
    A_orig = np.array([[1, 1000],
               [0.001, 1]], dtype=np.float64)
    A = A_orig.copy()
    
    R, H_vec = householder_qr(A) 
    Q = make_q(H_vec)
    

    print("Q^T * Q:")
    print(np.round(Q.T @ Q, decimals=6))
    
    A_check = Q @ R[:m, :n]
    print("\nOriginal matrix A:")
    print(A_orig)
    print("\nReconstructed matrix Q * R:")
    print(np.round(A_check, decimals=6))
    
    print("\nOrthogonal matrix Q:")
    print(np.round(Q, decimals=6))
    print("\nUpper triangular matrix R:")
    print(np.round(R[:m, :n], decimals=6))

    print("\nIs Q orthogonal?", np.allclose(Q.T @ Q, np.eye(m)))
    print("Is QR = A?", np.allclose(A_check, A_orig))
    
