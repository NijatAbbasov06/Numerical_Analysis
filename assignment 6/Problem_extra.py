import cupy as cp
from cupyx.scipy.sparse import diags
import time

def theor_eigenvalues(n, num_eigenvalues=20):
    h = 1 / n
    vals = []
    for mu in range(1, n):
        for v in range(1, n):
            lmv = 4 / (h * h) * (
                cp.sin(cp.pi * v * h / 2)**2 +
                cp.sin(cp.pi * mu * h / 2)**2
            )
            vals.append(lmv.item())
    return sorted(vals, reverse=True)[:num_eigenvalues]

def household_sim(A):
    n = A.shape[0]
    H = A.copy()
    Q = cp.eye(n)

    for i in range(n-2):
        x = H[i+1:, i]
        if cp.linalg.norm(x) < 1e-12:
            continue
        u = x.copy()
        u[0] += cp.sign(x[0]) * cp.linalg.norm(x)
        u /= cp.linalg.norm(u)
        P = cp.eye(n-i-1) - 2 * cp.outer(u, u)
        
        H[i+1:, i:] = P @ H[i+1:, i:]
        H[i:, i+1:] = H[i:, i+1:] @ P.T
        Q[:, i+1:] = Q[:, i+1:] @ P.T

    return H, Q
def create_matrix_A(n):
    h = 1/n
    N = (n-1)*(n-1)
    B = cp.zeros((n-1, n-1))
    cp.fill_diagonal(B, 4)
    
  
    for i in range(n-2):
        B[i, i+1] = -1
        B[i+1, i] = -1
    
    A = cp.zeros((N, N))
    blck = n-1
    
    for i in range(0, N, blck):
        A[i:i+blck, i:i+blck] = B
    
    for i in range(0, N-blck, blck):
        A[i:i+blck, i+blck:i+2*blck] = -cp.eye(blck)
        A[i+blck:i+2*blck, i:i+blck] = -cp.eye(blck)
    
    return A/(h*h)


def lanczos_iteration(A, k):
    n = A.shape[0]
    q = cp.random.rand(n)
    q = q / cp.linalg.norm(q)
    
    Q = cp.zeros((n, k))
    alpha = cp.zeros(k)
    beta = cp.zeros(k-1)
    
    Q[:, 0] = q
    
    for j in range(k-1):
        w = A @ Q[:, j]
        alpha[j] = cp.dot(Q[:, j].T, w)
        if j > 0:
            w = w - beta[j-1] * Q[:, j-1]
        w = w - alpha[j] * Q[:, j]
        beta[j] = cp.linalg.norm(w)
        if beta[j] < 1e-6:
            break
        Q[:, j+1] = w / beta[j]

    Hk = cp.diag(alpha[:k]) + cp.diag(beta[:k-1], 1) + cp.diag(beta[:k-1], -1)
    eigenvals, eigenvecs = cp.linalg.eigh(Hk)
    
    return cp.asnumpy(eigenvals), cp.asnumpy(Q[:, :k] @ eigenvecs)

def orthogonal_iteration(A, p, eigs_true, max_iter=1000):
    n = A.shape[0]
    X = cp.random.randn(n, p)
    lst = []
    
    for i in range(max_iter):
        Q, R = cp.linalg.qr(X)
        X = A @ Q
        
        eigs = cp.abs(cp.diag(R))
        eigs = sorted(cp.asnumpy(eigs), reverse=True)[:20]
        err = max(abs(e - t) for e, t in zip(eigs, eigs_true))
        lst.append(eigs)

        if err < 1e-6:
          print(f"Orthogonal iteration converged at iteration {i} with max error {err:.2e}")
          break
    
        if i == max_iter - 1:
           print("Orthogonal iteration did not converge")
           print("Orthogonal iteration error:", err)

    return X, lst
    


def qr_iteration_with_shift(H, eigs_true, maxit=1000, tol=1e-10):
    n = H.shape[0]
    Q = cp.eye(n)
    hist = []
    deflation_threshold = 1e-10
    
    for t in range(maxit):
        for i in range(n-1, 0, -1):
            if abs(H[i, i-1]) < deflation_threshold * (abs(H[i,i]) + abs(H[i-1,i-1])):
                H[i, i-1] = 0
        
        a = H[n-2:n, n-2:n]
        d = (a[0,0] - a[1,1]) / 2
        mu = a[1,1] + d - cp.sign(d) * cp.sqrt(d*2 + a[0,1]*2)
        
        Qt, R = cp.linalg.qr(H - mu * cp.eye(n))
        H = R @ Qt + mu * cp.eye(n)
        Q = Q @ Qt
        
        eigs = -cp.sort(-cp.abs(cp.diag(H)))[:20]
        hist.append(cp.asnumpy(eigs))
        
        err = cp.max(cp.abs(eigs - cp.array(eigs_true[:20])))
        if err < tol:
            print(f"QR iteration converged at iteration {t} with error {err:.2e}")
            return H, Q, cp.array(hist)
            
    print("QR iteration did not converge")
    return H, Q, cp.array(hist)


n = 100
h = 1/n
N = (n-1)**2
A = create_matrix_A(n)



eigs = theor_eigenvalues(n, num_eigenvalues=20)

t0 = time.time()
k = 100
lanczos_vals, _ = lanczos_iteration(A, k)
lanczos_vals = -cp.sort(-lanczos_vals)[:20]
lanczos_time = time.time() - t0


t0 = time.time()
empty, ortho_vals = orthogonal_iteration(A, 20, eigs)

ortho_time = time.time() - t0


t0 = time.time()

H = household_sim(A)[0]
empty1, empty2, qr_vals = qr_iteration_with_shift(H, eigs)
qr_vals = qr_vals[-1]
qr_time = time.time() - t0

lanczos_vals = cp.asarray(lanczos_vals)
ortho_vals = cp.asarray(ortho_vals)
qr_vals = cp.asarray(qr_vals)
eigs = cp.asarray(eigs)


print("\nMaximum errors compared to exact eigenvalues:")
print(f"Lanczos method: {cp.max(cp.abs(lanczos_vals - eigs)):.2e}")

print(f"QR iteration: {cp.max(cp.abs(qr_vals - eigs)):.2e}")

print("\nComputation times:")
print(f"Lanczos method: {lanczos_time:.2f} seconds")
print(f"Orthogonal iteration: {ortho_time:.2f} seconds")
print(f"QR iteration: {qr_time:.2f} seconds")