import numpy as np
from numpy.linalg import solve, norm
from numpy.random import rand

def newton_eigenpair(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    x_0 = rand(n)
    x_0 = x_0 / norm(x_0)
    lambda_0 = x_0.T @ A @ x_0
    
    x = x_0
    lambda_value = lambda_0
    
    for k in range(max_iter):
        J = np.zeros((n+1, n+1))
        J[:n, :n] = A - lambda_value * np.eye(n)
        J[:n, -1] = -x
        J[-1, :n] = 2 * x.T
        
        F = np.zeros(n+1)
        F[:n] = A @ x - lambda_value * x
        F[-1] = x.T @ x - 1
        
        try:
            delta = solve(J, -F)
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed to solve linear system - Jacobian may be singular")
        
        s = delta[:n]
        d_lambda = delta[-1]
        
        x = x + s
        lambda_value = lambda_value + d_lambda
        
        if norm(F) < tol:
            return x, lambda_value
            
    raise RuntimeError(f"Failed to converge after {max_iter} iterations")

def power_iteration(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    x = rand(n)
    x = x / norm(x)
    
    for i in range(max_iter):
        x_new = A @ x
        lambda_value = np.dot(x_new, x)
        x_new = x_new / norm(x_new)
        
        if norm(x_new - x) < tol:
            return x_new, lambda_value
            
        x = x_new
        
    raise RuntimeError(f"Power method failed to converge after {max_iter} iterations")

def examine_newton_eigenpair():
    examine_cases = [
        np.array([[2, 1], 
                 [1, 2]]),
        
        np.array([[1, 2],
                 [-1, 4]]),
        
        np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]),
        
        rand(4, 4)
    ]
    
    for i, A in enumerate(examine_cases):
        print(f"\nexamine case {i+1}:")
        print("Matrix A:")
        print(A)
        
        try:
            x_newton, lambda_newton = newton_eigenpair(A)
            x_power, lambda_power = power_iteration(A)
            eigenvals, eigenvecs = np.linalg.eig(A)
            
            print("\nNewton's method:")
            print(f"Eigenvalue: {lambda_newton:.6f}")
            print("Eigenvector:", x_newton)
            residual_newton = norm(A @ x_newton - lambda_newton * x_newton)
            print(f"Residual |Ax - λx|: {residual_newton:.2e}")
            
            print("\nPower method:")
            print(f"Eigenvalue: {lambda_power:.6f}")
            print("Eigenvector:", x_power)
            residual_power = norm(A @ x_power - lambda_power * x_power)
            print(f"Residual |Ax - λx|: {residual_power:.2e}")
            
            print("\nNumpy eigenvalues:", eigenvals)
            
            closest_newton = np.argmin(np.abs(eigenvals - lambda_newton))
            closest_power = np.argmin(np.abs(eigenvals - lambda_power))
            print(f"Closest eigenvalue to Newton: {eigenvals[closest_newton]:.6f}")
            print(f"Closest eigenvalue to Power: {eigenvals[closest_power]:.6f}")
            
        except Exception as e:
            print(f"Error: {e}")

examine_newton_eigenpair()