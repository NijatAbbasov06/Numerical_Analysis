import numpy as np
from math import cos, exp, log, sin
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x) + 1/(1 +np.exp(-2*x))

def df(x):
    return -sin(x) + 2*exp(-2*x)/((1 + exp(-2*x))**2)

def method_a(x):
    return np.arccos(-1/(1 + exp(-2*x)))

def method_b(x):
    return 0.5*log(-1/(1 + 1/cos(x)))

def newton_method(x):
    return x - f(x)/df(x)

def iterate_method(method, x0, tol=1e-10, max_iteration=100):
    x = x0
    iterations = [x]
    
    for i in range(max_iteration):
        try:
            x_new = method(x)
        except (ValueError, ZeroDivisionError):
            return iterations, False
            
        iterations.append(x_new)
        
        if abs(x_new - x) < tol:
            return iterations, True
            
        x = x_new
        
    return iterations, False

def convergence_inspection(iterations):
    if len(iterations) < 4:
        return None
        
    errors = np.abs(np.diff(iterations))
    convergence_rate = np.log(errors[1:]) / np.log(errors[:-1])
    return np.mean(convergence_rate[-3:])

x0 = 3.0
methods = [
    ("Method A (arccos)", method_a),
    ("Method B (log)", method_b),
    ("Newton's Method", newton_method)
]

print(f"Starting point x0 = {x0}")

plt.figure(figsize=(12, 8))
for name, method in methods:
    iterations, converged = iterate_method(method, x0)
    iterations = np.array(iterations)
    
    print(f"\n{name}:")
    if converged:
        rate = convergence_inspection(iterations)
        final_value = iterations[-1]
        error = abs(f(final_value))
        print(f"Converged to x = {final_value:.10f}")
        print(f"Function value: {f(final_value):.2e}")
        print(f"Iterations needed: {len(iterations)-1}")
        print(f"Empirical convergence rate: {rate:.2f}")
        
        plt.plot(range(len(iterations)), np.abs(iterations - iterations[-1]), 
                label=name, marker='o')
    else:
        print("Failed to converge")

plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)
plt.show()

x = np.linspace(0, 5, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label='f(x)')
plt.axhline(y=0, color='k', linestyle='--')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = cos(x) + 1/(1 + e^(-2x))')
plt.legend()
plt.show()

def theoretical_analysis():
    x_root =  3.0764211637927614  
    print("\nTheoretical Analysis:")
    
    for name, method in methods:
        print(f"\n{name}:")
        if name == "Method A (arccos)":
            print("Fixed point form: x = arccos(-1/(1 + e^(-2x)))")
            derivative = abs(-2*exp(-2*x_root)/(
                (1 + exp(-2*x_root))**2 * 
                np.sqrt(1 - (1/(1 + exp(-2*x_root)))**2)))
            print(f"Derivative at root ≈ {derivative:.4f}")
            
        elif name == "Method B (log)":
            print("Fixed point form: x = 0.5*log(-1/(1 + 1/cos(x)))")
            derivative = abs(0.5 * sin(x_root)/(cos(x_root) * (1 + 1/cos(x_root))))
            print(f"Derivative at root ≈ {derivative:.4f}")
            
        elif name == "Newton's Method":
            print("Quadratically convergent when initial guess is close enough")

theoretical_analysis()