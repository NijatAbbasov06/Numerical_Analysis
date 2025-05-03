from scipy.optimize import fsolve
import numpy as np

def sistem_a(vars):
    x1, x2 = vars
    f1 = x1 + x2 * (x2 * (5 - x2) - 2) - 13
    f2 = x1 + x2 * (x2 * (1 + x2) - 14) - 29
    return [f1, f2]


def sistem_b(vars):
    x1, x2, x3 = vars
    f1 = x1**2 + x2**2 + x3**2 - 5
    f2 = x1 + x2 - 1
    f3 = x1 + x3 - 3
    return [f1, f2, f3]


def sistem_e(vars):
    x1, x2 = vars
    f1 = 10**4 * x1 * x2 - 1
    f2 = np.exp(-x1) + np.exp(-x2) - 1.0001
    return [f1, f2]



initial_values_a = [15, -2]
initial_values_b = [(1 + np.sqrt(3)) / 2, (1 - np.sqrt(3)) / 2, np.sqrt(3)]
initial_values_e = [0, 1]


def solve_with_convergence(sistem, initial_guess, tol=1e-8
, max_iter=100):
    results = []
    convergence_rates = []
    
    def wrap_sistem(vars):
        res = sistem(vars)
        norm = np.linalg.norm(res)  
        results.append(norm)
        

        if len(results) > 1:
            rate = results[-1] / results[-2]
            convergence_rates.append(rate)
        return res

    solution = fsolve(wrap_sistem, initial_guess, xtol=tol, maxfev=max_iter)
    return solution, convergence_rates, results


solution_a, convergence_a, residual_a = solve_with_convergence(sistem_a, initial_values_a)
solution_b, convergence_b, residual_b= solve_with_convergence(sistem_b, initial_values_b)
solution_e, convergence_e, residual_e = solve_with_convergence(sistem_e, initial_values_e)

print("(a): Solution =", solution_a)
print("    Final Residual =", residual_a[-1])
print("    Convergence Rate =", convergence_a[-1])
print("    Iterations =", len(convergence_a))


print("(b) Solution", solution_b)
print("    Final Residual =", residual_b[-1])
print("    Convergence Rte =", convergence_b[-1])
print("    Iterations =", len(convergence_b))



print("(e): Solution =", solution_e)
print("    Final Residual =", residual_e[-1])
print("    Convergence Rate =", convergence_e[-1])
print("    Iterations =", len(convergence_e))


