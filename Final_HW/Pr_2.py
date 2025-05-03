import numpy as np
from scipy.optimize import root


def sistem(variables):
    x, y, z = variables
    return [
        16 * x**4 + 16 * y**4 + z**4 - 16,
        x**2 + y**2 + z**2 - 3,
        x**3 - y
    ]

def sistem_5_24(variables):

    x1, x2, w1, w2 = variables
    return[
        w1 + w2 - 2, 
        w1*x1 + w2*x2, 
        w1*x1**2 + w2 * x2**2 - 2/3,
        w1*x1**3 + w2*x2**3
        ]

def sistem_extra(variables):

    x1, x2, x3, w1, w2, w3 = variables
    return[
        w1 + w2 +w3 - 2, 
        w1*x1 + w2*x2 + w3*x3, 
        w1*x1**2 + w2 * x2**2 + w3 * x3**2 - 2/3,
        w1*x1**3 + w2*x2**3 + w3*x3**3,
        w1 * x1**4 + w2*x2**4 + w3*x3**4 - 2/5,
        w1*x1**5 + w2*x2**5 + w3*x3**5
        ]

def jacobian(variables):
    x, y, z = variables
    return [
        [64 * x**3, 64 * y**3, 4 * z**3],
        [2 * x, 2 * y, 2 * z],
        [3 * x**2, -1, 0]
    ]

def newton_m(func, jacob, guess, tol=1e-8, max_iter=100):
    variables = np.array(guess, dtype=float)
    rsdls = [] 
    for iter in range(max_iter):
        f_val = np.array(func(variables))
        rsdls.append(np.linalg.norm(f_val))
        j_val = np.array(jacob(variables))
        delta = np.linalg.solve(j_val, -f_val)
        variables += delta
        if np.linalg.norm(delta, ord=2) < tol:

            cnvrgnc_rates = [
                rsdls[i] / rsdls[i - 1] if i > 0 else None
                for i in range(1, len(rsdls))
            ]
            return variables, iter + 1, rsdls, cnvrgnc_rates


initial_values = [1, 1, 1]
initial_values_5_24 = [-1, 1, 1, 1]
initial_values_extra = [-0.7, 0.7, 0, 0.5, 0.5, 0.9]
sol_newton, iter_newton, rsdls_newton, rts_newton = newton_m(sistem, jacobian, initial_values)



result_scipy_newton = root(sistem, initial_values, method='hybr')
solution_scipy_newton = result_scipy_newton.x
iterations_scipy_newton = result_scipy_newton.nfev

result_5_24 = root(sistem_5_24, initial_values_5_24, method='hybr')
result_5_24_sol = result_5_24.x
result_5_24_iter = result_5_24.nfev


Extra = root(sistem_extra, initial_values_extra, method='hybr')
Extra_sol = Extra.x
Extra_iter = Extra.nfev


print("\nComparison of solutions:")
print(f"Custom Newton's: {sol_newton}")
print(f"Library routine: {solution_scipy_newton}")
print("\nComparison of iterations:")
print(f"Custom Newton's: {iter_newton}")
print(f"Library routine: {iterations_scipy_newton}")
print("\nConvergence rates for Custom Newton's:")
print(rts_newton)

print()
print("Problem 5_24 solution")
print(f"Problem 5_24 solution [x1, x2, w1, w2]: {result_5_24_sol}")
print(f"Problem 5_24 iterations: {result_5_24_iter}")

print()
print("Extra")
print(f"Extra solution [x1, x2, x3, w1, w2, w3]: {Extra_sol}")
print(f"Extra iterations: {Extra_iter}")