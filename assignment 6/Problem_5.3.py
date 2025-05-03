

import numpy as np
from math import log
from scipy.optimize import fsolve


def function_a(x):
    return x**3 - 2*x - 5


def derivative_function_a(x):
    return 3*x**2 - 2


def function_b(x):
    return np.exp(-x) - x


def derivative_function_b(x):
    return -np.exp(-x) - 1


def function_c(x):
    return (x * np.sin(x) - 1)


def derivative_function_c(x):
    return np.sin(x) + x * np.cos(x)


def function_d(x):
    return x**3 - 3*x**2 + 3*x - 1


def derivative_function_d(x):
    return 3*x**2 - 6*x + 3




def convergence_rate(approximations):

    if len(approximations) < 3:  
        return []
    
    rates = []
    for i in range(len(approximations) - 2):
        x_n = approximations[i]
        x_n1 = approximations[i + 1]
        x_n2 = approximations[i + 2]
        
        # Avoid division by zero or log(0)
        if abs(x_n1 - x_n) == 0:
            continue
            
        try:
            # Calculate rate using consecutive differences
            rate = abs(log(abs(x_n2 - x_n1)) / log(abs(x_n1 - x_n)))
            rates.append(rate)
        except (ValueError, ZeroDivisionError):
            continue
            
    return rates

def bisection(f, a, b, tol=1e-5, max_iter=1000):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None, [], []
    a_n, b_n = a, b
    errors = []
    approximations = []
    for n in range(1, max_iter+1):
        m_n = (a_n + b_n) / 2
        approximations.append(m_n)
        f_m_n = f(m_n)
        errors.append(abs(f_m_n))
        if abs(f_m_n) < tol:
            rates = convergence_rate(approximations)
            return m_n, n, rates
        elif f(a_n) * f_m_n < 0:
            b_n = m_n
        else:
            a_n = m_n
    rates = convergence_rate(approximations)
    return (a_n + b_n) / 2, n, rates

def newton(f, df, x0, tol=1e-5, max_iter=1000):
    x_n = x0
    errors = []
    approximations = [x0]
    for n in range(1, max_iter+1):
        f_x_n = f(x_n)
        df_x_n = df(x_n)
        errors.append(abs(f_x_n))
        if df_x_n == 0:
            print("Derivative zero. No solution found.")
            return None, [], []
        x_n = x_n - f_x_n / df_x_n
        approximations.append(x_n)
        if abs(f_x_n) < tol:
            rates = convergence_rate(approximations)
            return x_n, n, rates
    rates = convergence_rate(approximations)
    return x_n, n, rates

def secant(f, x0, x1, tol=1e-5, max_iter=1000):
    approximations = [x0, x1]
    errors = [abs(f(x0)), abs(f(x1))]
    for n in range(max_iter):
        f_x0, f_x1 = f(x0), f(x1)
        if (f_x1 - f_x0) == 0:
            print("Zero denominator. No solution found.")
            return None, [], []
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x0, x1 = x1, x2
        approximations.append(x2)
        errors.append(abs(f(x2)))
        if abs(f(x2)) < tol:
            rates = convergence_rate(approximations)
            return x2, n + 1, rates
    rates = convergence_rate(approximations)
    return x1, n, rates


def compare_with_fsolve(f, x0, method_name):
    root, info, ier, msg = fsolve(f, x0, full_output=True)
    if ier == 1:
        print(f"{method_name} Library Root: {root[0]}, Function calls: {info['nfev']}")
    else:
        print(f"{method_name} Library failed to converge: {msg}")


print("Problem a\n")
root_bisect, n_bisect, rates_bisect = bisection(function_a, 1, 3)
root_newton, n_newton, rates_newton = newton(function_a, derivative_function_a, 2)
root_secant, n_secant, rates_secant = secant(function_a, 1, 3)

print(f"Bisection root: {root_bisect}, iterations: {n_bisect}, rate: {rates_bisect[-1]}")
print(f"Newton root: {root_newton}, iterations: {n_newton}, rate: {rates_newton[-1]}")
print(f"Secant root: {root_secant}, iterations: {n_secant}, rates: {rates_secant[-1]}")

compare_with_fsolve(function_a, 2, "Problem a")

print("\n")


print("Problem b\n")
root_bisect, n_bisect, rates_bisect = bisection(function_b, 0.5, 1)
root_newton, n_newton, rates_newton = newton(function_b, derivative_function_b, 1)
root_secant, n_secant, rates_secant = secant(function_b, 0.5, 1)

print(f"Bisection root: {root_bisect}, iterations: {n_bisect}, rate: {rates_bisect[-1]}")
print(f"Newton root: {root_newton}, iterations: {n_newton}, rate: {rates_newton[-1]}")
print(f"Secant root: {root_secant}, iterations: {n_secant}, rates: {rates_secant[-1]}")

compare_with_fsolve(function_b, 1, "Problem b")

print("\n")

print("Problem c\n")
root_bisect, n_bisect, rates_bisect = bisection(function_c, 1, 1.2)
root_newton, n_newton, rates_newton = newton(function_c, derivative_function_c, 1.1)
root_secant, n_secant, rates_secant = secant(function_c, 1, 1.2)

print(f"Bisection root: {root_bisect}, iterations: {n_bisect}, rate: {rates_bisect[-1]}")
print(f"Newton root: {root_newton}, iterations: {n_newton}, rate: {rates_newton[-1]}")
print(f"Secant root: {root_secant}, iterations: {n_secant}, rates: {rates_secant[-1]}")
compare_with_fsolve(function_c, 1.1, "Problem c")

print("\n")

print("Problem d\n")
root_bisect, n_bisect, rates_bisect = bisection(function_d, 0.9, 1.2)
root_newton, n_newton, rates_newton = newton(function_d, derivative_function_d, 0.9)
root_secant, n_secant, rates_secant = secant(function_d, 0.9, 1.2)

print(f"Bisection root: {root_bisect}, iterations: {n_bisect}, rate: {rates_bisect[-1]}")
print(f"Newton root: {root_newton}, iterations: {n_newton}, rate: {rates_newton[-1]}")
print(f"Secant root: {root_secant}, iterations: {n_secant}, rates: {rates_secant[-1]}")
compare_with_fsolve(function_d, 0.9, "Problem d")

print("\n")