import numpy as np
import time

def newton_basis(new_x, degree, x_data):
    result = 1
    for i in range(degree):
        result *= (new_x - x_data[i])
    return result


def compute_newton_polynomial(x_data, y_data):
    L = np.zeros((len(x_data), len(x_data)))
   
    
    for i in range(len(x_data)):
        for j in range(i+1):
            basis = newton_basis(x_data[i], j, x_data)
            L[i,j] = basis
    y = np.array(y_data)
    x = np.linalg.solve(L, y)
    return x



def compute_newton_new_data(pol_param, x_data, new_x, new_y):
    n = len(pol_param)
    p_j = evaluate_pol(n, pol_param, new_x, x_data)  
    basis_value = newton_basis(new_x, len(x_data), x_data)
    
    new_param = (new_y - p_j) / basis_value  
    

    np.append(pol_param, new_param)
    return pol_param

def evaluate_pol(n, pol_param, t, x_data ):
    p = pol_param[0]
    for i in range(1, n):
        p += pol_param[i] * newton_basis(t, i, x_data[:i])
    return p 

def compute_newton_recursive(x_data, y_data):
    n = len(x_data)
    
    if n == 1:
        return [y_data[0]]
    
    
    pol_param = compute_newton_recursive(x_data[:-1], y_data[:-1])
    

    new_coeff = (y_data[-1] - evaluate_pol(len(pol_param), pol_param, x_data[-1], x_data[:-1])) / \
                newton_basis(x_data[-1], len(pol_param), x_data[:-1])
    
    return np.append(pol_param, new_coeff)

def gen_x_and_y_Data(num_points, x_ran, func_y):
    x_data = sorted(np.random.uniform(*x_ran, num_points))
    y_data = [func_y(x) for x in x_data]
    return np.array(x_data), np.array(y_data)


x_data = np.array([103, 2, 3, 4, 59, 8, 100, 31, 72])
y_data = np.array([129, 4, 9, 18, 84, 94, 1, 69, 151])

print("x_data:", x_data)
print("y_data:", y_data)


start_a = time.time()
pol_param_a = compute_newton_polynomial(x_data, y_data)
end_a = time.time()
print("\n(a)")
print(f"Polynomial parameters (coefficients): {pol_param_a}")
print(f"Time taken for (a): {end_a - start_a:.6f} seconds")

new_x, new_y = 12, 144
start_b = time.time()
pol_param_b = compute_newton_new_data(pol_param_a, x_data, new_x, new_y)
end_b = time.time()
print("\n(b)")
print(f"Polynomial parameters (coefficients): {pol_param_b}")
print(f"Time taken for (b): {end_b - start_b:.6f} seconds")


start_c = time.time()
pol_param_c = compute_newton_recursive(x_data, y_data)
end_c = time.time()
print("\n(c)")
print(f"Polynomial parameters (coefficients): {pol_param_c}")
print(f"Time taken for (c): {end_c - start_c:.6f} seconds")