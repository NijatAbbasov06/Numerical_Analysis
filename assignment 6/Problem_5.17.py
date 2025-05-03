import numpy as np
from numpy.linalg import solve


def cubic_solve(a, b, c, tol=1e-10, max_iter=100):
    
    def newton_step(x_1, x_2, x_3):
        sum_x = x_1 + x_2 + x_3
        sum_product = x_1*x_2 + x_2*x_3 + x_3*x_1
        product = x_1 * x_2 * x_3
        
        F = np.array([
            -sum_x - a,
            sum_product - b,
            -product - c
        ], dtype=complex)
        
        J = np.array([
            [-1, -1, -1],
            [x_2 + x_3, x_1 + x_3, x_1 + x_2],
            [-x_2*x_3, -x_1*x_3, -x_1*x_2]
        ], dtype=complex)

        dx = solve(J, -F)
        return dx
    
    x_1 = complex(1, 0)
    x_2 = complex(-0.5, 0.866) 
    x_3 = complex(-0.5, -0.866)  

    for element in range(max_iter):
        dx = newton_step(x_1, x_2, x_3)

        x_1 += dx[0]
        x_2 += dx[1]
        x_3 += dx[2]
        
        if max(abs(dx)) < tol:
            return (x_1, x_2, x_3)
    
    raise RuntimeError(f"Failed to converge after {max_iter} iterations")

def examine_solver():

    examine_cases = [
        [0, 0, -1],  
        [0, -2, 0],       
        [1, 1, 1],        
        [-3, 3, -1]          
    ]
    
    for coefficients in examine_cases:
        a, b, c = coefficients
        print(f"\nexamining x³ + {a}x² + {b}x + {c} = 0")
        
        try:
            roots = cubic_solve(a, b, c)
            print("Our roots:", [complex(round(r.real, 6), round(r.imag, 6)) for r in roots])
            

            sum_x = sum(roots)
            sum_product = sum(r1*r2 for i, r1 in enumerate(roots) 
                         for j, r2 in enumerate(roots) if i < j)
            product = np.prod(roots)
            
            print("Vieta's relations verification:")
            print(f"Sum = {-a:.6f} (should be {-a})")
            print(f"Sum of products = {sum_product:.6f} (should be {b})")
            print(f"Product = {-product: .6f} (should be {c})")
            

            numpy_roots = np.roots([1, a, b, c])
            print("NumPy roots:", [complex(round(r.real, 6), round(r.imag, 6)) for r in numpy_roots])
            
        except Exception as e:
            print(f"Error: {e}")

examine_solver()