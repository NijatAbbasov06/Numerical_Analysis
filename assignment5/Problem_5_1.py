import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def f(x):
    return np.sin(10 * x) - x


intervals = [(i * 0.1, (i + 1) * 0.1) for i in range(-20, 60)]
roots = []

for interval in intervals:
    try:
        result = root_scalar(f, bracket=interval, method='bisect')
        if result.converged:
            if not roots or all(abs(root - result.root) > 1e-5 for root in roots):
                roots.append(result.root)
    except ValueError:

        pass


print("Roots of f(x) = sin(10x) - x:")
for root in roots:
    print(root)


x_vals = np.linspace(-2, 6, 1000)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, color='blue', label=r"$f(x) = \sin(10x) - x$")
plt.axhline(0, color='gray', lw=0.8, linestyle='--') 
plt.scatter(roots, [0] * len(roots), color='green', label="Roots") 
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()