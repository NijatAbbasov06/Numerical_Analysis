import numpy as np
import matplotlib.pyplot as plt

def actual_function(x):
    return np.sin(x) - x

def taylor_series(x):
    return (-1)*((x**3 / 6) - (x**5 / 120) + (x**7 / 5040) - (x**9 / 362880))

def nested_multiplication(x):
    return (-1)*(x**3 / 6) * (1 - (x**2 / 20) * (1 - (x**2 / 42) * (1 - (x**2 / 72))))

# Generate x values
x = np.linspace(-15, 1, 15)

# Calculate y values for each method
y_actual = actual_function(x)
y_taylor = taylor_series(x)
y_nested = nested_multiplication(x)

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(x, y_actual, label='Actual: sin(x) - x')
plt.plot(x, y_taylor,"--", label='Taylor series')
plt.plot(x, y_nested, "--", label='Nested multiplication')
plt.plot(x, y_actual - y_taylor, "--" ,label='Error: Actual - Taylor', linestyle='--')
plt.plot(x, y_actual - y_nested, "--",label='Error: Actual - Nested', linestyle='--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of sin(x) - x Approximations')
plt.legend()
plt.grid(True)

plt.show()

# Print max errors
print(f"Max error (Taylor): {np.max(np.abs(y_actual - y_taylor))}")
print(f"Max error (Nested): {np.max(np.abs(y_actual - y_nested))}")