import numpy as np
import matplotlib.pyplot as plt

def factored_polynomial(x):
    return (x - 1)**6

def expanded_polynomial(x):
    return x**6 - 6*x**5 + 15*x**4 - 20*x**3 + 15*x**2 - 6*x + 1

# Generate 101 equally spaced points in the interval [0.995, 1.005]
x = np.linspace(0.995, 1.005, 101)

# Compute values for both forms of the polynomial
y_factored = factored_polynomial(x)
y_expanded = expanded_polynomial(x)

# Create the plot
plt.figure(figsize=(7, 7))

# Plot with different line styles, colors, and markers
plt.plot(x, y_factored, linestyle='--', color='black', marker='o', label='Factored form')
plt.plot(x, y_expanded, linestyle='-', color='red', label='Expanded form')

# Labels and title
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Factored vs Expanded Polynomial Forms - Customized Style', fontsize=16)

# Customize legend
plt.legend(loc='upper center', fontsize=12, frameon=True, shadow=True)

# Add a grid
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# Set x and y axis limits and scale
plt.xlim(0.995, 1.005)
plt.ylim(min(np.min(y_factored), np.min(y_expanded)), max(np.max(y_factored), np.max(y_expanded)))

# Add a log scale for y-axis to highlight small differences

# Show the plot
plt.show()

# Print some values for comparison
print("x\t\tFactored\t\tExpanded")
for i in range(0, 101, 10):
    print(f"{x[i]:.6f}\t{y_factored[i]:.6e}\t{y_expanded[i]:.6e}")