import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.30, 0.16, 0.01])
y = np.array([0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15])

A = np.column_stack([
    y**2,
    x * y,
    x,
    y,
    np.ones_like(x)
])
X = x**2

param, residuals, rank, s = np.linalg.lstsq(A, X, rcond=None)
a, b, c, d, e = param

print("Original Parameters:")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)

x_plot = np.linspace(-3, 3, 100)
y_plot = np.linspace(-1, 5, 100)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
Z = a * Y_plot**2 + b * X_plot * Y_plot + c * X_plot + d * Y_plot + e - X_plot**2

plt.figure(figsize=(8, 6))
plt.contour(X_plot, Y_plot, Z, levels=[0], colors="red", linewidths=2)
plt.scatter(x, y, color="black", label="Original Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Function and Original Data Points")
plt.grid()
plt.show()

np.random.seed(0)
x_per = x + np.random.uniform(-0.005, 0.005, x.shape)
y_per = y + np.random.uniform(-0.005, 0.005, y.shape)

A_per = np.column_stack([
    y_per**2,
    x_per * y_per,
    x_per,
    y_per,
    np.ones_like(x_per)
])
X_per = x_per**2

param_per, residuals, rank, s = np.linalg.lstsq(A_per, X_per, rcond=None)
a_per, b_per, c_per, d_per, e_per = param_per

print("\nPerturbed Parameters:")
print("a_per =", a_per)
print("b_per =", b_per)
print("c_per =", c_per)
print("d_per =", d_per)
print("e_per =", e_per)

print("\nDifference between Original and Perturbed Parameters:")
print("Delta a =", a_per - a)
print("Delta b =", b_per - b)
print("Delta c =", c_per - c)
print("Delta d =", d_per - d)
print("Delta e =", e_per - e)

Z_per = (a_per * Y_plot**2 + b_per * X_plot * Y_plot +
         c_per * X_plot + d_per * Y_plot + e_per - X_plot**2)

plt.figure(figsize=(8, 6))
plt.contour(X_plot, Y_plot, Z, levels=[0], colors="blue", linewidths=2, label="Original")
plt.contour(X_plot, Y_plot, Z_per, levels=[0], colors="green", linewidths=2, label="Perturbed")
plt.scatter(x, y, color="black", label="Original Points")
plt.scatter(x_per, y_per, color="purple", label="Perturbed Points", marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Original vs Perturbed Data and Fits")
plt.grid()
plt.show()