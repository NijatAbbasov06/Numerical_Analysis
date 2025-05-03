import numpy as np



def simp(f, a, b, n=10000):
    if n % 2 == 1:  
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:n:2])
    integral += 2 * np.sum(y[2:n-1:2])  
    return integral * h / 3


def f_a(x):
    return np.sqrt(x**3)

def f_b(x):
    return 1 / (1 + 10 * x**2)

def f_c(x):
    return (np.exp(-9 * x) + np.exp(-1024 * (x - 0.25)**2)) / np.sqrt(np.pi)

def f_d(x):
    return 50 / (np.pi * (2500 * x**2 + 1))

def f_e(x):
    return 1 / np.sqrt(np.abs(x))
 
def f_f(x):
    return 25 * np.exp(-25 * x)

def f_g(x):
    return np.log(x)


print("Results (Simpson's Rule):")
print(f"(a): {simp(f_a, 0, 1):.6f}")
print(f"(b): {simp(f_b, 0, 1):.6f}")
print(f"(c): {simp(f_c, 0, 1):.6f}")
print(f"(d): {simp(f_d, 0, 10):.6f}")
print(f"(e): {simp(f_e, -9, 100):.6f}")
print(f"(f): {simp(f_f, 0, 10):.6f}")
print(f"(g): {simp(f_g, 1e-10, 1):.6f}")  