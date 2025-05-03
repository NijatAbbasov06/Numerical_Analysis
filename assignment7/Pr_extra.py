import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def runge_fn(t):

    return 1 / (1 + 25 * t**2)

def pol_interp(x, x_pnts, y_pnts):

    n = len(x_pnts)
    result = 0.0
    
    for i in range(n):
        term = y_pnts[i]
        for j in range(n):
            if i != j:
                term *= (x - x_pnts[j]) / (x_pnts[i] - x_pnts[j])
        result += term
        
    return result

def compute_interp(n_pnts):


    x_interp = np.linspace(-1, 1, n_pnts)
    y_interp = runge_fn(x_interp)
    

    x_dns = np.linspace(-1, 1, 1000)
    y_org = runge_fn(x_dns)
    

    y_poly = np.array([pol_interp(x, x_interp, y_interp) for x in x_dns])
    
    
    spline = CubicSpline(x_interp, y_interp)
    y_spline = spline(x_dns)
    
    return x_dns, y_org, y_poly, y_spline, x_interp, y_interp


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle("Runge fn interp Comparison")


n1 = 11
x_dns, y_org, y_poly, y_spline, x_interp, y_interp = compute_interp(n1)

ax1.plot(x_dns, y_org, 'k-', label='org fn')
ax1.plot(x_dns, y_poly, 'r-', label='pol interp')
ax1.plot(x_dns, y_spline, 'b-', label='Cubic Spline')
ax1.plot(x_interp, y_interp, 'ko', label='interp pnts')
ax1.set_title(f'n = {n1} pnts')
ax1.set_xlabel('t')
ax1.set_ylabel('f(t)')
ax1.grid(True)
ax1.legend()


n2 = 21
x_dns, y_org, y_poly, y_spline, x_interp, y_interp = compute_interp(n2)

ax2.plot(x_dns, y_org, 'v-', label='org fn')
ax2.plot(x_dns, y_poly, 'b-', label='pol interp')
ax2.plot(x_dns, y_spline, 'b-', label='Cubic Spline')
ax2.plot(x_interp, y_interp, 'ko', label='interp pnts')
ax2.set_title(f'n = {n2} pnts')
ax2.set_xlabel('t')
ax2.set_ylabel('f(t)')
ax2.set_ylim(-3, 1)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


def compute_max_err(y_apprx, y_true):
    return np.max(np.abs(y_apprx - y_true))


x_dns11, y_orig11, y_poly11, y_spline11, x_interp11, y_interp11 = compute_interp(11)
print(f"\nmax abs errs for n=11:")
print(f"pol interp: {compute_max_err(y_poly11, y_orig11):.6f}")
print(f"Cubic spline interp: {compute_max_err(y_spline11, y_orig11):.6f}")


x_dns21, y_orig21, y_poly21, y_spline21, x_interp21, y_interp21  = compute_interp(21)
print(f"\nmax abs errs for n=21:")
print(f"pol interp: {compute_max_err(y_poly21, y_orig21):.6f}")
print(f"Cubic spline interp: {compute_max_err(y_spline21, y_orig21):.6f}")