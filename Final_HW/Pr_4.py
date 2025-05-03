import numpy as np
import matplotlib.pyplot as plt


def lotka_volterra(y):
    alpha1 = 1
    beta1 = 0.1
    alpha2 = 0.5
    beta2 =  0.02
    y1, y2 = y
    dy1_dt = y1 * (alpha1 - beta1 * y2)
    dy2_dt = y2 * (beta2* y1 - alpha2)
    return np.array([dy1_dt, dy2_dt])

def leslie_gower(y):
    alpha1 = 1
    beta1 = 0.1
    alpha2 = 0.5
    beta2 = 10
    
    y1, y2 = y
    dy1_dt_leslie = y1 * (alpha1 - beta1 * y2)
    dy2_dt_leslie = y2 * (alpha2 - beta2* y2 / y1)

    
    return np.array([dy1_dt_leslie, dy2_dt_leslie])



def Kermack(y):
    d = 5

    y1, y2, y3 = y

    dy1_dt_kermack = -y1*y2
    dy2_dt_kermack = y1 * y2 - d*y2
    dy3_dt_kermack = d *y2

    return np.array([dy1_dt_kermack, dy2_dt_kermack, dy3_dt_kermack])


def adaptive_runge_kutta_4(f, t0, y0, h, t_end, tol=1e-6, max_h=0.01, min_h=1e-8):

    times = [t0]
    solutions = [y0]
    
    t = t0
    y = np.array(y0, dtype=float)
    i = 0
    while t < t_end:
        i = i+1

        if t + h > t_end:
            h = t_end - t  
        k1 = f(y)
        k2 = f(y + h * k1 / 2)
        k3 = f(y + h * k2 / 2)
        k4 = f(y + h *k3)
        y_new = y + h*(k1 + 2 * k2 + 2 * k3 + k4) / 6




        t += h
        y = y_new
        times.append(t)
        solutions.append(y.copy())


    return np.array(times), np.array(solutions)


alpha1, beta1, alpha2, beta2 = 1, 0.1, 0.5, 0.02
y0 = [200, 10] 
t0, t_end, h = 0, 70, 0.01


y_kermack_0 = [95, 5, 0]
t_kermack_0, t_kermack_end, h_kermack = 0, 1, 0.01

times, solutions = adaptive_runge_kutta_4(
    lotka_volterra, t0, y0, h, t_end)

times_b, solutions_b = adaptive_runge_kutta_4(
    leslie_gower, t0,y0, h, t_end)

times_kermack, solutions_kermack = adaptive_runge_kutta_4(
    Kermack, t_kermack_0, y_kermack_0, h_kermack, t_kermack_end )

y1, y2 = solutions[:, 0], solutions[:, 1]

y_leslie_1, y_leslie_2 = solutions_b[:, 0], solutions_b[:, 1]

y_kermack_1, y_kermack_2, y_kermack_3 = solutions_kermack[:, 0], solutions_kermack[:, 1], solutions_kermack[:, 2]


plt.figure(figsize=(10, 6))
plt.plot(times_kermack, y_kermack_1, label='susceptibles', lw=2)
plt.plot(times_kermack, y_kermack_2, label='infectives_circul', lw=2)
plt.plot(times_kermack, y_kermack_3, label='infectives_removed', lw=2)
plt.xlabel('Time (t)')
plt.ylabel('Population')
plt.title('Popul vs. Time')
plt.legend()
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(times, y1, label='Prey(y1)', lw=2)
plt.plot(times, y2, label='Predator(y2)', lw=2)
plt.xlabel('Time (t)')
plt.ylabel('Popul')
plt.title('Popul vs. Time(Runge-Kutta)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(y1, y2, lw=2)
plt.xlabel('Prey(y1)')
plt.ylabel('Predator(y2)')
plt.title('Phase portrait (prey(x axis) vs. predator(y axis)) with RK4')
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(times_b, y_leslie_1, label='Prey(y1)', lw=2)
plt.plot(times_b, y_leslie_2, label='Predator(y2)', lw=2)
plt.xlabel('Time (t)')
plt.ylabel('Popul')
plt.title('Popul vs. Time (Leslie Gower method)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(y_leslie_1, y_leslie_2, lw=2)
plt.xlabel('Prey (y1)')
plt.ylabel('Predator(y2)')
plt.title('Phase Portrait (y1 vs. y2) with Leslie Gower method')
plt.grid(True)
plt.show()

