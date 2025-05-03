from math import *
import matplotlib.pyplot as plt



global relative_err
global abs_err
x = []
y = []
z = []
def error(n):
    global abs_err
    global relative_err
    true_val= factorial(n)
    approx_val = sqrt(2*pi*n)* (n/e)**(n)
    




    abs_err = abs(true_val - approx_val)
    relative_err =  abs_err/true_val
    x.append(n)
    y.append(abs_err)
    z.append(relative_err)




for i in range(1, 11):
    error(i)
    print(i, abs_err, relative_err, '\n')
    






plt.plot(x, y, label='absolute error', color='blue', marker='o')
# plt.plot(x, z, label='relative error', color='red', marker='x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Plots in One Graph')
plt.legend()
plt.show()