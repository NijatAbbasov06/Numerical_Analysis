import numpy as np
from scipy.linalg import solve
from numpy.linalg import cond, norm
import matplotlib.pyplot as plt
import statistics

np.random.seed(42)  
A = np.random.rand(5, 5) 
b = np.random.rand(5)   

x = solve(A, b)

def scale_and_solve(A, b, D):
    DA = D @ A      
    Db = D @ b     
    x_appr = solve(DA, Db) 
    return x_appr, DA


num_scalings = 30
scalings = [np.diag(10 ** np.random.uniform(-5, 5, 5)) for num in range(num_scalings)]
condition_numbers = []
relative_residuals = []
errors = []
standard_deviations = []


for i, D in enumerate(scalings):

    x_appr, DA = scale_and_solve(A, b, D)
    print(D)
    cond_DA = cond(DA)
    relative_residual = norm(b - A @ x_appr) / (norm(A, 2) * norm(x_appr))
    error = norm(x - x_appr) / norm(x)
    condition_numbers.append(cond_DA)
    relative_residuals.append(relative_residual)
    errors.append(error)
    std_dev = np.std(np.diag(D), ddof = 1)
    standard_deviations.append(std_dev)

    print("Condition number:", cond_DA)



max_error = max(errors)
finding_index = errors.index(max_error)


print("This is the Diagonal matrix that results in max error: " , scalings[finding_index], "\n and its index is ", finding_index)

print("\n the relative residual is", relative_residuals[finding_index])

plt.figure(figsize=(9, 8))
plt.scatter(condition_numbers, relative_residuals, color='b')
for i, (cn, rr) in enumerate(zip(condition_numbers, relative_residuals)):
    plt.annotate(str(i), (cn, rr), xytext=(5, 5), textcoords='offset points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Condition Number of DA (log scale)')
plt.ylabel('Relative Residual (log scale)')
plt.title('Relative Residual vs Condition Number')




plt.figure(figsize=(9, 8))
plt.scatter(standard_deviations, condition_numbers, color='r')
for i, (cn, err) in enumerate(zip(standard_deviations, condition_numbers)):
    plt.annotate(str(i), (cn, err), xytext=(5, 5), textcoords='offset points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('standard deviations of the elements of scaling matrix(to measure how skewed our values are)(log scale)')
plt.ylabel('Condition number (log scale)')
plt.title('Relation between skewness of the scaling matrix and condition number')







plt.figure(figsize=(9, 8))
plt.scatter(standard_deviations, errors, color='r')
for i, (cn, err) in enumerate(zip(standard_deviations, errors)):
    plt.annotate(str(i), (cn, err), xytext=(5, 5), textcoords='offset points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Standard Deviations of DA (log scale)')
plt.ylabel('Solution Relative Error (log scale)')
plt.title('Solution Relative Error vs Standard deviations')





plt.figure(figsize=(9, 8))
plt.scatter(condition_numbers, errors, color='r')
for i, (cn, err) in enumerate(zip(condition_numbers, errors)):
    plt.annotate(str(i), (cn, err), xytext=(5, 5), textcoords='offset points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Condition Number of D(log scale)')
plt.ylabel('Solution Relative Error (log scale)')
plt.title('Solution Relative Error vs Condition Number')






plt.figure(figsize=(9, 8))
plt.scatter(relative_residuals, errors, color='r')
for i, (cn, err) in enumerate(zip(relative_residuals, errors)):
    plt.annotate(str(i), (cn, err), xytext=(5, 5), textcoords='offset points')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Relative Residual (log scale)')
plt.ylabel('Solution Relative Error (log scale)')
plt.title('Solution Relative Error vs Relative Residuals')



print("Original matrix A:")
print(A)



plt.show()