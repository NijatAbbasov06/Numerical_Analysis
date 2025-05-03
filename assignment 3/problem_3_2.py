import numpy as np 



A = np.array([(1, 0, 0, 0), 
              (0, 1, 0, 0),
              (0, 0, 1, 0),
              (0, 0, 0, 1),
              (1, -1, 0, 0),
              (1, 0 , -1, 0),
              (1, 0 , 0, -1),
              (0, 1, -1, 0),
              (0, 1, 0, -1),
              (0, 0, 1, -1)])

b = np.array([2.95, 1.74, -1.45, 1.32, 1.23, 4.45, 1.61, 3.21, 0.45, -2.75])

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("Values from least square:",x)

print("Residual value:", residuals)

print("Rank value:",rank)



