import numpy as np
import math

# Function to compute machine epsilon
def compute_machine_epsilon():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2.0
    return epsilon * 2.0

# Function to compute underflow level (UFL)
def compute_underflow_level():
    ufl = 1.0
    while ufl / 2.0 > 0.0:
        ufl /= 2.0
    return ufl

# Function to compute overflow level (OFL)
def compute_overflow_level():
    ofl = 1.0
    try:
        while not math.isinf(ofl):
            prev_ofl = ofl
            ofl *= 2.0
        return prev_ofl
    except OverflowError:
        return sys.float_info.max

# Compute and print the results
epsilon_mach = compute_machine_epsilon()
underflow_level = compute_underflow_level()
overflow_level = compute_overflow_level()

print(f"Machine epsilon (ε_mach): {epsilon_mach}")
print(f"Underflow level (UFL): {underflow_level}")
print(f"Overflow level (OFL): {overflow_level}")
import numpy as np

# Function to compute machine epsilon
def compute_machine_epsilon():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2.0
    return epsilon * 2.0

# Function to compute underflow level (UFL)
def compute_underflow_level():
    ufl = 1.0
    while ufl / 2.0 > 0.0:
        ufl /= 2.0
    return ufl

# Function to compute overflow level (OFL)
def compute_overflow_level():
    ofl = 1.0
    try:
        while True:
            ofl *= 2.0
    except OverflowError:
        return ofl

# Compute and print the results
epsilon_mach = compute_machine_epsilon()
underflow_level = compute_underflow_level()
overflow_level = compute_overflow_level()

print(f"Machine epsilon (ε_mach): {epsilon_mach}")
print(f"Underflow level (UFL): {underflow_level}")
print(f"Overflow level (OFL): {overflow_level}")