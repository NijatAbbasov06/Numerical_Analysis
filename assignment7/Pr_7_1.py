import random


def evaluate_pol(n, x, t):
    
    p = x[0]
    for i in range(1, n+1):
        p = p * t + x[i]
    return p




def evaluate_pol_with_choice(n, x, t, choice):

    p = x[0]
    if choice == "d":
        dp = 0
        for i in range(1, n+1):
            dp = dp * t + p
            p = p * t + x[i]
        return dp
    elif choice == "i":
        integral = x[0] * t
        for i in range(1, n+1):
            integral += x[i] * t**(i+1) / (i+1)
        return integral
    else:
        for i in range(1, n+1):
            p = p * t + x[i]
        return p

n = 3
x = [1, 2, 3, 4]
t = 2.5
simple_result = evaluate_pol(n, x, t)
print("\nevaluating simple 3 degree pol with"
       f" x = [1, 2, 3, 4] and t = 2.5: \n {simple_result}")

derivative_result = evaluate_pol_with_choice(n, x, t, "d")
integral_result = evaluate_pol_with_choice(n, x, t, "i")

print(f"\n derivative : {derivative_result}")
print(f"\n integral : {integral_result}")






print("\ngenerating random values: \n")
n = random.randint(1, 10)
x = [random.uniform(-10, 10) for _ in range(n+1)]
t = random.uniform(-10, 10)





print(f" \n x vector = \n {x} \n")
print(f"n value = {n} \n")
print(f"t value = {t} \n ")
result = evaluate_pol(n, x, t)
print(f"resulting value of the pol at t value: \n {result}") 





derivative_result = evaluate_pol_with_choice(n, x, t, "d")
integral_result = evaluate_pol_with_choice(n, x, t, "i")

print(f"\n derivative : {derivative_result}")
print(f"\n integral : {integral_result}")