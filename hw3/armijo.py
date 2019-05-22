import math
import numpy as np
import random


def armijo(func):

    epsilon = 0.2
    lamda = float(random.randint(0, 1000000000))

    state = None
    while(1):
        if func(lamda) <= func(0) + lamda * epsilon * derivative(func, 0):
            if state == 2:
                break
            state = 1
            lamda *= 2
        else:
            if state == 1:
                break
            state = 2
            lamda /= 2


    return lamda

def derivative(func, x):

    dx = 0.00000000001

    return (func(x + dx) - func(x)) / dx


if '__main__' == __name__:

    func = lambda x: (x+10)**2
    res = armijo(func)

    print(res)
