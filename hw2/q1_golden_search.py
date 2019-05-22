import math
import numpy as np



def golden_search(func, interval, l = 0.0001):

    alpha = (math.sqrt(5) - 1)/2

    def calc_lamda():
        return interval[0] + (1 - alpha) * (interval[1] - interval[0])
    def calc_nu():
        return interval[0] + alpha * (interval[1] - interval[0])

    lamda = calc_lamda()
    nu = calc_nu()

    while (interval[1] - interval[0]) > l:

        if func(lamda) > func(nu):
            interval[0] = lamda

            lamda = nu
            nu = calc_nu()

        else:
            interval[1] = nu

            nu = lamda
            lamda = calc_lamda()

    return np.sum(interval)/2



if '__main__' == __name__:

    func = lambda x: x**2 * math.cos(x)
    res = golden_search(func, [2, 12])

    print(res)
