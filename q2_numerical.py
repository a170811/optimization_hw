import numpy as np
import sys
import time

from golden_search import golden_search
from armijo import armijo


class quadratic():
    def __init__(self, order, coefs = []):

        if len(coefs) != 0:
            assert order == len(coefs)
            self.coefs = np.array(coefs, dtype = np.float64)
        else:
            self.coefs = np.ones(order)

        self.order = order

    def val(self, x, coefs):
        if type(x) != ('int' or 'float'):
            res = np.zeros(len(x))
        else:
            x = float(x)

        for i in range(self.order):
            res += x**i * coefs[i]
        return res

    def loss_func(self, x, y, coefs):
        res = self.val(x, coefs)
        return np.sum((y - res)**2)

    def fletcher_reeves(self, x, y, epsilon = 0.0005):

        x = np.array(x, dtype = np.float64)
        y = np.array(y, dtype = np.float64)

        loss = lambda coefs: self.loss_func(x, y, coefs)

        g = gradient(loss, self.coefs)
        s = -g

        counter = 0
        while np.sqrt(np.sum(s**2)) > epsilon:

            f = lambda lamda: loss(self.coefs + lamda * s)
            ran = armijo(f)
            res = golden_search(f, [0, ran*1.2])
            self.coefs += s * res

            g_next = gradient(loss, self.coefs)
            s = -g_next + g_next.dot(g_next)/g.dot(g) * s
            counter += 1
            if counter < self.order:
                g = g_next
            else:
                g = gradient(loss, self.coefs)
                s = -g



def gradient(func, coefs):

    coefs = np.array(coefs, dtype = np.float64)
    dx = 0.00000000001
    res = []
    for i in range(len(coefs)):
        d_coefs = coefs.copy()
        d_coefs[i] += dx
        res.append((func(d_coefs) - func(coefs)) / dx)

    return np.array(res)


def q1_result():

    t1 = time.time()
    print('Part1-1: Linear regression with fletcher_reeves\n')
    func = quadratic(2)
    func.fletcher_reeves(x, y)
    t2 = time.time()

    print('\tThe coefficients a0 and a1 for function (a0 + a1 * x) is :\n\ta0 = {:.3f}\n\ta1 = {:.3f}'.format(func.coefs[0], func.coefs[1]))
    print('\n\tIt cost {:.3f} seconds.'.format(t2 - t1))
    print('\n\n')



def q2_result():

    t1 = time.time()
    order = 3
    print('Part1-2: Quadratic regression with fletcher_reeves\n')
    func = quadratic(order)
    func.fletcher_reeves(x, y)

    t2 = time.time()
    assert order == len(func.coefs)
    print('\tFor order k = {}'.format(order))
    print('\tcoefficients are as below:\n')
    for i, c in enumerate(func.coefs):
        print('\ta{} = {:.5f}'.format(i, c))

    print('\n\tIt cost {:.3f} seconds.'.format(t2 - t1))

if '__main__' == __name__:

    x = [0, 1, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9]
    y = [30, 27, 29, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32, 28, 22, 20, 27, 40]

    q1_result()
    q2_result()
