import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def my_func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    val = a0
    val += a1 * x
    val += a2 * x**2
    val += a3 * x**3
    val += a4 * x**4
    val += a5 * x**5
    val += a6 * np.cos(x)
    val += a7 * np.cos(x)**2
    val += a8 * np.sin(x)
    val += a9 * np.sin(x)**2
    return val


def result():
    x = np.array([0, 1, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9])
    y = np.array([30, 27, 29, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32, 28, 22, 20, 27, 40])

    parms, _ = curve_fit(my_func, x, y)

    loss = 0
    predict = my_func(x, *parms)
    loss = np.sum((y - predict)**2)

    print('[y_value] & [y_predict]')
    for i in range(len(y)):
        print('  {:5.2f} \t{:5.2f}'.format(y[i], predict[i]))
    print('\nloss = {:.3f}'.format(loss))

    plt.figure(1)
    plt.title('my function')
    plt.plot(x, y)
    plt.plot(x, predict)

if '__main__' == __name__:
    result()
