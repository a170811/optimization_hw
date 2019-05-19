import numpy as np
import matplotlib.pyplot as plt

import q1_regression as regression


def loss_func(coef, matrix, x, y):

    assert len(x) == len(y)
    x_ = np.array([x])
    y_ = np.array([y])
    vecs = tuple([x_**i for i in range(len(coef))])
    matrix = np.concatenate(vecs, 0).transpose()
    predict = matrix.dot(coef)

    loss = np.sum((y-predict)**2)
    return predict, loss

def result():
    x = np.array([0, 1, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9])
    y = np.array([30, 27, 29, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32, 28, 22, 20, 27, 40])

    for i in range(1, 8):
        order = i + 6
        coef, matrix = regression.quadratic(x, y, order)
        predict, loss = loss_func(coef, matrix, x, y)

        plt.figure(i)
        plt.title('order {}, loss = {:.3f}'.format(order, loss))
        plt.plot(x, y)
        plt.plot(x, predict)


if '__main__' == __name__:
    result()
