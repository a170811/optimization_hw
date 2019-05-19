import numpy as np

def linear(x, y):
    sigma_x = np.sum(x)
    sigma_xx = np.sum(x**2)
    sigma_xy = np.sum(x*y)
    sigma_y = np.sum(y)

    assert(len(x) == len(y))
    n = len(x)

    m = (sigma_x * sigma_y - n * sigma_xy) / (sigma_x**2 - n * sigma_xx)
    b = (sigma_y - m * sigma_x) / n

    return m.item(), b.item()


def quadratic(x, y, n_args):

    assert len(x) == len(y)
    x = np.array([x])
    y = np.array([y])

    vecs = tuple([x**i for i in range(n_args)])
    matrix_t = np.concatenate(vecs, 0)
    matrix = matrix_t.transpose()

    a = np.linalg.inv(np.matmul(matrix_t, matrix)).dot(matrix_t).dot(y.transpose()).flatten()

    val = np.zeros(len(x[0]))
    for i, c in enumerate(a):
        val += x[0]**i * c
    loss = np.sum((y - val)**2)

    return a, loss

def result():
    x = np.array([0, 1, 2.2, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 11, 12.4, 14.1, 15.2, 16.8, 18.7, 19.9])
    y = np.array([30, 27, 29, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36, 32, 28, 22, 20, 27, 40])
    print('''
    Note:
        The arg, k, to pass to function result()
        means that there are k args in quadratic regression.
        i.e. For k = 2,
        the result returned is an array of two coefs.
        That is, when k = 2, quadraic regression is equal to linear regression
    ''')


    print('a. Linear Regression: \n')
    m, b = linear(x, y)

    print('\tThe coefficients b and m for function (mx + b) is :\n\tb = {:.3f}\n\tm = {:.3f}'.format(b, m))
    print('\n\n')

    print('b. Quadratic Regression(order = 2): \n')

    k = 2
    vector_a, _ = quadratic(x, y, k)

    print('\tFor order k = {}'.format(k))
    print('\tcoefficients are as below :\n')
    for i in range(k):
        print('\ta{} = {:.5f}'.format(i, vector_a[i]))

    print()

    k = 3
    print('b. Quadratic Regression(order = {}): \n'.format(k))
    vector_a, _ = quadratic(x, y, k)

    print('\tFor order k = {}'.format(k))
    print('\tcoefficients are as below :\n')
    for i in range(k):
        print('\ta{} = {:.5f}'.format(i, vector_a[i]))

if '__main__' == __name__:

    result()
