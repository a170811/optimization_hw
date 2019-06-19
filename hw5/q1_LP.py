import numpy as np
from scipy.optimize import linprog
from timer import timer


def small_question(method):

    A = np.array([[-1, -1, -1], [-1,2, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0]])
    b = np.array([-1000, 0, -340, 0, 0])
    c = np.array([10,15,25])

    print('In small question:')
    print()
    with timer(method):
        res = linprog(c, A_ub=A, b_ub=b,bounds=(0, None), method=method)

    print('Optimal value:', res.fun, '\nX:', res.x)

    print("""
    ---------------------------------
    """)

def big_question(method):

    A = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [500, 0, 0, 700, 0, 0, 600, 0, 0, 400, 0, 0, 900, 0, 0],
        [0, 500, 0, 0, 700, 0, 0, 600, 0, 0, 400, 0, 0, 900, 0],
        [0, 0, 500, 0, 0, 700, 0, 0, 600, 0, 0, 400, 0, 0, 900]
    ])
    b = np.array([12, 18, 10, 20, 16, 25, 13, 7000, 9000, 5000])
    A_ = np.array([
        [3, -2, 0, 3, -2, 0, 3, -2, 0, 3, -2, 0, 3, 2, 2],
        [5, 0, -6, 5, 0, -6, 5, 0, -6, 5, 0, -6, -1, 7, 4]
    ])
    b_ = np.array([0, 0])
    c = np.array([320, 320, 320, 400, 400, 400, 360, 360, 360, 290, 290, 290, 100, 100, 100]) * -1

    print('In big question:')
    print()
    with timer(method):
        res = linprog(c, A_ub=A, b_ub=b, A_eq = A_, b_eq=b_, bounds=(0, None), method=method)

    print('Optimal value: ', res.fun, '\nX:', res.x)

    print("""
    ---------------------------------
    """)

if '__main__' == __name__:

    small_question('simplex')
    small_question('interior-point')
    big_question('simplex')
    big_question('interior-point')
