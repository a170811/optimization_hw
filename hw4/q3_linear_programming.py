import pulp as lp
from scipy.optimize import minimize
import numpy as np

def q1_linear_programming():

    x1 = lp.LpVariable('x1', 0)
    x2 = lp.LpVariable('x2', 0)
    x3 = lp.LpVariable('x3', 0)

    prob = lp.LpProblem('myProblem', lp.LpMaximize)
    prob += 2*x1 + x2 + x3 <= 2
    prob += x1 + 2*x2 + 3*x3 <= 5
    prob += 2*x1 + 2*x2 + x3 <= 6
    prob += 3*x1 + x2 + 3*x3

    status = prob.solve()

    print('1. linear programming:', end = '\n\n')
    print('The optimal solution \n x1 = {},\n x2 = {},\n x3 = {}'.format(lp.value(x1), lp.value(x2), lp.value(x3)))
    print()
    print('The result is {}'.format(3*lp.value(x1)+lp.value(x2)+3*lp.value(x3)))


def q2_nonlinear_programming():

    def obj(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return -3.*x1 - x2 - 3.*x3

    def con1(x):
        return 2. - 2*x[0] - x[1] - x[2]

    def con2(x):
        return 5. - 1*x[0] - 2*x[1] - 3*x[2]

    def con3(x):
        return 6. - 2*x[0] - 2*x[1] - x[2]

    b = (0., 100.)
    bnd = (b, b, b)

    c1 = {'type': 'ineq', 'fun': con1}
    c2 = {'type': 'ineq', 'fun': con2}
    c3 = {'type': 'ineq', 'fun': con3}
    condi = [c1, c2, c3]

    x0 = [0., 0., 0.]
    sol = minimize(obj, x0, method = 'SLSQP', bounds = bnd, constraints = condi)
    print('2. non-linear programming(Sequential Least SQuares Programming (SLSQP)):', end = '\n\n')
    print('The optimal solution \n x1 = {:.3f},\n x2 = {:.3f},\n x3 = {:.3f}'.format(sol.x[0], sol.x[1], sol.x[2]))
    print()
    print('The result is {:.3f}'.format(obj(sol.x)*-1))

if '__main__' == __name__:

    q1_linear_programming();
    print()
    print()
    print()
    q2_nonlinear_programming()
