from scipy.optimize import minimize
import numpy as np

def obj(x):
    x = np.array(x)
    a = np.array([11.5, 92.5, 44.3, 98.1, 20.1, 6.1, 45.5, 31.0, 44.3])
    return np.sum((x/a)**2)

def con1(x):
    return 0.0298*x[0] - 0.044*x[1] - 0.044*x[2] - 4

def con2(x):
    return -0.0138*x[2] + 0.0329*x[3] + 0.0329*x[4] + 0.025*x[5] - 0.025*x[6] - 33

def con3(x):
    return 0.0279*x[4] - 0.0619*x[6] + 0.0317*x[7] - 0.0368*x[8] - 31

b = (0., 10000000.)
bnd = tuple([b for i in range(9)])

c1 = {'type': 'eq', 'fun': con1}
c2 = {'type': 'eq', 'fun': con2}
c3 = {'type': 'eq', 'fun': con3}
condi = [c1, c2, c3]

x0 = np.zeros(9)
sol = minimize(obj, x0, method = 'SLSQP', bounds = bnd, constraints = condi)
for i in range(9):
    print('F{} is {:.3f}'.format(i+1, sol.x[i]))
print()
print('And the total result is {:.3f}'.format(obj(sol.x)))
