import numpy as np
from scipy.optimize import minimize
from timer import timer

small_question = {
    'n': 3,
    'type': 'small question',
    'func': lambda x: x[0]**2 + x[1]**2 + x[2]**2 + 40*x[0] + 20*x[1],
    'con': [
        {'type':'ineq', 'fun': lambda x: x[0]-50},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]-100},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]+x[2]-150}
    ]
}

big_question = {
    'n': 5,
    'type': 'big question',
    'func': lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + 40*x[0] + 20*x[1] -15*x[2] + 30*x[3],
    'con': [
        {'type':'ineq', 'fun': lambda x: x[0]-50},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]-100},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]+x[2]-150},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]-200},
        {'type':'ineq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]+x[4]-250}
    ]
}


def question(method, func):
    x = np.arange(func['n'])
    print(func['type'], ':')
    print()
    with timer(method):
        res = minimize(func['func'], x, method=method, constraints=func['con'])
    print(res.x)
    print('The optimal value is : {}'.format(func['func'](res.x)))

    print("""
    -----------------------------------
    """)


if '__main__' == __name__:

    question('SLSQP', small_question)
    question('trust-constr', small_question)
    question('SLSQP', big_question)
    question('trust-constr', big_question)
