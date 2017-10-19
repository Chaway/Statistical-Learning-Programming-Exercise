# -*- coding: utf-8 -*-
import scipy.optimize as opt
import numpy as np

def func(x):
    """Define object function"""
    print x 
    # print x in every iteration
    return (10 - x[0]**2 - x[1]**2)

def func_deriv(x):
    """Derivative of object function"""
    return np.array([-2 * x[0], -2 * x[1]])

# Define constraint conditions of object function
cons = ({'type': 'eq', 'fun': lambda x: np.array([x[0] + x[1]]),
         'jac': lambda x: np.array([1.0, 1.0])},
        {'type': 'ineq', 'fun': lambda x: np.array([-x[0]**2 + x[1]]),
         'jac': lambda x: np.array([-2*x[0], 1.0])})

"""
Call the optimize module in scipy python library to
solve the optimization problem
"""
res = opt.minimize(func, [-2, 5], jac = func_deriv, \
                   constraints = cons, method = 'SLSQP', \
                   options = {'disp': True})
print res.x
