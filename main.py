from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum
import random
import pylab

class Kernel(Enum): 
    linear = 1
    polynomial = 2
    rbf = 3
    sigmoid = 4

class GenerateData():
    np.random.seed(100)
    classA = [(random.normalvariate(-1.5, 0.5), random.normalvariate(0.5, 1), 1.0) for i in range(30)] + \
              [(random.normalvariate(1.5, 0.3), random.normalvariate(0.5, 1), 1.0) for i in range(30)]
               
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(60)]
    
    data = classA + classB
    random.shuffle(data)


def linear_kernel(x, y):
    return np.dot(x, y) + 1

def polynomial_kernel(x, y, p):
    return pow(np.dot(x, y) + 1, p)
    
def rbf_kernel(x, y, theta):
    sub = [a[0] - a[1] for a in zip(x, y)]
    return math.exp(-(np.dot(sub, sub))/(2*pow(theta, 2)))

def sigmoid_kernel(x, y, k, delta):
    return math.tanh(k*np.dot(x, y) - delta)

def p_matrix(x, kernel, p=2, theta=1, delta=1):
    P = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if kernel == Kernel.linear:
                P[i, j] = x[i][2]*x[j][2]*linear_kernel(x[i][0:2], x[j][0:2])
            elif kernel == Kernel.polynomial:
                P[i, j] = x[i][2]*x[j][2]*polynomial_kernel(x[i][0:2], x[j][0:2], p)
            elif kernel == Kernel.rbf:
                P[i, j] = x[i][2]*x[j][2]*rbf_kernel(x[i][0:2], x[j][0:2], theta)
            else:
                P[i, j] = x[i][2]*x[j][2]*sigmoid_kernel(x[i][0:2], x[j][0:2], delta)
    return P

def QP_function(x, P, slack=False, C=1):
    if slack == True:
        h = np.row_stack((np.zeros((len(x), 1)), C*np.ones((len(x), 1))))
        G = np.row_stack((np.diag([-1.0]*len(x)), np.diag([1.0]*len(x))))
    else:
        h = np.zeros((len(x), 1))
        G = np.diag([-1.0]*len(x))
    q = -1*np.ones((len(x), 1))

    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    
    sup_vec = list()
    for i in range(len(alpha)):
        if slack == True:
            if (alpha[i] > 1e-5) and (alpha[i] < C):
                sup_vec.append((x[i][0], x[i][1], x[i][2], alpha[i])) 
        else:
            if alpha[i] > 1e-5:
                sup_vec.append((x[i][0], x[i][1], x[i][2], alpha[i]))
    return sup_vec

def Indicator(x, sup_vec, kernel, p=2, theta=1, delta=1):
    ind = 0
    for i in range(len(sup_vec)):
        if kernel == Kernel.linear:
            ind += sup_vec[i][3]*sup_vec[i][2]*linear_kernel(x, sup_vec[i][0:2])
        elif kernel == Kernel.polynomial:
            ind += sup_vec[i][3]*sup_vec[i][2]*polynomial_kernel(x, sup_vec[i][0:2], p)
        elif kernel == Kernel.rbf:
            ind += sup_vec[i][3]*sup_vec[i][2]*rbf_kernel(x, sup_vec[i][0:2], theta)
        else:
            ind += sup_vec[i][3]*sup_vec[i][2]*sigmoid_kernel(x, sup_vec[i][0:2], delta)
    return ind

def dicision_boundary(sup_vec, kernel):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[Indicator([x, y], sup_vec, kernel) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))

    pylab.plot([p[0] for p in GenerateData.classA], [p[1] for p in GenerateData.classA], 'bo')
    pylab.plot([p[0] for p in GenerateData.classB], [p[1] for p in GenerateData.classB], 'ro')
    pylab.show()

def main():    
    kernel = Kernel.rbf
    slack = True
    C = 0.1
    P = p_matrix(GenerateData.data, kernel)
    sup_vec = QP_function(GenerateData.data, P, slack, C)
    
    plt.figure()
    dicision_boundary(sup_vec,kernel)

if __name__ == "__main__":
	main()
