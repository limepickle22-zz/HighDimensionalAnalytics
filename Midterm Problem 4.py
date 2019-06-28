import scipy.io
import numpy as np
import math
import tensorly as tl
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import spline_filter

Z = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\Z.mat')
x = np.array([Z['x'][i][0] for i in range(len(Z['x']))])
y = np.array([Z['y'][i][0] for i in range(len(Z['y']))])

def f1(x, y, B):
    value = np.sum([y[i]*math.log((math.e**(B[0] + B[1]*x[i]))/(1 + math.e**(B[0] + B[1]*x[i])))
    + (1 - y[i])*math.log(1 - (math.e**(B[0] + B[1]*x[i]))/(1 + math.e**(B[0] + B[1]*x[i]))) for i in range(len(x))])
    return value

def gradf1(x, y, B):
    partial1 = np.sum([y[i] -(math.e**(B[0] + B[1]*x[i]))/(1 + math.e**(B[0] + B[1]*x[i])) for i in range(len(x))])
    partial2 = np.sum([y[i]*x[i] -(x[i]*math.e**(B[0] + B[1]*x[i]))/(1 + math.e**(B[0] + B[1]*x[i])) for i in range(len(x))])
    return [partial1, partial2]

beta = np.squeeze(np.random.rand(1, 2))
epsilon = .0001
step = .01
grad = np.array(gradf1(x, y, beta))

while np.matmul(grad, grad.T) > epsilon:
    beta = beta + step*grad
    grad = np.array(gradf1(x, y, beta))

print(beta)

def f2(x, y, B):
    return np.sum([(y[i] - (math.e**(B[0] + B[1]*x[i]))/(1 + math.e**(B[0] + B[1]*x[i])))**2 for i in range(len(x))])
def g(x, y, B):
    return y - (np.power(math.e, B[0] + B[1]*x))/(1 + np.power(math.e, B[0] + B[1]*x))

def J(x, y, B):
    first = -(1 + np.power(math.e, B[0] + B[1]*x)*np.power(math.e, B[0] + B[1]*x)
             + np.power(math.e, B[0] + B[1]*x)**2)/(1 + np.power(math.e, B[0] + B[1]*x))**2
    second = -(1 + np.power(math.e, B[0] + B[1]*x)*np.power(math.e, B[0] + B[1]*x)*x
             + x*np.power(math.e, B[0] + B[1]*x)**2)/(1 + np.power(math.e, B[0] + B[1]*x))**2
    return np.array([first, second]).T

epsilon = .001
lam = .000000001
alpha = 1
betant = np.squeeze(np.random.rand(1, 2))
g0 = g(x, y,  betant)
J0 = J(x, y, betant)
f0 = f2(x, y, betant)
O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(2)), np.matmul(-J0.T, g0))
while max(O) > epsilon:
    beta = betant + alpha*O
    f = f2(x, y, beta)
    while f > f0:
        alpha = .1*alpha
        beta = betant + alpha*O
        f = f2(x, y, beta)
    alpha = alpha**.5
    betant = beta
    f0 = f
    g0 = g(x, y, betant)
    J0 = J(x, y, betant)
    O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(2)), np.matmul(-J0.T, g0))

print(beta)

beta = np.squeeze(np.random.rand(1, 2))
betaprev = beta + 1
epsilon = .0001
step = .01
grad = np.array(gradf1(x, y, beta))
start = time.time()
while np.linalg.norm(np.array(beta) - np.array(betaprev), ord = 1) > .001:
    betaprev = beta
    beta = beta + step*grad
    grad = np.array(gradf1(x, y, beta))

print(time.time() - start)
print(beta)
print(f1(x, y, beta))

epsilon = .001
lam = .000000001
alpha = 1
betant = np.squeeze(np.random.rand(1, 2))
betaprev = betant + 1
g0 = g(x, y,  betant)
J0 = J(x, y, betant)
f0 = f2(x, y, betant)
O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(2)), np.matmul(-J0.T, g0))
start = time.time()
while np.linalg.norm(np.array(beta) - np.array(betaprev), ord = 1) > .001:
    betaprev = betant
    beta = betant + alpha*O
    f = f2(x, y, beta)
    while f > f0:
        alpha = .1*alpha
        beta = betant + alpha*O
        f = f2(x, y, beta)
    alpha = alpha**.5
    betant = beta
    f0 = f
    g0 = g(x, y, betant)
    J0 = J(x, y, betant)
    O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(2)), np.matmul(-J0.T, g0))

print(time.time() - start)
print(beta)
print(f1(x, y, beta))