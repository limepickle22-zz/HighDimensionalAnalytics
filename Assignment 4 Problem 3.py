import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt

leakmat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\leak-1.mat')
b = np.array(leakmat['b'])
partmatrices = [leakmat['leak'][int(len(leakmat['leak'])/b.shape[0]*i):int(len(leakmat['leak'])/b.shape[0]*(i+1))] for i in range(b.shape[0])]
t = np.array([partmatrices[i][:, 1] for i in range(len(partmatrices))])
T = np.array([partmatrices[i][:, 2] for i in range(len(partmatrices))])
F = np.array([partmatrices[i][:, 3] for i in range(len(partmatrices))])

def L(F, T, t, beta):
    return np.sum((F - (beta[0] + beta[1]*T)*(1 - np.power(math.e, -2*beta[2]*t)))**2)

def g(F, T, t, beta):
    return F - (beta[0] + beta[1]*T)*(1 - np.power(math.e, -2*beta[2]*t))

def J(F, T, t, beta):
    dg1 = -1 + np.power(math.e, -2*beta[2]*t)
    dg2 = -T + T*np.power(math.e, -2*beta[2]*t)
    dg3 = -2*t*beta[0]*np.power(math.e, -2*beta[2]*t) - 2*t*beta[1]*T*np.power(math.e, -2*beta[2]*t)
    return np.array([dg1, dg2, dg3]).T

np.random.seed(0)

betalist = []
rmse = []

for i in range(5):
    epsilon = .001
    lam = .000000001
    alpha = 1
    betant = np.squeeze(np.random.rand(b[0].shape[0]))
    g0 = g(F[i], T[i], t[i], betant)
    J0 = J(F[i], T[i], t[i], betant)
    f0 = L(F[i], T[i], t[i], betant)
    O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(3)), np.matmul(-J0.T, g0))

    while max(O) > epsilon:
        beta = betant + alpha*O
        f = L(F[i], T[i], t[i], beta)
        while f > f0:
            alpha = .1*alpha
            beta = betant + alpha*O
            f = L(F[i], T[i], t[i], beta)
        alpha = alpha**.5
        betant = beta
        f0 = f
        g0 = g(F[i], T[i], t[i], betant)
        J0 = J(F[i], T[i], t[i], betant)
        O = np.linalg.solve((np.matmul(J0.T, J0) + lam*np.identity(3)), np.matmul(-J0.T, g0))

    rmse.append(np.sum((beta - b[i])**2)**.5)
    betalist.append(beta)

print(np.array(rmse))
print(np.array(betalist))
print(b)