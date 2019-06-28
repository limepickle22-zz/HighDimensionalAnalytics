import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt

emissionmat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\emission-1.mat')
truemat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\true-1.mat')

p = emissionmat['p']
y = np.squeeze(emissionmat['y'])
truemu = truemat['mu']

def loglikelihood(mu, y, p):
    sumlist = []
    for j in range(len(y)):
        sum = [-np.sum([p[j][i]*mu[i] for i in range(len(mu))]) + y[j]*math.log(np.sum([p[j][i]*mu[i] for i in range(len(mu))]))
               - math.log(math.factorial(y[j]))]
        sumlist.append(sum)
    return np.sum(sumlist)

def gradient(mu, y, p):
    grad = []
    for k in range(len(mu)):
        sum = 0
        for j in range(len(y)):
            sum = sum - p[j][k] + (p[j][k]*y[j]/np.sum([p[j][i]*mu[i] for i in range(len(mu))]))
        grad.append(sum)
    return np.squeeze(np.array(grad))

def stochgrad(mu, y, p, j):
    stochgrad = []
    for k in range(len(mu)):
        sum = 0
        sum = sum - p[j][k] + (p[j][k]*y[j]/np.sum([p[j][i]*mu[i] for i in range(len(mu))]))
        stochgrad.append(sum)
    return np.squeeze(np.array(stochgrad))

def Hessian(mu, y, p):
    mat = []
    for k in range(len(mu)):
        for l in range(len(mu)):
            sum = 0
            for j in range(len(y)):
                sum = sum - (p[j][k]*p[j][l] * y[j] / np.sum([p[j][i] * mu[i] for i in range(len(mu))])**2)
            mat.append(sum)
    return np.reshape(np.array(mat), (3, 3))


np.random.seed(0)
mu = np.squeeze(np.random.rand(1, 3))
epsilon = .0001
step = 1
loglikelist = []
grad = gradient(mu, y, p)

while np.matmul(grad, grad.T) > epsilon:
    mu = mu + step*grad
    grad = gradient(mu, y, p)
    loglikelist.append(loglikelihood(mu, y, p))

error = np.sum([(mu[i] - truemu[i])**2 for i in range(len(mu))])/len(mu)

print(mu)
print(truemu)
print(error)
plt.plot(loglikelist)
plt.show()

mu = np.squeeze(np.random.rand(1, 3))
mutwo = np.squeeze(np.random.rand(1, 3))
mu0 = mu
epsilon = .0001
k = 1
grad = gradient(mutwo, y, p)
loglikelist = []

while np.matmul(grad, grad.T) > epsilon:
    mu = mutwo + .01*grad
    mutwo = mu + (k-1)/(k+2)*(mu - mu0)
    grad = gradient(mu, y, p)
    mu0 = mu
    loglikelist.append(loglikelihood(mu, y, p))
    k = k + 1

error = np.sum([(mu[i] - truemu[i])**2 for i in range(len(mu))])/len(mu)

print(mu)
print(truemu)
print(error)
plt.plot(loglikelist)
plt.show()

mu = np.squeeze(np.random.rand(1, 3))
step = 1
K = 10000
m = 5
loglikelist = []

for k in range(K):
    for i in range(m):
        s = np.random.randint(0, m)
        mu = mu + step*stochgrad(mu, y, p, s)
    loglikelist.append(loglikelihood(mu, y, p))

error = np.sum([(mu[i] - truemu[i])**2 for i in range(len(mu))])/len(mu)

print(mu)
print(truemu)
print(error)
plt.plot(loglikelist)
plt.show()


mu = np.squeeze(np.random.rand(1, 3))
mu0 = mu
epsilon = .000000001
lam = .000000001
alpha = 1
f0 = -loglikelihood(mu, y, p)
g0 = -gradient(mu, y, p)
H = -Hessian(mu, y, p)
O = np.linalg.solve((H + lam*np.identity(len(mu))), -g0)
loglikelist = [-f0]

while max(O) > epsilon:
    mu = mu0 + alpha*O
    f = -loglikelihood(mu, y, p)
    while f > f0:
        alpha = .1*alpha
        mu = mu0 + alpha*O
        f = loglikelihood(mu, y, p)
    alpha = alpha**.5
    mu0 = mu
    f0 = f
    loglikelist.append(-f0)
    g0 = -gradient(mu0, y, p)
    H = -Hessian(mu0, y, p)
    O = np.linalg.solve((H + lam * np.identity(len(mu))), -g0)

error = np.sum([(mu[i] - truemu[i])**2 for i in range(len(mu))])/len(mu)

print(mu)
print(truemu)
print(error)
plt.plot(loglikelist)
plt.show()

