import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def S(tau, y):
    u, sigma, v = np.linalg.svd(y, full_matrices = False)
    transformed = np.array([max(item - tau, 0) for item in sigma])
    return np.matmul(np.matmul(u, np.diag(transformed)), v)
	
M = np.array(scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\image_anomaly.mat')['X'])
L = np.random.rand(M.shape[0], M.shape[1])
SP = np.random.rand(M.shape[0], M.shape[1])
gamma = 10
lam = .01

for i in range(50):
    u, sigma, v = np.linalg.svd(M, full_matrices = False)
    L = np.matmul(np.matmul(u, np.diag([max(item - gamma, 0) for item in sigma])), v)
    SP = S(lam, M - SP)

plt.imshow(M)
plt.show()
plt.imshow(L)
plt.show()