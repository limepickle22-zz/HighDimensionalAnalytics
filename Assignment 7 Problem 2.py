import numpy as np
import scipy.io
import matplotlib.pyplot as plt

np.random.seed(0)
imagemat = np.array(scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\Image-1.mat')['X'])
X = imagemat[:]
shape = X.shape
X = np.asarray(X)
X = X.flatten()
indices = np.random.choice(X.size, size = int(X.size*.15))
X[indices] = 0
X = X.reshape(shape)

def S(tau, y):
    u, sigma, v = np.linalg.svd(y, full_matrices = False)
    transformed = np.array([max(item - tau, 0) for item in sigma])
    return np.matmul(np.matmul(u, np.diag(transformed)), v)

Y = np.zeros((X.shape[0], X.shape[1]))
lam = 1
delta = .1
for k in range(200):
    Z = S(lam*delta, Y)
    Y = Z + delta*(X - Z)

plt.imshow(imagemat)
plt.show()
plt.imshow(X)
plt.show()
plt.imshow(Z)
plt.show()

errorMissing = sum((np.asarray(imagemat).flatten()[indices] - np.asarray(Z).flatten()[indices])**2)
errorImage = sum(sum((imagemat - Z)**2))
errorMissingPercentage = errorMissing/len(indices)
errorImagePercentage = errorImage/(imagemat.shape[0]*imagemat.shape[1])

print(errorMissingPercentage, errorImagePercentage)