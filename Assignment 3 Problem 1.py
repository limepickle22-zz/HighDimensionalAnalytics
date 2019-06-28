import scipy.io
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.tenalg import inner
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import cv2



aminomat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\aminoacid.mat')
X = aminomat['X']
X = X[0,0][0]
X = tl.tensor(X)
X1 = tl.unfold(X, 0)
X2 = tl.unfold(X, 1)
X3 = tl.unfold(X, 2)
XV = [X1, X2, X3]
Y = aminomat['Y']
ylam = np.linalg.norm(Y, axis = 0)
Ynorm = Y/ylam

R = 3
iterations = 1000

A1 = np.random.rand(X.shape[0], R)
A2 = np.random.rand(X.shape[1], R)
A3 = np.random.rand(X.shape[2], R)
A = [A1, A2, A3]
lam = [[], [], []]

for i in range(iterations):
    for k in range(3):
        AO = A[:k] + A[k+1:]
        V = np.multiply(np.matmul(AO[0].T, AO[0]), np.matmul(AO[1].T, AO[1]))
        KR = tl.tenalg.khatri_rao(AO)
        A[k] = np.matmul(np.matmul(XV[k], KR), np.matmul(np.linalg.inv(np.matmul(V.T, V)), V.T))
        lam[k] = np.linalg.norm(A[k], axis = 0)
        A[k] = A[k]/lam[k]

print(np.arange(len(A[1][:, 0])))
print(A[1][:, 0])

plt.plot(A[1][:, 0])
plt.plot(A[1][:, 1])
plt.plot(A[1][:, 2])
plt.show()

plt.plot(A[2][:, 0])
plt.plot(A[2][:, 1])
plt.plot(A[2][:, 2])
plt.show()