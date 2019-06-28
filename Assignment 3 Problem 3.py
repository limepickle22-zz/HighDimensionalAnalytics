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

heatmat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\heatT.mat')
T1 = heatmat['T1']
T1 = T1[0,0][0]
T1 = tl.tensor(T1)
T2 = heatmat['T2']
T2 = T2[0,0][0]
T2 = tl.tensor(T2)
T3 = heatmat['T3']
T3 = T3[0,0][0]
T3 = tl.tensor(T3)

tuckerrank = [10, 10, 10]
AIC1 = []
AIC2 = []
AIC3 = []

for i in range(20):
    cpfactorsT1 = parafac(T1, i + 1)
    cpfactorsT2 = parafac(T2, i + 1)
    cpfactorsT3 = parafac(T3, i + 1)
    AIC1.append(2*np.linalg.norm(T1 - tl.kruskal_to_tensor(cpfactorsT1))**2 + 2*i)
    AIC2.append(2*np.linalg.norm(T2 - tl.kruskal_to_tensor(cpfactorsT2))**2 + 2*i)
    AIC3.append(2*np.linalg.norm(T3 - tl.kruskal_to_tensor(cpfactorsT3))**2 + 2*i)

plt.plot(np.arange(1, 21), AIC1)
plt.plot(np.arange(1, 21), AIC2)
plt.plot(np.arange(1, 21), AIC3)
plt.show()

R1 = 5
R2 = 5
R3 = 5

TAIC1 = np.zeros(shape = (R1, R2, R3))
TAIC2 = np.zeros(shape = (R1, R2, R3))
TAIC3 = np.zeros(shape = (R1, R2, R3))
T1performance = []
T2performance = []

for i in range(R1):
    for j in range(R2):
        for k in range(R3):
            tuckerT1core, tuckerT1factors = tucker(T1, [i + 1, j + 1, k + 1])
            tuckerT2core, tuckerT2factors = tucker(T2, [i + 1, j + 1, k + 1])
            tuckerT3core, tuckerT3factors = tucker(T3, [i + 1, j + 1, k + 1])
            TAIC1[i][j][k] = (2*np.linalg.norm(T1 - tl.tucker_to_tensor(tuckerT1core, tuckerT1factors))**2 + 2*(i + j + k))
            TAIC2[i][j][k] = (2 * np.linalg.norm(T2 - tl.tucker_to_tensor(tuckerT2core, tuckerT2factors)) ** 2 + 2 * (i + j + k))
            TAIC3[i][j][k] = (2 * np.linalg.norm(T3 - tl.tucker_to_tensor(tuckerT3core, tuckerT3factors)) ** 2 + 2 * (i + j + k))

print(np.argwhere(TAIC1 == TAIC1.min()))
print(np.argwhere(TAIC2 == TAIC2.min()))
print(np.argwhere(TAIC3 == TAIC3.min()))

cpfactorsT1 = parafac(T1, 3)
cpfactorsT2 = parafac(T2, 3)
cpfactorsT3 = parafac(T3, 3)
print(np.linalg.norm(tl.kruskal_to_tensor(cpfactorsT3) - tl.kruskal_to_tensor(cpfactorsT1)))
print(np.linalg.norm(tl.kruskal_to_tensor(cpfactorsT3) - tl.kruskal_to_tensor(cpfactorsT2)))

tuckerT1core, tuckerT1factors = tucker(T1, [3, 3, 3])
tuckerT2core, tuckerT2factors = tucker(T2, [3, 3, 3])
tuckerT3core, tuckerT3factors = tucker(T3, [3, 3, 3])
print(np.linalg.norm(tl.tucker_to_tensor(tuckerT3core, tuckerT3factors) - tl.tucker_to_tensor(tuckerT1core, tuckerT1factors)))
print(np.linalg.norm(tl.tucker_to_tensor(tuckerT3core, tuckerT3factors) - tl.tucker_to_tensor(tuckerT2core, tuckerT2factors)))