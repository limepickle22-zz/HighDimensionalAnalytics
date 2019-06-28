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

imgtrainvec = [Image.open('C:\\Users\\SSamtani\\Downloads\\CatsBirds\\CatsBirds\\train' + str(i+1) + '.jpg') for i in range(28)]
imgtrainvec = [tl.tensor(np.ndarray.astype(np.asarray(img), 'f')) for img in imgtrainvec]
imgtestvec = [Image.open('C:\\Users\\SSamtani\\Downloads\\CatsBirds\\CatsBirds\\Test' + str(i+1) + '.jpg') for i in range(12)]
imgtestvec = [tl.tensor(np.ndarray.astype(np.asarray(img), 'f')) for img in imgtestvec]
coretrain = [tucker(imgtrainvec[i], ranks = [10, 10, 3])[0] for i in range(28)]
coretrainvec = [tl.base.tensor_to_vec(coretrain[i]) for i in range(28)]
coretest = [tucker(imgtestvec[i], ranks = [10, 10, 3])[0] for i in range(12)]
coretestvec = [tl.base.tensor_to_vec(coretest[i]) for i in range(12)]
factorstrainvec = [tucker(imgtrainvec[i], ranks = [10, 10, 3])[1] for i in range(28)]
factorstestvec = [tucker(imgtestvec[i], ranks = [10, 10, 3])[1] for i in range(12)]
trainlabels = [0]*14 + [1]*14
testlabels = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]

rf = RandomForestRegressor(n_estimators = 100, random_state = 1)
rf.fit(coretrainvec, trainlabels)
predictions = rf.predict(coretestvec)
predictions = [int(prediction + .5) for prediction in predictions]
print(predictions)
print(confusion_matrix(testlabels, predictions))

threshold = 150
for i in range(len(imgtrainvec)):
    sobelx = cv2.Sobel(cv2.cvtColor(imgtrainvec[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize = 1)
    sobely = cv2.Sobel(cv2.cvtColor(imgtrainvec[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize = 1)
    f = np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely)
    fc = f > threshold
    fc = 255*fc
    cv2.imwrite('SobelTrain' + str(i + 1) + '.jpg', fc)

for i in range(len(imgtestvec)):
    sobelx = cv2.Sobel(cv2.cvtColor(imgtestvec[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(cv2.cvtColor(imgtestvec[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=1)
    f = np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely)
    fc = f > threshold
    fc = 255 * fc
    cv2.imwrite('SobelTest' + str(i + 1) + '.jpg', fc)

sobeltrainvec = [Image.open('SobelTrain' + str(i+1) + '.jpg') for i in range(28)]
sobeltrainvec = [tl.tensor(np.ndarray.astype(np.asarray(img), 'f')) for img in sobeltrainvec]
sobeltestvec = [Image.open('SobelTest' + str(i+1) + '.jpg') for i in range(12)]
sobeltestvec = [tl.tensor(np.ndarray.astype(np.asarray(img), 'f')) for img in sobeltestvec]
sobelcoretrain = [tucker(sobeltrainvec[i], ranks = [10, 10])[0] for i in range(28)]
sobelcoretrainvec = [tl.base.tensor_to_vec(sobelcoretrain[i]) for i in range(28)]
sobelcoretest = [tucker(sobeltestvec[i], ranks = [10, 10])[0] for i in range(12)]
sobelcoretestvec = [tl.base.tensor_to_vec(sobelcoretest[i]) for i in range(12)]
sobelfactorstrainvec = [tucker(sobeltrainvec[i], ranks = [10, 10, 3])[1] for i in range(28)]
sobelfactorstestvec = [tucker(sobeltestvec[i], ranks = [10, 10, 3])[1] for i in range(12)]
trainlabels = [0]*14 + [1]*14
testlabels = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]

rf = RandomForestRegressor(n_estimators = 100, random_state = 1)
rf.fit(sobelcoretrainvec, trainlabels)
sobelpredictions = rf.predict(sobelcoretestvec)
sobelpredictions = [int(prediction + .5) for prediction in sobelpredictions]
print(sobelpredictions)
print(confusion_matrix(testlabels, sobelpredictions))