import scipy.io
import numpy as np
import math
import tensorly as tl
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import spline_filter

Set1 = SetA[:, :, :25]
Set2 = SetA[:, :, 25:]
Set1factors, lam = CPD(Set1, 3, 1000)
V1 = Set1factors[0]
V2 = Set1factors[1]
W = tl.tenalg.khatri_rao([V2, V1])
WTWinv = np.linalg.inv(np.matmul(W.T, W))

lamnewlist = [np.matmul(np.matmul(WTWinv, W.T), np.ndarray.flatten(Set2[:, :, i])) for i in range(25)]
lowpercentile = [np.percentile([lamnewlist[i][j] for i in range(25)], 1.67) for j in range(3)]
highpercentile = [np.percentile([lamnewlist[i][j] for i in range(25)], 98.33) for j in range(3)]

classificationlist = []
for i in range(SetB.shape[2]):
    value = np.matmul(np.matmul(WTWinv, W.T), np.ndarray.flatten(SetB[:, :, i]))
    if value[0] >= lowpercentile[0] and value[0] <= highpercentile[0] and value[1] >= lowpercentile[1] and value[1] <= highpercentile[1] and value[2] >= lowpercentile[2] and value[2] <= highpercentile[2]:
        classificationlist.append(0)
    else:
        classificationlist.append(1)

print(classificationlist)

def sobel(set):
    sobelset = []
    for i in range(set.shape[2]):
        sobelx = cv2.Sobel(set[:, :, i], cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(set[:, :, i], cv2.CV_64F, 0, 1, ksize=1)
        f = np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely)
        sobelset.append(f)
    return np.swapaxes(np.swapaxes(np.array(sobelset), 0, 2), 0, 1)
Set1BW = sobel(Set1)
Set2BW = sobel(Set2)
Set1BWfactors, lamBW = CPD(Set1BW, 3, 1000)
V1 = Set1BWfactors[0]
V2 = Set1BWfactors[1]
W = tl.tenalg.khatri_rao([V2, V1])
WTWinv = np.linalg.inv(np.matmul(W.T, W))
lamnewlist = [np.matmul(np.matmul(WTWinv, W.T), np.ndarray.flatten(Set2BW[:, :, i])) for i in range(25)]

lowpercentile = [np.percentile([lamnewlist[i][j] for i in range(25)], 1.67) for j in range(3)]
highpercentile = [np.percentile([lamnewlist[i][j] for i in range(25)], 98.33) for j in range(3)]

SetBBW = sobel(SetB)
classificationlist = []
for i in range(SetBBW.shape[2]):
    value = np.matmul(np.matmul(WTWinv, W.T), np.ndarray.flatten(SetBBW[:, :, i]))
    if value[0] >= lowpercentile[0] and value[0] <= highpercentile[0] and value[1] >= lowpercentile[1] and value[1] <= highpercentile[1] and value[2] >= lowpercentile[2] and value[2] <= highpercentile[2]:
        classificationlist.append(0)
    else:
        classificationlist.append(1)

print(classificationlist)
print(np.sum(np.array(classificationlist)))