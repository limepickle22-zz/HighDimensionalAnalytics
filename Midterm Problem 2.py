import scipy.io
import numpy as np
import math
import tensorly as tl
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import spline_filter

SetAmat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\SetA.mat')
SetBmat = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\SetB.mat')
SetA = SetAmat['Xn'][0,0][0]
SetB = SetBmat['Xd'][0,0][0]

def CPD(X, R, iterations):
    np.random.seed(0)
    X = tl.tensor(X)
    X1 = tl.unfold(X, 0)
    X2 = tl.unfold(X, 1)
    X3 = tl.unfold(X, 2)
    XV = [X1, X2, X3]

    A1 = np.random.rand(X.shape[0], R)
    A2 = np.random.rand(X.shape[1], R)
    A3 = np.random.rand(X.shape[2], R)
    A = [A1, A2, A3]
    lam = [[], [], []]

    for i in range(iterations):
        for k in range(3):
            AO = A[:k] + A[k + 1:]
            V = np.multiply(np.matmul(AO[0].T, AO[0]), np.matmul(AO[1].T, AO[1]))
            KR = tl.tenalg.khatri_rao(AO)
            A[k] = np.matmul(np.matmul(XV[k], KR), np.matmul(np.linalg.inv(np.matmul(V.T, V)), V.T))
            lam[k] = np.linalg.norm(A[k], axis = 0)
            A[k] = A[k] / lam[k]

    return A, lam[0]

imagesA, lamA = CPD(SetA, 3, 1000)
imagesB, lamB = CPD(SetB, 3, 1000)

firstImageA = tl.tenalg.kronecker([imagesA[0][:, 0], imagesA[1][:, 0]]).reshape(SetA.shape[0], SetA.shape[1])
secondImageA = tl.tenalg.kronecker([imagesA[0][:, 1], imagesA[1][:, 1]]).reshape(SetA.shape[0], SetA.shape[1])
thirdImageA = tl.tenalg.kronecker([imagesA[0][:, 2], imagesA[1][:, 2]]).reshape(SetA.shape[0], SetA.shape[1])
meanImageA = np.sum(SetA, axis = 2)/50

plt.imshow(firstImageA, interpolation = 'none')
plt.savefig('MidtermSetAFirstImage.jpg')
plt.imshow(secondImageA, interpolation = 'none')
plt.savefig('MidtermSetASecondImage.jpg')
plt.imshow(thirdImageA, interpolation = 'none')
plt.savefig('MidtermSetAThirdImage.jpg')
plt.imshow(meanImageA, interpolation = 'none')
plt.savefig('MidtermSetAMeanImage.jpg')
plt.plot(imagesA[2][:, 0])
plt.plot(imagesA[2][:, 1])
plt.plot(imagesA[2][:, 2])
plt.show()

firstImageB = tl.tenalg.kronecker([imagesB[0][:, 0], imagesB[1][:, 0]]).reshape(SetB.shape[0], SetB.shape[1])
secondImageB = tl.tenalg.kronecker([imagesB[0][:, 1], imagesB[1][:, 1]]).reshape(SetB.shape[0], SetB.shape[1])
thirdImageB = tl.tenalg.kronecker([imagesB[0][:, 2], imagesB[1][:, 2]]).reshape(SetB.shape[0], SetB.shape[1])
meanImageB = np.sum(SetB, axis = 2)/50

plt.imshow(firstImageB, interpolation = 'none')
plt.savefig('MidtermSetBFirstImage.jpg')
plt.imshow(secondImageB, interpolation = 'none')
plt.savefig('MidtermSetBSecondImage.jpg')
plt.imshow(thirdImageB, interpolation = 'none')
plt.savefig('MidtermSetBThirdImage.jpg')
plt.imshow(meanImageB, interpolation = 'none')
plt.savefig('MidtermSetBMeanImage.jpg')
plt.plot(imagesB[2][:, 0])
plt.plot(imagesB[2][:, 1])
plt.plot(imagesB[2][:, 2])
plt.show()

SetASpline = []
SetBSpline = []
SetABlur = []
SetBBlur = []
for i in range(50):
    blurmask = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                     [4, 16, 24, 16, 4],
                                     [6, 24, 36, 24, 6],
                                     [4, 16, 24, 16, 4],
                                     [1, 4, 6, 4, 1]])
    blurredphotoA = cv2.filter2D(SetA[:, :, i], cv2.CV_64F, blurmask)
    blurredphotoB = cv2.filter2D(SetB[:, :, i], cv2.CV_64F, blurmask)

    splinephotoA = spline_filter(SetA[:, :, i])
    splinephotoB = spline_filter(SetB[:, :, i])

    SetASpline.append(splinephotoA)
    SetBSpline.append(splinephotoB)
    SetABlur.append(blurredphotoA)
    SetBBlur.append(blurredphotoB)

plt.imshow(SetABlur[0])
plt.savefig('MidtermSetABlur.jpg')
plt.imshow(SetASpline[0])
plt.savefig('MidtermSetASpline.jpg')

print(np.mean(SetA[:, :, 0] - SetASpline[0]), np.std(SetA[:, :, 0] - SetASpline[0]))
print(np.mean(SetA[:, :, 0] - SetABlur[0]), np.std(SetA[:, :, 0] - SetABlur[0]))