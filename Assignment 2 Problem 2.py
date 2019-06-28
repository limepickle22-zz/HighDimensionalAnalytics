from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img = Image.open("C:\\Users\\SSamtani\\Downloads\\FlowerN.jpg")
img.save("FlowerOriginal.jpg")

imggray = Image.open("C:\\Users\\SSamtani\\Downloads\\FlowerN.jpg").convert('L')
imggray.save("FlowerGray.jpg")

imgblackwhite = imggray.point(lambda p: p > 255*.6 and 255)
imgblackwhite.save("FlowerBW.jpg")

imgquarter = img.resize([int(.5*size) for size in img.size]) #.5 in both dimensions to reduce to .25 of original
imgquarter.save("FlowerQuarter.jpg")

grayfreq = imggray.histogram()
bins = [i for i in range(256)]
plt.bar(bins, grayfreq)
plt.show()

imgthreshold = imggray.point(lambda p: p > 150 and 255)
imgthreshold.save("FlowerThresh.jpg")
threshfreq = imgthreshold.histogram()
threshfreq = [item for item in threshfreq if item != 0]
plt.bar([0, 1], threshfreq)
plt.show()

U = 225
L = 25
s = 50

graymatrix = np.asarray(imggray)
shiftmatrix = np.copy(graymatrix)

for i in range(graymatrix.shape[0]):
    for j in range(graymatrix.shape[1]):
        if graymatrix[i][j] > U - s:
            shiftmatrix[i][j] = U
        elif graymatrix[i][j] <= L - s:
            shiftmatrix[i][j] = L
        else:
            shiftmatrix[i][j] = shiftmatrix[i][j] + s

imgshift = Image.fromarray(shiftmatrix)
imgshift.save("FlowerShift.jpg")
shiftfreq = imgshift.histogram()
plt.bar([i for i in range(50, 226)], shiftfreq[50:226])
plt.show()

stretchmatrix = np.copy(graymatrix)
mingray = np.amin(graymatrix)
maxgray = np.amax(graymatrix)
lam = 205

for i in range(graymatrix.shape[0]):
    for j in range(graymatrix.shape[1]):
        stretchmatrix[i][j] = ((stretchmatrix[i][j] - mingray)/(maxgray - mingray))*lam

imgstretch = Image.fromarray(stretchmatrix)
imgstretch.save("FlowerStretch.jpg")
stretchfreq = imgstretch.histogram()
plt.bar(bins, stretchfreq)
plt.show()

logmatrix = np.copy(graymatrix)
c = 40

for i in range(graymatrix.shape[0]):
    for j in range(graymatrix.shape[1]):
        logmatrix[i][j] = c*(math.log(graymatrix[i][j] + 1))

imglog = Image.fromarray(logmatrix)
imglog.save("FlowerLog.jpg")
logfreq = imglog.histogram()
plt.bar(bins, logfreq)
plt.show()

powermatrix = np.copy(graymatrix)
c = .1
gamma = 1.4

for i in range(graymatrix.shape[0]):
    for j in range(graymatrix.shape[1]):
        powermatrix[i][j] = c*(graymatrix[i][j]**gamma)

imgpower = Image.fromarray(powermatrix)
imgpower.save("FlowerPower.jpg")
powerfreq = imgpower.histogram()
plt.bar(bins, powerfreq)
plt.show()

imgcolor = cv2.imread("C:\\Users\\SSamtani\\Downloads\\FlowerN.jpg")
blurmask = (1/256)*np.array([[1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]])
blurredphoto = cv2.filter2D(imgcolor, cv2.CV_64F, blurmask)
cv2.imwrite("FlowerBlur.jpg", blurredphoto)

sharpmask = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])
sharpphoto = cv2.filter2D(imgcolor, cv2.CV_64F, sharpmask)
cv2.imwrite("FlowerSharp.jpg", sharpphoto)

embossmask = np.array([[-1, -1, -1, -1, 0],
                       [-1, -1, -1, 0, 1],
                       [-1, -1, 0, 1, 1],
                       [-1, 0, 1, 1, 1],
                       [0, 1, 1, 1, 1]])
embossedphoto = cv2.filter2D(imgcolor, cv2.CV_64F, embossmask)
cv2.imwrite("FlowerEmbossed.jpg", embossedphoto)

edgemask = np.array([[-1, 0, 0, 0, 0],
                     [0, -2, 0, 0, 0],
                     [0, 0, 6, 0, 0],
                     [0, 0, 0, -2, 0],
                    [0, 0, 0, 0, -1]])
edgephoto = cv2.filter2D(imgcolor, cv2.CV_64F, edgemask)
cv2.imwrite("FlowerEdge.jpg", edgephoto)

grayimage = cv2.imread("C:\\Users\\SSamtani\\Downloads\\FlowerN.jpg", 0)
thresh = cv2.threshold(grayimage, 0, 255, cv2.THRESH_OTSU)
print(thresh[0])

Z = imgcolor.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(imgcolor.shape)

cv2.imwrite("FlowerKMeans" + str(K) + ".jpg", res2)

sobelx = cv2.Sobel(imgcolor, cv2.CV_64F, 1, 0, ksize = 1)
sobely = cv2.Sobel(imgcolor, cv2.CV_64F, 0, 1, ksize = 1)
f = np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely)
cv2.imwrite("FlowerSobel.jpg", f)

print(f)

cutoff = 1000
fc = f > cutoff
print(fc)
fc = 255*fc
cv2.imwrite("FlowerSobelCutoff" + str(cutoff) + ".jpg", fc)