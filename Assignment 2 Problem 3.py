smoothmask = (1/1115)*np.array([[1, 4, 7, 10, 7, 4, 1],
        [4, 12, 26, 33, 26, 12, 4],
        [7, 26, 55, 71, 55, 26, 7],
        [10, 33, 71, 91, 71, 33, 10],
        [1, 4, 7, 10, 7, 4, 1],
        [4, 12, 26, 33, 26, 12, 4],
        [7, 26, 55, 71, 55, 26, 7]])
smoothphoto = cv2.filter2D(imgcolor, cv2.CV_64F, smoothmask)
cv2.imwrite("FlowerSmooth.jpg", smoothphoto)

G = np.zeros(shape = smoothphoto.shape)
theta = np.zeros(shape = smoothphoto.shape)

for k in range(smoothphoto.shape[2]):
    for i in range(smoothphoto.shape[0]):
        for j in range(smoothphoto.shape[1]):
            if i + 1 >= smoothphoto.shape[0] or j + 1 >= smoothphoto.shape[1]:
                G[i][j][k] = smoothphoto[i][j][k]
                theta[i][j][k] = 0
            else:
                G[i][j][k] =((.5*(smoothphoto[i+1][j][k] - smoothphoto[i][j][k] + smoothphoto[i+1][j+1][k]
                                  - smoothphoto[i][j+1][k]))**2 + (.5*(smoothphoto[i][j+1][k] - smoothphoto[i][j][k]
                                  + smoothphoto[i+1][j+1][k] - smoothphoto[i+1][j][k]))**2)**.5
                theta[i][j][k] = math.atan2(smoothphoto[i][j+1][k] - smoothphoto[i][j][k] + smoothphoto[i+1][j+1][k]
                                            - smoothphoto[i+1][j][k], smoothphoto[i+1][j][k] - smoothphoto[i][j][k]
                                            + smoothphoto[i+1][j+1][k] - smoothphoto[i][j+1][k])

cv2.imwrite("FlowerG.jpg", G)

phi = np.zeros(shape = smoothphoto.shape)
for k in range(smoothphoto.shape[2]):
    for i in range(smoothphoto.shape[0]):
        for j in range(smoothphoto.shape[1]):
            if i + 1 >= smoothphoto.shape[0] or j + 1 >= smoothphoto.shape[1]:
                phi[i][j][k] = 0
            else:
                if theta[i][j][k] >= math.pi*(-1/8) and theta[i][j][k] <= math.pi*(1/8):
                    if G[i][j][k] >= G[i][j-1][k] and G[i][j][k] >= G[i][j+1][k]:
                        phi[i][j][k] = G[i][j][k]
                    else:
                        phi[i][j][k] = 0
                elif theta[i][j][k] >= math.pi*(1/8) and theta[i][j][k] <= math.pi*(3/8):
                    if G[i][j][k] >= G[i+1][j-1][k] and G[i][j][k] >= G[i-1][j+1][k]:
                        phi[i][j][k] = G[i][j][k]
                    else:
                        phi[i][j][k] = 0
                elif theta[i][j][k] >= math.pi*(-3/8) and theta[i][j][k] <= math.pi*(-1/8):
                    if G[i][j][k] >= G[i-1][j-1][k] and G[i][j][k] >= G[i+1][j+1][k]:
                        phi[i][j][k] = G[i][j][k]
                    else:
                        phi[i][j][k] = 0
                else:
                    if G[i][j][k] >= G[i-1][j][k] and G[i][j][k] >= G[i+1][j][k]:
                        phi[i][j][k] = G[i][j][k]
                    else:
                        phi[i][j][k] = 0

cv2.imwrite("FlowerPhi.jpg", phi)

t1 = 5
t2 = 10
E = np.zeros(shape = smoothphoto.shape)
count = 1

while count != 0:
    count = 0
    for k in range(E.shape[2]):
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if phi[i][j][k] >= t2 and E[i][j][k] == 0:
                    E[i][j][k] = 255
                    count = count + 1
                elif phi[i][j][k] >= t1 and E[i][j][k] == 0:
                    if E[i-1][j-1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i][j-1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i-1][j][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i-1][j+1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i][j+1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i+1][j-1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i+1][j][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1
                    if E[i+1][j+1][k] == 1:
                        E[i][j][k] = 255
                        count = count + 1

cv2.imwrite("FlowerE.jpg", E)