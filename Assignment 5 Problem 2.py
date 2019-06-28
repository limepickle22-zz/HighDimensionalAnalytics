MRI = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\MRI.mat')
img = scipy.io.loadmat('C:\\Users\\SSamtani\\Downloads\\img.pb3.mat')

X1 = MRI['X1']
X2 = MRI['X2']
X3 = MRI['X3']
X = [X1, X2, X3]
y1 = MRI['y1']
y2 = MRI['y2']
y3 = MRI['y3']
y = [y1, y2, y3]

img = img['img']

rho = .5
lam = 1
B1 = np.random.rand(961)
B2 = np.random.rand(961)
B3 = np.random.rand(961)
B = [B1, B2, B3]
theta = np.random.rand(961)
mu1 = np.random.rand(961)
mu2 = np.random.rand(961)
mu3 = np.random.rand(961)
mu = [mu1, mu2, mu3]

for i in range(1000):
    for i in range(3):
        A = np.matmul(X[i].T, X[i]) + rho*np.identity(961)
        b = np.squeeze(np.matmul(X[i].T, y[i])) + rho*(theta - mu[i])
        B[i] = np.linalg.solve(A, b)
    Bbar = np.mean(B, axis = 0)
    mubar = np.mean(mu, axis = 0)
    together = Bbar + mubar
    theta = np.where(together > lam/(rho*3), together - lam/(rho*3), theta)
    theta = np.where(abs(together) <= lam/(rho*3), 0, theta)
    theta = np.where(together < lam/(rho*3), together + lam/(rho*3), theta)

plt.imshow(np.reshape(theta, (31, 31)), cmap = 'gray')
plt.show()

plt.imshow(img, cmap = 'gray')
plt.show()
