import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

realEstateData = pd.read_csv("C:\\Users\\SSamtani\\Downloads\\RealEstate-1.csv")
realEstateData = realEstateData.values

y = realEstateData[:, 0]
y = [(item - min(y))/(max(y) - min(y)) for item in y]

bedrooms = realEstateData[:, 1]
bedrooms1 = [1 if item == 1 else 0 for item in bedrooms]
bedrooms2 = [1 if item == 2 else 0 for item in bedrooms]
bedrooms3 = [1 if item == 3 else 0 for item in bedrooms]
bedrooms4 = [1 if item == 4 else 0 for item in bedrooms]

bathrooms = realEstateData[:, 2]
bathrooms2 = [1 if item == 2 else 0 for item in bathrooms]
bathrooms3 = [1 if item == 3 else 0 for item in bathrooms]
bathrooms4 = [1 if item == 4 else 0 for item in bathrooms]

priceSF = realEstateData[:, 3]
priceSF = [(item - min(priceSF))/(max(priceSF) - min(priceSF)) for item in priceSF]

status = realEstateData[:, 4]
status2 = [1 if item == 2 else 0 for item in status]
status3 = [1 if item ==3 else 0 for item in status]

Z1 = np.array([bedrooms1, bedrooms2, bedrooms3, bedrooms4])
Z2 = np.array([bathrooms2, bathrooms3, bathrooms4])
Z3 = np.array([priceSF])
Z4 = np.array([status2, status3])
Z5 = np.array([np.ones(len(y))])

theta1 = np.random.rand(4)
theta2 = np.random.rand(3)
theta3 = np.random.rand(1)
theta4 = np.random.rand(2)
theta5 = np.random.rand(1)

t = .001
lam = .012

def prox(theta, t, Z, r, lam):
    R = r - np.matmul(Z.T, theta)
    a = theta - t*np.matmul(-Z, R)
    if t*lam < np.linalg.norm(a):
        return (1 - (t*lam/np.linalg.norm(a)))*a
    else:
        return a*0

for k in range(10000):
    r = y - np.matmul(Z2.T, theta2) - np.matmul(Z3.T, theta3) - np.matmul(Z4.T, theta4) - np.matmul(Z5.T, theta5)
    theta1 = prox(theta1, t, Z1, r, lam)
    r = y - np.matmul(Z1.T, theta1) - np.matmul(Z3.T, theta3) - np.matmul(Z4.T, theta4) - np.matmul(Z5.T, theta5)
    theta2 = prox(theta2, t, Z2, r, lam)
    r = y - np.matmul(Z2.T, theta2) - np.matmul(Z1.T, theta1) - np.matmul(Z4.T, theta4) - np.matmul(Z5.T, theta5)
    theta3 = prox(theta3, t, Z3, r, lam)
    r = y - np.matmul(Z2.T, theta2) - np.matmul(Z3.T, theta3) - np.matmul(Z1.T, theta1) - np.matmul(Z5.T, theta5)
    theta4 = prox(theta4, t, Z4, r, lam)
    r = y - np.matmul(Z2.T, theta2) - np.matmul(Z3.T, theta3) - np.matmul(Z4.T, theta4) - np.matmul(Z1.T, theta1)
    theta5 = prox(theta5, t, Z5, r, lam)

print(theta1, theta2, theta3, theta4, theta5)