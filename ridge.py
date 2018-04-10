import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import data
np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

def RegressionInv(X):
    return X.T.dot(X)

def LSRegression(X,y):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn
    res = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def RidgeRegression(X,y,reg):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn
    res = np.linalg.inv(X.T.dot(X) + np.eye(n+1) * reg).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def LSLoss(X,y,a,b):
    l = X.shape[0]

    loss = 0
    for i in range(l):
        loss += (a.dot(X[i,])+b-y[i])**2
    return loss * 0.5 / l

X_train, Y_train, X_control, Y_control = data.DataFactory().createData('plant_with_control')

def CostForRidge(reg):
    a, b = RidgeRegression(X_train, Y_train,reg)
    return LSLoss(X_control,Y_control, a, b)

ridges = [10**i for i in range(-4,5)]
values = []
for r in ridges:
    values.append(CostForRidge(r))

plt.plot(np.log(ridges) / np.log(10), values)
plt.ylim(ymin=0)
plt.show()

# a, b = RidgeRegression(X_train, Y_train,0.001)
# print(LSLoss(X_train,Y_train, a, b))
# print(LSLoss(X_control,Y_control, a, b))
# print(a,b)
