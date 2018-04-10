import nn
import matplotlib.pyplot as plt, numpy as np
import random
import os
import data
import conjugate_gradient
import classification_map
np.random.seed(271828182)

# X, Y = data.DataFactory().createData('checkerboard')
X, Y = data.DataFactory().createData('cool')
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1

NN = nn.NeuralNetwork([2, 10, 1], ["tanh", "tanh"])
def gradient_wrapper(weights):
    return NN.cost_grad(weights, X, Y)
def cost_wrapper(weights):
    return NN.cost(weights, X, Y)
def classifier_wrapper(x, y):
    inp = np.atleast_2d([x,y])
    result = NN.forward(inp)
    return np.sign(result)

w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=600)
NN.params(w_hat)

# count errors
cnt = 0
N = X.shape[0]
for i in range(X.shape[0]):
    px, py = X[i,:]
    if (Y[i,0] != classifier_wrapper(px, py)):
        cnt += 1
print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

classification_map.plot(classifier_wrapper, X, Y.ravel())

