import nn
import matplotlib.pyplot as plt, numpy as np
import random
import os
import data
import conjugate_gradient
import classification_map

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})
np.random.seed(271828182)

# X, Y = data.DataFactory().createData('checkerboard')
X, Y = data.DataFactory().createData('cool')
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1

NN = nn.NeuralNetwork([2, 10, 1], ["tanh", "tanh"])

def SecondDerivatives(NN, weights, X, Y, h=1e-5):
    grad_zero = NN.cost_grad(weights, X, Y)
    
    grads_forward = np.empty((weights.size, weights.size))
    for i in range(weights.size):
        old = weights[i]
        weights[i] += h
        grads_forward[i,...] = NN.cost_grad(weights, X, Y)
        weights[i] = old
    
    res = np.empty(weights.size)
    for i in range(weights.size):
        gij = grads_forward
        res[i] = (grads_forward[i,i] - grad_zero[i]) / h
    return res

def Salience(NN, weights, X, Y):
    sd = SecondDerivatives(NN, weights, X, Y)
    res = np.zeros(weights.size)
    for i in range(weights.size):
        res[i] = weights[i] ** 2 * sd[i]
    return res

def OBD():
    NN = nn.NeuralNetwork([2, 10, 1], ["tanh", "tanh"])
    def gradient_wrapper(weights):
        return NN.cost_grad(weights, X, Y)
    def cost_wrapper(weights):
        return NN.cost(weights, X, Y)

    for i in range(5):
        w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
        s = Salience(NN,w_hat, X, Y)
        useless = np.argmin(np.abs(s))
        w_hat[useless] = 0
        print("Removing axon {} with salience {}".format(useless, s[useless]))
        NN.params(w_hat)
    return NN

NN = OBD()

# count errors
def classifier_wrapper(x, y):
    inp = np.atleast_2d([x,y])
    result = NN.forward(inp)
    return np.sign(result)
cnt = 0
N = X.shape[0]
for i in range(X.shape[0]):
    px, py = X[i,:]
    if (Y[i,0] != classifier_wrapper(px, py)):
        cnt += 1
print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

classification_map.plot(classifier_wrapper, X, Y.ravel())

