
import nn
import matplotlib.pyplot as plt, numpy as np
import random
import os
import data
import conjugate_gradient
import classification_map

# X, Y = data.DataFactory().createData('checkerboard')
X, Y = data.DataFactory().createData('cool')
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1

def AddNeuron(NN):
    a, b, c = NN.layers
    NN.layers[1] += 1
    NN.b[0] = np.concatenate((NN.b[0], [0]))
    n = NN.W[0].shape[1]
    NN.W[0] = np.hstack((NN.W[0], np.random.randn(a,1) * 1e-5))
    NN.W[1] = np.vstack((NN.W[1], np.random.randn(1,c) * 1e-5))


start = 3
NN = nn.NeuralNetwork([2, start, 1], ["tanh", "tanh"])
# AddNeuron(NN)
# quit()

def gradient_wrapper(weights):
    return NN.cost_grad(weights, X, Y)
def cost_wrapper(weights):
    return NN.cost(weights, X, Y)
def trained_cost():
    w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
    return NN.cost(w_hat, X, Y)


now = start
neurons = []
costs = []
for i in range(10):
    print("Making {} neurons".format(now))
    neurons.append(now)
    costs.append(trained_cost())
    now += 1
    AddNeuron(NN)

plt.plot(neurons, costs)
plt.show()


# count errors
# cnt = 0
# N = X.shape[0]
# for i in range(X.shape[0]):
    # px, py = X[i,:]
    # if (Y[i,0] != classifier_wrapper(px, py)):
        # cnt += 1
# print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))


