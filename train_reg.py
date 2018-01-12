
import nn
import matplotlib.pyplot as plt, numpy as np
import random
import os
import data
import conjugate_gradient
import classification_map


X_train, Y_train, X_control, Y_control = data.DataFactory().createData('plant_with_control')
Y_train = np.atleast_2d(Y_train).T
Y_control = np.atleast_2d(Y_control).T


def NeuronControl(neurons):
    NN = nn.NeuralNetwork([4, 10, 1], ["tanh", "tanh"])
    def gradient_wrapper(weights):
        return NN.cost_grad(weights, X_train, Y_train)
    def cost_wrapper(weights):
        return NN.cost(weights, X_control, Y_control)

    w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=100, printEvery=10)
    NN.params(w_hat)
    return NN.cost(NN.params(), X_control, Y_control)

neurons = []
costs = []
for i in range(1, 11):
    neurons.append(i)
    costs.append(NeuronControl(i))
    
f, ax = plt.subplots(1)
ax.plot(neurons, costs)
ax.set_ylim(ymin=0)
plt.show(f)
quit()

NN = nn.NeuralNetwork([4, 10, 1], ["tanh", "tanh"])
def gradient_wrapper(weights):
    return NN.cost_grad(weights, X_train, Y_train)
def cost_wrapper(weights):
    return NN.cost(weights, X_control, Y_control)

w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=100, printEvery=10)
NN.params(w_hat)
print(NN.cost(NN.params(), X_train, Y_train))
print(NN.cost(NN.params(), X_control, Y_control))

for i in range(X_control.shape[0]):
    xnow = np.atleast_2d(X_control[i,...])
    y = Y_control[i,0]
    y_hat = NN.forward(xnow)[0,0]
    print(y, y_hat, np.abs(y - y_hat))
    