
import nn
import matplotlib.pyplot as plt, numpy as np
from numpy.linalg import norm
import random
import os
import data

# X, Y = data.DataFactory().createData('checkerboard')
X, Y = data.DataFactory().createData('cool')
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1
# print(Y)
# quit()

def conjugrad(f, g, x0, maxiter=2000, gtol=1e-6, steptol=1e-6):
    def bins(f, l, r, eps=1e-7):
        d = r-l
        if f(r) < 0: return r
        while d > eps:
            m = l+d/2
            if f(m) < 0: l += d/2
            d /= 2
        return l + d/2

    x = x0.copy()
    n = x0.size

    i = 0
    for iter in range(maxiter):
        grad = g(x)
        gradnorm = norm(grad)

        if i == 0:
            d = -grad
        else: # conjugate direction
            den = d_1.dot(grad-grad_1) # Hestenes-Stiefel formula ( grad . (grad-grad_1) / d_1 . (grad-grad_1) )
            if abs(den)<gtol:
                d = -grad
            else:
                num = grad.dot(grad-grad_1)
                beta=num/den
                d = -grad + beta * d_1

        # ensure descent direction
        if d.dot(-grad) <= 0:
            i = 0
            d = -grad

        dphi = lambda a: g(x + a*d).dot(d)

        alpha = bins(dphi, 0, 1, gtol)

        x_1 = x
        d_1 = d
        grad_1 = grad

        x = x + alpha * d
        i += 1
        print(iter, gradnorm, f(x))

    return x

def nesterov(f, g, x0, maxiter=400, mu=0.9, speed=0.02):
    x = x0.copy()
    v = np.zeros(x0.size)
    for iter in range(maxiter):
        x_ahead = x + mu * v
        dx = g(x_ahead)
        v = mu * v - speed *dx
        x += v
        print(iter, f(x))
    return x
    
def classification_map(classifier, inp, out, ticks=200):
    # ranges
    xfrom = inp[:,0].min()*1.1
    xto = inp[:,0].max()*1.1
    yfrom = inp[:,1].min()*1.1
    yto = inp[:,1].max()*1.1

    # meshgrid
    h = (xto - xfrom) / ticks
    xx, yy = np.arange(xfrom, xto, h), np.arange(yfrom, yto, h)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.empty(xx.shape, dtype=float)

    # classify meshgrid
    pos = 0
    for x in range(xx.shape[0]):
        for y in range(xx.shape[1]):
            zz[x][y] = classifier(xx[x][y], yy[x][y])

    plt.clf()
    plt.contourf(xx, yy, zz, alpha=0.5) # class separations
    plt.scatter(inp[:,0], inp[:,1], c=out, s=50) # dataset points
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()


NN = nn.NeuralNetwork([2, 10,10, 1], ["tanh", "tanh", "tanh"])
def gradient_wrapper(weights):
    return NN.cost_grad(weights, X, Y)
def cost_wrapper(weights):
    return NN.cost(weights, X, Y)
def classifier_wrapper(x, y):
    inp = np.atleast_2d([x,y])
    result = NN.forward(inp)
    return np.sign(result)

w_hat = conjugrad(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
#w_hat = nesterov(cost_wrapper, gradient_wrapper, NN.params(), maxiter=3000)
NN.params(w_hat)


# count errors
cnt = 0
N = X.shape[0]
for i in range(X.shape[0]):
    px, py = X[i,:]
    if (Y[i,0] != classifier_wrapper(px, py)):
        cnt += 1
print("{} errors from {}, {}".format(cnt, N, 1 - cnt / N))

classification_map(classifier_wrapper, X, Y.ravel())

