import numpy as np


def sigmoid(x):
    x = np.clip(np.copy(x), -45, 45)
    val = np.exp(-x)
    return 1 / (1 + val)

def sigmoid_prime(x):
    val = sigmoid(x)
    return val * (1 - val)

def linear(x):
    return x

def linear_prime(x):
    return np.ones(x.shape, dtype=float)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    x = np.clip(np.copy(x), -45, 45)
    exps = np.exp(x)
    try:
        sum = np.sum(exps, axis=1).reshape(x.shape[0], 1)
    except:
        sum = np.sum(exps)
    return exps / sum

def relu(x):
    x = x.copy()
    select = x < 0
    x[select] *= 0
    return x

def relu_prime(x):
    x = x.copy()
    select = x < 0
    x[select] = 0
    x[np.logical_not(select)] = 1
    return x
    
def lrelu(x):
    x = x.copy()
    select = x < 0
    x[select] *= 0.01
    return x

def lrelu_prime(x):
    x = x.copy()
    select = x < 0
    x[select] = 0.01
    x[np.logical_not(select)] = 1
    return x

def sin(x):
    return np.sin(x)
def sin_prime(x):
    return np.cos(x)

func = {
    "logistic": sigmoid,
    "linear": linear,
    "softmax": softmax,
    "tanh": tanh,
    "relu": relu,
    "lrelu": lrelu,
    "sin": sin,
}

funcprime = {
    "logistic": sigmoid_prime,
    "linear": linear_prime,
    "softmax": None,
    "tanh": tanh_prime,
    "relu": relu_prime,
    "lrelu": lrelu_prime,
    "sin": sin_prime,
}
