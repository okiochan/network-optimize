
import activations
import numpy as np, math, json, time, sys, os


class NeuralNetwork:
    # I[i] = W[i-1].dot(O[i-1])
    # O[i] = f[i-1](I[i])
    # W[i] = weights between layer i and i+1
    # b[i] = biases between layer i and i+1
    # delta[i] = dCostFuncion/dI[i]
    # examples:
    # O[0] = input data
    # I[0] = nothing, rubbish
    # O[l-1] = output of neural net

    def __init__(self, layers=None, functions=None):
        # weight matrices, bias vectors
        self.layers = layers
        self.len = len(layers)
        self.W, self.b = [], []
        for i in range(self.len - 1):
            shape = self.__get_shape(i)
            #initialize special random weights
            self.W.append(np.random.randn(shape[0], shape[1]) * math.sqrt(2 / shape[0]))
            self.b.append(np.zeros(shape[1]))

        # functions
        assert len(layers) - 1 == len(functions), "function count doesn't match"
        self.__func_list = functions
        self.__init_functions(functions)

    #matrix to vector, if params != 0; otherwise: vector to matrix
    def params(self, param=None):
        if param is None: # return params
            #ravel: matrix to string
            ret = self.W[0].ravel()
            for i in range(1, self.len - 1):
                ret = np.concatenate((ret, self.W[i].ravel()))
            for i in range(0, self.len - 1):
                ret = np.concatenate((ret, self.b[i].ravel()))
            return ret
        else: # set params
            pos = 0
            W, b = [], []
            for i in range(self.len - 1):
                shape = self.__get_shape(i)
                size = shape[0] * shape[1]
                W.append(param[pos:pos+size].reshape(shape[0], shape[1]))
                pos += size
            for i in range(self.len - 1):
                shape = self.__get_shape(i)
                size = shape[1]
                b.append(param[pos:pos+size].reshape(shape[1]))
                pos += size
            self.W, self.b = W, b

    def forward(self, x):
        # propogate inputs through the network
        self.I = [0]
        self.O = [x]
        ret = x
        for i in range(self.len - 1):
            I = ret.dot(self.W[i]) + self.b[i]
            O = self.f[i](I)
            self.I.append(I)
            self.O.append(O)
            ret = O
        return ret

    def cost(self, params, x, y):
        self.params(params)
        self.forward(x)
        S = np.sum((y - self.O[-1]) ** 2) / 2 / x.shape[0]
        return S

    #cheslennyi gradient
    def cost_grad_num(self, params, x, y):
        self.params(params)

        h = 1e-7
        sz = params.size
        ret = np.empty((sz,), dtype=float)
        H = np.zeros((sz,), dtype=float)
        
        fm = self.cost(params, x, y)
        for i in range(sz):
            if i >= 1: H[i - 1] = 0.0
            H[i] += h
            f1 = self.cost(params + H, x, y)
            ret[i] = (f1 - fm) / h
        return ret

    #backprop
    def cost_grad(self, params, x, y):
        self.params(params)

        delta = [np.empty((x.shape[0], self.layers[q])) for q in range(self.len)]
        dW = [np.empty(self.__get_shape(q)) for q in range(self.len - 1)]
        db = [np.empty(self.layers[q + 1]) for q in range(self.len - 1)]

        T = x.shape[0] # test cases
        m = self.layers[-1] # output size
        self.forward(x) # forward all test cases

        # compute delta for last layer
        delta[-1] = (self.O[-1] - y) * self.fp[-1](self.I[-1]) / T # least squares for regression

        # compute delta for previous layers
        for q in range(self.len - 2, 0, -1):
            delta[q] = np.dot(delta[q + 1], self.W[q].T) * self.fp[q - 1](self.I[q])

        # compute gradient from delta
        for q in range(self.len - 1):
            O = self.O[q].reshape(T, self.layers[q], 1)
            D = delta[q + 1].reshape(T, 1, self.layers[q + 1])
            dW[q] = np.sum(O * D, axis=0)
            db[q] = np.sum(delta[q + 1], axis=0)

        # serialize gradient
        ret = dW[0].ravel()
        for q in range(1, self.len - 1):
            ret = np.concatenate((ret.ravel(), dW[q].ravel()))
        for q in range(0, self.len - 1):
            ret = np.concatenate((ret.ravel(), db[q].ravel()))

        return ret

    def __init_functions(self, functions):
        # convert strings of functions to real functions and their derivatives
        self.f, self.fp = [], []
        for i in range(len(functions)):
            self.f.append(activations.func[functions[i]])
            self.fp.append(activations.funcprime[functions[i]])

    def __get_shape(self, i):
        return (self.layers[i], self.layers[i + 1]) # shape of the weight matrix
