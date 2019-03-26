import numpy as np

def tanh(x):
    return np.tanh(x)

def tan_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1 / ( 1 + np.exp(-x) )

def logistic_deriv(x):
    return logistic(x) * ( 1 - logistic(x) )

class NeuralNetwork:
    def __init__(self, layers, activation = 'logistic'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tan_deriv
        self.weights = []
        for i in range(1,len(layers) - 1):
            if i == 1:
                self.weights.append((2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1) * 0.25)
            if i < len(layers) -  2:
                self.weights.append((2 * np.random.random((layers[i] + 1, layers[i+1] + 1)) - 1) * 0.25)
            else:
                self.weights.append((2 * np.random.random((layers[i] + 1, layers[i+1])) - 1) * 0.25)
        print("shape of weights: ",[item.shape for item in self.weights])

    def fit(self, X, y, learning_rate = 0.25, epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        print("shape of X: ",X.shape)
        temp[:, 0: -1] = X
        X = temp
        y = np.array(y)
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]# a 为从 X 中随机抽取的一行

            #正向更新
            for line in range(len(self.weights)):
                a.append(self.activation(np.dot(a[line], self.weights[line])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            #print("shape of delta: ",np.array(deltas).shape)
            #反向更新
            for l in range(1, len(a) - 1):
                print("l euqals:",str(l))
                deltas.append(deltas[-1].dot(self.weights[-l].T) * self.activation_deriv(a[-1-l]))
                deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        x = temp
        for i in range(len(self.weights)):
            x = self.activation(np.dot(x, self.weights[i]))
        return x



