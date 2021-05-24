import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        # Annul gradient
        pass

    def forward(self, X):
        # Calculi la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        # Met a jour la value du gradient
        pass

    def backward_delta(self, input, delta):
        # Calculate la derivee de l'erreur
        pass


class Linear(Module):

    def __init__(self, n, d, bias=True, type=2):
        self._n = n
        self._d = d
        self._gradient = np.zeros((n, d))
        if type == 0:
            self._parameters = np.random.random((n, d)) - 0.5
        if type == 1:
            self._parameters = np.random.normal(0, 1, (n, d))
        if type == 2:
            self._parameters = np.random.normal(0, 1, (n, d)) * np.sqrt(2 / (n + d))

        if bias:
            if type == 0:
                self._bias = np.random.random((1, d)) - 0.5
            if type == 1:
                self._bias = np.random.normal(0, 1, (1, d))
            if type == 2:
                self._bias = np.random.normal(0, 1, (1, d)) * np.sqrt(2 / (n + d))
            self._bias_grad = np.zeros((1, d))
        else:
            self._bias = None

    def forward(self, X):
        assert X.shape[1] == self._n  # batch * input

        if self._bias is not None:
            return np.dot(X, self._parameters) + self._bias  # batch * output
        else:
            return np.dot(X, self._parameters)  # batch * output

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d
        assert input.shape[0] == delta.shape[0]

        self._gradient += np.dot(input.T, delta)

        if self._bias is not None:
            self._bias_grad += np.sum(delta, axis=0)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

        if self._bias is not None:
            self._bias -= gradient_step * self._bias_grad

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d

        return np.dot(delta, self._parameters.T)

    def zero_grad(self):
        self._gradient = np.zeros((self._n, self._d))
        self._bias_grad = np.zeros((1, self._d))


class TanH(Module):

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return (1 - np.tanh(input) ** 2) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoid(Module):

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        tmp = 1 / (1 + np.exp(-input))
        return delta * (tmp * (1 - tmp))

    def update_parameters(self, gradient_step=1e-3):
        pass


class Softmax(Module):

    def forward(self, X):
        expo = np.exp(X)
        return expo / np.sum(expo, axis=1).reshape((-1, 1))

    def backward_delta(self, input, delta):
        expo = np.exp(input)
        out = expo / np.sum(expo, axis=1).reshape((-1, 1))
        return delta * (out * (1 - out))

    def update_parameters(self, gradient_step=1e-3):
        pass


class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride):
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self._parameters = np.ones((chan_out, chan_in, k_size))
        self.dw = np.zeros((chan_out, chan_in, k_size))

    def forward(self, X):
        len_img = np.size(X[0])
        nb_img = np.size(X, axis=0)

        l = int((len_img - self._k_size + 1) / self._stride)
        r = len_img - self._k_size + 1

        output = np.zeros((nb_img, self._chan_out, l))

        for img in range(nb_img):
            for k in range(self._chan_out):
                i = -self._stride
                j = 0
                while i + self._stride < r:
                    output[img, k, j] = (
                        np.sum(self._parameters[k] * X[img][i + self._stride:i + self._stride + self._k_size]))
                    i += self._stride
                    j += 1

        return output

    def backward_delta(self, X, delta):
        len_img = np.size(delta[0], axis=1)
        nb_img = np.size(X, axis=0)
        l = int((len_img - self._k_size + 1) / self._stride)
        r = len_img - self._k_size + 1

        dw = np.zeros((self._chan_out, self._chan_in, self._k_size))

        for img in range(nb_img):
            for p in range(len_img):
                temp = []
                for c in self._chan_out:
                    temp.append(X[img, c, p:p + self._k_size] * delta)
                dw[p:p + self._k_size] += temp * delta
        return dw

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= self.dw * self._gradient


class MaxPool1D:
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride
        self.idx = []

    def forward(self, X):
        len_img = np.size(X[0])
        nb_img = np.size(X, axis=0)

        l = int((len_img - self._k_size + 1) / self._stride)
        r = len_img - self._k_size + 1

        output = np.zeros((nb_img, self._chan_out, l))

        for img in range(nb_img):
            for k in range(self._chan_out):
                i = -self._stride
                j = 0
                while i + self._stride < r:
                    output[img, k, j] = (
                        np.max(X[img][i + self._stride:i + self._stride + self._k_size]))
                    self.idx.append(img)
        return output

    def backward_delta(self, X, delta):
        return [np.repeat(range(len(self.idx)), 3), self.idx, list(range(3)) * len()(self.idx)]


class flatten:

    def __init__(self):
        pass

    def forward(self, X):
        return X.reshape(self.batch, self.chan_in * self.length)

    def backward(self, X):
        return X.reshape(self.batch, self.length, self.chan_in)


def relu(x):
    if not (type(x) in [list, tuple, np.ndarray]):
        if x < 0:
            return 0
        else:
            return x
    elif type(x) in [list, tuple]:
        x = np.array(x)

    result = x
    result[x < 0] = 0

    return result
