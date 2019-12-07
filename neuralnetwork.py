import pickle
import numpy as np
import math

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"'',
                 test=0
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}
        if test == 0:
            if datapath is not None:
                u = pickle._Unpickler(open(datapath, 'rb'))
                u.encoding = 'latin1'
                self.train, self.valid, self.test = u.load()
            else:
                self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f'b{layer_n}'] = np.zeros((1, all_dims[layer_n]))
            self.weights[f'W{layer_n}'] = np.random.uniform(low=-1/math.sqrt(all_dims[layer_n - 1]), high=1/math.sqrt(all_dims[layer_n - 1]), size=(all_dims[layer_n-1] , all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return x > 0
        return np.maximum(0, x)

    def sigmoid(self, x, grad=False):
        if grad:
            return np.exp(-x)/((1 + np.exp(-x))**2)
        return 1/(1 + np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            return 1 - (((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))**2)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        exp = np.exp(x.T - np.amax(x.T, axis=0))
        return (exp/np.sum(exp, axis=0)).T

    def forward(self, x):
        cache = {f"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        x = np.array(x)
        for i in range(1, self.n_hidden + 1):
            x = np.dot(x, self.weights[f'W{i}']) + self.weights[f'b{i}']
            cache[f"A{i}"] = x
            x = self.activation(x)
            cache[f"Z{i}"] = x

        x = np.dot(x, self.weights[f'W{self.n_hidden + 1}']) + self.weights[f'b{self.n_hidden + 1}']
        cache[f"A{self.n_hidden + 1}"] = x
        cache[f"Z{self.n_hidden + 1}"] = self.softmax(x)
        #np.array([self.softmax(x[i]) for i in range(x.shape[0])])

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE

        grads[f'dA{self.n_hidden + 1}'] = output - labels
        grads[f'db{self.n_hidden + 1}'] = output - labels
        grads[f'dW{self.n_hidden + 1}'] = np.dot(cache[f'Z{self.n_hidden}'].T, grads[f'dA{self.n_hidden + 1}']) / cache['Z0'].shape[0]
        for i in reversed(range(2, self.n_hidden + 1)):
            output = grads[f"dA{i + 1}"].dot(self.weights[f'W{i + 1}'].T)
            grads[f'dZ{i}'] = output
            grads[f'dA{i}'] = np.multiply(self.activation(cache[f'A{i}'], grad=True), output)
            grads[f'dW{i}'] = np.dot(cache[f'Z{i - 1}'].T, grads[f'dA{i}']) / cache['Z0'].shape[0]
            grads[f'db{i}'] = grads[f"dA{i}"]

        output = np.dot(grads["dA2"], self.weights['W2'].T)
        grads['dZ1'] = output
        grads['dA1'] = np.multiply(self.activation(cache[f'A1'], grad=True), output)
        grads['dW1'] = np.dot(cache["Z0"].T, grads['dA1']) / cache['Z0'].shape[0]
        grads['db1'] = grads["dA1"]

        for index in range(1, self.n_hidden + 2):
            grads[f'db{index}'] = np.mean(grads[f'db{index}'], axis=0).reshape(1, -1)

        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f'W{layer}'] -= self.lr * grads[f'dW{layer}']

    def one_hot(self, y):
        onehot = np.zeros((y.shape[0], self.n_classes))
        onehot[np.arange(y.shape[0]), y] = 1
        return onehot

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return - np.sum(np.log(prediction) * labels)/prediction.shape[0]

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                foward_cache = self.forward(minibatchX)
                grads_cache = self.backward(foward_cache, minibatchY)
                self.update(grads_cache)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)

        return valid_loss, valid_accuracy