import random
from typing import Optional, Tuple
import os
import time

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []

    def save_weights_and_biases(self, path: str):
        """
        Saves the network's weights and biases to ``@path/weights.npy`` and ``@path/biases.npy``.

        If ``@path`` does not exist, it will be created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/weights.npy", "wb+") as f:
            np.save(f, np.asarray(self.weights, dtype=object))

        with open(f"{path}/biases.npy", "wb+") as f:
            np.save(f, np.asarray(self.biases, dtype=object))

    def load_weights_and_biases(self, path: str):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: Weights and biases could no be loaded because the folder '{path}' does not exist")

        try:
            with open(f"{path}/weights.npy", "rb") as f:
                loaded_weights = np.load(f, allow_pickle=True)
                loaded_weights = np.asarray(loaded_weights, dtype=object)
                self.weights = loaded_weights

            with open(f"{path}/biases.npy", "rb") as f:
                loaded_biases = np.load(f, allow_pickle=True)
                loaded_biases = np.asarray(loaded_biases, dtype=object)
                self.biases = loaded_biases
        except FileNotFoundError:
            raise SystemExit(f"ERROR: Weights and biases could no be loaded because the file 'weights.npy' or 'biases.npy' does not exist in folder {path}.")


    def initialize_weights_and_biases(self, path: Optional[str] = None):
        """
        Initializes the network's weights and biases.

        ``@path`` is an optional string to a folder where weights and biases will be loaded from. Ex: 'data/test'. \n
        If ``@path`` does not exist, or does not contain 'weights.npy' and 'biases.npy', the program will exit. 
        """
        if path == None:
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        else:
            self.load_weights_and_biases(path)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, epoch_result_path: str,
            test_data=None):
        """Trains the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""

        
        if os.path.exists(epoch_result_path):
            raise SystemExit(f"ERROR: Epoch path: {epoch_result_path}, already exists.")
        else: 
            os.makedirs(epoch_result_path)

        training_data = list(training_data)
        n = len(training_data)
        n_test = -1

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        epoch_file =  open(f"{epoch_result_path}/epoch-result.txt", "a")

        start_time = time.time()

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if test_data:
                result = f"{j+1}: {self.evaluate(test_data) / n_test}: {elapsed_time}"
                print(result)
                epoch_file.write(result + "\n")
            else:
                print(f"Epoch {j+1} complete")

        epoch_file.close()
            

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, input, wanted_output):
        """Returns a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = input
        activations = [input]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BP1,                                  Hadamard product
        delta = (2 * (activations[-1] - wanted_output)) * sigmoid_prime(zs[-1])
        # BP3
        nabla_b[-1] = delta
        # BP4
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            # BP2                                           Hadamard product
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            # BP3
            nabla_b[-l] = delta
            # BP4
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Returns the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Returns the vector of partial derivatives partial C_x partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
