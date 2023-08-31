import network
import numpy as np


def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


tr_d = [
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [0, 0, 0, 1]
]


training_inputs = [np.reshape(x, (2, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = zip(training_inputs, training_results)

te_d = [
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [0, 0, 0, 1]
]

test_inputs = [np.reshape(x, (2, 1)) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])

net = network.Network([2, 1, 2])
net.SGD(training_data, 1000, 10, 3.0, test_data=test_data)


print(net.feedforward([[0], [0]]))
print(net.feedforward([[1], [0]]))
print(net.feedforward([[0], [1]]))
print(net.feedforward([[1], [1]]))

