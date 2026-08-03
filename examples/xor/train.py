"""
Start simple:

  1. Define the XOR dataset:
      • Inputs X: shape  (4, 2)  →  [[0,0], [0,1], [1,0], [1,1]]
      • Labels Y: shape  (4, 1)  →  [[0], [1], [1], [0]]
  2. Initialize weights and biases with small random numbers:
      • W₁ (Input → Hidden): shape  (2, 4)  or  (2, 2)
      • b₁: shape  (1, 4)  or  (1, 2)
      • W₂ (Hidden → Output): shape  (4, 1)  or  (2, 1)
      • b₂: shape  (1, 1)
  3. Compute the forward pass step by step:
      • Hidden layer pre-activation: Z₁ = X·W₁ + b₁
      • Hidden layer activation: A₁ = σ(Z₁)
      • Output layer pre-activation: Z₂ = A₁·W₂ + b₂
      • Output prediction: Ŷ = σ(Z₂)
  4. Print the output shape and raw predictions to confirm matrix multiplication works.#
"""

import numpy as np

X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.matrix([[0], [1], [1], [0]])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for y, x in zip(sizes[:-1], sizes[1:])]
        """
        print(self.weights[0])
        print(self.biases[0])
        print(self.weights[1])
        print(self.biases[1])
        """

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a


nw = Network([2, 3, 1])
for row in np.asarray(X):
    sample = row.reshape(2, 1)
    pred = nw.feedforward(sample)
    print(sample.T, pred)
