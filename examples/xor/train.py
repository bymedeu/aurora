"""
Start simple:

  1. Define the XOR dataset:
      • Inputs X: shape  (4, 2)  →  [[0,0], [0,1], [1,0], [1,1]]
      • Labels Y: shape  (4, 1)  →  [[0], [1], [1], [0]]
  2. Use vertical activation vectors and store one row of outgoing
     weights per source neuron for a 2 → 3 → 1 network:
      • a₀ (Input): shape  (2, 1)
      • W₁ (Input → Hidden): shape  (2, 3)
      • b₁: shape  (3, 1)
      • W₂ (Hidden → Output): shape  (3, 1)
      • b₂: shape  (1, 1)
  3. Compute the forward pass step by step:
      • Hidden layer pre-activation: Z₁ = W₁ᵀ·a₀ + b₁
      • Hidden layer activation: A₁ = σ(Z₁)
      • Output layer pre-activation: Z₂ = W₂ᵀ·A₁ + b₂
      • Output prediction: Ŷ = σ(Z₂)
  4. Print the output shape and raw predictions to confirm matrix multiplication works.#
"""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)
learning_rate = 0.1
epochs = 50001
seed = 0
np.random.seed(seed)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.zs = [np.array(0) for _ in zip(sizes[:-1], sizes[1:])]
        """
        print(self.weights[0])
        print(self.biases[0])
        print(self.weights[1])
        print(self.biases[1])
        """

    def feedforward(self, a):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            self.zs[i] = w.T @ a + b
            a = sigmoid(self.zs[i])
        return a

    def backprop(self, a0, target, prediction):
        layers = len(self.weights)
        wgradients = [None] * (layers)
        bgradients = [None] * (layers)
        activations = [a0]
        for z in self.zs:
            activations.append(sigmoid(z))

        deltai = prediction - target
        wgradients[layers - 1] = activations[layers - 1] @ deltai.T
        bgradients[layers - 1] = deltai
        for L in range(layers - 2, -1, -1):
            deltai = (self.weights[L + 1] @ deltai) * sigmoid_prime(self.zs[L])
            wgradients[L] = activations[L] @ deltai.T
            bgradients[L] = deltai

        return wgradients, bgradients

    def update(self, wgradients, bgradients, learning_rate):
        for i, (w, b) in enumerate(zip(wgradients, bgradients)):
            self.weights[i] -= learning_rate * w
            self.biases[i] -= learning_rate * b


nw = Network([2, 3, 1])
bce = 0.0
for epoch in range(epochs):
    bce = 0.0
    for index, row in enumerate(X):
        sample = row.reshape(2, 1)
        pred = nw.feedforward(sample)
        target = Y[index].reshape(1, 1)
        stable_pred = np.clip(pred, 1e-12, 1 - 1e-12)
        bce += float(
            -(
                target * np.log(stable_pred) + (1 - target) * np.log(1 - stable_pred)
            ).item()
        )
        wgrads, bgrads = nw.backprop(sample, target, pred)
        nw.update(wgrads, bgrads, learning_rate)
    bce /= len(X)
    if not epoch % 500:
        # print(f"grad_w1={wgrads[0]}\ngrad_b1={bgrads[0]}\n")
        # print(f"grad_w2={wgrads[1]}\ngrad_b2={bgrads[1]}\n")
        print("\rBCE: ", bce, end="")
print()
probas = []
final = []
expected = []
for index, row in enumerate(X):
    sample = row.reshape(2, 1)
    val = nw.feedforward(sample)[0][0]
    probas.append(np.round(val, 4).item())
    final.append(1 if val.item() >= 0.5 else 0)
    expected.append(Y[index].reshape(1, 1)[0][0].astype(int).item())
print(f"probas: {probas}")
print(f"final: {final}")
print(f"expected: {expected}")
assert final == expected
assert bce < 0.01
