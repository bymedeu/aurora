"""
1. Create a script:  examples/single_neuron/train.py
  2. Generate a 2D Dataset: Generate two clusters (blobs) of points in 2D space. For instance:
      • Class 0: Points clustered around (-1.5, - 1.5)
      • Class 1: Points clustered around (1.5,1.5)
  3. Define the Neuron:
      • Inputs: x₁ and x₂ (so your inputs array has shape  (N, 2) ).
      • Parameters: w₁, w₂, and b.
  4. Implement Sigmoid and BCE:
      • Write a helper for σ(z).
      • Write a helper for BCE loss.
  5. Optimize:
      • You will need the derivatives of BCE loss with respect to w₁, w₂, and b.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

classes = [(-1.5, -1.5), (1.5, 1.5)]
size = (1000 // 2) * 2
dist = 1.0
epochs = 400
alpha = 0.1

x = classes[0][0] + dist * np.random.randn(2, size // 2)
y = classes[1][0] + dist * np.random.randn(2, size // 2)
z = np.append(x.transpose(), y.transpose(), axis=0)
YTrue = np.append(np.zeros(size // 2), np.ones(size // 2))


class single_neuron:
    w1_: float
    w2_: float
    b_: float

    def __init__(self, w1=1, w2=1, b=0) -> None:
        self.w1_ = w1
        self.w2_ = w2
        self.b_ = b
        return

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(np.clip(-z, -50, 50)))

    def getGroup(self, z) -> np.ndarray:
        mat = np.array([self.w1_, self.w2_]).transpose()
        nz = z @ mat + self.b_
        return self._sigmoid(nz)

    def BCE(self, Z) -> float:
        Z = np.clip(Z, 1e-7, 1 - 1e-7)
        return -np.average(YTrue * np.log(Z) + (1 - YTrue) * np.log(1 - Z))

    def updateWeights(self, w1, w2, b):
        self.w1_ -= alpha * w1
        self.w2_ -= alpha * w2
        self.b_ -= alpha * b


neuron = single_neuron()

for epoch in range(epochs):
    groups = neuron.getGroup(z)
    BCE = neuron.BCE(groups)

    if epoch % 20 == 0:
        print(epoch, BCE)

    Wgrad = (z.T @ ((groups - YTrue) * groups * (1 - groups))).flatten() / size
    Bgrad = np.average(groups - YTrue)
    neuron.updateWeights(Wgrad[0], Wgrad[1], Bgrad)
else:
    print(f"{epochs}/{epochs} ran total")


res = neuron.getGroup(z) >= 0.5
# Boolean mask for detecting wheter the prediction was correct or no
mistakes = res != YTrue
plt.scatter(z[~mistakes, 0], z[~mistakes, 1], label="Correct")
plt.scatter(z[mistakes, 0], z[mistakes, 1], label="Incorrect")
x1_vals = np.linspace(z[:, 0].min(), z[:, 0].max(), 100)
x2_vals = -(neuron.w1_ * x1_vals + neuron.b_) / neuron.w2_
plt.plot(x1_vals, x2_vals, label="Decision boundary")
print(
    f"Group delimiter = {neuron.w1_:.3f} * x + {neuron.w2_:.3f} * y + {neuron.b_:.3f}"
    "x and y being the (x,y) coordinates of a point"
)
plt.legend()
plt.show()
