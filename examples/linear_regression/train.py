import numpy as np
import matplotlib.pyplot as plt

a = 0.0
b = 0.0
epochs = 400
alpha = 0.1
n = 100
lst = (np.random.rand(n) * 4) - 2
est_lst = (
    2 * lst
    + 1
    + (
        np.random.randn(
            100,
        )
        * 0.1
    )
)


for epoch in range(epochs):
    estimated_y = np.array(a * lst + b)

    loss = np.mean((estimated_y - est_lst) ** 2)

    grad_a = -(2 / n) * np.sum((lst * (est_lst - (a * lst + b))))
    grad_b = -(2 / n) * np.sum((est_lst - (a * lst + b)))
    a = a - alpha * grad_a
    b = b - alpha * grad_b

    if loss <= 10**-8 or epoch == epochs - 1:
        print(f"y = {a:.4f} * x + {b:.4f} (loss = {loss:.8f})")
        print(f"Finished in {epoch} epochs")
        break
    if epoch % 5 == 0:
        print(f"y = {a:.4f} * x + {b:.4f} (loss = {loss:.8f})")


x1, y1 = 0, b
x2, y2 = 5, a * 5 + b
plt.plot(lst, est_lst, "ro")
plt.axline((x1, y1), (x2, y2))
plt.axis((-2, 2, -4, 4))
plt.ylabel("y")
plt.xlabel("x")
plt.show()
