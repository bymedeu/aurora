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
