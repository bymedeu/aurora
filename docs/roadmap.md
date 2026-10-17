# Aurora Learning Roadmap

This document outlines the milestones, learning objectives, restrictions, and completion criteria for the Aurora project, moving from numerical foundations to a character-level transformer from scratch.

---

## Milestone M0 — Linear Regression From Scratch
**Goal:** Understand prediction, loss, gradients, and gradient descent.

### Restrictions
* Start with plain Python if needed.
* Use NumPy only when useful.
* Do not use PyTorch.
* Do not use scikit-learn.

### Completion Criteria
- [x] The student can explain the model $y = ax + b$.
- [x] The student can explain mean squared error (MSE).
- [x] The student can manually derive the update direction (gradients).
- [x] The student can train a tiny model on synthetic data.
- [x] The student can write a short note explaining the result.

---

## Milestone M1 — Single Neuron
**Goal:** Understand weights, bias, activation, and binary classification.

### Restrictions
* No PyTorch.
* No high-level ML library.

### Completion Criteria
- [x] The student can explain what a neuron computes.
- [x] The student can explain why activation functions exist.
- [x] The student can train a single neuron on a toy dataset.
- [x] The code must be clean, optimised, and well‑structured; AI may pinpoint specific lines for improvement.

---

## Milestone M2 — Small Neural Network
**Goal:** Understand hidden layers, forward pass, backpropagation, and non-linearity.

### Completion Criteria
- [ ] The student can solve XOR.
- [ ] The student can explain why a single linear model cannot solve XOR.
- [ ] The student can explain forward and backward passes.

---

## Milestone M3 — Autograd
**Goal:** Build a tiny automatic differentiation engine.

**Status:** Implementation complete on **October 17, 2026**, 59 days before the
December 15 deadline. The owner requested the milestone implementation closeout
and a switch to MyStarCeiling M2. Aurora's next phase is M4 after a short entry
review; independent conceptual mastery is not inferred from passing tests.

### Delivered implementation
- [x] Scalar computation graphs with parent identity and local backward rules.
- [x] Reverse topological backward traversal and shared-gradient accumulation.
- [x] Explicit gradient clearing and fresh parameter leaves after updates.
- [x] Arithmetic, non-negative integer powers, and stable sigmoid.
- [x] Differentiable MSE and gradient-descent parameter updates.
- [x] An importable linear Aurora prototype and a seven-parameter nonlinear experiment.
- [x] Numerical derivative checks, training regression checks, and 61 passing tests.

### Learning review carried into the M4 entry session
- [ ] Independently explain a hidden-weight chain-rule contribution, accumulation,
      graph lifetime, and average traversal time/memory costs.
- [ ] Compare the implementation conceptually to PyTorch autograd.

See [M3 progress and completion evidence](milestones/M3_progress.md).

---

## Milestone M4 — PyTorch Reconstruction
**Goal:** Learn PyTorch by recognizing concepts already implemented in Aurora.

### Completion Criteria
- [ ] The student can use PyTorch tensors.
- [ ] The student can use PyTorch autograd.
- [ ] The student can write a manual training loop.
- [ ] The student can explain `nn.Module`, `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.

---

## Milestone M5 — Attention
**Goal:** Understand query, key, value, dot-product attention, masking, and softmax.

### Completion Criteria
- [ ] The student can implement a small attention mechanism.
- [ ] The student can explain what attention weights represent.
- [ ] The student can visualize attention on a toy sequence.

---

## Milestone M6 — Tiny Transformer
**Goal:** Assemble embeddings, positional encoding, attention, feed-forward layers, residual connections, and normalization.

### Completion Criteria
- [ ] The student can train a tiny character-level model.
- [ ] The student can explain every major component.
- [ ] The student can write a technical blog post about the implementation.
