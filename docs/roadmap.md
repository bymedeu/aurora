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

**Status:** Complete. Implementation delivered on **October 17, 2026**, 59 days
before the December 15 deadline; milestone closure confirmed on November 4, 2026.
Aurora continues with M4, without an additional M3 entry review.

### Delivered implementation
- [x] Scalar computation graphs with parent identity and local backward rules.
- [x] Reverse topological backward traversal and shared-gradient accumulation.
- [x] Explicit gradient clearing and fresh parameter leaves after updates.
- [x] Arithmetic, non-negative integer powers, and stable sigmoid.
- [x] Differentiable MSE and gradient-descent parameter updates.
- [x] An importable linear Aurora prototype and a seven-parameter nonlinear experiment.
- [x] Numerical derivative checks, training regression checks, and 61 passing tests.

See [M3 progress and completion evidence](milestones/M3_progress.md).

---

## Milestone M4 — NumPy Tensors and Autograd
**Goal:** Build array-valued automatic differentiation and a small trainable
network using Aurora's own mechanisms, Python, and NumPy.

### Current focus
- Tensor meaning, shapes, examples, features, and batches.
- Broadcasting and reductions, explained with small numerical examples.

### Completion Criteria
- [ ] Explain tensor values, shapes, dependencies, and gradients.
- [ ] Implement a bounded set of array operations with automatic differentiation.
- [ ] Explain and handle broadcasting, reductions, and gradient accumulation.
- [ ] Train a small network using Aurora's own layers, loss, and parameter updates.
- [ ] Verify the implemented mechanisms and document their behavior and limitations
      when concrete implementation work is available.

### Restrictions
- Python and NumPy; no PyTorch or external automatic differentiation engine.
- Keep correctness checks and cost analysis proportional to concrete work;
  these are later implementation activities, not the current introductory lesson.

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
