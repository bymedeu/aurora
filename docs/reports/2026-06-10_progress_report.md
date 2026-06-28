# Aurora Progress Report — 2026-06-10

## 1. Global Status

Current main focus: Transitioning to Milestone M1 (Single Neuron)
Current milestone: M1 — Single Neuron
Estimated global progress: 14% (1 out of 7 milestones complete)
Current priority: Understand weights, bias, activation, and binary classification.

## 2. Milestone Progress

| Milestone | Status | Estimated % | Evidence | Missing |
|----------|--------|-------------|----------|---------|
| M0 — Linear Regression | Done | 100% | `docs/notes/linear_regression.md`, `examples/linear_regression/train.py` | None |
| M1 — Single Neuron | Not started | 0% | None | Conceptual study, math, implementation |
| M2 — Small Neural Network | Not started | 0% | None | ... |
| M3 — Autograd | Not started | 0% | None | ... |
| M4 — PyTorch Reconstruction | Not started | 0% | None | ... |
| M5 — Attention | Not started | 0% | None | ... |
| M6 — Tiny Transformer | Not started | 0% | None | ... |

## 3. Evidence Reviewed

- [linear_regression.md](file:///home/amade/projects/aurora/docs/notes/linear_regression.md) - Geometrical and mathematical documentation of linear regression model, MSE loss, and gradient updates.
- [train.py](file:///home/amade/projects/aurora/examples/linear_regression/train.py) - Vectorized implementation of gradient descent optimization with synthetic noisy data generation and Matplotlib visualization.

## 4. What Is Solid

- Analytical grasp of $y = ax + b$ and its geometric visualization.
- Vectorized gradient derivations for weight $a$ and bias $b$ under Mean Squared Error (MSE).
- Understanding optimization dynamics (gradient direction and subtraction update rules).
- Visualization of optimization via scatter plots of noisy targets vs. fitted lines.

## 5. What Is Weak or Missing

- Milestone M0 is fully solid. Ready to step up difficulty.

## 6. Current Risks

- Transitioning to neural networks and activation functions requires understanding probability boundaries and non-linear mappings.

## 7. Next Smallest Useful Step

- Create `docs/notes/single_neuron.md` and research the purpose of an **activation function** (e.g., Sigmoid).

## 8. Resources to Study Next

- **[StatQuest: Odds and Log-Odds (Logistic Regression)](https://www.youtube.com/watch?v=ARfXDSkQf1Y)** — for moving from regression to binary classification.
- **[But what is a neural network? (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk)** — introduction to neurons and activations.

## 9. Do Not Do Yet

- Avoid PyTorch entirely. Continue using NumPy and pure Python.
- Do not build multi-layered architectures yet. Focus purely on a single unit.

## 10. AI Mentor Notes

- The student successfully derived and vectorized basic linear regression and MSE.
- The mentoring style should be highly subtle: do not point out syntax or logic bugs directly. Instead, ask general structural questions or link to documentation/articles when troubleshooting.
