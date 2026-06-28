# Aurora Progress Report — 2026-06-27

## 1. Global Status

Current main focus: M2 — Small Neural Network
Current milestone: M2
Estimated global progress: 2 / 6 milestones complete
Current priority: Implement XOR with a hidden layer (M2)

---

## 2. Milestone Progress

| Milestone | Status | Estimated % | Evidence | Missing |
|-----------|--------|-------------|----------|---------|
| M0 — Linear Regression | Done | 100% | `examples/linear_regression/`, `docs/notes/linear_regression.md` | None |
| M1 — Single Neuron | Done | 100% | `examples/single_neuron/train.py`, `docs/notes/single_neuron.md`, MQC passed | None |
| M2 — Small Neural Network | Not started | 0% | — | Everything |
| M3 — Autograd | Not started | 0% | — | Everything |
| M4 — PyTorch Reconstruction | Not started | 0% | — | Everything |
| M5 — Attention | Not started | 0% | — | Everything |
| M6 — Tiny Transformer | Not started | 0% | — | Everything |

---

## 3. Evidence Reviewed

- `examples/single_neuron/train.py` — working training loop, correct sigmoid with clipping, BCE loss, complete gradient (includes sigmoid derivative factor), decision boundary visualization, loss logging every 20 epochs.
- `docs/notes/single_neuron.md` — covers neuron definition, sigmoid, BCE formula, decision boundary geometry, training dynamics with concrete numerical results (w1=1.175, w2=1.197, b=-0.026, BCE=0.0915 after 400 epochs).
- MQC assessment passed (answers B, C, A, A, C).
- `docs/roadmap.md` — M0 and M1 all criteria checked off.

---

## 4. What Is Solid

- Full understanding of what a neuron computes: linear combination → sigmoid → probability.
- BCE loss derivation and numerical stability via clipping.
- Chain rule application: sigmoid derivative factor correctly included in gradient.
- Decision boundary geometry: the line w1·x + w2·y + b = 0 and why it appears at sigmoid output = 0.5.
- Clean, structured code.

---

## 5. What Is Weak or Missing

- No experience yet with hidden layers or non-linear decision boundaries.
- No understanding of why a single neuron cannot solve XOR (to be explored in M2).
- No backpropagation through multiple layers.

---

## 6. Current Risks

- Jumping to M2 before fully internalising the chain rule may cause confusion when backprop spans multiple layers.

---

## 7. Next Smallest Useful Step

Implement a 2-layer network (one hidden layer, one output neuron) from scratch and attempt to solve the XOR problem.

---

## 8. Resources to Study Next

1. "Yes you should understand backprop" — Andrej Karpathy: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
2. "Neural Networks and Deep Learning" — Chapter 2 (backpropagation): http://neuralnetworksanddeeplearning.com/chap2.html

---

## 9. Do Not Do Yet

- PyTorch (reserved for M4).
- Autograd engines (reserved for M3).
- Attention or transformers.
- Multi-class classification.

---

## 10. AI Mentor Notes

The student has completed M0 and M1 with solid evidence. They understand the single-neuron forward pass, BCE loss, numerical stability, and the chain rule for a one-layer network. The next focus is M2: XOR with a hidden layer. The student must not be given the backpropagation implementation — guide them to derive the gradient layer by layer using the chain rule they already know from M1. Do not write code for them.
