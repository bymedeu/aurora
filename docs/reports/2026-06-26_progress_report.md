# Aurora Progress Report — 2026-06-26

## 1. Global Status
- **Current main focus:** Debugging the single‑neuron training loop (BCE stability, proper weight update, visualisation).
- **Current milestone:** M1 – Single Neuron.
- **Estimated global progress:** ~55 % (M0 complete, M1 in‑progress).
- **Current priority:** Finish the conceptual fixes for BCE, gradient descent, and add a decision‑boundary plot.

## 2. Milestone Progress
| Milestone | Status | Estimated % | Evidence | Missing |
|----------|--------|-------------|----------|---------|
| M0 — Linear Regression | Done | 100% | `docs/milestones/M0_progress.md` shows completed loss curve and notes. | – |
| M1 — Single Neuron | In progress | 55% | `examples/single_neuron/train.py` (data generation, neuron class, forward pass). <br> `docs/notes/single_neuron.md` (conceptual write‑up). <br> Runtime warnings from BCE (`log(0)`). | Correct BCE sign & clipping, proper gradient descent step, decision‑boundary visualisation, validation loss check. |
| M2 — Small Neural Network | Not started | 0% | – | – |
| M3 — Autograd | Not started | 0% | – | – |
| M4 — PyTorch Reconstruction | Not started | 0% | – | – |
| M5 — Attention | Not started | 0% | – | – |
| M6 — Tiny Transformer | Not started | 0% | – | – |

## 3. Evidence Reviewed
- `examples/single_neuron/train.py` (implementation status, warnings).
- `docs/notes/single_neuron.md` (theoretical coverage).
- `docs/milestones/M0_progress.md` (completed linear regression).
- Console output showing BCE warnings and low negative loss.

## 4. What Is Solid
- Dataset generation and basic neuron class structure are in place.
- Conceptual understanding of sigmoid, BCE, and gradient descent is documented.
- Overall project file layout follows the Aurora roadmap.

## 5. What Is Weak or Missing
- BCE implementation lacks a leading minus sign and numerical clipping → produces negative loss and runtime warnings.
- Weight update overwrites gradients rather than performing a gradient‑descent step.
- No visualisation of the decision boundary or class predictions.
- No validation set to assess generalisation.

## 6. Current Risks
- Continued numerical instability may mask learning progress.
- Over‑writing weights can lead to divergence or “random walk” behaviour.
- Without visual feedback, it’s hard to confirm that the model separates the two clusters.

## 7. Next Smallest Useful Step
**Add three conceptual safeguards:**
1. Clip the sigmoid output before feeding it into BCE (`np.clip(p, 1e-7, 1-1e-7)`).
2. Insert a leading minus sign in the BCE formula to make the loss non‑negative.
3. Change the weight update to `self.w1_ -= alpha * Wgrad[0]` (and similarly for `w2_` and `b_`).
After those changes, run a few epochs, print the loss, and produce a scatter plot with the decision line `w1*x1 + w2*x2 + b = 0`.

## 8. Resources to Study Next
- **Binary‑cross‑entropy implementation and clipping** – https://machinelearningmastery.com/binary-crossentropy-loss-function/
- **Gradient descent update rule** – https://distill.pub/2020/understanding-gradient-descent/ (Chapter 2)
- **Plotting a decision line** – https://realpython.com/python-matplotlib-guide/#plotting-a-line

## 9. Do Not Do Yet
- Do not increase the number of epochs or change the architecture before the numerical issues are resolved.
- Avoid adding a validation split before the loss function is stable.

## 10. AI Mentor Notes
The student is at the **implementation‑debugging** stage of M1. The primary blockers are the sign and stability of the BCE loss and an incorrect weight‑update rule. The next step is to add the three conceptual fixes listed above, then verify loss behavior and visualize the learned separator. Once those are confirmed, the student can proceed to validation and then to the next milestone.
