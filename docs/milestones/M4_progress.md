Milestone: M4 — NumPy Tensors and Autograd

Status: IN PROGRESS — tensor construction, addition, full-array sum, and clearing verified.
Updated: 2026-11-21.

M3 is complete, with implementation delivered on 2026-10-17 and milestone closure
confirmed on 2026-11-04. No additional M3 entry review blocks M4.

Goal:
- Combine NumPy array calculations with Aurora's own automatic differentiation
  to train a small neural network. PyTorch is not a dependency or requirement.

Verified implementation evidence:
- Tensor is exported from aurora.tensor. Numerical scalar, sequence, and array
  inputs become independently owned float64 arrays with their shapes preserved.
- Default gradients are zero arrays matching the value's shape. Valid explicitly
  supplied gradients use independent float64 storage.
- A directly constructed tensor has no parents and no local backward effect.
- Equal-shape addition preserves parent identity and input state. Its local
  backward rule accumulates nonuniform incoming gradients and both contributions
  for repeated operands.
- Scalar tensor addition works with the existing backward entry point, retaining
  zero-dimensional array gradients.
- Full-array sum returns a scalar Tensor linked to its input. Its backward rule
  accumulates arbitrary scalar incoming gradients while preserving input shape.
  Tests cover backward through addition, repeated operands, a later scalar
  operation, and a composed array gradient against finite differences.
- The shared engine clears current AutoNode and Tensor graphs with class-specific
  handling. Valid tensor gradients retain their shapes, values and graph metadata
  remain unchanged, and unrelated nodes are untouched. Repeated clearing and
  backward reproduce nonuniform array gradients.
- The full suite passes all 91 tests: the existing 61 tests and 30 tensor checks.

Known boundaries:
- This checkpoint verifies construction, equal-shape addition, full-array sum,
  and clearing for the current node classes, not every copied arithmetic method
  or a complete tensor differentiation engine.
- Gradient-shape validation still needs explicit coverage. General mutation and
  invalid-input policies remain bounded.
- Broadcasting derivatives, axis-specific and mean reductions, reshape/transpose,
  matrix multiplication, and tensor-based network training are not delivered yet.
- Passing these tests is implementation evidence, not M4 completion or a new
  conceptual-understanding score.

Next smallest useful step:
- Verify equal-shape multiplication of distinct tensors through a scalar loss,
  including nonuniform incoming gradients and numerical derivative comparisons.

Remaining milestone scope:
- Establish tensor gradient accumulation, seeding, and clearing contracts.
- Implement and verify the needed elementwise operations and activation.
- Add reductions, shape operations, matrix multiplication, and broadcasting
  backward rules while preserving each parent's gradient shape.
- Train a small network using Aurora parameters, loss, and updates; retain
  reproducible correctness checks and a technical explanation of the mechanisms.

Planning:
- February 28, 2027 is the previously recorded M4 target. Reassess it against
  the revised NumPy implementation scope; no new estimate is asserted here.
- M4 remains incomplete. No new competency percentage has been assigned.
