Milestone: M3 — Autograd

Status: COMPLETE — 2026-10-17.

Completion confirmed on 2026-11-04. M3 is closed; no additional entry review is
required before M4. The earlier implementation-only status and carried-over review
requirement are superseded. This correction records milestone closure; it does
not invent a new competency score or claim an additional assessment took place.
The implementation was delivered 59 days before the December 15, 2026 deadline.

Evidence reviewed:
- Scalar nodes record values, parent identity, and operation-specific backward
  behavior. Public values are read-only; gradient fields remain mutable.
- Arithmetic, non-negative integer powers, stable sigmoid, reverse topological
  traversal, shared/repeated dependencies, cycle detection, gradient accumulation,
  and explicit graph clearing are implemented and tested.
- Traversal uses visiting/visited sets and an ordered result; its average cost is
  O(V + E) time and O(V) additional space. Recursive depth remains a limitation.
- MSE builds a differentiable scalar loss. Gradient descent reads stored
  gradients and returns fresh parameter leaves without mutating old graph values.
- The owner implemented Aurora in src/aurora/model.py, exported from the package
  root. predict returns a connected AutoNode; fit trains the linear model using
  the existing loss, backward, and update functions.
- The linear caller trains on five pairs following y = 2x + 1. After 50 updates
  at learning rate 0.1, MSE is approximately 2.037e-10 and predict(4).value is
  approximately 8.999986.
- The owner implemented a deeper example with two sigmoid hidden neurons, a
  linear output, and seven explicit parameter nodes. All seven output gradients
  agree with finite differences at two parameter/input configurations.
- The nonlinear caller completes 50 updates at learning rate 0.1, reducing MSE
  from 0.31281292252 to 0.00285574309. Final parameters are finite fresh leaves.
- The current complete suite passes all 61 tests. The nonlinear training check
  executes the actual example and evaluates its retained parameters independently
  of the last pre-update loss graph.

Debugging and design evidence:
- The owner corrected detached numerical predictions, nested node values,
  forward activation versus backward callbacks, retaining updated parameters,
  and prediction lists that accidentally spanned multiple training steps.
- The Aurora class remains a specific two-parameter linear model. The nonlinear
  experiment calls the reusable engine/loss/optimizer directly; no configurable
  model framework was added to close M3.
- Assistant-written tests and the linear caller verify behavior. The nonlinear
  forward/training implementation and the model integration were owner work.
  Assistant-maintained documentation is not owner-authored mastery evidence.

Next milestone:
- M4 — NumPy Tensors and Autograd: extend the scalar foundation to array-valued
  operations and small trainable networks, using Python and NumPy.
- Begin with tensor meaning, batches, broadcasting, and reductions. Revisit an
  older concept only when a concrete question in the new work needs it.

Known boundaries:
- Reusing an unchanged graph requires explicit clearing before another backward
  pass; stale intermediate gradients must not be reused accidentally.
- Valid, equal-length, nonempty batches are the current training contract; a full
  invalid-input policy has not been designed.
- Scalar nodes, recursive traversal, and the fixed linear Aurora wrapper remain
  deliberate limits. Tensor machinery and a generic model API are not M3 work.

Resume commands (from the repository root):

    python3 -B tests/run.py
    PYTHONPATH=src python3 -B examples/autograd/train.py
    PYTHONPATH=src python3 -B examples/autograd/nonlinear.py

The nonlinear script currently runs silently; its test checks the learned result.
Historical reports retain their dated assessments and are not current status.
