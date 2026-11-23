Milestone: M4 — NumPy Tensors and Autograd

Status: IN PROGRESS — construction, addition, multiplication, full-array sum,
and gradient clearing verified.
Updated: 2026-11-23.

M3 is complete, with implementation delivered on 2026-10-17 and milestone closure
confirmed on 2026-11-04. No additional M3 entry review blocks M4.

Goal:
- Train a small array-based neural network using Aurora's own differentiation,
  loss, and parameter updates, with Python and NumPy. PyTorch and external
  automatic differentiation engines are not dependencies or requirements.

Verified implementation evidence:
- Tensor is exported from aurora.tensor. Scalars, sequences, and arrays become
  independently owned float64 arrays with their numerical shapes preserved.
- Default gradients are zero arrays matching the value shape. Valid supplied
  gradients are copied into independent float64 storage.
- Directly constructed leaves have no parents and a no-op backward callback.
- Equal-shape addition and elementwise multiplication retain parent identity,
  preserve input state, accumulate nonuniform incoming gradients, and account
  for repeated operands. Distinct multiplication operand gradients agree with
  finite differences of a weighted scalar objective.
- Full-array sum returns a scalar Tensor connected to its input and distributes
  arbitrary scalar incoming gradients across the parent array. Tests cover
  composed graphs and gradient flow through earlier operations.
- Graph clearing delegates to each node's clear_gradient method, preserving
  valid gradient shapes, values and graph metadata. Unrelated nodes are untouched;
  repeated clearing and backward reproduce nonuniform array gradients.
- Engine annotations accept AutoNode or Tensor. Current output seeding is
  class-specific: scalar one for AutoNode, or ones matching the current Tensor
  gradient shape. For valid shapes and supported operations, a non-scalar seed
  differentiates the sum of output elements.
- Docstrings describe current behavior using Markdown-friendly sections and
  bullet lists. Actual Pyright Markdown and plaintext hover responses were checked.
- All 97 tests pass: the existing 61 scalar/model checks and 36 tensor checks.

Known boundaries:
- This checkpoint does not verify every copied arithmetic method or provide a
  complete tensor differentiation engine. Subtraction, division, and powers still
  need focused array review; division's denominator validation is scalar-oriented.
- NumPy accepting a broadcast forward expression does not establish correct
  broadcasting derivatives. Parent gradient shapes must be restored explicitly.
- Mean, selected-axis reductions, reshape/transpose, matrix multiplication, an
  array activation, and tensor-based network training are not delivered yet.
- Supplied and assigned gradient shapes are not validated. Mutation and invalid
  input policies remain limited to the current supported use cases.
- Basic Pyright checks accept both current node classes, but traversal containers
  and callbacks still contain inferred Unknown types. A Protocol or file-layout
  refactor is optional and does not block the next numerical implementation step.
- Passing tests is implementation evidence; it does not establish M4 completion
  or assign a conceptual-understanding score.

Next smallest useful step:
- Implement a full-array Tensor.mean() for non-empty inputs, retaining a scalar
  graph result. For N elements, each input receives incoming_gradient / N,
  accumulated into its existing gradient. Selected axes are a later extension.

Remaining milestone scope:
- Finish the needed elementwise arithmetic and verify its array gradients.
- Add mean and the selected-axis reductions needed by subsequent operations.
- Implement broadcasting backward rules and the required reshape/transpose rules.
- Add matrix multiplication and one array activation, sufficient for a small network.
- Enforce gradient-shape invariants and document the supported mutation/seeding
  behavior without building a general-purpose validation framework.
- Supply minimum layer/parameter structure, a tensor loss, and tensor parameter
  updates. Existing M3 model, loss, and update helpers remain scalar-oriented.
- Train a small network on batches, with reproducible correctness and loss-reduction
  evidence, then explain its shapes, gradient flow, and main computational costs.

Planning:
- February 28, 2027 is the previously recorded M4 target. Reassess it against the
  revised NumPy scope; no new estimate or competency percentage is asserted here.
- M4 remains incomplete. Dataset loaders, GPU execution, a broad framework API,
  and external-autograd comparisons are not prerequisites for this checkpoint.
