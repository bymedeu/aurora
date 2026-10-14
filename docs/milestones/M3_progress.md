Milestone: M3 — Autograd

Estimated completion: 80%

Breakdown:
- Conceptual Understanding: 18/25
- Mathematical Understanding: 14/20
- Implementation: 25/25
- Debugging Ability: 12/15
- Written Explanation: 11/15

What is solid:
- Scalar forward operations construct graph nodes with parent identity and local
  backward behavior.
- Reverse topological traversal handles chains, branches, shared intermediates,
  and repeated operands.
- Traversal uses separate visiting and visited sets, runs in average O(V + E)
  time, and rejects cycles before backward mutates gradients.
- Local rules accumulate the incoming gradient multiplied by the local derivative.
- Explicit gradient clearing supports deliberate reuse of an unchanged graph.
- Addition, subtraction, multiplication, division, and a numerically stable
  sigmoid are implemented.
- Forward values are exposed through a read-only property so recorded graphs keep
  the values used by backward.
- `aurora.autograd` exposes the intended public API, while its tests live outside
  the source package and run through one repository-wide discovery command.
- The local suite has 24 passing tests, including finite-difference comparisons,
  sigmoid edge cases, the read-only value contract, and cycle rejection.

What is missing:
- Demonstrate the completed engine in one end-to-end learning loop where loss
  decreases and parameters move toward a known solution.
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Revisit the existing M0 linear-regression problem using AutoNode parameters and
  an AutoNode mean-squared-error graph. Preserve the original manual-gradient
  implementation as the comparison baseline.

Recommended resources:
- The existing `examples/linear_regression/train.py` manual-gradient baseline.
- The M3 autograd design notes, especially graph and gradient lifecycle.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer abstractions merely
  to expand the framework.
- Do not add optimizer, neuron, tensor, or training-loop abstractions. Keep the
  validation loop explicit so every forward, backward, and update step is visible.
