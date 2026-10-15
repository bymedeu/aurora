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
- The local suite has 33 passing tests, including finite-difference comparisons,
  sigmoid edge cases, the read-only value contract, cycle rejection, integer
  powers, and MSE gradients reaching model parameters.
- Power arguments are checked at runtime; the assistant added validation and
  corrected the MSE package import with explicit approval. The power composition
  and differentiable MSE calculation were implemented by the owner.
- The owner implemented gradient_descent_step using fresh parameter leaves. A
  direct 50-update check learning y = 2x + 1 reduced MSE from 9 to approximately
  2.04e-10, with weight approaching 2 and bias approaching 1. This is execution
  evidence, not a new persistent test or proof of owner conceptual mastery.

What is missing:
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Trace how the shared weight and bias receive contributions from two squared
  errors, including the averaging factor, and explain why the next forward uses
  fresh parameter leaves. Then complete the bounded conceptual comparison with
  PyTorch autograd required by the roadmap; do not add further engine features
  merely to postpone the remaining explanation work.

Recommended resources:
- The existing M0 mean-squared-error definition in docs/notes/linear_regression.md.
- The M3 autograd design notes, especially graph and gradient lifecycle.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer classes merely
  to expand the framework.
- Do not add optimizer, neuron, tensor, or training-loop abstractions. Keep the
  composition of the loss graph explicit and inspectable.
