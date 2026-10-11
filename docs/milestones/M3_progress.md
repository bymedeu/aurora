Milestone: M3 — Autograd

Estimated completion: 78%

Breakdown:
- Conceptual Understanding: 18/25
- Mathematical Understanding: 14/20
- Implementation: 24/25
- Debugging Ability: 11/15
- Written Explanation: 11/15

What is solid:
- Scalar forward operations construct graph nodes with parent identity and local
  backward behavior.
- Reverse topological traversal handles chains, branches, shared intermediates,
  and repeated operands.
- Local rules accumulate the incoming gradient multiplied by the local derivative.
- Explicit gradient clearing supports deliberate reuse of an unchanged graph.
- Addition, subtraction, multiplication, division, and a numerically stable
  sigmoid are implemented.
- Forward values are exposed through a read-only property so recorded graphs keep
  the values used by backward.
- `aurora.autograd` exposes the intended public API, while its tests live outside
  the source package and run through one repository-wide discovery command.
- The local suite has 23 passing tests, including finite-difference comparisons,
  sigmoid edge cases, and the read-only value contract.

What is missing:
- Explain current traversal time and memory costs, including list membership and
  recursive depth limits.
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Explain the current traversal complexity, then replace linear list membership
  checks with explicit visiting and visited sets while preserving topological
  order and cycle detection.

Recommended resources:
- Python documentation: Set Types.
- Python documentation: Recursion and recursive functions.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer abstractions merely
  to expand the framework.
- Do not add optimizer, neuron, tensor, or training-loop abstractions before the
  traversal contract and remaining M3 evidence are settled.
