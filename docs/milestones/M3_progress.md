Milestone: M3 — Autograd

Estimated completion: 74%

Breakdown:
- Conceptual Understanding: 18/25
- Mathematical Understanding: 14/20
- Implementation: 23/25
- Debugging Ability: 11/15
- Written Explanation: 8/15

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
- The local suite has 22 passing tests, including finite-difference comparisons
  and sigmoid edge cases.

What is missing:
- Add one focused regression test for the read-only value contract.
- Make `aurora.autograd` importable as a package with a small public API; current
  imports and test execution still assume the autograd directory is on sys.path.
- Group tests under the repository test tree after the package boundary is clear.
- Explain current traversal time and memory costs, including list membership and
  recursive depth limits.
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Define the `aurora.autograd` package boundary and public imports without adding
  neural-network abstractions.

Recommended resources:
- Python documentation: Packages.
- Python documentation: The import system.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer abstractions merely
  to expand the framework.
- Do not optimize graph traversal before the package contract and remaining M3
  evidence are settled.
