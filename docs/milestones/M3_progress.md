Milestone: M3 — Autograd

Progress assessment: reassessment required. The previous assistant-estimated 80%
is withdrawn because it overstated owner understanding based on working code and
assistant-assembled integration checks.

Competency evidence:
- Conceptual understanding: the owner reports feeling lost about how components
  form a complete usable model. Resolve this before assigning a percentage.
- Mathematical understanding: local derivative discussions exist, but independent
  end-to-end explanation is not yet established.
- Implementation: the inspected components pass their existing tests, and direct
  integration checks show learning; this is not proof of independent mastery.
- Debugging ability: owner corrections and assistant-authored checks provide
  partial evidence; do not infer a competency score from a green suite.
- Written explanation: assistant-maintained design notes are engineering records,
  not owner-authored explanation evidence.

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
- The suite is now organized into autograd, nn, optim, model, and integration
  groups, with component and filename selectors. The latest run discovers 57
  tests: 37 pass, eight fit behavior checks fail, and 12 checks encounter package
  import errors. All 32 autograd tests and four model prediction tests pass. The
  prior 33-test all-green snapshot does not describe the current working tree.
- Power arguments are checked at runtime; the assistant added validation and
  corrected the MSE package import with explicit approval. The power composition
  and differentiable MSE calculation were implemented by the owner.
- The owner implemented gradient_descent_step using fresh parameter leaves. A
  direct 50-update check learning y = 2x + 1 reduced MSE from 9 to approximately
  2.04e-10, with weight approaching 2 and bias approaching 1. This is execution
  evidence, not a new persistent test or proof of owner conceptual mastery.

What is missing:
- Implement the saved fit stub and correct the nn/optim package imports. Test
  failures must remain visible; no source changes were made during suite cleanup.
- The Aurora class imports and predicts correctly. Its saved fit method is still
  a stub: the remaining integration work is training through the existing loss,
  backward, and update components, with owner understanding of those connections.
  The owner prefers tests or caller snippets over an executable package entrypoint.
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Connect node-based prediction and MSE to the existing backward function, then
  retain the fresh parameters returned by gradient_descent_step. backward fills
  gradients on existing objects; the update function does not calculate them.
- Resume from the M3 return brief at the top of docs/notes/autograd_design.md:
  distinguish the persistent model from a particular forward/loss graph, and
  understand why MSE records arithmetic operations beyond model neurons.
- Guide owner design of a bounded Aurora class holding parameters and exposing
  fit and predict, proposed in src/aurora/model.py with a root-package export.
  The empty __main__.py is not required. Agree on state and prediction behavior
  before training implementation. No source edits have been authorized by merely
  requesting this design guidance.

Recommended resources:
- The existing M0 mean-squared-error definition in docs/notes/linear_regression.md.
- The M3 autograd design notes, especially graph and gradient lifecycle.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer classes merely
  to expand the framework.
- Do not add a generic training framework or optimizer classes. A concrete,
  inspectable fitting function may connect the existing components when needed
  for understanding; keep prediction, loss, backward, and update visible.
