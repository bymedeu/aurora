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
- A visible, runnable Aurora program connecting the implemented components, and
  owner understanding of its data, model parameters, training, and predictions.
- Give a concise owner-authored account of graph construction, chain-rule
  propagation, accumulation, graph lifetime, and known limitations.
- Compare the learned concepts with PyTorch autograd after the first bounded
  PyTorch study in M4; this has not been demonstrated yet.

Next smallest useful step:
- Resume from the M3 return brief at the top of docs/notes/autograd_design.md:
  understand the four distinct roles (data, model, learned state, training), then
  implement and explain one complete update in the owner-created entrypoint.
- Guide the owner in implementing a visible Aurora prototype that connects the
  existing components and can train and predict. The owner created an empty
  src/aurora/__main__.py; they own its implementation. Explain and agree on the
  design before assigning implementation details. Do not interpret agreement on
  the goal or "let's go" as authorization for assistant-written source.

Recommended resources:
- The existing M0 mean-squared-error definition in docs/notes/linear_regression.md.
- The M3 autograd design notes, especially graph and gradient lifecycle.

Do not do yet:
- Do not add scalar neuron, optimizer, tensor, or transformer classes merely
  to expand the framework.
- Do not add a generic training framework or optimizer classes. A concrete,
  inspectable fitting function may connect the existing components when needed
  for understanding; keep prediction, loss, backward, and update visible.
