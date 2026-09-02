Milestone: M2 — Small Neural Network

Status: Complete

Completed: 2026-09-01

Estimated completion: 100%

Breakdown:
- Conceptual Understanding: 25/25
- Mathematical Understanding: 20/20
- Implementation: 25/25
- Debugging Ability: 15/15
- Written Explanation: 15/15

What is solid:
- The network uses vertical activation vectors and a consistent outgoing-weight orientation.
- `feedforward` supports an arbitrary sequence of layer sizes and saves the pre-activations required by backpropagation.
- `backprop` supports any number of hidden layers while keeping every gradient aligned with its corresponding parameter.
- The relationship between output delta, hidden deltas, activation derivatives, and weight gradients has been studied and debugged explicitly.
- The training loop solves XOR reproducibly with a fixed random seed.
- Final BCE is approximately `0.00558`, with predicted classes `[0, 1, 1, 0]`.
- Gradient shapes were checked on a deeper `2 -> 4 -> 3 -> 2 -> 1` network.
- Analytical gradients were compared with numerical gradients; the maximum observed error was approximately `1.2e-10`.

What is missing:
- Nothing required for M2 completion.
- Optional future polish: revise `docs/notes/xor.md` for clarity and consistency with the final matrix orientation.

Next smallest useful step:
- Begin M3 by understanding how a scalar computation can be stored as a graph and how the chain rule moves gradients backward through that graph.
