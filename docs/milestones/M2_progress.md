Milestone: M2 — Small Neural Network

Estimated completion: 35%

Breakdown:
- Conceptual Understanding: 15/25
- Mathematical Understanding: 10/20
- Implementation: 5/25
- Debugging Ability: 0/15
- Written Explanation: 5/15

What is solid:
- Documented matrix shapes, hidden layer definitions, and forward pass structure in `docs/notes/xor.md`.
- Identified target network topology for XOR.

What is missing:
- Implementation of full 2-layer forward and backward pass in `examples/xor/train.py`.
- Working training loop that solves XOR.
- Formal explanation of non-linear decision boundaries and why a single neuron fails on XOR.

Next smallest useful step:
- Write the forward pass for a 2-layer neural network (2 inputs -> 2 or 4 hidden neurons -> 1 output neuron) in `examples/xor/train.py` and print output shapes.
