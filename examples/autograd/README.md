# M3: training through automatic differentiation

Run from the repository root, with the local source package on Python's import path:

```bash
PYTHONPATH=src python3 -B examples/autograd/train.py
```

The example imports the real `Aurora` class. It starts with zero weight and bias,
passes five input/target pairs to `fit`, then predicts for three new inputs.
There are no additional dependencies or hand-written parameter-gradient formulas.

The data follows `y = 2x + 1`; the model receives only the pairs. After 50 updates,
weight should be approximately 2, bias approximately 1, and the training MSE about
`2.04e-10`. The prediction for input 4 changes from 0 to approximately 9.

`predict` returns a connected `AutoNode`. Use `.value` when printing a number;
keep the node when constructing a differentiable loss. The final MSE shown here
is an evaluation of the trained model; it does not perform another update.

Unlike the M0 example's explicit derivative formulas, training here uses MSE,
`backward`, and the existing gradient-descent helper inside `Aurora.fit`.
The implementation remains in `src/aurora`; this file is a caller showing its use.
The current model is the scalar linear rule `weight * input + bias`.

## Deeper nonlinear model

```bash
PYTHONPATH=src python3 -B examples/autograd/nonlinear.py
```

The owner-written experiment uses two sigmoid hidden neurons and a linear output,
with seven trainable parameter nodes. Its forward rule and training caller use
the existing autograd, MSE, and gradient-descent functions directly. The Aurora
class above remains the specific linear model; its fit method does not train
this separate network.

On inputs `[-2, -1, 0, 1, 2]` and targets `[0.1, 0.2, 0.5, 0.8, 0.9]`, 50 updates
at learning rate 0.1 reduce MSE from approximately `0.312813` to `0.002856`.
The script currently runs silently. Its regression test executes the actual
caller, checks its final parameter state, and independently evaluates the final
MSE. A separate finite-difference check covers all seven output gradients.

```bash
python3 -B tests/run.py integration -p test_nonlinear_forward.py
```

For correctness checks, run:

```bash
python3 -B tests/run.py
```
