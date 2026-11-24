# Aurora verification

Run commands from the repository root. The runner uses the checkout's `src/`
directly; installing Aurora or another test framework is not required.

```bash
python3 -B tests/run.py
```

## Component suites

| Folder | What it checks |
| --- | --- |
| `autograd/` | Scalar operations, ordering, backward, clearing, sigmoid, and powers. |
| `nn/` | MSE values, averaging, gradients, and its public package export. |
| `optim/` | Updates from supplied gradients, fresh leaves, old-state preservation, and the public export. |
| `model/` | `from aurora import Aurora`, parameter state, connected prediction nodes, and `fit` behavior. |
| `integration/` | Connections between loss construction, backward, and parameter updates. |
| `tensor/` | Value and gradient normalization/ownership, leaf state, equal-shape addition, subtraction, multiplication, and division, local gradient accumulation, full-array sums and non-empty means, shape-preserving clearing, and backward integration. |

```bash
python3 -B tests/run.py --list
python3 -B tests/run.py autograd
python3 -B tests/run.py tensor
python3 -B tests/run.py tensor -p test_construction.py
python3 -B tests/run.py tensor -p test_addition.py
python3 -B tests/run.py tensor -p test_subtraction.py
python3 -B tests/run.py tensor -p test_multiplication.py
python3 -B tests/run.py tensor -p test_division.py
python3 -B tests/run.py tensor -p test_power.py
python3 -B tests/run.py tensor -p test_shapes.py
python3 -B tests/run.py tensor -p test_sum.py
python3 -B tests/run.py tensor -p test_mean.py
python3 -B tests/run.py tensor -p test_clearing.py
python3 -B tests/run.py nn optim
python3 -B tests/run.py model -p test_prediction.py
python3 -B tests/run.py model -p test_fit.py --failfast
```

Each test name describes observable behavior. `FAIL` means an expectation did not
hold; `ERROR` means the check could not finish, for example because a package import
failed. Failures and errors produce a nonzero exit status. Unknown suite names or
a filename pattern matching no tests also produce a nonzero status.

## Adding checks

Add `test_*.py` modules to the relevant component folder. Name test methods
`test_*` and use `unittest.TestCase`. A new component folder needs an `__init__.py`;
the runner automatically discovers it and offers it as a selectable suite. The
root `tests/__init__.py` keeps imported test names under `tests.*`, separate from
Aurora's implementation packages.

Prefer externally observable mathematical or state guarantees. Tests for local
derivative rules can inspect graph internals; model tests should exercise the
public interface. Integration tests prove connections without replacing missing
library behavior with mocks. Invalid-input expectations should follow an agreed
contract rather than inventing extra milestone requirements.

The M3 implementation closeout on 2026-10-17 passes all 61 tests.
`model/test_fit.py` verifies the implemented one-step and repeated linear training
behavior. `integration/test_nonlinear_forward.py` checks nonlinear forward values,
all seven output gradients against finite differences, and the actual training
example's loss reduction and final parameter state. Regressions must remain
visible; do not skip or mark them as expected failures to obtain a green suite.

The teaching example in `integration/test_learning_step.py` shows that MSE builds
the loss, backward writes gradients onto existing parameter nodes, and the update
helper creates new parameter leaves. It is verification code, not a replacement
for the owner implementing and understanding `Aurora.fit`.

## Tensor checkpoint

The tensor suite requires NumPy and is discovered by the existing runner. The
construction checks cover scalar, vector, matrix, and higher-dimensional values;
floating-point normalization; independent input and gradient storage; default
zero gradients; and explicitly supplied gradients. The addition checks cover
same-shaped operands, graph parent identity, forward state preservation,
nonuniform incoming gradients, repeated operands, and the scalar backward entry
point. Seven sum checks cover full-array scalar results, parent identity, input
preservation, accumulation of nonunit incoming gradients, backward through
addition, repeated operands, and a composed gradient against finite differences.
Three tensor clearing checks cover scalar, vector, matrix, and empty-array
gradient shapes; preserved values and graph metadata; untouched unrelated nodes;
and repeated clearing followed by backward on a graph with nonuniform gradients.
Six multiplication checks cover scalar/vector/matrix and empty-array forward
results, parent identity, input preservation, nonuniform incoming gradients,
accumulation into existing gradients, repeated operands, and a weighted scalar
loss. Both distinct operand gradients are compared with finite differences.
Seven mean checks cover non-empty scalar/vector/matrix and higher-rank averages,
input preservation, total element counts, scaled incoming gradient accumulation,
backward through addition, repeated operands, and finite differences of the mean
of squared values.
Three subtraction checks cover equal-shape forward values and operand order,
opposite gradient accumulation through a weighted loss, and cancellation
when the same tensor appears on both sides. Four division checks cover equal-shape
forward values, zero denominators, gradient accumulation, and both operand
gradients against finite differences. Four power checks cover shape preservation,
constant ones for exponent zero, composed gradients, and invalid exponents. The
shape check covers rejection of broadcastable operand shapes across the four
binary arithmetic operators. The full suite passes 116 tests, including 55 tensor
checks.

This M4 checkpoint does not cover every copied arithmetic method or the whole
milestone. Broadcasting derivatives, axis-specific reductions, empty-input means,
matrix multiplication, network training, and a broader invalid-input policy are
outside this suite. A failure or error remains visible and returns a nonzero status;
missing or broken behavior is not skipped or marked as an expected failure.
