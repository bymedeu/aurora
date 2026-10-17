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
| `model/` | `from aurora import Aurora`, parameter state, numerical prediction, and `fit` behavior. |
| `integration/` | Connections between loss construction, backward, and parameter updates. |

```bash
python3 -B tests/run.py --list
python3 -B tests/run.py autograd
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

`model/test_fit.py` contains normal checks for the agreed training behavior. An
unfinished `fit` should fail these checks visibly. Do not skip them or mark them
as expected failures just to obtain a green suite. Use the prediction or autograd
selectors to check completed parts independently.

The teaching example in `integration/test_learning_step.py` shows that MSE builds
the loss, backward writes gradients onto existing parameter nodes, and the update
helper creates new parameter leaves. It is verification code, not a replacement
for the owner implementing and understanding `Aurora.fit`.
