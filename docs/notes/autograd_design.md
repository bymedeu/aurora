# Autograd design decisions

Aurora M3 develops a scalar reverse-mode differentiation engine. This log records implementation contracts, validation evidence, owner preferences, and unresolved design questions. It is assistant-maintained engineering documentation, not owner-authored evidence of mathematical mastery.

## M3 working preferences

Recorded on 2026-10-11:

- The active checkout is /home/amade/projects/portfolio/aurora. Inspect and test this checkout rather than using GitHub as the source of truth for current work.
- Record substantive decisions and changes in repository docs. Distinguish agreed behavior, implementation evidence, proposals, and open questions.
- Discuss and obtain approval before changing README or roadmap priorities, completion criteria, or scoring values.
- The owner prioritizes AI and systems engineering rather than a PhD. Keep writing and mathematics focused on explaining, debugging, and defending the implementation. No change to the existing milestone scoring has been approved.
- Prefer Mermaid diagrams with TD direction.
- Keep verification outside the source package and provide one command that discovers the repository test suite.
- Do not add demonstration files merely to expand the repository. A proposed standalone tiny-model example was declined.

## M3 learning objective

For M3, the central learning target is the lifecycle of dynamic reverse-mode
automatic differentiation: forward execution calculates values and records
dependencies; backward execution applies the chain rule over that recorded graph;
an optimization step changes parameters; the next forward execution records a
fresh graph. This lifecycle matters more than building a broad framework API.

## Implemented scalar graph model

AutoNode stores a forward value, accumulated gradient, immediate parent references, and an operation callable.

Arithmetic operators calculate values and construct result nodes. Each result retains its own local backward callable. The operation field contains backward behavior, not the forward operation itself. Leaves use a no-op callable.

Implemented operations are addition, subtraction, multiplication, division, and sigmoid. Operations currently expect node operands; mixed Python-number arithmetic is not part of the implemented contract.

Parent connections preserve object identity. Two nodes with equal values are distinct graph objects. Repeated operands retain both operand connections so both derivative contributions are included.

## Traversal and backward

_topological_order(output) discovers reachable nodes and produces a
parent-before-result order. backward(output) seeds the output gradient with one,
then executes local callables in reverse order.

The engine is generic: it does not contain neuron, prediction, loss, or activation derivative logic. Forward graph construction does not call the traversal helper.

Current traversal uses lists for active and completed membership checks. For these unary and binary scalar graphs, repeated linear searches can make traversal quadratic in node count. Set-based bookkeeping is deferred until correctness and understanding are established. The graph and traversal storage scale with nodes and edges; recursive traversal is also limited by Python recursion depth.

The traversal helper is named _topological_order. The leading underscore marks it
as an internal implementation detail used by backward and gradient clearing,
rather than part of the public aurora.autograd API.

## Gradient lifecycle

Local backward rules accumulate incoming gradient multiplied by the local derivative. Shared paths contribute additively.

clear_gradients(output) explicitly clears gradients of every reachable node, including leaves and intermediates, while preserving values, parent connections, and operation callables. Nodes outside the reachable graph are unaffected.

Validated usage includes:
- one backward on a fresh graph with initially zero intermediate gradients;
- accumulation into shared leaves across fresh graphs;
- clearing a graph followed by another backward on that same graph, with unchanged forward values.

Repeated backward on the same graph without clearing is not a supported accumulation contract: old intermediate gradients are propagated again. Clearing is not hidden inside backward.

## Sigmoid implementation and numerical behavior

AutoNode.sigmoid performs forward construction and records one parent. op_sigmoid accumulates the incoming gradient multiplied by s * (1 - s), where s is the recorded sigmoid output value.

The incoming gradient and output value are distinct: the gradient carries downstream sensitivity, while the value determines sigmoid's local derivative.

Forward evaluation chooses an algebraically equivalent expression according to the input sign so that the evaluated exponential has a nonpositive exponent. It uses math.exp and returns a scalar value. The final implementation was supplied by the assistant at the owner's explicit request; correctness tests do not establish independent implementation mastery.

Earlier attempts exposed:
- overflow in the direct exponential for large negative input;
- overflow warnings despite finite selected outputs when np.where evaluated both expressions;
- an invalid assumption that Python int or float types expose a max attribute.

## Validation evidence

The last local run on 2026-10-11 executed:

python3 -B tests/run.py

All 23 tests passed without overflow warnings. The checker discovers every
`test_*.py` module under `tests/` and adds the local `src/` directory to the
import path. Coverage includes forward graph identity, local rule scaling and
accumulation, repeated operands, shared intermediates, ordering, analytical and
finite-difference gradients, gradient clearing, sigmoid at large positive and
negative finite inputs, and the read-only value contract.

Most tests are assistant-authored verification artifacts; the owner added the
read-only value regression before the suite was reorganized. The owner must still
explain the tested mechanisms and diagnose representative failures. M3 completion
has not been declared; the progress estimate was reassessed after the package and
test boundaries were completed.

## Value mutation and graph lifetime

Current backward rules read recorded node values. Arbitrary reassignment can make those values inconsistent with the forward calculation being differentiated.

Example: a=2, b=3, c=a*b gives c=6. Reassigning a to 10 before backward makes the current multiplication rule contribute 10 to b's gradient instead of the original derivative 2.

Decision approved on 2026-10-11: AutoNode values are fixed after construction.
Training updates produce new parameter leaf nodes, and the next forward execution
constructs a fresh graph. Gradients remain mutable and can be explicitly cleared.

This policy applies to the M3 scalar reference engine. It does not require a future
tensor implementation to copy large parameter arrays or use identical ownership
semantics.

Saved forward data can make gradients of the old calculation valid, but does not make the old output a prediction with updated parameters. Snapshot references alone are not sufficient if their underlying values remain mutable.

The owner proposed private-style fields plus a setter that updates dependent calculations. This was rejected for M3. Effects propagate toward consumers, not toward operand parents. Current nodes store parent links, not lists of their consumers. Automatic downstream recomputation would introduce a separate dependency-propagation mechanism and would require a defined policy for invalidating gradients and saved data.

The public contract is a readable value with no supported reassignment after
construction. Python convention and properties can express this API, even though
Python does not provide C++-style absolute privacy. Direct modification of an
implementation backing attribute remains outside the supported interface.

Implementation status: the owner implemented an internal backing attribute and a
read-only value property with no setter. On 2026-10-11 the full 23-test suite
passed. A permanent regression test confirms that public value reassignment raises
AttributeError, leaves the recorded value unchanged, and does not prevent gradient
mutation.

Ordinary scalar-engine training calculates all updates after backward, creates the
new parameter leaves, and then executes one fresh forward per next step or batch.
It does not rebuild after each individual parameter update.

This matches the useful conceptual part of dynamic autograd systems such as
PyTorch: the graph describes one executed forward computation. Production systems
usually keep mutable parameter storage and add saved tensors, version tracking,
and checks around in-place mutation. Those mechanisms are deliberately outside
the M3 scalar engine.

## Milestone boundary

The controlling M3 contract requires a scalar graph with backward and explanations of the chain rule, traversal, accumulation, ownership, complexity, and conceptual relation to PyTorch autograd.

A tensor engine, optimizer class, full neuron framework, transformer, broad activation catalogue, and additional standalone examples are not M3 completion requirements. Future useful neural-network computation should build on the generic differentiation mechanism without model-specific logic in the traversal engine.
