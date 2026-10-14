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

## Learning ownership and approved supporting work

The owner implements the AI mechanisms that Aurora exists to teach. Necessary
supporting code should still be correct and usable, but routine maintenance must
not displace the active learning objective. The assistant may propose handling
input validation, imports, packaging, test infrastructure, or organization when
doing so improves the project without removing meaningful learning.

Every source or test edit still requires explicit approval for its specific scope.
An invitation to propose assistance is not permission to apply the edit. Before
seeking approval, explain the defect or responsibility, the bounded proposed change,
and how it will be verified. Existing authorization for a specific task remains
valid; do not repeatedly ask for the same approved work.

Show proposed code edits directly in fenced code blocks in the conversation,
with their affected files and scope, rather than linking to temporary patch files.

When personally implementing a supporting function would develop valuable AI,
systems, or engineering judgment toward the owner's career goals, explain that
value and recommend owner implementation. Do not frame minor housekeeping as an
employer requirement or count assistant-written code as owner implementation
mastery. The distinction depends on learning value, not merely on whether a
function is located under src.

Current example: the owner approved assistant-written checks for unsupported power
arguments and the MSE package import correction. Power composition, graph
preservation, loss construction, and the meaning of gradients remain owner learning
work. Both supporting fixes are applied, and the full 33-test suite passes.

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

Current traversal keeps the ordered result in a list and uses separate visiting and
visited sets for membership checks. Each reachable node is completed once and each
parent edge is followed once, giving average O(V + E) traversal time and O(V)
additional bookkeeping space. Recursive traversal remains limited by Python's
recursion depth.

Encountering a node already in the visiting set proves that the current parent
path contains a cycle. The traversal raises ValueError because a cyclic graph has
no topological order. backward requests the order before seeding the output
gradient, so cycle rejection does not partially execute that backward call.

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

The last local run on 2026-10-14 executed:

python3 -B tests/run.py

All 24 tests passed without overflow warnings. The checker discovers every
`test_*.py` module under `tests/` and adds the local `src/` directory to the
import path. Coverage includes forward graph identity, local rule scaling and
accumulation, repeated operands, shared intermediates, ordering, analytical and
finite-difference gradients, gradient clearing, sigmoid at large positive and
negative finite inputs, the read-only value contract, and cycle rejection before
gradient mutation.

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
read-only value property with no setter. On 2026-10-14 the full 24-test suite
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

## Practical continuation and package responsibilities

The owner reaffirmed the preference for reusable Aurora code rather than another
standalone training example. The earlier suggestion to add train_autograd.py and
the progress entry describing a learning loop as missing work were assistant
proposals, not requirements from the roadmap. A new training example is not
required to complete M3.

The proposed next integration step is a mean_squared_error function in
src/aurora/nn/losses.py. Its name describes the mathematical quantity and its
module describes the reusable responsibility. It would construct a loss graph
using existing AutoNode operations; backward would derive gradients through that
composition. Dataset selection, printing, epoch counts, and training orchestration
do not belong in the generic autograd traversal package. The scalar loss API is
bounded to the current scalar engine; a future tensor loss can use the same
mathematical definition without requiring identical implementation details.

## Power composition and MSE verification

The owner implemented __pow__ by repeated squaring, composing existing AutoNode
multiplications rather than adding a dedicated power backward rule. Consequently,
parent references and multiplication callbacks retain the differentiation graph;
there is no separate exponent metadata required by backward. For positive integer
exponents, the composition takes O(log n) multiplication operations.

The requested tests cover squares, cubes, fifteenth powers, additional integer
exponents, negative and zero bases, scaling of incoming gradients, accumulation,
shared computations, and a composed finite-difference comparison. All eight power
tests pass, including the zero-exponent convention that x ** 0 returns a
constant-one node. Runtime guards reject non-integer exponents with TypeError and
negative integer exponents with ValueError. These checks were assistant-written
with explicit approval; the power composition itself was owner-written.

An additional integration test constructs two predictions from shared weight and
bias nodes and checks the MSE value and gradients back to both parameters. It
passes after the approved package import correction. For weight 2, bias 1, inputs
[1, 3], and targets [5, 6], the loss is 2.5, prediction gradients are [-2, 1],
weight gradient is 1, and bias gradient is -1. The MSE graph construction was
owner-written; its package import correction was assistant-written with approval.
The integration check covers a valid non-empty batch; it is not evidence of a
completed policy for empty or mismatched inputs.

The latest full checker run passes all 33 tests: 24 existing autograd tests,
eight power tests, and one MSE integration test. This supersedes the initial
verification run with four failing unsupported-input subtests and one import
error. The next learning step is interpreting and using the validated parameter
gradients for a gradient-descent update with fresh parameter leaves, rather than
extending the operation catalogue or adding a new standalone training example.
