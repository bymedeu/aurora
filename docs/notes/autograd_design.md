# Autograd design decisions

Aurora M3 develops a scalar reverse-mode differentiation engine. This log records implementation contracts, validation evidence, owner preferences, and unresolved design questions. It is assistant-maintained engineering documentation, not owner-authored evidence of mathematical mastery.

## M3 return brief — updated 2026-10-17

### Session objective and ownership

The owner now wants an importable Aurora object, consumed by tests or small caller
snippets, rather than a package executable. This supersedes the earlier __main__.py
proposal. The initial object should represent one small trainable linear model,
hold its current parameters, and expose training and prediction. The owner writes
it with guidance; the assistant does not create the class or export without
specific source-edit approval. The existing empty __main__.py is not a blocker.

### What M3 is and why it matters

M3 is scalar reverse-mode automatic differentiation. It records a calculation as
it executes, then propagates derivatives backward through that recorded structure.
It supplies gradients for learning; it does not itself select a model or target,
choose an update rule, or make a trained model appear automatically.

In M0–M2, parameter-gradient formulas were written for particular calculations.
M3's capability is that supported operations can be composed differently while
the same backward engine calculates the resulting gradients. Changing wx + b to
(wx + b)^3, for example, does not require a new model-specific backpropagation
function. Existing operation rules and graph connections supply the chain rule.
The significance is this reusable mechanism, not the number or length of functions.

### Four distinct things in the prototype

| Thing | Meaning | Where it lives |
| --- | --- | --- |
| Data | Known inputs and observed target answers. | Lists chosen by the prototype caller; values enter the graph as leaves. |
| Model | The chosen prediction rule w*x + b, together with current parameter values. | The proposed Aurora object's prediction behavior and parameter state. |
| Learned state | The current numerical values of w and b. | Parameter AutoNode objects owned by the model object. |
| Training procedure | Repeated evaluation, differentiation, and parameter updates. | The proposed fit method using the existing library functions. |

For the first controlled case, use inputs [-2, -1, 0, 1, 2] and target answers
[-3, -1, 1, 3, 5]. The answers follow 2x + 1, but w and b must start at zero.
The model receives the observations, not the target-generating formula or the
answer coefficients. It learns coefficients within the chosen model form; it
does not learn an arbitrary program or decide the model architecture.

After training, the relation is represented by the learned coefficients plus the
forward expression. There is no stored string "2x + 1" that the engine executes.
Parameters remain in memory while the prototype runs; saving/loading a model is
not implemented. Prediction on a new input uses those coefficients without a
target, loss, backward pass, or update.

A model is a parameterized prediction rule with current parameter values. The
AutoNode graph is the record of a particular execution of that rule and, during
training, the objective calculation following it. The persistent model object is
not identical to a saved graph: a new forward creates a new graph, and MSE extends
that graph beyond the predictions. A neuron is a model-level unit and may contain
several scalar operations; the graph records those operations and also loss
subtraction, squaring, summation, and division. It is not a list of neurons.

MSE's additions and division are explicitly part of the definition
sum((prediction - target)^2) / N. They are required for its value and derivatives.
The mathematical loss is a scalar; loss.value is that scalar in this implementation.
The returned AutoNode additionally retains the calculation needed by backward.

### How the mechanism works

1. Forward node arithmetic computes numerical values and creates result nodes
   with immediate parents and operation-specific backward behavior. Predictions
   and the MSE loss are part of this same recorded graph.
2. backward seeds loss.gradient with 1 because the derivative of loss with
   respect to itself is 1. The traversal supplies a reverse topological order.
3. Each operation accumulates its incoming gradient multiplied by its local
   derivative into its parents. Contributions from different paths add. This is
   how parameters shared by multiple predictions receive all sample contributions.
4. Parameter.gradient means the derivative of the final loss with respect to
   that parameter. It is neither the parameter value nor an automatic new value.
5. gradient_descent_step reads the selected parameters' values and gradients,
   computes new numerical values, and returns fresh parameter leaves. Those new
   leaves are the model state for the next forward calculation.

Graph construction belongs to autograd/scalar_node.py; traversal and backward
belong to autograd/engine.py. nn/losses.py defines the MSE calculation, and
optim/gradient_descent.py defines the update. The caller connects them. Folders
do not execute each other automatically. Tests exercise these responsibilities;
they do not add hidden learning logic to the library.

Parameters and intermediates are fresh each step under the chosen immutable-value
policy, so the prototype does not need to clear an old graph before moving to a
new one. Repeating backward on the same old graph without clearing is a different,
unsupported accumulation usage. Keep inputs and targets out of the parameter
update even though they also use AutoNode as their representation.

### The owner implementation task, expressed as behavior

Proposed bounded interface: create an Aurora instance, call fit with input/target
data and training settings, and call predict with a new input. The instance owns
the current parameters. The caller supplies the data; the object does not need to
store the dataset permanently. fit coordinates the existing loss, backward, and
update functions. predict returns a numerical prediction; a training forward
retains the AutoNode graph. Do not move every autograd helper into the class.

A proposed location is src/aurora/model.py, with a root-package export in
src/aurora/__init__.py so a caller can use "from aurora import Aurora". These are
proposals for owner implementation, not permission for assistant edits. First
agree on the model's state and prediction contract, then implement training using
the existing components. Do not rewrite derivatives or initialize parameters to
the known answer.

Begin with one complete update. These are independent reference values for the
five-pair dataset above, zero initial parameters, and learning rate 0.1:

| Observable | Expected value |
| --- | --- |
| Initial predictions | [0, 0, 0, 0, 0] |
| Initial mean squared error | 9 |
| Gradients after backward | weight: -8; bias: -2 |
| Updated parameter values | weight: 0.8; bias: 0.2 |
| Loss from a fresh forward after the update | 3.52 |

These are checkpoints to explain and verify, not constants to put into the update.
The first-session target is being able to run and explain this complete step.

Once that is understood, repeat the same procedure. In the assistant's prior
50-update check, weight approached 2, bias approached 1, and MSE reached about
2.04e-10. A held-out input 4 then predicts approximately 9. The owner prefers to
observe this through an importable model used by tests or caller snippets. An
executable __main__.py is not required for that interface.

### Readiness and remaining milestone evidence

Use the running prototype to explain the distinction between data, model state,
forward calculation, derivative calculation, and update. Be able to trace one
operation's chain-rule contribution and a shared parameter's accumulated gradient,
and explain why the next calculation uses fresh leaves. Traversal costs average
O(V + E) time and O(V) additional space; scalar graph construction also costs
objects and references per recorded operation, and recursion depth remains a limit.

Existing tests verify local rules, graph ordering, finite-difference agreement,
and MSE-to-parameter connections. New prototype verification should check the
owner's implementation rather than impose an assistant-written architecture.
The current valid-batch MSE check does not establish a full invalid-input policy.

Before declaring M3 complete, retain a short owner-authored explanation and make
the bounded conceptual comparison with PyTorch autograd named in the roadmap.
A running prototype is evidence of integration, not an automatic competency score.
Do not assign a new percentage until owner understanding has been reassessed.

The longer-term goal is one usable prototype whose capabilities improve with each
milestone. "from aurora import Aurora" is now the requested prototype direction;
the first API should be bounded to one understandable model rather than a general
framework. Source implementation remains owner work unless explicitly delegated.

Planning estimate for this bounded scope: roughly 10–20 focused hours for the
remaining concept clarification, importable model integration, verification,
debugging, short owner explanation, and introductory conceptual PyTorch comparison.
A first runnable model may take around 3–6 hours within that work. These are rough
estimates, not competency scores or a promised deadline; revisit after the first
owner-written model. Do not expand into general network configuration or new
autograd features while using this estimate.

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

## M3 responsibility map and gradient-descent review

Explain each component's purpose and its relationship to the training calculation
before assigning implementation work. Folder names organize responsibilities;
they are not automatically executed stages or additional milestone requirements.

- src/aurora/autograd owns scalar graph construction, local backward rules,
  traversal, and gradient clearing. It does not choose a model, objective, or
  parameter update rule.
- src/aurora/nn currently owns the differentiable MSE loss. It composes existing
  graph operations and returns a loss node; it does not call backward or update
  parameters.
- src/aurora/optim owns the proposed gradient-descent step: consume selected
  parameter values and their numerical gradients and return updated parameter
  leaves. It does not calculate predictions or gradients.
- A caller connects prediction, loss construction, backward, update, and a fresh
  forward pass. Current focused tests exercise these responsibilities; no new
  standalone example or general training-loop framework is needed.
- tests verifies the mechanisms; docs records decisions, explanations, and
  limitations. Empty packages are reserved structure, not M3 implementation work.

AutoNode arithmetic already performs numerical arithmetic on ordinary stored
values and additionally records parent references and backward behavior. Plain
number arithmetic should remain plain where no differentiable graph is needed,
including derivative accumulation and numerical parameter updates. The M3 scalar
engine intentionally pays Python-object and graph-storage costs per recorded
operation for inspectability. Reimplementing a separate numerical arithmetic
library is not required.

The owner's initial gradient_descent_step prototype was reviewed without source
changes. A direct execution with weight 2, bias 1, and MSE gradients [1, -1]
returned the original parameter objects with values [2, 1], leaving the loss at
2.5. The intended fresh-leaf update at learning rate 0.1 produces [1.9, 1.1] and a
fresh-forward loss of 2.32. The unresolved concepts are rebinding a loop variable
versus replacing container contents, and producing a parameter leaf versus an
operation result connected to the previous graph. This targeted check does not
establish correctness of the new optimizer function or change the existing
33-test validation result for the previously tested components.

The owner subsequently changed the update to calculate each new value with
ordinary numerical arithmetic and collect fresh AutoNode leaves in a new list.
A targeted execution verified values [1.9, 1.1], a fresh-forward loss of about
2.32, empty parent tuples and zero initial gradients on updated parameters, and
preservation of both the old parameters and their gradients after backward on the
new graph. All 33 existing suite tests also pass. This targeted optimizer check
was executed without adding a new persistent optimizer test module.

The current autograd, nn, and optim packages share a representation but separate
responsibilities: differentiation, model/loss calculation, and parameter updates.
Their separation is a small organizational convention, not a mathematical
requirement. Grouping all three under nodes or weights would obscure the fact
that losses and inputs are also nodes, while only selected leaves are trainable
parameters. Do not reorganize folders merely to maintain this convention.

Python iteration binds the loop variable to the existing element object without
copying that object. Rebinding the loop name changes that name's binding, not the
list entry. Mutating a mutable object through either reference affects the shared
object; replacing a list entry is a separate operation. The fresh-parameter API
intentionally returns new objects rather than mutating old graph values.

## Repeated learning check and remaining M3 work

A direct execution on 2026-10-15 used inputs [-2, -1, 0, 1, 2], targets 2x + 1,
initial weight and bias zero, and learning rate 0.1. Each update constructed a
fresh prediction/MSE graph, called backward once, and used gradient_descent_step
to create fresh parameter leaves. No training script or additional source function
was added for this verification.

The loss was 9 initially, 3.52 after one update, about 0.01182 after ten updates,
and about 2.04e-10 after fifty updates. The final weight was approximately
1.999999999984 and bias 0.999985727523. Thus the existing components can execute
learning together, beyond merely returning individual correct local gradients.
Targets were reused as constant leaves; their unused gradients may accumulate,
but they are never included in the parameter update and do not affect prediction
gradients under the current leaf no-op rule.

Remaining learning evidence is the owner's explanation of how gradients from
multiple samples accumulate into shared parameters, how graph lifetime relates to
updates, and the roadmap's bounded conceptual comparison with PyTorch autograd.
The successful numerical run does not by itself establish those competencies or
justify adding more framework features.

## Making the training caller visible to the owner

The owner reports that seeing isolated source functions and an assistant-assembled
test or terminal learning loop does not establish understanding of how they work
together. Do not treat those successful checks as a substitute for owner-created
integration. Explain where data, model state, and orchestration live before
assigning more components or assessing conceptual mastery.

In the repeated-learning verification, 2x + 1 was only the rule used to generate
target data. The trainable model was wx + b, with w and b initialized to zero.
The training procedure received input/target pairs and adjusted those parameter
leaves; it did not receive or store the target-generating formula. The MSE unit
test's initial w = 2 and b = 1 are a separate, deliberately chosen gradient-check
case, not evidence of learning from initialization.

Proposed next owner implementation: a concrete fit_linear_regression function in
src/aurora/training.py receiving inputs, targets, learning rate, and update count,
and returning the learned weight and bias nodes. The module's responsibility is
coordinating existing prediction arithmetic, MSE, backward, and parameter update.
It is not another isolated example, a new autograd operation, or a generic
training framework. Keep the mathematical model form and the existing function
calls visible. No new source file was created by the assistant for this proposal.

## Owner correction of progress and integration priority

On 2026-10-15 the owner explicitly rejected the previous 80% understanding estimate
and reported feeling lost despite the working individual functions. The estimate
has been withdrawn pending reassessment. Successful assistant-run integration and
assistant-written notes must not be converted into owner understanding scores.

The owner wants a working Aurora program that visibly connects the components.
The proposed training helper above is superseded as the immediate next step by a
minimal executable entrypoint in src/aurora/__main__.py: use the existing scalar
model arithmetic, MSE, backward, and fresh-leaf update to train and then predict
for a supplied input. Explain where the known dataset, unknown parameters, and
training procedure live in that program. Avoid adding a model hierarchy, generic
trainer, or more disconnected functions to address this gap.

The owner will implement the prototype. On 2026-10-15 the assistant incorrectly
interpreted agreement to proceed as approval to implement the entrypoint. The
owner explicitly corrected this: wanting a prototype and saying "let's go" do not
authorize assistant-written source. The assistant restored the owner's empty
entrypoint and removed its own implementation-specific prototype tests. No core
components or committed source were changed by this restoration.

Outside docs and tests, require an unambiguous request for the assistant to edit
the specific code. For supporting work with little AI-learning value, propose a
bounded edit and wait for explicit approval; do not infer delegation merely
because the work seems routine. The prototype's model/training integration is
owner learning work and must not be implemented on their behalf without a new
explicit request.

The owner wants one evolving, usable prototype with meaningful behavior at each
milestone, and may later want an external consumer or tests using
"from aurora import Aurora". This is a desired direction, not authorization to add
a new facade class, project, or source API now. Relate M3's meaningful behavior to
training through automatic gradients and understanding the complete caller,
rather than counting helper functions or using green tests as proof of mastery.
