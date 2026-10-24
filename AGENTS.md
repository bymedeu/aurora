# AGENTS.md — Aurora Technical Contract

## Scope

This file contains the public technical contract for this repository. Keep personal
background, career direction, coaching, assessments, collaboration preferences,
private planning and local equipment details out of this file.

Use repository artifacts as the source of truth. Treat dated status and planning
as historical evidence until checked against the working tree and recent history.
Keep volatile implementation status in the linked project records rather than
copying it into this contract.

## Aurora's Identity and Point

Aurora is a long-term educational and research-oriented project for building deep
understanding of artificial intelligence from first principles. It progressively
constructs a small machine-learning framework and research notebook, moving from
numerical foundations to neural networks, automatic differentiation, attention,
and a tiny transformer.

Aurora exists to prove that the owner can understand, explain, implement, test,
and debug the mechanisms behind modern AI. Its principle is:

> Understanding before abstraction.

Aurora is not:

- a tutorial repository;
- a collection of copied notebooks;
- an API-wrapper project;
- an attempt to reproduce all of PyTorch;
- a general-purpose machine-learning framework;
- a race to produce impressive demos.

Implement only enough framework structure to expose and study the mechanism in
scope. Educational transparency is more important than speed, API breadth,
feature count, or production-style abstraction.

Aurora's portfolio point is to demonstrate:

- deep knowledge of ML mathematics and mechanisms;
- the ability to implement important ideas from scratch;
- precise reasoning about tensors, shapes, gradients, numerical behavior, and
  complexity;
- disciplined tests and debugging;
- clear learning notes and technical communication;
- independence and potential for research.

Its long-term systems direction includes tensor storage and metadata, layouts and
strides, autograd graphs, allocation and saved-intermediate lifetime, vectorized
execution, profiling, C++/CUDA kernels, training and inference behavior, ML
infrastructure, foundation-model engineering, distributed training and inference,
and rigorous agent evaluation. These are future directions, not permission to
skip the current milestone.

## Repository inspection and evidence

Before assessing or changing a project:

1. Read the relevant README, roadmap, milestone notes, source, examples, tests,
   and recent relevant history.
2. Inspect that repository's working tree so uncommitted owner work is not
   overwritten or mistaken for committed evidence.
3. Treat inspected repository artifacts as the source of truth for implementation
   status.
4. Distinguish verified state, intended roadmap, proposed work, and unresolved
   questions.
5. Preserve unrelated changes and useful existing work.
6. Prefer the smallest useful, reviewable step.

Strong evidence includes:

- inspected working code;
- focused tests with meaningful assertions;
- mathematical derivations and explicit conventions;
- debugging notes and failure analysis;
- controlled experiments;
- benchmarks and profiles when performance is discussed;
- written explanations and justified architectural decisions.

For meaningful performance work:

1. establish correctness;
2. define a representative workload;
3. record a baseline;
4. profile;
5. form a concrete hypothesis;
6. make one focused change;
7. measure again.

Ask about time complexity, memory complexity, data access, scaling at 10× and
100×, the dominant bottleneck, and whether optimization is premature. Do not add
concurrency, SIMD, GPU execution, or distributed machinery without a measured need.

## Aurora Repository Model

The intended structure is:

```text
aurora/
├── README.md
├── docs/
│   ├── roadmap.md
│   ├── milestones/
│   ├── notes/
│   └── reports/
├── research/
│   ├── papers/
│   ├── articles/
│   └── experiments/
├── examples/
│   ├── linear_regression/
│   ├── xor/
│   ├── mnist/
│   └── attention/
├── src/
│   └── aurora/
└── tests/
```

- `docs/` contains explanations written by the owner, roadmap state, milestone
  evidence, and reports.
- `research/` contains article and paper summaries and controlled experiments.
- `examples/` contains small demonstrations.
- `src/aurora/` contains framework code.
- `tests/` contains correctness tests.

NumPy is a prerequisite and tool, not a milestone or permanent top-level area.

## Aurora Roadmap

Proceed sequentially. Later topics may be recorded for future study, but must not
displace the current completion criteria.

### M0 — Linear Regression From Scratch

Understand prediction, mean squared error, gradients, and gradient descent. Train
a tiny synthetic model and explain `y = ax + b`, the loss, and the update direction.
Use plain Python or NumPy; no PyTorch or scikit-learn.

### M1 — Single Neuron

Understand weights, bias, activation, and binary classification. Train one neuron
on a toy dataset and explain why activation functions exist. No PyTorch or
high-level ML library.

### M2 — Small Neural Network

Understand hidden layers, forward passes, backpropagation, non-linearity, and XOR.
Explain why a single linear model cannot solve XOR and explain both passes.

### M3 — Automatic Differentiation

Build a small scalar computation graph with `backward()`. Explain the chain rule,
topological traversal, gradient accumulation, ownership, and how the design relates
conceptually to PyTorch autograd.

### M4 — PyTorch Reconstruction

Recognize already-understood mechanisms inside PyTorch. Use tensors, autograd, and
a manual training loop; explain `nn.Module`, `zero_grad()`, `backward()`, and
`step()`. Investigate the layers from Python tensor operations through storage,
metadata, dispatch, kernels, devices, and synchronization without trying to rebuild
all of PyTorch.

### M5 — Attention

Understand query, key, value, scaled dot-product attention, masking, and softmax.
Implement a small attention mechanism, explain attention weights, and visualize a
toy sequence.

### M6 — Tiny Transformer

Assemble embeddings, positional information, attention, feed-forward layers,
residual connections, and normalization. Train a tiny character model, explain
every major component, and write a technical article.

Once correct baselines exist, relevant M5–M6 systems questions include attention
time and memory complexity, batching, autoregressive inference, KV caching,
training versus inference, latency versus throughput, numerical precision,
profiling, kernel fusion, and distributed communication. These are investigation
topics, not initial production requirements.

## Canonical project records

Read [README.md](README.md), [the roadmap](docs/roadmap.md), the relevant
`docs/milestones/MX_progress.md`, notes, examples, tests and recent history before
assessing progress or proposing work. Milestone sequence and scope do not establish
current completion. Keep implementation evidence and limitations in those records.
