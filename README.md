<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=180&color=0:0B1026,30:7C3AED,60:EC4899,85:F97316,100:FDE047&text=Aurora&fontColor=ffffff&fontSize=60&animation=fadeIn&fontAlignY=38"/>

# Aurora

### From Tensors to Transformers — A Journey Toward Understanding AI From First Principles

[![Status](https://img.shields.io/badge/Status-In%20Development-1E3A8A)]()
[![Focus](https://img.shields.io/badge/Focus-Understanding%20Over%20Abstraction-FBBF24)]()
[![Language](https://img.shields.io/badge/Primary-Python-3776AB)]()
[![Goal](https://img.shields.io/badge/Goal-Research%20Grade%20Understanding-F59E0B)]()

</div>

---

## What is Aurora?

Aurora is a long-term educational and research-oriented project whose purpose is to understand modern artificial intelligence from first principles.

Rather than relying on high-level frameworks and hidden abstractions, Aurora progressively rebuilds the core concepts behind machine learning systems:

* Linear Regression
* Neural Networks
* Backpropagation
* Automatic Differentiation
* Attention Mechanisms
* Transformers

The objective is not to build another machine learning library.

The objective is to understand every major component deeply enough to explain it, implement it, debug it, and eventually improve upon it.

---

## Current Status — November 21, 2026

**M3 is complete.** Its implementation was delivered on October 17, 2026,
59 days before the December 15 deadline. Aurora includes scalar reverse-mode
autograd, an importable linear model, a seven-parameter nonlinear training
experiment, and a recorded baseline of 61 passing tests.

**M4 — NumPy Tensors and Autograd** is now the active learning milestone.
It extends Aurora's own differentiation machinery to NumPy arrays, without
PyTorch. The first tensor checkpoint now covers floating-point array storage,
independent values and gradients, graph metadata, and equal-shape addition with
gradient accumulation. Full-array sum now connects array calculations to a
scalar result and propagates gradients through earlier operations. All **88 tests
pass**, including 27 tensor checks.
No additional M3 entry review is required.

M4 remains in progress. Further reductions, matrix operations, broadcasting
derivatives, and network training remain to be implemented and verified.
See [the M4 record](docs/milestones/M4_progress.md) for scope and limitations.

Run from the repository root:

```bash
python3 -B tests/run.py
PYTHONPATH=src python3 -B examples/autograd/train.py
PYTHONPATH=src python3 -B examples/autograd/nonlinear.py
```

The linear example prints its results. The nonlinear experiment currently runs
silently; its regression test checks the final learned state. See
[the example guide](examples/autograd/README.md) and
[the M3 completion record](docs/milestones/M3_progress.md).

---

## Philosophy

Aurora follows a simple principle:

> Understanding before abstraction.

Every concept is approached in three stages:

1. Understand the mathematics.
2. Implement the mechanism.
3. Compare with production-grade tools.

Aurora implements its learning mechanisms using Python and NumPy. Comparisons
with production tools may provide context, but do not require a framework
dependency or replace the from-scratch implementation.

---

## Roadmap

| Milestone | Topic                     |
| --------- | ------------------------- |
| M0        | Linear Regression         |
| M1        | Single Neuron             |
| M2        | Neural Networks           |
| M3        | Automatic Differentiation — complete; implementation delivered October 17, 2026 |
| M4        | NumPy Tensors and Autograd |
| M5        | Attention                 |
| M6        | Tiny Transformer          |

Each milestone includes:

* Notes
* Experiments
* Implementations
* Progress Reports
* Research Summaries

---

## Delivery Timeline — September 2027 Portfolio Target

Aurora is intended to reach a portfolio-quality first release before September 2027. The target is a technically credible project that can support applications to demanding AI and software-engineering opportunities, including Google, rather than a collection of unfinished experiments.

### Target deliverable

By **August 31, 2027**, Aurora should contain:

* completed and explained M0–M6 milestone artifacts;
* a reproducible tiny-transformer demonstration built from understood components;
* focused correctness tests for the central mechanisms;
* one substantial technical write-up connecting the mathematics, implementation, debugging evidence, and measured behaviour;
* a concise portfolio presentation that makes the student's individual engineering and research contribution clear.

September 2027 is reserved for final presentation work and applications, not unfinished foundational implementation.

### Calendar deadlines

| Deadline | Deliverable |
| -------- | ----------- |
| September 30, 2026 | Complete M2: train XOR successfully and explain forward propagation, backpropagation, and the role of non-linearity. |
| December 15, 2026 | M3 complete; implementation delivered October 17, 2026, 59 days early: scalar autograd, tested training prototypes, debugging evidence, and design notes. |
| February 28, 2027 | Previous M4 target, to reassess for the revised scope: NumPy tensor autograd and a small trainable network using Aurora mechanisms. |
| May 14, 2027 | Complete M5: implement, test, explain, and visualise attention on a small sequence. |
| May 15–June 30, 2027 | LeetCode becomes the primary use of free time. Keep Aurora to light maintenance, review, and documentation; no critical milestone deadline falls in this period. |
| August 15, 2027 | Complete M6: reproducible tiny character-level transformer with correctness evidence and an initial technical write-up. |
| August 31, 2027 | Freeze the Aurora portfolio release: clean repository, reproducible demonstration, tests, final report, and project presentation. |
| September 1–15, 2027 | Prepare and tailor application material, project summary, and outreach using the frozen release. |

The schedule should be reviewed at the end of every month. If a milestone slips, reduce optional scope first; do not hide missing understanding behind copied code or premature framework features.

---

## Repository Structure

```text
aurora/
├── docs/
├── research/
├── examples/
├── src/
└── tests/
```

* `docs/` — learning notes and milestone tracking
* `research/` — article and paper summaries
* `examples/` — demonstrations and experiments
* `src/` — Aurora framework source code
* `tests/` — validation and correctness tests

---

## Long-Term Vision

Aurora is designed as a multi-year journey.

By the end of the project, the goal is to:

* understand modern deep learning architectures;
* implement core AI systems from scratch;
* read and reproduce research papers;
* develop the foundations required for advanced AI research.

---

<div align="center">

*"Every sunrise begins below the horizon."*

</div>
