# Differentiable Optimization with PyTorch

A compact, tutorial-oriented implementation of differentiable quadratic optimization in PyTorch.

The repository demonstrates how an optimization problem can be placed inside a neural-network forward pass while preserving end-to-end gradient flow.

## What this repository teaches

The core problem is the strictly convex unconstrained quadratic program

\[
x^*(Q,p)
=
\arg\min_x
\left(
\frac{1}{2}x^\top Qx + p^\top x
\right),
\qquad Q \succ 0.
\]

The first-order optimality condition is

\[
Qx^* + p = 0,
\]

which gives

\[
x^* = -Q^{-1}p.
\]

Numerically, the implementation does **not** form the inverse. It solves the linear system directly:

```python
torch.linalg.solve(Q, -p.unsqueeze(-1)).squeeze(-1)
```

because solving the linear system is more stable and efficient.

PyTorch can differentiate through `torch.linalg.solve`, so a loss depending on `x*` can propagate gradients back to `p`, to `Q`, and to any neural network that produced those quantities.

## Why this version is structured this way

A common first attempt at differentiable optimization is to generate a new random `Q` matrix inside every forward pass and train against unrelated random targets. That can produce a decreasing loss curve, but the result is difficult to interpret because the optimization problem itself changes between passes and the target contains no learnable signal.

This repository instead uses:

- a fixed positive-definite `Q` in the introductory model;
- a neural network that predicts the linear cost vector `p`;
- synthetic targets generated from a known ground-truth mapping;
- separate training and validation losses;
- exact unconstrained QP solutions via `torch.linalg.solve`;
- tests that verify both optimality and gradient correctness.

The constrained example is deliberately separated from the PyTorch-only implementation. General constrained QPs require a proper solver; sequential hand-written projection steps are not a substitute for a robust constrained optimization method.

## Repository layout

```text
differentiable-optimization-pytorch/
├── differentiable_qp/
│   ├── __init__.py
│   ├── layer.py
│   └── model.py
├── examples/
│   ├── unconstrained_qp.py
│   └── constrained_qp_cvxpylayers.py
├── tutorials/
│   └── differentiable_optimization_tutorial.ipynb
├── tests/
│   └── test_layer.py
├── .github/workflows/tests.yml
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation

Create and activate a virtual environment, then install the project and test dependency:

```bash
pip install -r requirements.txt
```

The repository uses an editable install, so imports work consistently from the examples and tests.

## Jupyter tutorial

For a line-by-line instructional walkthrough, open:

```text
tutorials/differentiable_optimization_tutorial.ipynb
```

The notebook covers the QP derivation, exact solution, autograd gradient verification, a predict-then-optimize neural model, meaningful synthetic data, training and validation analysis, gradient-flow verification, positive-definite matrix construction, and the distinction between unconstrained and constrained differentiable optimization.

## Run the main tutorial

From the repository root:

```bash
python examples/unconstrained_qp.py
```

The script trains a small predict-then-optimize model and plots training and validation MSE on a logarithmic scale.

The computational path is:

```text
features
   |
neural network
   |
predicted p
   |
differentiable QP solve
   |
optimal decision x*
   |
loss
```

During backpropagation, gradients travel through the QP solution and into the neural-network parameters.

## The PyTorch QP layer

The essential layer is intentionally small:

```python
class UnconstrainedQPLayer(nn.Module):
    def forward(self, Q, p):
        return torch.linalg.solve(Q, -p.unsqueeze(-1)).squeeze(-1)
```

For a positive-definite `Q`, the solution is unique.

The repository also provides `positive_definite_from_factor`, which constructs

\[
Q = LL^\top + \varepsilon I
\]

from an unconstrained matrix `L`. This is useful when a model needs to learn a quadratic cost matrix while preserving positive definiteness.

## Constrained differentiable optimization

For a constrained problem such as

\[
\begin{aligned}
\min_x \quad & \frac{1}{2}x^\top Qx + p^\top x \\
\text{s.t.} \quad & Gx \le h,
\end{aligned}
\]

use a solver designed for differentiable convex optimization rather than an ad hoc projection loop.

An optional CVXPYLayers example is included:

```bash
pip install cvxpy cvxpylayers
python examples/constrained_qp_cvxpylayers.py
```

That example keeps the quadratic and constraint matrices fixed and differentiates with respect to parameterized vectors in a DPP-compliant CVXPY problem.

## Tests

Run:

```bash
pytest -q
```

The tests check:

- satisfaction of the QP first-order optimality condition;
- the autograd gradient with respect to `p` against the analytic gradient;
- positive definiteness of constructed quadratic matrices;
- gradient propagation through the neural network and optimization layer.

A local verification run of this repository passed all four tests. The tutorial training example reduced validation MSE from approximately `0.668` at epoch 1 to approximately `0.0026` at epoch 200 with the documented seed and configuration.

## Important distinction

This PyTorch-only unconstrained layer does not implement a custom implicit-function-theorem backward pass. Instead, it relies on PyTorch autograd differentiating through the linear solve.

That is still differentiable optimization: the optimization solution is part of the differentiable computational graph. More advanced solver-backed systems may use implicit differentiation of KKT conditions or related techniques for constrained problems.

## Scope

This repository is intended for teaching and experimentation. It is deliberately small enough to explain line by line while still separating mathematical assumptions, solver behavior, training logic, and validation.

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

Commercial use is not permitted. The complete license terms are referenced in the `LICENSE` file and are available from the PolyForm Project.
