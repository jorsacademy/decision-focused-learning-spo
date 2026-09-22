# Differentiable Optimization for Portfolio Decisions

A compact research sandbox for **differentiable convex optimization layers** in PyTorch.

The model predicts expected asset returns from context, solves a long-only mean-variance
portfolio problem in the forward pass, and backpropagates decision loss through the convex
program using `cvxpylayers`.

## Optimization layer

\[
\min_w \; \lambda w^\top \Sigma w - \mu^\top w
\]

subject to

\[
\mathbf{1}^\top w = 1,\qquad w \ge 0.
\]

The covariance matrix and risk-aversion coefficient are fixed layer data; predicted expected
returns are differentiable parameters.

## Run

```bash
pip install -e ".[dev]"
python scripts/train.py
pytest
```

## Research scope

This is a transparent sandbox, not a production portfolio optimizer. It demonstrates
predict -> optimize -> differentiate -> update end-to-end training and is grounded in
Agrawal et al., *Differentiable Convex Optimization Layers*, NeurIPS 2019.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
