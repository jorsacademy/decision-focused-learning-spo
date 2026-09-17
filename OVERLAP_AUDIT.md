# Repository Overlap Audit — Predict-then-Optimize, Decision-Focused, and Differentiable Optimization

This document records portfolio overlap without merging, archiving, renaming, or deleting repositories.

## Status legend

- **Keep separate** — materially different optimization layer, gradient mechanism, uncertainty model, or research question.
- **Overlap but justified** — same research family, but a distinct benchmark or educational role.
- **Potential consolidation** — unusually high duplication; requires another review before any action.

## SPO / decision-focused learning

### `decision-focused-learning-spo`

**Keep separate.**

Role: method-centered SPO/SPO+ benchmark using controlled downstream problems such as fixed-cardinality selection and contextual shortest path.

Distinctive value: isolates the decision-focused loss itself and compares prediction error with downstream regret under exact optimization oracles.

### `predict-then-optimize-production-planning-spo-plus-pytorch`

**Overlap but justified.**

Role: application-centered SPO+ benchmark for contextual multi-period production planning with an exact LP downstream model.

Distinctive value: richer industrial planning structure, production/inventory balance, capacity, decision-activity-weighted MSE, and application-specific feasibility auditing.

These two should remain separate: one is the general methodological sandbox, the other is a production-planning case study.

### `distributionally-robust-decision-focused-learning`

**Keep separate.**

Role: decision-focused learning under ambiguity/distributional robustness. The uncertainty model changes the research question materially.

## Differentiable optimization layers

### `differentiable-optimization-pytorch`

**Keep separate.**

Role: tutorial/foundation for differentiating through quadratic optimization, including direct autograd through linear solves and a constrained CVXPYLayers example.

### `differentiable-optimization-portfolio`

**Overlap but justified.**

Role: compact application sandbox for differentiable convex portfolio decisions through `cvxpylayers`.

This is an application of the differentiable-layer idea rather than a duplicate tutorial.

### `differentiable-black-box-supplier-selection-pytorch`

**Keep separate.**

Role: discrete mixed-integer downstream optimization with black-box gradient estimators such as signed identity and perturb-and-resolve.

The gradient mechanism and optimization class are fundamentally different from differentiable convex layers.

## Contextual and inverse optimization

### `contextual-optimization-newsvendor`

**Keep separate.**

Role: context-conditioned optimization/decision rules rather than an end-to-end differentiable solver layer.

### `inverse-optimization-shortest-path`

**Keep separate.**

Role: infer objective coefficients from observed optimal path decisions. This reverses the direction of the usual predict-then-optimize pipeline.

### `inverse-optimization-industrial-decisions`

**Overlap but justified** relative to the shortest-path inverse project.

Role: richer inverse-optimization benchmark for production/resource-allocation LPs with challenger generation, noisy near-optimal observations, held-out/OOD decision-regret evaluation, and coefficient-stability analysis.

The shortest-path repository is the compact graph-based sandbox; the industrial-decisions repository is the broader research benchmark.

## Audit conclusion

There is no current consolidation candidate in this family. The projects form a useful methodological progression:

`predict then optimize -> SPO/SPO+ -> differentiable convex layers -> black-box discrete differentiation -> robust decision-focused learning -> contextual optimization -> inverse optimization`.

Repositories should be consolidated only when both the downstream model and the learning/differentiation mechanism are substantially the same. Sharing the phrase `end-to-end optimization` is not sufficient evidence of duplication.
