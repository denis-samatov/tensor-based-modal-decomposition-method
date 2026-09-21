# E9 preregistration: third-order versus fourth-order property modes

Frozen on 2026-09-21 before reading any newly computed E9 outer-test metric.
Earlier E8 RMSE results are known, so E9 is a post-development, procedurally
untouched re-evaluation rather than an external preregistration.

## Hypotheses

- H1: matched-capacity 4D reduces property-wise anomaly-relative Frobenius error.
- H2: matched-capacity 4D increases property-wise anomaly mask-aware SSIM.
- H3: the paired effect depends on sensor budget.
- H4: the effect depends on train-only pressure-saturation anomaly coupling.
- H5: 4D may preserve structure despite worse pointwise physical error.

## Primary endpoints fixed before outer evaluation

The primary reconstruction metric is the property-wise Frobenius norm of the
physical reconstruction residual divided by the Frobenius norm of the held-out
field anomaly relative to the outer-training ensemble mean at the same cell
and report step. This removes the pressure datum that makes raw relative error
artificially small. Raw relative Frobenius error remains a required sensitivity
analysis.

The primary structural metric is local Gaussian SSIM applied to physical-field
anomalies. Local moments are renormalised over active cells. A window is used
only when its centre is active and at least half its in-domain Gaussian weight
falls on active cells. The 7 by 7 window, sigma 1.5, K1 0.01, K2 0.03, and the
property-specific dynamic range are fixed from training data. Raw-field SSIM,
train-range NRMSE, and physical-unit RMSE remain mandatory diagnostics.

## Leakage control and matched comparisons

Each of ten P2 folds holds out one scenario. The other nine scenarios are split
into three deterministic inner scenario folds. Representation, preprocessing,
property rank/weight, recovery solver, and regularisation are selected inside
those folds. Outer evaluation is performed once after freezing the choices.

Both capacity regimes use identical outer folds, measurement rows, timestamps,
and solver candidate protocol. For each matched 3D--4D pair, one common recovery
solver and regularisation level is selected by the mean of the two methods'
inner-validation losses; the frozen common choice is then applied to both
members of the pair on the outer fold. Equal-total capacity compares 16+16 independent
coefficients with 32 joint coefficients. Equal-per-property capacity compares
32+32 independent coefficients with 64 joint coefficients. Existing-well
budgets are 5, 10, 15, 20, and 30 in configured order. Grid budgets are 30,
100, and 300 scalar channels from one common train-only anomaly POD-E QR-DG
order, never a placement optimised separately for either dimensionality.

## Decision rule

A metric-specific empirical advantage requires a favourable median paired
difference and at least seven favourable outer folds out of ten. A broad
novelty claim additionally requires replication in both capacity regimes or at
two adjacent sensor budgets. A structural-only advantage is reported as such;
it cannot be restated as an accuracy improvement. Failure to meet these rules
is retained as a negative or regime-specific result.

The machine-readable frozen protocol is
studies/brugge_sparse_sensing/e9_protocol.json.

## Pre-outer computational amendment

During the first incomplete train-only fold, LASSO ADMM dominated runtime.
Before any E9 outer metric existed or was inspected, LASSO was moved from the
exhaustive inner Cartesian search to a sensitivity analysis on the frozen final
3D and 4D configurations. Least squares and four relative-ridge levels remain
in every inner comparison. Metrics, hypotheses, candidates, folds, sensor
budgets, capacity matching, and decision rules did not change. The interrupted
fold produced no registry or selection artifact and was restarted from scratch.

A second implementation clarification was frozen before outer evaluation: the
same solver requirement is enforced pairwise, rather than merely searching the
same solver candidates independently for 3D and 4D. Existing train-only
registries already contain every candidate, so this clarification requires
only deterministic reselection and no recomputation or outer information.
