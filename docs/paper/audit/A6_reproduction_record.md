# A6. Independent reproduction record

## Run 1 — complete end-to-end execution (2026-09-17)

Command (from `studies/brugge_sparse_sensing`, output to a scratch directory):

```bash
BSS_OUT=<scratch>/outputs BSS_FIG=<scratch>/figures BSS_LOG=<scratch>/run_logs ./run_all.sh 10
```

Result: exit code 0, `run_all finished in 1498 s` (12-core Apple M2 Pro, 10 parallel jobs).
All stages ran from empty output directories (S0, E0, V1, E1, E2, E3, E4, E5, E7, E6, tables,
number macros, figures). **The scratch directory was cleared by a session restart before the
comparison step**, so the numerical comparison for this run is not available. The run demonstrates
that the single-command pipeline executes end-to-end.

## Run 2 — repeated into a persistent directory (historical `rerun_verification/`)

The temporary outputs were subsequently removed from the active study and preserved by the previous workspace cleanup. The compact comparison below and A6_rerun_comparison.txt remain the verification record; use the rerun command to regenerate outputs.

Stages compared with the outputs used in the manuscript so far:

| Stage | Comparison | Result |
|---|---|---|
| S0 data manifest | all fields except git/platform | identical |
| V1 library verification | all fields | identical (QR pivots equal; ADMM iterations 110/84; state difference 2.76e-08); Proposition 1 residuals at machine precision (1.8e-14 vs 1.1e-14 — both < 1e-13, reported as a bound) |
| E1 representation, P1 and P2, all 20 folds (merged `e1_representation.csv`) | all numeric fields except timings | max relative difference **0** (bit-for-bit) |
| E2 main benchmark (`e2_summary.csv`, 98,240 rows × 12 numeric fields, timings excluded) | all entries | max relative difference **0** (1,178,880 values bit-for-bit) |
| E0 re-execution of the reviewed configuration (original library) | wells, grid, soil, cluster table | wells metrics identical to quoted precision; **cluster counts and failing grid budgets differ** (original library non-deterministic) → reported in Supp. S2; run-dependent E0 macros removed |

Run 2 completed: `run_all finished in 1561 s` (exit code 0). Full comparison (`compare_rerun.py`,
output in `A6_rerun_comparison.txt`):

```
macros compared: 873; identical: 868; timing macros differing: 5; non-timing differing: 0
  timing CostBlockDg: manuscript=3.2 rerun=3.0
  timing CostPodSvd: manuscript=1.18 rerun=1.22
  timing CostQr: manuscript=2.1 rerun=2.0
  timing CostSynth: manuscript=4.8 rerun=4.7
  timing CostTucker: manuscript=8.9 rerun=8.7
e2_summary.csv: rows=98240, numeric cols=12, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e1_representation.csv: rows=420, numeric cols=9, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e3_noise.csv: rows=16820, numeric cols=6, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e4_l1_sensitivity.csv: rows=900, numeric cols=5, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e5_stability.csv: rows=60, numeric cols=7, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e7_spatial_structure.csv: rows=10, numeric cols=7, max rel diff=0.00e+00, entries rel diff>1e-9: 0
stats_wilcoxon.csv: rows=64, numeric cols=6, max rel diff=0.00e+00, entries rel diff>1e-9: 0
e0_cluster_table.csv: rows=5, numeric cols=7, max rel diff=5.66e-02, entries rel diff>1e-9: 9
e0_archived_pressure_wells.csv: rows=30, numeric cols=15, max rel diff=6.62e-03, entries rel diff>1e-9: 360
```

**Conclusion.** All 868 non-timing number macros used to build the manuscript are identical between
the two independent runs; the only differing macros are the five wall-clock timings of Table 3
(wall-clock variability; see the timing correction below). All study result files (E1, E2, E3, E4, E5, E7, Wilcoxon statistics) are
bit-for-bit identical. The only non-identical files are the E0 re-executions of the *original*
library (cluster table, per-noise-draw PSNR values), whose non-determinism is documented in
Supplementary Section S2 and A1; the E0 values quoted in the manuscript are unaffected at their
printed precision.

To repeat: `cd studies/brugge_sparse_sensing && BSS_OUT=... BSS_FIG=... BSS_LOG=... ./run_all.sh 10`,
then `python scripts/compare_rerun.py <BSS_OUT>`.

## Timing-bound correction (repository audit, 2026-09-18)

The earlier prose claimed a maximum timing difference of 4%; that bound is not supported by
the primary recorded timings. The historical rerun record was recovered byte for byte from
the previous workspace recovery archive and is retained here as `A6_rerun_timing.json`. Compared
with canonical `outputs/e6_cost.json`, the largest relative difference among the five changed
timing macros is 4.70% for block-DG (0.0031784170132596046 s versus 0.003029041999980109 s).
Rounded display macros 3.2 versus 3.0 ms differ by 6.25%; rounding must not be confused with
raw timing variability. Other timing fields that round to the same macro can also differ.
The unsupported 4% bound is removed from the active supplement/study guide; all original
canonical timings, scientific results and printed comparison pairs remain unchanged.

## Run 3 — clean exact-environment regeneration (2026-09-19)

The repository-local `.venv/bin/python` was found to be a broken symlink. The prior shell fallback
therefore selected a different Python/SciPy stack, which changed Wilcoxon-derived macros. A clean
temporary environment was built from `requirements-lock.txt` with Python 3.12.14 and was checked by
the new fail-closed `scripts/verify_environment.py` gate:

```
ENVIRONMENT VERIFIED: Python 3.12; h5py=3.16.0, hdbscan=0.8.44, matplotlib=3.11.1,
numpy=2.5.2, pandas=3.0.5, scikit-image=0.26.0, scikit-learn=1.9.0, scipy=1.18.1,
tensorly=0.9.0, torch=2.14.0
```

All outputs and figures were regenerated from empty scratch directories under
`<temporary-rerun-directory>`. The wrapper was edited while that shell process was still
reading it, so it stopped after the completed E1/E2 stages (`command not found`). This is recorded
as an orchestration failure, not a successful single-command run. The remaining published stages
(E3, E4, V2, E5, E7, E6, tables, macros and figures) were then run explicitly with the same data,
environment, seeds and scratch outputs.

The final comparison is in `A7_rerun_comparison_2026-09-19.txt`: all 869 non-timing macros were
identical; E1, E2, E3, E4, E5 and Wilcoxon files were bit-for-bit identical; E7 differed only at
floating-point roundoff (maximum relative difference 9.04e-14); and four wall-clock macros varied.
The manuscript claim gate passed for all 161 used result macros after being corrected to reject any
scientific-macro drift while explicitly reporting allowed `Cost*` wall-clock drift. The original
E0 helper remains informational and non-deterministic, as documented in A1 and Supplementary S2.
