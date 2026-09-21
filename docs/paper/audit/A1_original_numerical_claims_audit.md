# A1. Numerical-claim provenance audit of the reviewed manuscript (CAGEO-D-26-01439)

Historical source paths below identify the reviewed version, preserved in the local workspace recovery archive. Original notebook paths are members of `historical_notebooks.zip` (manifest/checksums inside); these notebooks are not current executable pipeline inputs.

Source audited: `docs/latex/main.tex` (13 Aug 2026 build) and
`docs/latex/submission_package/supplementary_materials.tex`.
Reproduction scripts: `tensor-based-modal-decomposition-method/studies/brugge_sparse_sensing/scripts/e0_reproduce_archived.py`,
`e0_cluster_table.py`, `s0_data_manifest.py`; outputs in `studies/brugge_sparse_sensing/outputs/e0_*`.
The E0 scripts re-execute the configuration recorded in
`notebooks/experiments/exp_tbmd_2.1&2.2.ipynb` and `exp_tbmd_2.3_probability.ipynb` with the public
TBMD library at commit `35c910a` (library code unmodified).

Status legend: VERIFIED · CORRECTED · NOT REPRODUCIBLE · REQUIRES NEW EXPERIMENT · UNSUPPORTED.

## A1.1 Protocol facts stated in the manuscript

| Claim | Location | Source checked | Finding | Status |
|---|---|---|---|---|
| Tensor per run is 139×48×2×134 | Abstract; Supp. S5 | `data_exp_4_.h5` (`pressure`, `soil` are 10×139×48×133) | 133 time steps, not 134 | CORRECTED (133) |
| 107 training / 27 held-out snapshots | Sec. 3.1; Supp. S5 | `split_data_in_memory_ordered`: `int(0.8·133)=106` | 106 / 27 | CORRECTED |
| 1,788,096 values per run; 17,880,960 total | Supp. S5 | 139·48·2·133 = 1,774,752 | arithmetic based on wrong T | CORRECTED |
| Each run processed independently; W=480 | Sec. 3.4, 3.7 | `TuckerDecomposer` on the dict of 10 runs + `ModalTensorStacker` | W=480 = 10 runs × 48 modes pooled into **one** dictionary; runs are not independent | UNSUPPORTED (text contradicts code) |
| Multilinear ranks [48,48,2,48] (joint) | Sec. 3.7; Supp. S4 | `ranks=None` → `[min(shape)]*4` | joint tensor gives ranks [2,2,2,2], W=20 (`outputs/e0_joint_ranks.json`); [48,48,48] applies to pressure-only | UNSUPPORTED |
| ε=10⁻² is a truncation (energy) threshold | Sec. 3.3 | `tucker(..., tol=epsilon)` | ε is the HOOI convergence tolerance; ranks are not chosen by an error criterion | UNSUPPORTED |
| "HOSVD" | Sec. 3.3 | `tensorly.decomposition.tucker` | HOOI with SVD initialisation | CORRECTED |
| QR pivots by trailing ℓ1 residual norm | Sec. 3.4; Supp. Eq. S2, Alg. S1 | `OptimizedPivotSelector._compute_residual_norms` | ℓ2 norm (standard Businger–Golub pivoting) | CORRECTED |
| ADMM δ₀=1.0, tol 10⁻⁴ | Sec. 3.7 | notebook `ExperimentConfig` used for the curves | δ₀=0.1, tol 10⁻⁷ in the experiment runner | CORRECTED |
| SSIM with 11×11 Gaussian window | Sec. 1, 3.1 | `compute_metrics` | 7×7 Gaussian window; non-standard constants C1=0.012, C2=0.032 | CORRECTED |
| PSNR with MAX=1 | Sec. 1, 3.1 | `compute_metrics` | MAX = data range of the second argument | CORRECTED |
| Metrics on normalised held-out fields | Sec. 3.1 | `run_single_slice_*` | ground truth and reconstruction passed in swapped order; inactive cells (1,722 of 6,672) included; 3-decimal rounding of reconstructions | UNSUPPORTED (metric defects) |
| Metric curves aggregate 10 runs × held-out snapshots | Sec. 5; Figs. 2–4 captions | `run_single_slice_*` in notebook | one run (`case3`), one held-out snapshot (index 10), 11 samples = 1 clean + 10 noisy measurement draws (noise 10 % of max|y|) | UNSUPPORTED |
| Envelopes are descriptive | Fig. captions | runner `_compute_confidence_intervals` | 95 % normal CI over noise draws of a single snapshot | CORRECTED (nature identified) |
| Joint-property well-only curves | Sec. 5; Fig. 4 | `run_single_slice_wells_experiments` with property `all` | raises `ValueError` (mask 139×48 vs 139×48×2); public code cannot produce these curves | NOT REPRODUCIBLE |
| Runtime/memory Table 1 (TBMD 0.2 s / 31 MB; QR+CS 0.01 s / 2 MB) | Table 1 | `measure_brugge_runtime.py` | QR timed on a **random** matrix, CS on a dummy problem; MPS vs CPU mixed | UNSUPPORTED → replaced by E6 |

## A1.2 Quantitative results

| Claim | Location | Source experiment | Parameters | Reproduced value (public code) | Status |
|---|---|---|---|---|---|
| Wells, single property: rel. error 0.62 → 0.17 (N=1→10) | Sec. 5.1; Supp. S6.2.1 | E0 `pressure`/`wells` | case3, slice 10, noise 0.1 | 0.051 → 0.023 | NOT REPRODUCIBLE |
| Wells, single property: SSIM 0.47 → 0.90 | same | same | same | 0.9977 → 0.9993 | NOT REPRODUCIBLE |
| Wells, single property: PSNR 27 → 34.5 dB | same | same | same | 33.1 → 37.5 dB | NOT REPRODUCIBLE |
| Wells, joint: rel. error 0.57 → 0.20; 0.11 at N=20 | Abstract; Sec. 5.1; Conclusions | E0 `all`/`wells` | — | execution fails (A1.1) | NOT REPRODUCIBLE |
| Wells, joint: SSIM 0.47 → 0.88; PSNR 33.8 → 37 dB | Abstract; Sec. 5.1 | same | — | execution fails | NOT REPRODUCIBLE |
| Grid-wide curves improve with budget up to N=299 | Sec. 5.1; Fig. 2 | E0 `pressure`/`grid` | N=1,11,…,351 | improves to N≈131, then NaN / error jumps at run-dependent budgets above ≈200 (ADMM breakdown with 480 dependent atoms) | NOT REPRODUCIBLE (unstable) |
| Cluster area fractions 0.088/0.201/0.167/0.306/0.239 | Supp. Table S1 | E0 cluster | k-means k=5 on wells, N=300 | 0.088/0.201/0.167/0.306/0.239 | VERIFIED |
| Sensors per cluster 53/104/35/39/68 (299 inside mask) | Supp. Table S1; Sec. 6 | E0 cluster | N=300 | run 1: 53/101/38/41/64; run 2: 56/101/37/39/64 (297 inside; 3 QR sensors in inactive cells) — original library non-deterministic | NOT REPRODUCIBLE |
| P(C1\|S)=0.177, P(C2\|S)=0.348, C4 0.130; densities 0.122/0.105/0.026 | Sec. 6 | E0 cluster | N=300 | run 1: 0.178/0.340/0.138; 0.122/0.102/0.027 (run-dependent) | NOT REPRODUCIBLE (diagnostic removed) |
| "k=5 selected by silhouette and confirmed by Gap statistic" | Supp. S7.1 | notebooks | `KMeans(n_clusters=5)` hard-coded; no silhouette/Gap code found | UNSUPPORTED |

## A1.3 Determinism of the original library

Two complete executions of the E0 scripts (manuscript outputs and `rerun_verification/`) agree for the well-budget metrics to the precision quoted, but differ in QR cluster counts and in the grid budgets at which ADMM fails. The original single-precision, multi-threaded library code is therefore not bit-for-bit reproducible; the study package is (A6).

## A1.4 Consequence

No quantitative result of the reviewed manuscript's Abstract, Results or Conclusions is reproducible
with the public code and local data, and the stated protocol differs materially from the executed one.
All such claims are removed from the revised manuscript and replaced by results of the new,
scripted experiments E1–E6 (see A2).
