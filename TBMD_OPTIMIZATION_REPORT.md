# TBMD Optimization Report

## Evidence boundary

This report describes a fresh leakage-controlled optimization run on the checksummed Brugge exports. Each of the ten scenarios was used once as an untouched outer test within this run; all ranks, scaling, property architecture, solvers, regularization and placement settings were selected from the other nine scenarios only. The benchmark itself and older outer results had been inspected before this study, so the result is not a preregistered external validation on a previously unseen data set. It is a nested re-evaluation of a previously studied benchmark.

Canonical evidence is under `studies/brugge_sparse_sensing/outputs/optimization/`. The representation registry has 3,300 train/validation rows; the sparse registry has 42,000 rows; the single-pass outer file has 20,890 rows; the prespecified placement audit has 7,200 rows. No planned fit failed. Independent frozen refitting reproduced 2,160 deterministic outer rows with maximum reconstruction-metric difference `2.84e-14` (`reproduction/validation.json`). Runtime columns were deliberately excluded from exact reproducibility claims.

## 1. Original TBMD bottlenecks

The complete chain is:

`10 scenarios x 2 properties x 4,950 active cells x 133 snapshots` -> train-fitted property transform -> zero-filled `139 x 48 x 2 x T_train` tensor -> Tucker/HOSVD spatial factors -> energy-scaled time-insensitive modal basis -> sensor-row operator -> regularized coefficient estimate -> inverse property transform -> active-cell RMSE/MAE.

The original implementation used HOOI ranks `(48,48,2,r)`, min--max scaling, a training-energy depth, QR/DG placement and either minimum-norm least squares or fixed L1 ADMM. Its factor count at `r=16` was 101,860 floats. The new structured implementation computes mode covariances, retains deterministic spatial eigenspaces, performs SVD in the compressed spatial tensor and expands energy-scaled modes. For a joint model with ranks `(R1,R2,R3,r)`, factor storage is `139 R1 + 48 R2 + 2 R3 + R1 R2 R3 r`; independent-property storage is the sum of the two spatial factors and property cores. The measurement matrix has shape `m x r_total`, where a joint well contributes two rows and a pressure-only well contributes one.

| Source of loss | Diagnosis | Evidence |
|---|---|---|
| Representational error | Dominant in original TBMD | Full-state pressure RMSE `0.3537`; optimized joint rank gives `0.0261` |
| Spatial truncation | `R2<48` and `R1=48` discard material structure | Inner-validation pressure RMSE falls from about `1.30` at `(32,24)` to `0.060` at `(139,48)`, `r=16` |
| Temporal/modal truncation | Secondary after spatial ranks are adequate | Raising `r` from 16 to 32 at `(80,48)` reduces validation pressure error further |
| Property coupling | Joint coefficients compromise pressure/saturation | Independent heads give the best normalized full-state objective (`0.190`) |
| Coefficient recovery | Material for undersampled/ill-conditioned systems | Fold-specific ridge/L1/elastic-net selections substantially outperform unregularized POD and original-basis fits |
| Sensor placement | Material but regime dependent | QR is best at 30 grid channels; clustering becomes competitive only at 100+ channels |
| Regularization | Essential near and below modal depth | Train-selected POD-E uses ridge throughout small budgets; optimized TBMD frequently selects L1 |
| Scaling | Useful but not the main gain | No scaling selected in 7/10 accuracy folds and 8/10 compression folds; RMS scaling selected otherwise |
| Rank mismatch | Original fixed ranks were too small spatially | `(80,48,2,32)` joint TBMD nearly reaches POD pressure representation accuracy |
| Conditioning | Explains part, not all, of sparse loss | At 30 joint wells median `kappa(B_S)` is `8.64e6` for independent TBMD versus `1.92e4` for POD-E; representation remains important even after regularization |

The main causal conclusion is therefore not “tensorization is worse.” The original loss was mainly a spatial-rank floor, followed by an over-restrictive joint-property coefficient model and an ill-conditioned sparse inverse problem.

## 2. Search space

The staged, bounded search covered:

- spatial ranks `(32,24)`, `(48,32)`, `(48,48)`, `(64,48)`, `(80,48)`, `(96,48)`, `(139,48)`;
- modal ranks 8, 16, 24 and 32, property ranks 1 and 2, plus train-energy ranks at 0.99, 0.999 and 0.9999;
- joint, shared-spatial/independent-coefficient, independent-property, weighted-joint and joint-plus-POD-residual architectures;
- none, min--max, standard-deviation and RMS/energy property scaling, with joint property weights from 0.25 to 4;
- LS, four relative ridge penalties, three L1 penalties and two elastic-net settings;
- configured, DG/QR, whitened QR, condition-aware, pressure-objective, cluster-constrained and well-region placement;
- all 1--30 existing-well budgets and 2--300 grid-channel budgets.

Every attempted candidate and its inner score is retained. Energy-adaptive candidates and the residual hybrid were not removed when fixed ranks performed better.

## 3. Nested-validation protocol

For outer fold `q`, scenario `q` is loaded only for the final evaluation. The remaining nine scenario identifiers are deterministically divided into three inner validation groups; each training scenario validates exactly once. Stage A selects representation using the equal-property loss

`0.5 * (RMSE_pressure / prior_RMSE_pressure + RMSE_saturation / prior_RMSE_saturation)`.

Stage B selects recovery on configured/QR anchors, Stage C selects placement with the frozen Stage-B solver, and Stage D jointly rechecks the selected placement against all solver candidates. Stage E refits on all nine outer-training scenarios and evaluates the test scenario once. Nearby un-tuned budgets inherit the nearest logarithmic anchor strategy. The accuracy configuration minimizes mean inner loss; the compression configuration is the smallest model within 5% of that minimum. Random placement reports the within-fold median of 20 deterministic seeds rather than treating seeds as independent folds.

## 4. Rank optimization

Rank expansion is the largest improvement. The original full-state pressure/saturation RMSE was `0.3537/0.004176`. The train-selected joint `(80,48,2,32)` model reached `0.0261/0.000432` with 259,188 floats. The untruncated direction approaches POD/POD-E as predicted by the TBMD--POD-E equivalence, but `(139,48)` no longer has a storage advantage at the same depth.

Fixed ranks were selected over singular-decay energy thresholds in this data set. This is a negative result for global adaptive-energy rank rules, not a reason to remove them: their validation records remain in `representation_registry.csv`.

## 5. Property formulation

The lowest train-selected full-state joint loss uses two independent property decompositions with 16 pressure and 16 saturation coefficients. It gives outer full-state RMSE `0.0424 +/- 0.0068` for pressure and `0.000269 +/- 0.000072` for saturation, versus POD-E `0.0133 +/- 0.0040` and `0.000359 +/- 0.000077`. Thus POD-E remains substantially better for pressure, while independent TBMD is better for saturation and for the declared equal-property normalized objective (`0.190` versus `0.231`).

Shared-spatial/independent-head and residual hybrids did not match the fully independent validation objective. Property weights are not identifiable in the independent model because multiplication before each property SVD and division during expansion cancel; their fold-specific values are recorded but are not credited as an improvement. The effective changes are spatial rank, property separation, modal depth and scaling.

## 6. Sparse recovery optimization

No single solver is universally selected. At 30 joint wells, the accuracy TBMD uses L1 in seven folds, ridge in two and elastic net in one. At 300 grid channels, the compressed model uses LS in eight folds and L1 in two. POD-E relies mainly on relative ridge at sparse budgets. This budget dependence supports adaptive sparse recovery, but all adaptation is train-selected.

The original basis remains poor even with optimized sensing/recovery (`0.4289/0.004235` at 30 joint wells), showing that solver tuning cannot remove its representation floor. Conversely, independent TBMD cannot infer the unmeasured property: with pressure-only wells its saturation reconstruction is effectively zero after inverse scaling (`RMSE=0.26897`). Pressure-only results must therefore not be presented as joint-state reconstruction.

## 7. Sensor-placement optimization

The selected objective changes with budget. QR/DG is most reliable once enough channels observe the modal space; regularized condition-aware or configured placements are often selected at low well budgets. Placement alone does not cure the highly conditioned independent block basis. The complete outer audit is `outer_placement_audit.csv` and does not feed back into selection.

For optimized TBMD grid sensing, QR has the best mean joint score at 30 channels (`0.385` for the selected strategy). At 100 channels QR (`0.272`) and cluster-constrained placement (`0.280`) are close; at 300 channels the clustered, QR and condition-aware prefixes coincide numerically (`0.239`). The cluster method is therefore not a general replacement for global placement.

## 8. Well-placement optimization

Pressure-objective rankings are stable: mean pairwise Kendall tau is `0.882` for condition-aware pressure ranking and `0.908` for pressure DG. Joint rankings are less stable (`0.754` and `0.739`). The ten consistently early pressure wells are configured indices 18, 14, 21, 20, 12, 26, 17, 29, 13 and 9. Their mean pressure variability is `5.03 u_p`, versus `4.07 u_p` over all wells; median rank correlates with pressure variability at `-0.624`. Their saturation variability is slightly below the all-well mean, consistent with the pressure-specific objective. This is an ensemble-dynamics interpretation, not a causal geological claim.

## 9. Clustering experiment

Clustering uses active cells only, standardized coordinates plus train-only pressure/saturation mean and standard deviation, deterministic seeds, and selection by mean silhouette plus 0.1 times adjusted-Rand stability. All outer-training fits select `k=5` without hard-coding it.

The corrected experiment does not justify a main-paper clustering claim. At 30 grid channels its outer score (`1.994`) is much worse than QR (`0.385`); at 100 it is competitive but not best on average; at 300 its prefix is effectively identical to QR. It is retained in the registry and Supplement as a negative/conditional result.

## 10. Ablation results

At 30 joint wells:

| Version | Pressure RMSE | Saturation RMSE | Normalized joint error | Stored floats |
|---|---:|---:|---:|---:|
| Original basis + optimized recovery | 0.4289 | 0.004235 | 2.864 | 101,860 |
| Rank-optimized joint TBMD | 0.0892 | 0.000643 | 0.451 | 259,188 |
| Joint TBMD + POD residual | 0.2117 | 0.000788 | 0.591 | 188,708 |
| Compressed independent TBMD | 0.1754 | 0.000651 | 0.495 | 120,704 |
| Accuracy-selected independent TBMD | 0.1787 | 0.000644 | 0.499 | 149,728 |

Rank correction supplies the dominant gain. Property separation supplies the best full-state and high-grid-budget joint objective, but the joint rank-optimized basis is better at existing joint wells. The residual hybrid is dominated here. No gain is attributed to independent-model property weights, which algebraically cancel.

## 11. Final outer-test benchmark

At all 30 joint wells, POD-E is the strongest relevant method: pressure/saturation RMSE is `0.0840 +/- 0.0273 / 0.000575 +/- 0.000130`. Rank-optimized joint TBMD is the strongest tensor comparator at `0.0892 +/- 0.0234 / 0.000643 +/- 0.000132`; the pressure difference is small but POD-E has the lower joint score. The compressed independent TBMD is worse in error but uses 120,704 rather than 316,800 stored floats. Prior+IDW remains visible at `0.3127/0.000780`.

Fold-level pressure RMSE at 30 joint wells:

| Fold | POD-E | Joint rank-optimized TBMD | Compressed independent TBMD |
|---:|---:|---:|---:|
| 1 | 0.1328 | 0.1294 | 0.1202 |
| 2 | 0.0704 | 0.0816 | 0.1346 |
| 3 | 0.0838 | 0.0817 | 0.1184 |
| 4 | 0.1199 | 0.0874 | 0.1580 |
| 5 | 0.0487 | 0.0680 | 0.1815 |
| 6 | 0.1043 | 0.0957 | 0.0959 |
| 7 | 0.0803 | 0.0785 | 0.2534 |
| 8 | 0.0749 | 0.0800 | 0.1317 |
| 9 | 0.0716 | 0.1297 | 0.3755 |
| 10 | 0.0531 | 0.0595 | 0.1845 |

At 300 grid channels, compressed TBMD has higher pressure error (`0.0663` versus `0.0223`) but lower saturation error (`0.000324` versus `0.000443`) and lower normalized joint error (`0.238` versus `0.290`) than POD-E in all ten folds. The paired raw Wilcoxon p-value is `0.001953`; it is not significant after a deliberately conservative Holm correction across all 360 comparisons in the same metric family (`0.703`). We therefore report the fold-consistent effect without claiming broad statistical superiority.

At 30 pressure-only wells, compressed TBMD pressure RMSE is `0.1343`, joint rank-optimized TBMD is `0.1164`, and POD-E is `0.2904`. This is a pressure-only operating regime; no claim is made for saturation reconstruction.

## 12. Accuracy/compression Pareto analysis

The scientific decision is **Outcome C (regime-specific), with an Outcome-B storage trade-off elsewhere**:

- POD-E is best for joint reconstruction from the 30 existing wells and for pressure accuracy in grid sensing.
- Rank-optimized joint TBMD is close to POD-E at 30 wells and can have lower pressure error at some 20-well folds, but it does not win the joint objective.
- Independent TBMD is best on the declared joint normalized objective at 100--300 grid channels because it improves saturation; at 300 channels it wins all ten folds.
- Independent/joint TBMD is best for pressure-only well fitting, but it cannot reconstruct unobserved saturation reliably.
- The accuracy and compression TBMD bases use 149,728 and 120,704 floats, respectively: `2.12x` and `2.62x` less storage than rank-32 POD/POD-E. Their basis fits are also faster (`0.57 s` and `0.48 s` versus `1.49 s`), while regularized online solves at 30 wells are slower (about `1.2--1.5 ms/snapshot` versus `0.60 ms/snapshot` for POD-E in this run).

Model-storage memory is 1.20 MB for accuracy TBMD, 0.97 MB for compressed TBMD and 2.53 MB for POD-E using float64. These are model-storage figures, not measured process peak RSS. Computing all clustering and placement candidates dominates the offline context time (about 12 s in the placement audit) and is not an intrinsic cost of applying one frozen placement.

## 13. Final TBMD configuration

There is no scientifically defensible single configuration for every regime. The frozen train-selected family is:

| Role | Architecture | Spatial ranks | Modal ranks | Scaling selection | Storage |
|---|---|---|---|---|---:|
| Accuracy joint objective | Independent properties | `(80,48)` per property | `16+16` | none in 7 folds; RMS in 3 | 149,728 |
| Compression objective | Independent properties | `(64,48)` per property | `16+16` | none in 8 folds; RMS in 2 | 120,704 |
| Joint-well tensor comparator | Joint properties | `(80,48,2)` | 32 | min--max in all folds | 259,188 |

The independent models use deterministic randomized SVD seed 0 and six power iterations. Clustering seeds are 20260917--20260919; random-placement seeds are derived from master seed 20260917 and outer fold. Solver and placement are fold- and budget-specific nested selections stored verbatim in `selected_sparse_strategies.json`; reporting one global penalty would be false. At the main 30-well budget, accuracy TBMD selects L1 in seven folds, ridge in two and elastic net in one.

## 14. Changes to manuscript

The manuscript now centers on spatial truncation, nested optimization and regime-specific accuracy--compression trade-offs rather than claiming superiority for fourth-order joint TBMD. The research questions, methods, protocol, results, discussion, limitations and conclusion have been updated. Figures 8--14 show rank sensitivity, representation compression, sensor budgets, conditioning, well ranking, ablation and the Pareto map. `tab_optimization_main.tex` and `tab_tbmd_evolution.tex` are generated from canonical CSV outputs; `optimization_numbers.tex` supplies manuscript macros.

## 15. Remaining limitations

- Only ten fixed-geology control scenarios from one 2-D Brugge export are available; folds are related scenarios, not independent reservoirs.
- The pressure unit is absent from the HDF5 metadata, so it remains `u_p`.
- Outer folds are untouched within this run but the benchmark was seen during prior development.
- The equal-property normalized objective is a declared design choice; applications may weight pressure and saturation differently.
- Pressure-only TBMD cannot infer saturation with independent heads.
- Model storage is measured exactly; process peak memory is not measured for every method.
- Runtime is hardware-specific and is not part of deterministic reproduction.
- The transformed HDF5/JSON inputs are checksummed, but the original simulator deck/export pipeline is unavailable.
- No geological uncertainty, well-bore response, correlated noise or field qualification is represented. These results are a reproducible numerical benchmark, not operational validation.
