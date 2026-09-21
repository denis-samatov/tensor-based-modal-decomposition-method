# Third-order versus fourth-order property-mode analysis

## Evidence boundary

E9 is a post-development but leakage-controlled re-evaluation of the Brugge
control ensemble. Earlier E8 results were known, but the E9 endpoints,
candidate set, folds and decision rule were frozen before any new E9 outer
metric was computed. The ten leave-one-scenario-out tests were evaluated once.
All preprocessing, ranks, fourth-order architecture, recovery and
regularisation choices came from three grouped inner folds drawn from the other
nine scenarios. The frozen protocol and preregistration have SHA-256 values
`102d8140f901364b81c2c53a18b662dda499b8e3d093ccb1ce4a3dc026325db0`
and `f5b8ca20d485121fe03de4c276a6841fed959568f388c3a92387f5154057b56b`,
respectively; both hashes are stored in every fold selection record.

Canonical evidence is in
`studies/brugge_sparse_sensing/outputs/e9_property_mode/`. The train-only
registry contains 2,160 representation records and 19,200 sparse-recovery
records, all successful. The single-pass outer registry contains 2,080 records
over all 133 report steps. This is validation on ten related control scenarios
at fixed geology, not external validation on independent reservoirs.

## 1. Original hypothesis

The historical hypothesis was that a space--space--property--time tensor can
use pressure--saturation dependence that is unavailable to two separately
fitted space--space--time tensors. E9 separated five statements that had
previously been conflated:

1. an explicit property axis may improve estimation of spatial factors;
2. property rank one may compress a genuinely shared response;
3. shared coefficients may transfer information across properties;
4. partial sharing may regularise a sparse inverse problem;
5. a fourth-order model may preserve structure even when physical-unit RMSE is
   not lower.

The preregistered H1--H5 test reconstruction error, structural similarity,
sensor-budget dependence, dependence on measured coupling, and metric
discordance. A metric-specific advantage requires a favourable median paired
difference and at least 7 favourable outer folds out of 10. A broad claim also
requires replication across capacities or adjacent budgets.

## 2. Problems with previous metric implementation

The original article intended to report relative Frobenius reconstruction
error and SSIM. The reviewed Brugge implementation did not faithfully realise
that intent.

| Original metric or claim | Problem found in audit | Corrected E9 metric | Used now |
|---|---|---|---|
| `||Xhat-X||F / ||X||F` | pressure datum makes the denominator very large; inactive cells and rounded fields entered the historical call | property-wise anomaly-relative Frobenius norm on active cells after inverse scaling | Headline |
| Historical SSIM | inactive zeros entered rectangular windows; arguments were swapped; reconstruction was rounded; 7x7 window and non-standard constants disagreed with the text | mask-renormalised 7x7 Gaussian SSIM on train-referenced anomalies | Headline |
| Raw relative error | scale-dependent and strongly diluted for pressure | correctly implemented on active physical fields | Sensitivity |
| Raw-field SSIM | saturated close to one because common baselines dominate | correctly masked, but retained only as a diagnostic | Sensitivity |
| RMSE | valid physical-unit diagnostic but not scale comparable across properties | active-cell physical-unit RMSE | Mandatory diagnostic |
| PSNR | monotone re-expression of MSE/RMSE under a fixed range | omitted | No |

Historical quantitative claims remain excluded: the old curves used one
scenario, one snapshot and repeated noise draws rather than the stated
scenario/time aggregation; the joint-well path also failed a mask/tensor shape
check. E9 preserves the original scientific intent, not the defective code.

## 3. Corrected reconstruction error

For held-out property `k`, the primary error is

`E'_k = ||Xhat_k - X_k||F / ||X_k - mu_train,k||F`,

where `mu_train,k` is the outer-training ensemble mean at the same active cell
and report step. The prediction and target are inverse-transformed before the
metric is evaluated. Pressure and saturation are never mixed in this scalar.
The denominator therefore measures the scenario-specific signal that the
sensor reconstruction must recover, rather than the large pressure datum.

Raw relative Frobenius error, train-range NRMSE and RMSE are generated for every
same-fold, same-measurement comparison. The primary normalization was fixed
before outer evaluation. It was not selected because it favoured 4D.

## 4. Mask-aware SSIM

For a Gaussian window with weights `g` and active mask `m`, local moments are
computed with weights `g*m / sum(g*m)`. Thus inactive values cannot contribute
to local means, variances or covariance. A window is retained only when its
centre is active and active cells contain at least 50% of its in-domain
Gaussian weight. E9 uses a 7x7 window, sigma 1.5, `K1=0.01` and `K2=0.03`.
The stabilising constants use the property-specific anomaly range fitted only
from the outer-training scenarios. Scores are averaged over valid windows and
all 133 report steps, then retained as one value per outer fold.

Tests cover identity, symmetry, determinism, inactive-background invariance,
invalid support/ranges, vectorised-versus-snapshot equivalence and train-only
reference fitting. Raw SSIM is indeed saturated: over optimized outer records
its median is 0.99931 for pressure and 0.99994 for saturation. Anomaly SSIM has
medians 0.9653 and 0.9002 and spans -0.380 to 0.9999 for pressure and 0.745 to
0.984 for saturation. The median absolute 3D--4D separation rises from
`8.8e-5` to `0.0274` for pressure and from `9.6e-6` to `0.0187` for saturation.
Anomaly SSIM is therefore the primary structural score; an additional gradient
headline metric was unnecessary.

## 5. Cross-property structure analysis

All dependence diagnostics use outer-training scenarios only.

| Diagnostic | Mean across folds | Interpretation |
|---|---:|---|
| raw cell/time Pearson correlation | -0.618 | dominated by absolute field evolution and scaling |
| anomaly Pearson correlation | 0.240 | weak-to-moderate pointwise shared variation |
| temporal correlation of spatial-mean anomalies | 0.824 | strong shared global scenario response |
| mean top-16 spatial subspace cosine | 0.474 | partial, not complete, shared spatial structure |
| min/max top-16 cosine | 0.011 / 0.922 | some directions nearly orthogonal, one strongly aligned |
| standardized property rank-1 energy | 0.620 | one leading shared property direction, but rank one discards 38% |

This spectrum predicts that rigid rank-one or fully shared coefficients should
be too restrictive, while shared spatial factors plus property-specific heads
can be useful. Exploratory Spearman associations between fold coupling and 4D
gain include large coefficients, but their signs vary across metrics/budgets;
none survives Benjamini--Hochberg control over the 288 exploratory tests.
H4 is therefore not established as a monotone fold-level relationship.

## 6. 3D vs 4D experimental protocol

Two capacity controls were evaluated with identical folds, fields, time steps,
measurement rows and one common train-selected solver per matched pair:

- equal total: independent 3D uses 16+16 coefficients and 4D uses 32;
- equal per property: independent 3D uses 32+32 and 4D uses 64.

Configured existing-well prefixes use 5, 10, 15, 20 and 30 wells, each with
pressure and saturation. Grid sensing uses 30, 100 and 300 common scalar
channels selected by a train-only rank-64 POD-E QR/DG order, so neither model
receives placement tailored to itself. Full-observation projection isolates
representation. The common solver is selected from least squares and four
relative-ridge levels by the mean inner loss of the paired 3D and 4D methods.
LASSO at `lambda=0.001` is a frozen-final sensitivity, not an additional model
selection pass.

The first 5,000-iteration LASSO diagnostic reached its ceiling in 140 of 320
fits. Keeping lambda and tolerance fixed, the final sensitivity raised only the
ceiling to 20,000 iterations; 50 fits still reached it. Among the resulting 64
paired regime/metric comparisons, 49 satisfy the directional 7/10-fold rule,
but convergence-limited fits prevent treating this as confirmation on the same
level as the LS/ridge headline. The complete iterations and metrics are retained
in `lasso_sensitivity.csv` rather than silently dropped.

The bounded fourth-order variants are: 4D-A rigid joint coefficients; 4D-B
jointly learned shared spatial factors with separate property coefficients;
4D-C weighted joint fitting; 4D-D partially shared spatial factors; and 4D-E a
joint basis plus property-specific tensor residual modes. Raw/min--max,
mean-centred, ensemble-anomaly and standardized-anomaly transforms, ranks
`(64,48)` and `(80,48)`, property ranks one/two where applicable, and the two
capacity controls were selected by inner validation.

## 7. Matched-capacity results

The original/current 4D formulation is not rescued by corrected metrics: it
meets the empirical-advantage rule in only 3 of 72 headline comparisons and has
a clear median disadvantage in 68. In particular, its saturation anomaly error
is about 4.7 under full observation, compared with 0.34 for current independent
3D. E8's negative finding therefore remains valid for the rigid formulation.

Optimization changes the answer. Under equal-total full observation, optimized
4D-B improves pressure anomaly error from `0.0456 +/- 0.0230` to
`0.0392 +/- 0.0204` and pressure anomaly SSIM from `0.9937 +/- 0.0032` to
`0.9953 +/- 0.0025` (9/10 folds for both). Saturation error is slightly worse
(`0.3407` versus `0.3312`) and saturation SSIM is indistinguishable by the
decision rule. Equal-per-property capacity gives the same qualitative result.
Fourth order therefore improves pressure representation modestly, not the
full-state approximation of both variables.

The full-observation equal-total 4D-B stores 136,304 floats versus 149,728 for
independent 3D; equal-per-property stores 259,184 versus 272,608. The saving
comes from sharing spatial factors. Property rank two is the full two-property
space and is not itself compression.

## 8. Sensor-budget results

Equal-total outer means (`mean +/- fold SD`) are representative:

| Geometry/budget | Method | E'p | E'So | SSIM'p | SSIM'So | RMSEp | RMSESo |
|---|---|---:|---:|---:|---:|---:|---:|
| wells 5 | independent 3D | 0.498+/-0.265 | 2.071+/-0.378 | 0.818+/-0.145 | 0.762+/-0.011 | 0.486 | 0.00184 |
| wells 5 | optimized 4D | 0.421+/-0.287 | 0.950+/-0.174 | 0.922+/-0.039 | 0.866+/-0.020 | 0.411 | 0.00086 |
| wells 15 | independent 3D | 0.212+/-0.120 | 1.059+/-0.242 | 0.939+/-0.035 | 0.850+/-0.029 | 0.210 | 0.00096 |
| wells 15 | optimized 4D | 0.207+/-0.092 | 0.890+/-0.078 | 0.968+/-0.016 | 0.868+/-0.023 | 0.216 | 0.00083 |
| wells 30 | independent 3D | 0.191+/-0.120 | 0.733+/-0.154 | 0.920+/-0.062 | 0.882+/-0.032 | 0.189 | 0.00067 |
| wells 30 | optimized 4D | 0.121+/-0.092 | 0.722+/-0.274 | 0.965+/-0.035 | 0.894+/-0.043 | 0.121 | 0.00067 |
| grid 30 | independent 3D | 1.167+/-0.619 | 0.844+/-0.287 | 0.406+/-0.392 | 0.856+/-0.034 | 1.166 | 0.00078 |
| grid 30 | optimized 4D | 0.350+/-0.162 | 0.465+/-0.135 | 0.770+/-0.254 | 0.932+/-0.019 | 0.372 | 0.00041 |
| grid 300 | independent 3D | 0.148+/-0.071 | 0.375+/-0.114 | 0.941+/-0.036 | 0.927+/-0.027 | 0.147 | 0.00033 |
| grid 300 | optimized 4D | 0.096+/-0.039 | 0.348+/-0.089 | 0.967+/-0.021 | 0.948+/-0.018 | 0.102 | 0.00031 |

The clearest replicated advantage is severe undersampling: at 5 wells and 30
grid channels, 4D passes the decision rule for both properties and both
headline metrics in both capacity regimes. At 300 grid channels equal-total 4D
again passes for all four endpoints. Intermediate and high-well budgets are
metric/property dependent: pressure structural similarity is particularly
consistent, whereas saturation reconstruction error does not always pass.
Thus H3 is supported as regime dependence, not as a monotone statement that
coupling helps only when sensors are few.

## 9. Structural reconstruction results

RMSE, reconstruction error and anomaly SSIM select the same optimized method in
32 of 36 capacity/geometry/budget/property regimes. The four discordant regimes
are informative rather than suppressed:

- equal-total, 15 wells, pressure: 3D has lower mean RMSE, while 4D has lower
  anomaly-relative error and higher anomaly SSIM; only SSIM meets 7/10 wins;
- equal-total full observation, saturation: 3D wins RMSE/error, while mean SSIM
  is microscopically higher for 4D without a fold-consistent advantage;
- equal-per-property, grid 300, saturation: 3D wins RMSE/error while 4D wins
  SSIM in 9/10 folds;
- equal-per-property, 10 wells, saturation: mean SSIM favours 4D, but only 6/10
  folds do so.

The fixed visualization (scenario 1, report step 132, 15 wells, chosen before
outer evaluation) also shows the intended H5 pattern: for pressure, 4D-B has
higher snapshot RMSE/anomaly error (`0.286/0.462` versus `0.142/0.229`) but
higher anomaly SSIM (`0.950` versus `0.937`); for saturation it improves both
error (`0.691` versus `1.811`) and SSIM (`0.842` versus `0.784`). Figure 19
shows reference, both reconstructions and matched error maps.

## 10. Ablation

The main ablation result is architectural. The best train-only choice is 4D-B
in all 20 full-observation folds and 122 of 160 sparse fold/budget selections.
It uses shared spatial factors, independent property coefficients, ranks
`(80,48)`, and usually ensemble-anomaly centring. Removing the independent
heads (4D-A/C) reintroduces the pressure--saturation compromise; removing shared
factors gives independent 3D and its severe sparse-conditioning failures;
removing anomaly treatment returns toward the poor current formulation.

Property rank one is never selected for 4D-A. Property rank two preserves the
full property space, so the useful fourth-order operation is estimating shared
spatial factors from the property tensor, not compressing two properties into
one scalar mode. Weighted 4D-C and residual 4D-E can help at individual budgets
but are not stable global winners. At 5 wells, partial/residual candidates are
selected more often because they regularise the most undersampled cases.

## 11. Statistical comparison

Every comparison is paired by held-out scenario, measurement rows, report
steps and solver. Canonical files report fold values, mean, median, sample SD,
IQR, signed fold differences and two-sided Wilcoxon diagnostics. The decision
uses direction and 7/10 wins, not averaged score or p-value alone. Optimized 4D
meets that rule in 56 of 72 headline regime/metric comparisons; current 4D does
so in only 3. P-values are descriptive because folds share geology and training
data, and the many regime analyses are not independent population tests.

## 12. Interpretation

The central scientific answer is conditional but positive. The fourth-order
axis is useful when it regularises spatial-factor estimation while retaining
property-specific coefficients. It is not useful when pressure and saturation
are forced to share the same coefficients. The sparse gains are much larger
than the full-state gains, so observability and conditioning are part of the
mechanism. At 30 grid channels, for example, both methods are underdetermined,
but the shared-factor 4D family reduces both errors and improves both SSIMs;
at 300 channels it retains smaller but fold-consistent benefits.

This is not evidence that an additional tensor dimension is causally superior
in general. It is matched evidence that a specific partially coupled
space--space--property--time representation outperforms independent spatial
factor learning on this ensemble and these measurement geometries. Saturation
full-state approximation and some intermediate budgets remain better under 3D.

## 13. Novelty implications

The old claim, "joint 4D improves accuracy," remains unsupported. A defensible
central claim is:

> On the Brugge control ensemble, retaining an explicit property mode improves
> sparse reconstruction when it is used to learn shared spatial factors while
> pressure and saturation retain separate coefficients. Under nested,
> matched-capacity evaluation, this partially coupled fourth-order model
> improves anomaly-relative reconstruction and mask-aware structural fidelity
> most clearly under severe undersampling; rigid joint coefficients remain
> inferior and full-state saturation accuracy does not improve consistently.

This formulation makes clear what is gained by the property mode (shared
spatial-factor estimation, storage reuse and sparse regularisation), what is
not gained (rank-one property compression or universally better approximation),
and where the evidence applies.

## 14. Manuscript changes

E8 is retained as provenance for the negative rigid-joint result and E9 now
supersedes it as the dimensionality experiment. The manuscript metrics section
defines anomaly-relative reconstruction error and mask-aware anomaly SSIM,
keeps RMSE as a physical-unit diagnostic and removes PSNR. E9 methods,
matched-capacity controls, train-only selection, current-versus-optimized
results, budget curves, cross-property diagnostics, architecture ablation,
fixed reconstruction maps and limitations are incorporated in the main text
and Supplement. Figures 15--19 and `tab_e9_headline.tex` are generated from
canonical E9 outputs. Numerical prose uses generated E9 macros and is checked
against the same output directory.

The final independent `--reproduce` run refitted all ten held-out folds into a
separate output directory. All 2,080 outer records matched the canonical
identity fields and every non-runtime numeric field exactly (maximum absolute
difference `0.0`, acceptance tolerance `1e-10`). The manuscript claim checker
then passed all E9 and non-timing macro checks. The machine-readable receipt is
`outputs/e9_property_mode/reproduction_validation.json`.

Remaining limitations are the ten related fixed-geology scenarios, idealised
cell measurements, no measurement noise in the E9 headline, post-development
rather than external preregistration, and insufficient evidence to identify a
causal link between fold-level correlation strength and 4D gain.
