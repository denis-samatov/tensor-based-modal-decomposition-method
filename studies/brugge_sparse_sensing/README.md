# Brugge sparse-sensing study

Study package for the manuscript *"Nested optimization of tensor-based modal decomposition for sparse reservoir-state reconstruction: accuracy–compression regimes on Brugge"*
(D. Samatov, B. Merzlikin, G. Shishaev; target journal: *Applied Computing and Geosciences*).
Numerical results, tables and figures of the manuscript and its supplement are computed by the code
in this directory; results enter LaTeX through `outputs/numbers.tex` and generated table files.
The [canonical manuscript and build](../../docs/paper/README.md) are in this checkout and reference
these outputs directly. Historical notebooks are preserved as [scientific provenance](../../docs/paper/audit/README.md).

The study extends the original third-order space-space-time TBMD chain to a fourth-order
space-space-property-time tensor. Pressure and oil saturation form the property mode; the resulting
spatial-property modes feed mode-4 fibre QR placement and joint or single-property reconstruction.
`scripts/v1_verify_against_library.py` checks QR/reconstruction parity, while E1 and E2 test the
full-rank identity, spatial truncation and property-channel ablations.

V1 compares selected operations with **this authors' library**, not Zhong et al.'s source code.
V3 checks explicit 3D/4D contractions, row ordering and all training-fold scaler bounds. E8 fits
independent third-order property models against joint fourth-order TBMD with identical observed
channels and two coefficient-capacity controls. All 240 fold/design records and the generated
supplementary table are in `outputs/e8_property_coupling*` and `outputs/tabS_coupling.tex`.
The property rank is full (2); its orthogonal factor can be absorbed into the core. Shared spatial
factors and coefficients, not a new QR or decomposition primitive, define the joint constraint.

Workflow: **environment → data → `run_all.sh` → verification → tables/figures.** If you do not have
the simulation data, section 4.1 verifies all tables, statistics and number macros from the
committed result files.

## Contents

| Path | Content |
|---|---|
| `config.json` | All parameters: split, energy threshold, TBMD ranks, Tucker/ADMM settings, budgets, seeds, noise levels, sensitivity grids |
| `inputs.sha256` | SHA-256 of the two input files |
| `bss/` | Study library: data loading and normalisation (`data.py`), bases (`bases.py`), placement (`placement.py`), estimators (`estimators.py`), metrics (`metrics.py`), experiment loop (`experiment.py`) |
| `scripts/` | One script per stage (table below), verification scripts |
| `run_all.sh` | Full pipeline from the input files (one command) |
| `verify_outputs.sh` | Data-free verification from the committed result files |
| `outputs/` | Result files, statistics, LaTeX tables and `numbers.tex` used in the manuscript |
| `figures/` | Canonical manuscript/graphical-abstract PDFs; regenerated PNG previews are ignored |
| `SHA256SUMS` | SHA-256 of every canonical file in `outputs/` and publication PDF in `figures/` |
| `run_logs/` | Logs of the complete independent re-run of 2026-09-17 (absolute paths replaced by `<repo>`/`<venv>`) |
| `requirements-lock.txt` | Exact package versions of the environment used |

| Stage | Script | Output | Manuscript item |
|---|---|---|---|
| Input checksums | `scripts/verify_inputs.py` | `run_logs/inputs.log` | Supp. Table S1 |
| S0 data manifest | `scripts/s0_data_manifest.py` | `outputs/s0_data_manifest.json` | Sec. 3.1, Fig. 2, Supp. Table S1 |
| E0 legacy benchmark lineage | `scripts/e0_reproduce_archived.py`, `scripts/e0_cluster_table.py` | `outputs/e0_*` | Supp. Sec. S2 |
| V1/V2 implementation checks | `scripts/v1_verify_against_library.py`, `scripts/v2_admm_convergence.py` | `outputs/v1_verification.json`, `outputs/v2_admm_convergence.json` | Sec. 4.5, Proposition 1 |
| E1 representation | `scripts/e1_representation.py` | `outputs/e1_representation.csv` | Fig. 3 |
| E2 main benchmark | `scripts/e2_budget_sweep.py` | `outputs/e2_summary.csv`, `outputs/e2_snapshots.csv.gz` | Figs. 4–5, Table 1, Supp. Tables S3–S4 |
| E3 noise | `scripts/e3_noise.py` | `outputs/e3_noise.csv` | Fig. 6 |
| E4 ℓ1 weight | `scripts/e4_l1_sensitivity.py` | `outputs/e4_l1_sensitivity.csv` | Supp. Fig. S1 |
| E5 stability | `scripts/e5_stability.py` | `outputs/e5_*` | Fig. 7 |
| E6 cost | `scripts/e6_cost.py` | `outputs/e6_cost.json` | Table 2 |
| E7 spatial structure | `scripts/e7_spatial_structure.py` | `outputs/e7_spatial_structure.csv` | Sec. 6.1 |
| E8 property coupling | `scripts/e8_property_coupling.py` | `outputs/e8_property_coupling*.csv`, `outputs/tabS_coupling.tex` | Supp. Table S5 |
| V3 shape/operator checks | `scripts/v3_shape_contract.py` | `outputs/v3_shape_contract.json` | Supp. Sec. S3 |
| Tables and statistics | `scripts/make_tables.py`, `scripts/make_latex_tables.py` | `outputs/table_*.csv`, `outputs/stats_wilcoxon.csv`, `outputs/key_numbers.json`, `outputs/tab*.tex` | Tables 1–2, S3–S4 |
| Number macros | `scripts/make_numbers_tex.py` | `outputs/numbers.tex` | all numbers in the text |
| Figures | `scripts/make_figures.py`, `scripts/make_graphical_abstract.py` | `figures/` | Figs. 2–7, S1–S2, graphical abstract |
| Manuscript consistency | `scripts/check_claims.py` | `run_logs/claims.log` | — |

Figure 1 is a TikZ diagram in the manuscript source.

## 1. Environment

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r studies/brugge_sparse_sensing/requirements-lock.txt
python -m pip install --no-deps -e .
```

The lock file is the complete environment used for the manuscript (Python 3.12, macOS arm64); it
also contains development tools that the study does not need. The TBMD library code used by the
study (`src/`) is that of this repository revision. CPU only; no GPU. `run_all.sh` uses
`../../.venv/bin/python` when it is executable, otherwise `python3`; override with `PYTHON=...`.
Both study entry points fail before computation if Python is not 3.12 or if a result-sensitive
package differs from `requirements-lock.txt`. This prevents a broken virtual-environment symlink
from silently changing the Wilcoxon statistics or generated number macros.

## 2. Data

The study reads two files from `$TBMD_DATA_DIR/brugge/` (default `TBMD_DATA_DIR`: the repository's
`data/` directory, which is not tracked by git):

| File | Content | Bytes | SHA-256 |
|---|---|---|---|
| `data_exp_4_.h5` | `pressure` (supplied export unit $u_p$; physical unit unconfirmed) and `soil` (oil saturation), shape (10, 139, 48, 133); scenario `names` | 141,982,268 | `d55ae5e5f6a45167b67251ab21af6b864334a97c68051754f50dc7566813227e` |
| `all_wells_exp_4.json` | grid coordinates of the 30 wells for each scenario | 3,151 | `e3b8c99688efcbff4d41ba36b9714f6a32cf0d23df3933465b20387af8e425e7` |

**Provenance.** The authors confirm that the supplied states were simulated with t-Navigator for ten
well-control scenarios on the Brugge benchmark reservoir model (Peters et al., 2010,
https://doi.org/10.2118/119094-PA). A separate experiment-description file states that the 20
producers operate from month 1 to 15 at a random constant bottom-hole pressure in [60, 120] bar
and the 10 injectors start in month 15 at 175 bar. Those control units do not identify the unit of
the stored pressure array (Supplementary Section S1).

**Rights and access.** The Brugge data set is © TNO (The Hague, the Netherlands), whose public
notice reserves all rights, permits non-commercial use and asks users to acknowledge TNO
(https://github.com/TNO/Brugge; the full data set is requested via www.isapp2.com). Unrestricted
public redistribution of these transformed simulation exports has not been established, so the two
state files are **not** included in this repository. The transformed HDF5/JSON inputs are available
from the corresponding author upon reasonable request, subject to applicable data-use and access
conditions. Once you have authorised copies:

```bash
mkdir -p /path/to/data/brugge && cp data_exp_4_.h5 all_wells_exp_4.json /path/to/data/brugge/
TBMD_DATA_DIR=/path/to/data python studies/brugge_sparse_sensing/scripts/verify_inputs.py
```

`run_all.sh` stops if either checksum does not match.

**Boundary:** the simulator is t-Navigator, but its version, the simulator-to-HDF5 exporter, the
original deck and the nine-layer-to-areal reduction rule are not available here. Historical
t-Navigator CSV helpers operate on already two-dimensional arrays and do not establish the HDF5
creation process. The HDF5 file contains no unit attributes, and surviving materials conflict
between bar and psi labels. Values are not converted and are reported as the supplied export unit
$u_p$; no physical pressure unit is claimed.
Obtaining TNO's original model does not recreate these particular exports. Without authorised
copies of the exact two inputs, only stored-output verification is currently possible. Do not
redistribute raw files, derived state arrays or trained bases under the code's MIT licence.

## 3. Run everything

```bash
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data ./run_all.sh 8                  # 8 parallel jobs; overwrites canonical outputs/ and figures/; logs default to ignored rerun_verification/run_logs/
BSS_OUT=/tmp/bss/outputs BSS_FIG=/tmp/bss/figures BSS_LOG=/tmp/bss/logs \
  TBMD_DATA_DIR=/path/to/data ./run_all.sh 8                # or write into a separate directory
```

Hardware and runtime of the recorded runs: Apple MacBook Pro, M2 Pro (12 cores), 16 GB memory,
macOS arm64. The complete pipeline took 1,498 s and 1,561 s with 10 parallel jobs (the largest
stage, E2, runs about 140 s per P1 fold and 660 s per P2 fold single-threaded). Peak memory is below
2 GB per job. Parallelism is shell-level (`xargs -P`); `./run_all.sh 1` runs serially. Stage E6
measures wall-clock time and should run on an otherwise idle machine.

### 3.1 Nested TBMD optimization study

The final optimization pipeline is separate from the legacy E0--E8 chain:

```bash
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data ./run_tbmd_optimization.sh 3
```

It validates the pinned environment and inputs, runs ten outer folds with three train-only inner
scenario folds, merges the representation/sparse registries, executes the prespecified placement
audit, generates Tables/Figs. 8--14 and their manuscript macros, and independently refits the frozen
accuracy-TBMD, compression-TBMD and POD-E configurations. Canonical results are under
`outputs/optimization/`; `reproduction/validation.json` is the final deterministic comparison.
Runtime columns are recorded but are not required to reproduce bit-for-bit.

### 3.2 E9 property-mode study

E9 tests independent third-order property tensors against five fourth-order coupling architectures
with matched capacities, measurements and pairwise common train-selected solvers. The default command
independently refits the frozen selections; pass `--full` to repeat nested selection and the one-pass
outer evaluation:

```bash
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data ./run_e9_property_mode.sh --reproduce
TBMD_DATA_DIR=/path/to/data E9_JOBS=3 ./run_e9_property_mode.sh --full
```

Canonical metrics, fold selections, paired statistics, cross-property diagnostics, generated table
and manuscript macros are under `outputs/e9_property_mode/`; Figs. 15--19 are in `figures/`.
`E9_PREREGISTRATION.md` and `e9_protocol.json` record the pre-outer endpoints and decision rule.
The main comparison uses anomaly-relative Frobenius error and mask-aware anomaly SSIM; physical-unit
RMSE, raw relative error and raw SSIM are retained as diagnostics. The frozen-final LASSO run is a
sensitivity analysis only and never changes the selected headline model.

## 4. Verification

### 4.1 Without the simulation data

```bash
./verify_outputs.sh [manuscript_dir]  # canonical manuscript is checked by default
```

1. checks `outputs/` and `figures/` against `SHA256SUMS`;
2. verifies Python 3.12 and the pinned result-sensitive numerical packages;
3. rebuilds all tables, Wilcoxon statistics, `key_numbers.json`, `numbers.tex` and the LaTeX tables
   from the committed result files in a temporary directory and requires them to be byte-identical;
4. regenerates the graphical abstract, and all figures if the input files are available (the figure
   script also plots simulated fields);
5. runs `scripts/check_claims.py` on it.

### 4.2 After a full re-run

```bash
BSS_OUT=/tmp/bss/outputs BSS_FIG=/tmp/bss/figures BSS_LOG=/tmp/bss/logs TBMD_DATA_DIR=/path/to/data ./run_all.sh 8
python scripts/compare_rerun.py /tmp/bss/outputs
python scripts/check_claims.py /path/to/manuscript            # optional: manuscript LaTeX directory
```

`compare_rerun.py` compares all number macros and the result files of E1–E5, E7 and the statistics
and exits with an error if any value other than a wall-clock timing differs. Expected result (the
recorded independent re-run of 2026-09-17): 873 macros, 868 identical, 5 timing macros differing (Table 2; timing variability is separate from scientific results), all E1–E5/E7 and Wilcoxon entries identical. The E0 files re-execute the
original library, which is not deterministic (Supplementary Section S2); they are reported for
information only.

Scratch verification outputs are removed on success or failure. This command never overwrites the canonical result files.

`check_claims.py` fails if a generated number source referenced by the manuscript differs from the
canonical output, if a manuscript macro is not generated, or if a result-like decimal is typed
directly into the text. The nested studies add `outputs/optimization/optimization_numbers.tex` and
`outputs/e9_property_mode/e9_numbers.tex`.

### 4.3 Determinism

Random placements use `numpy.random.default_rng(master_seed + s)`, `s = 0,…,19`, with
`master_seed = 20260917` (`config.json`); noise draws use `master_seed + 1000 + fold`. SVD, Tucker
(HOOI with SVD initialisation) and QR are deterministic, and BLAS threads are pinned to one.
Results were bit-identical between two runs on the same machine; other BLAS builds or CPUs can
change trailing digits.

## 5. Licence and citation

Code in this directory: MIT (repository `LICENSE`). The result files in `outputs/` and `figures/`
are statistics and figures derived from simulations of the TNO-owned Brugge model; when using them,
acknowledge TNO as requested in its data-use agreement. The Brugge data themselves are not covered by
the MIT licence. Release v2.1.0 (https://doi.org/10.5281/zenodo.22814377) predates E8/V3 and must
not be cited as the archive of this manuscript revision. The proposed reviewed tag is
`v2.2.0-rc1`; it becomes citable only after author approval and archival.

Some stored result columns retain `_bar` in their names for compatibility with the original output
schema. This is a legacy identifier, not unit evidence; manuscript-facing tables, figures and prose
use $u_p$ until the exporter metadata are confirmed.
