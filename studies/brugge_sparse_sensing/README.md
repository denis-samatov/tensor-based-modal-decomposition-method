# Brugge sparse-sensing study

Study package for the manuscript *"Sparse-sensor reconstruction of reservoir states with
tensor-based and matrix modal bases: a reproducible benchmark on the Brugge model"*
(D. Samatov, B. Merzlikin, G. Shishaev; submitted to *Applied Computing and Geosciences*).
Every number, table and figure of the manuscript and its supplement is computed by the code in this
directory; numbers enter the LaTeX source only through the generated macro file `outputs/numbers.tex`.

Workflow: **environment → data → `run_all.sh` → verification → tables/figures.** If you do not have
the simulation data, section 4.1 verifies all tables, statistics, macros and figures from the
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
| `figures/` | Figures (PDF used in the manuscript; PNG previews) and graphical abstract |
| `SHA256SUMS` | SHA-256 of every file in `outputs/` and `figures/` |
| `run_logs/` | Logs of the complete independent re-run of 2026-09-17 (absolute paths replaced by `<repo>`/`<venv>`) |
| `requirements-lock.txt` | Exact package versions of the environment used |

| Stage | Script | Output | Manuscript item |
|---|---|---|---|
| Input checksums | `scripts/verify_inputs.py` | `run_logs/inputs.log` | Supp. Table S1 |
| S0 data manifest | `scripts/s0_data_manifest.py` | `outputs/s0_data_manifest.json` | Sec. 3.1, Fig. 2, Supp. Table S1 |
| E0 audit of the reviewed version | `scripts/e0_reproduce_archived.py`, `scripts/e0_cluster_table.py` | `outputs/e0_*` | Supp. Sec. S2 |
| V1/V2 implementation checks | `scripts/v1_verify_against_library.py`, `scripts/v2_admm_convergence.py` | `outputs/v1_verification.json`, `outputs/v2_admm_convergence.json` | Sec. 4.5, Proposition 1 |
| E1 representation | `scripts/e1_representation.py` | `outputs/e1_representation.csv` | Fig. 3 |
| E2 main benchmark | `scripts/e2_budget_sweep.py` | `outputs/e2_summary.csv`, `outputs/e2_snapshots.csv.gz` | Figs. 4–5, Table 1, Supp. Tables S3–S4 |
| E3 noise | `scripts/e3_noise.py` | `outputs/e3_noise.csv` | Fig. 6 |
| E4 ℓ1 weight | `scripts/e4_l1_sensitivity.py` | `outputs/e4_l1_sensitivity.csv` | Supp. Fig. S1 |
| E5 stability | `scripts/e5_stability.py` | `outputs/e5_*` | Fig. 7 |
| E6 cost | `scripts/e6_cost.py` | `outputs/e6_cost.json` | Table 2 |
| E7 spatial structure | `scripts/e7_spatial_structure.py` | `outputs/e7_spatial_structure.csv` | Sec. 6.1 |
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

The lock file is the complete environment used for the manuscript (Python 3.12.12, macOS arm64); it
also contains development tools that the study does not need. The TBMD library code used by the
study (`src/`) is that of this repository revision. CPU only; no GPU. `run_all.sh` uses
`../../.venv/bin/python` when it exists, otherwise `python3`; override with `PYTHON=...`.

## 2. Data

The study reads two files from `$TBMD_DATA_DIR/brugge/` (default `TBMD_DATA_DIR`: the repository's
`data/` directory, which is not tracked by git):

| File | Content | Bytes | SHA-256 |
|---|---|---|---|
| `data_exp_4_.h5` | `pressure` (bar) and `soil` (oil saturation), shape (10, 139, 48, 133); scenario `names` | 141,982,268 | `d55ae5e5f6a45167b67251ab21af6b864334a97c68051754f50dc7566813227e` |
| `all_wells_exp_4.json` | grid coordinates of the 30 wells for each scenario | 3,151 | `e3b8c99688efcbff4d41ba36b9714f6a32cf0d23df3933465b20387af8e425e7` |

**Provenance.** The authors generated the files by simulating ten well-control scenarios on the
Brugge benchmark reservoir model (Peters et al., 2010, https://doi.org/10.2118/119094-PA). In all
scenarios the 20 producers operate from month 1 to 15 at a random constant bottom-hole pressure in
[60, 120] bar and the 10 injectors start in month 15 at 175 bar (Supplementary Section S1).

**Rights and access.** The Brugge data set is © TNO (The Hague, the Netherlands), which distributes
it under a data-use agreement that reserves all rights, permits non-commercial use only and asks
users to acknowledge TNO (https://github.com/TNO/Brugge; the full data set is requested via
www.isapp2.com). The agreement grants no right to redistribute the model or data derived from it,
so the two simulated-state files are **not** included in this repository. The manuscript's data
availability statement gives the access route. Once you have the files:

```bash
mkdir -p /path/to/data/brugge && cp data_exp_4_.h5 all_wells_exp_4.json /path/to/data/brugge/
TBMD_DATA_DIR=/path/to/data python studies/brugge_sparse_sensing/scripts/verify_inputs.py
```

`run_all.sh` stops if either checksum does not match.

## 3. Run everything

```bash
cd studies/brugge_sparse_sensing
TBMD_DATA_DIR=/path/to/data ./run_all.sh 8                  # 8 parallel jobs; overwrites outputs/, figures/, run_logs/
BSS_OUT=/tmp/bss/outputs BSS_FIG=/tmp/bss/figures BSS_LOG=/tmp/bss/logs \
  TBMD_DATA_DIR=/path/to/data ./run_all.sh 8                # or write into a separate directory
```

Hardware and runtime of the recorded runs: Apple MacBook Pro, M2 Pro (12 cores), 16 GB memory,
macOS arm64. The complete pipeline took 1,498 s and 1,561 s with 10 parallel jobs (the largest
stage, E2, runs about 140 s per P1 fold and 660 s per P2 fold single-threaded). Peak memory is below
2 GB per job. Parallelism is shell-level (`xargs -P`); `./run_all.sh 1` runs serially. Stage E6
measures wall-clock time and should run on an otherwise idle machine.

## 4. Verification

### 4.1 Without the simulation data

```bash
./verify_outputs.sh [manuscript_dir]
```

1. checks `outputs/` and `figures/` against `SHA256SUMS`;
2. rebuilds all tables, Wilcoxon statistics, `key_numbers.json`, `numbers.tex` and the LaTeX tables
   from the committed result files in a temporary directory and requires them to be byte-identical;
3. regenerates all figures into the temporary directory;
4. if the manuscript LaTeX directory is given, runs `scripts/check_claims.py` on it.

### 4.2 After a full re-run

```bash
BSS_OUT=/tmp/bss/outputs BSS_FIG=/tmp/bss/figures BSS_LOG=/tmp/bss/logs TBMD_DATA_DIR=/path/to/data ./run_all.sh 8
python scripts/compare_rerun.py /tmp/bss/outputs
python scripts/check_claims.py /path/to/manuscript            # optional: manuscript LaTeX directory
```

`compare_rerun.py` compares all number macros and the result files of E1–E5, E7 and the statistics
and exits with an error if any value other than a wall-clock timing differs. Expected result (the
recorded independent re-run of 2026-09-17): 873 macros, 868 identical, 5 timing macros differing by
at most 4 % (Table 2), all E1–E5/E7 and Wilcoxon entries identical. The E0 files re-execute the
original library, which is not deterministic (Supplementary Section S2); they are reported for
information only.

`check_claims.py` fails if the manuscript copy of `numbers.tex` differs from `outputs/numbers.tex`,
if a manuscript macro is not generated, or if a result-like decimal is typed directly into the text.

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
the MIT licence. Cite the article and the software release (repository `CITATION.cff`).
