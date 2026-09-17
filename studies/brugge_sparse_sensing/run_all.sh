#!/usr/bin/env bash
# Full reproduction of every computed number, table and figure in the revised manuscript.
# Usage:  TBMD_DATA_DIR=/path/to/data ./run_all.sh [JOBS]      (data layout: $TBMD_DATA_DIR/brugge/<files>)
# Optional: PYTHON=/path/to/python (default: ../../.venv/bin/python if present, else python3)
# Optional: BSS_OUT / BSS_FIG to write into other directories; MANUSCRIPT_DIR to copy
#           numbers.tex, LaTeX tables and figures into the manuscript sources.
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PYTHON:-}" ]; then
  if [ -x ../../.venv/bin/python ]; then PYTHON=../../.venv/bin/python; else PYTHON=python3; fi
fi
PY=$PYTHON
JOBS=${1:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bss-mpl}
OUTDIR=${BSS_OUT:-outputs}; FIGDIR=${BSS_FIG:-figures}; LOG=${BSS_LOG:-run_logs}
mkdir -p "$OUTDIR/parts" "$FIGDIR" "$LOG"
rm -f "$OUTDIR"/parts/*.pkl
t0=$(date +%s)
$PY scripts/verify_inputs.py                     > "$LOG/inputs.log" 2>&1 || { cat "$LOG/inputs.log"; exit 1; }
$PY scripts/s0_data_manifest.py                  > "$LOG/s0.log" 2>&1
$PY scripts/e0_reproduce_archived.py --property pressure --regime wells > "$LOG/e0_wells.log" 2>&1
$PY scripts/e0_reproduce_archived.py --property pressure --regime grid  > "$LOG/e0_grid.log" 2>&1
$PY scripts/e0_reproduce_archived.py --property soil --regime wells     > "$LOG/e0_soil.log" 2>&1
$PY scripts/e0_reproduce_archived.py --property all --regime wells      > "$LOG/e0_joint.log" 2>&1 || echo "E0 joint wells failed as documented (shape mismatch in public helper)" >> "$LOG/e0_joint.log"
$PY scripts/e0_cluster_table.py                  > "$LOG/e0_cluster.log" 2>&1
$PY scripts/v1_verify_against_library.py         > "$LOG/v1.log" 2>&1
for tag in e1_representation e2_budget_sweep; do
  for p in P1 P2; do for k in $(seq 0 9); do echo "$p $k"; done; done \
    | xargs -P "$JOBS" -n 2 sh -c "$PY scripts/$tag.py part \$0 \$1" > "$LOG/$tag.log" 2>&1
  $PY scripts/$tag.py merge >> "$LOG/$tag.log" 2>&1
done
for tag in e3_noise e4_l1_sensitivity; do
  seq 0 9 | xargs -P "$JOBS" -n 1 sh -c "$PY scripts/$tag.py part \$0" > "$LOG/$tag.log" 2>&1
  $PY scripts/$tag.py merge >> "$LOG/$tag.log" 2>&1
done
$PY scripts/v2_admm_convergence.py                > "$LOG/v2.log" 2>&1
$PY scripts/e5_stability.py                      > "$LOG/e5.log" 2>&1
$PY scripts/e7_spatial_structure.py              > "$LOG/e7.log" 2>&1
$PY scripts/e6_cost.py                           > "$LOG/e6.log" 2>&1   # run last, on an otherwise idle machine
$PY scripts/make_tables.py                       > "$LOG/tables.log" 2>&1
$PY scripts/make_numbers_tex.py                  > "$LOG/numbers.log" 2>&1
$PY scripts/make_latex_tables.py                 >> "$LOG/numbers.log" 2>&1
$PY scripts/make_figures.py                      > "$LOG/figures.log" 2>&1
$PY scripts/make_graphical_abstract.py           >> "$LOG/figures.log" 2>&1
if [ -n "${MANUSCRIPT_DIR:-}" ]; then
  cp "$OUTDIR/numbers.tex" "$MANUSCRIPT_DIR/numbers.tex"
  cp "$OUTDIR/tab_main.tex" "$OUTDIR/tab_cost.tex" "$MANUSCRIPT_DIR/sections/"
  cp "$OUTDIR/tabS_P1.tex" "$OUTDIR/tabS_So.tex" "$MANUSCRIPT_DIR/"
  cp "$FIGDIR"/*.pdf "$MANUSCRIPT_DIR/figures/"
  $PY scripts/check_claims.py "$MANUSCRIPT_DIR"  > "$LOG/claims.log" 2>&1 || echo "check_claims failed: see $LOG/claims.log"
fi
echo "run_all finished in $(( $(date +%s) - t0 )) s" | tee -a "$LOG/run_all.log"
