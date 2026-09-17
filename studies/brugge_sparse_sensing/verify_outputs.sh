#!/usr/bin/env bash
# Data-free verification: rebuild all tables, statistics, number macros and figures from the
# committed result files in outputs/ and compare them with the committed versions.
# Steps 0, 1 and 3 do not need the simulation data; figures are regenerated only when the inputs
# are available. Usage: ./verify_outputs.sh [manuscript_dir]
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PYTHON:-}" ]; then
  if [ -x ../../.venv/bin/python ]; then PYTHON=../../.venv/bin/python; else PYTHON=python3; fi
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
TMP=$(mktemp -d "${TMPDIR:-/tmp}/bss-verify.XXXXXX")
export MPLCONFIGDIR="$TMP/mpl" BSS_OUT="$TMP/outputs" BSS_FIG="$TMP/figures"
mkdir -p "$BSS_OUT" "$BSS_FIG"
echo "0. checksums of committed result files and figures"
"$PYTHON" scripts/check_sha256.py SHA256SUMS
cp outputs/*.csv outputs/*.csv.gz outputs/*.json outputs/*.npz "$BSS_OUT"/
rm -f "$BSS_OUT"/table_*.csv "$BSS_OUT"/stats_wilcoxon.csv "$BSS_OUT"/key_numbers.json
echo "1. tables, statistics and number macros"
"$PYTHON" scripts/make_tables.py > "$TMP/tables.log"
"$PYTHON" scripts/make_numbers_tex.py > "$TMP/numbers.log"
"$PYTHON" scripts/make_latex_tables.py >> "$TMP/numbers.log"
status=0
for f in outputs/table_*.csv outputs/stats_wilcoxon.csv outputs/key_numbers.json outputs/numbers.tex outputs/tab_main.tex outputs/tab_cost.tex outputs/tabS_P1.tex outputs/tabS_So.tex; do
  if cmp -s "$f" "$BSS_OUT/$(basename "$f")"; then echo "  identical  $(basename "$f")"; else echo "  DIFFERENT  $(basename "$f")"; status=1; fi
done
echo "2. figures"
"$PYTHON" scripts/make_graphical_abstract.py > "$TMP/figures.log"
if "$PYTHON" scripts/verify_inputs.py > "$TMP/inputs.log" 2>&1; then
  "$PYTHON" scripts/make_figures.py >> "$TMP/figures.log"
  echo "  all figures regenerated into $BSS_FIG (compare with figures/)"
else
  echo "  graphical abstract regenerated into $BSS_FIG; Figs. 2-7 and S1-S2 also plot simulated fields"
  echo "  and need the input files (README section 2) - skipped"
fi
if [ -n "${1:-}" ]; then
  echo "3. manuscript numbers"
  "$PYTHON" scripts/check_claims.py "$1" || status=1
fi
[ $status -eq 0 ] && echo "VERIFY OUTPUTS: PASSED" || echo "VERIFY OUTPUTS: FAILED"
exit $status
