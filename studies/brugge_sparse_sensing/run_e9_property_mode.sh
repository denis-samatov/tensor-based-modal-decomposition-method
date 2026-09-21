#!/usr/bin/env bash
# E9 nested selection, untouched outer evaluation, analysis, and frozen reproduction.
set -euo pipefail

cd "$(dirname "$0")/../.."
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python3
  fi
fi

MODE=${1:---reproduce}
JOBS=${E9_JOBS:-3}
STUDY=studies/brugge_sparse_sensing
DEST=$STUDY/outputs/e9_property_mode
REPRO_BASE=$STUDY/outputs/e9_reproduction
LOG_DIR=${E9_LOG_DIR:-$STUDY/rerun_verification/e9_logs}
mkdir -p "$LOG_DIR" "$DEST/parts" "$REPRO_BASE/e9_property_mode/parts"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/tbmd-e9-mpl}

"$PYTHON" "$STUDY/scripts/verify_environment.py" > "$LOG_DIR/environment.log" 2>&1
"$PYTHON" "$STUDY/scripts/verify_inputs.py" > "$LOG_DIR/inputs.log" 2>&1

if [ "$MODE" = "--full" ]; then
  "$PYTHON" "$STUDY/scripts/e9_cross_property.py" > "$LOG_DIR/coupling.log" 2>&1
  seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
    "$STUDY/scripts/e9_nested_property_mode.py" inner > "$LOG_DIR/inner.log" 2>&1
  seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
    "$STUDY/scripts/e9_nested_property_mode.py" reselect >> "$LOG_DIR/inner.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_nested_property_mode.py" merge-inner >> "$LOG_DIR/inner.log" 2>&1
  seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
    "$STUDY/scripts/e9_nested_property_mode.py" outer > "$LOG_DIR/outer.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_nested_property_mode.py" merge-outer >> "$LOG_DIR/outer.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_analyze.py" > "$LOG_DIR/analysis.log" 2>&1
  seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
    "$STUDY/scripts/e9_lasso_sensitivity.py" part > "$LOG_DIR/lasso.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_lasso_sensitivity.py" merge >> "$LOG_DIR/lasso.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_reconstruction_maps.py" > "$LOG_DIR/maps.log" 2>&1
  "$PYTHON" "$STUDY/scripts/e9_make_numbers.py" >> "$LOG_DIR/analysis.log" 2>&1
elif [ "$MODE" != "--reproduce" ]; then
  echo "usage: $0 [--full|--reproduce]" >&2
  exit 2
fi

cp "$DEST"/parts/selection_*.json "$REPRO_BASE/e9_property_mode/parts/"
seq 0 9 | xargs -P "$JOBS" -n 1 env BSS_OUT="$REPRO_BASE" "$PYTHON" \
  "$STUDY/scripts/e9_nested_property_mode.py" outer > "$LOG_DIR/reproduction.log" 2>&1
env BSS_OUT="$REPRO_BASE" "$PYTHON" "$STUDY/scripts/e9_nested_property_mode.py" \
  merge-outer >> "$LOG_DIR/reproduction.log" 2>&1
"$PYTHON" "$STUDY/scripts/e9_compare_reproduction.py" \
  "$REPRO_BASE/e9_property_mode/outer_results.csv" >> "$LOG_DIR/reproduction.log" 2>&1

if [ -f docs/paper/manuscript/main.tex ]; then
  "$PYTHON" "$STUDY/scripts/check_claims.py" docs/paper/manuscript > "$LOG_DIR/claims.log" 2>&1
fi
echo "E9 property-mode study reproduction completed."
