#!/usr/bin/env bash
# Complete leakage-controlled TBMD optimization, outer evaluation, analysis, and frozen reproduction.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON=$(command -v python3.12)
    if [ -d .venv/lib/python3.12/site-packages ]; then
      export PYTHONPATH="$(pwd)/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    fi
  else
    echo "Python 3.12 is required; set PYTHON to the pinned study interpreter." >&2
    exit 1
  fi
fi

JOBS=${1:-3}
STUDY=studies/brugge_sparse_sensing
LOG_DIR=${TBMD_OPT_LOG:-$STUDY/rerun_verification/optimization_logs}
mkdir -p "$LOG_DIR" "$STUDY/outputs/optimization/parts" \
  "$STUDY/outputs/optimization/reproduction/parts"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/tbmd-optimization-mpl}

"$PYTHON" "$STUDY/scripts/verify_environment.py" > "$LOG_DIR/environment.log" 2>&1
"$PYTHON" "$STUDY/scripts/verify_inputs.py" > "$LOG_DIR/inputs.log" 2>&1

seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
  "$STUDY/scripts/o1_nested_representation.py" part > "$LOG_DIR/stage_a.log" 2>&1
"$PYTHON" "$STUDY/scripts/o1_nested_representation.py" merge >> "$LOG_DIR/stage_a.log" 2>&1

seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
  "$STUDY/scripts/o2_nested_sparse.py" part > "$LOG_DIR/stages_b_e.log" 2>&1
"$PYTHON" "$STUDY/scripts/o2_nested_sparse.py" merge >> "$LOG_DIR/stages_b_e.log" 2>&1

seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
  "$STUDY/scripts/o2b_outer_placement_audit.py" part > "$LOG_DIR/placement_audit.log" 2>&1
"$PYTHON" "$STUDY/scripts/o2b_outer_placement_audit.py" merge >> "$LOG_DIR/placement_audit.log" 2>&1

"$PYTHON" "$STUDY/scripts/o3_analyze_optimization.py" > "$LOG_DIR/analysis.log" 2>&1

seq 0 9 | xargs -P "$JOBS" -n 1 "$PYTHON" \
  "$STUDY/scripts/o4_reproduce_frozen.py" part > "$LOG_DIR/reproduction.log" 2>&1
"$PYTHON" "$STUDY/scripts/o4_reproduce_frozen.py" merge >> "$LOG_DIR/reproduction.log" 2>&1

if [ -f docs/paper/manuscript/main.tex ]; then
  "$PYTHON" "$STUDY/scripts/check_claims.py" docs/paper/manuscript > "$LOG_DIR/claims.log" 2>&1
fi

echo "TBMD optimization study and independent frozen reproduction completed."
