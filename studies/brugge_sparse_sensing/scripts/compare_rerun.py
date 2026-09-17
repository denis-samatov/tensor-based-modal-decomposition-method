"""Compare an independent re-run of run_all.sh (written with BSS_OUT=<dir>) with outputs/.

Scientific results must agree exactly; wall-clock timings (Cost* macros, timing columns) may differ.
Usage: python scripts/compare_rerun.py <rerun_outputs_dir>   (exit code 1 if a non-timing value differs)"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = Path(__file__).resolve().parents[1] / "outputs"
RR = Path(sys.argv[1])
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
m0 = dict(MACRO.findall((ORIG / "numbers.tex").read_text()))
m1 = dict(MACRO.findall((RR / "numbers.tex").read_text()))
names = sorted(set(m0) | set(m1))
diff = {k: (m0.get(k), m1.get(k)) for k in names if m0.get(k) != m1.get(k)}
timing = {k: v for k, v in diff.items() if k.startswith("Cost")}
other = {k: v for k, v in diff.items() if not k.startswith("Cost")}
print(f"macros compared: {len(names)}; identical: {len(names) - len(diff)}; "
      f"timing macros differing: {len(timing)}; non-timing differing: {len(other)}")
for k, (a, b) in timing.items():
    print(f"  timing {k}: outputs={a} rerun={b}")
for k, (a, b) in other.items():
    print(f"  NON-TIMING {k}: outputs={a} rerun={b}")
failed = bool(other)
# E0 re-executes the original, non-deterministic library; its files are reported but not required to match.
for fn, drop, required in (("e2_summary.csv", ["sec_per_snapshot"], True), ("e1_representation.csv", ["seconds"], True),
                           ("e3_noise.csv", [], True), ("e4_l1_sensitivity.csv", [], True), ("e5_stability.csv", [], True),
                           ("e7_spatial_structure.csv", [], True), ("stats_wilcoxon.csv", [], True),
                           ("e0_cluster_table.csv", [], False), ("e0_archived_pressure_wells.csv", [], False)):
    a, b = pd.read_csv(ORIG / fn), pd.read_csv(RR / fn)
    if a.shape != b.shape:
        print(f"{fn}: SHAPE DIFFERS {a.shape} vs {b.shape}")
        failed |= required
        continue
    num = [c for c in a.select_dtypes("number").columns if c not in drop]
    A, B = a[num].to_numpy(float), b[num].to_numpy(float)
    with np.errstate(invalid="ignore"):
        rel = np.abs(A - B) / np.maximum(np.abs(A), 1e-12)
    rel = np.where((np.isnan(A) & np.isnan(B)) | (np.isinf(A) & np.isinf(B) & (A == B)), 0.0, rel)
    n_bad = int((rel > 1e-9).sum())
    print(f"{fn}: rows={len(a)}, numeric cols={len(num)}, max rel diff={np.nanmax(rel):.2e}, "
          f"entries rel diff>1e-9: {n_bad}{'' if required else ' (informational: original library)'}")
    failed |= required and n_bad > 0
print("RERUN MATCHES (timings excluded)" if not failed else "RERUN DIFFERS")
sys.exit(1 if failed else 0)
