"""Compare an independent re-run of run_all.sh (written with BSS_OUT=<dir>) with outputs/.

Non-timing macros must agree exactly; CSVs use rtol=1e-9, atol=1e-12 and check labels/NaNs.
Wall-clock timings (Cost* macros, timing columns) may differ.
Usage: python scripts/compare_rerun.py <rerun_outputs_dir>   (exit code 1 if a non-timing value differs)"""
import re
import sys
from pathlib import Path

import pandas as pd

ORIG = Path(__file__).resolve().parents[1] / "outputs"
RR = Path(sys.argv[1])
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
m0 = dict(MACRO.findall((ORIG / "numbers.tex").read_text()))
m1 = dict(MACRO.findall((RR / "numbers.tex").read_text()))
names = sorted(set(m0) | set(m1))
diff = {k: (m0.get(k), m1.get(k)) for k in names if m0.get(k) != m1.get(k)}
timing = {k: v for k, v in diff.items() if k.startswith("Cost") and k in m0 and k in m1}
other = {k: v for k, v in diff.items() if k not in timing}
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
                           ("e2_snapshots.csv.gz", [], True), ("e5_well_ranks.csv", [], True),
                           ("e8_property_coupling.csv", [], True), ("e8_property_coupling_summary.csv", [], True),
                           ("e0_cluster_table.csv", [], False), ("e0_archived_pressure_wells.csv", [], False)):
    a, b = pd.read_csv(ORIG / fn), pd.read_csv(RR / fn)
    if a.shape != b.shape:
        print(f"{fn}: SHAPE DIFFERS {a.shape} vs {b.shape}")
        failed |= required
        continue
    try:
        pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop),
                                      check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
        print(f"{fn}: MATCH ({len(a)} rows; all non-timing fields, labels and NaNs)")
    except AssertionError as error:
        print(f"{fn}: {'FAIL' if required else 'INFORMATIONAL DIFFERENCE'}: {str(error)[:500]}")
        failed |= required
print("RERUN MATCHES (timings excluded)" if not failed else "RERUN DIFFERS")
sys.exit(1 if failed else 0)
