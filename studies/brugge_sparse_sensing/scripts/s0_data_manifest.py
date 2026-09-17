"""S0 -- Data provenance and descriptive statistics of the Brugge control ensemble used.
Output: outputs/s0_data_manifest.json"""
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import data  # noqa: E402

def _git():
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, cwd=Path(__file__).parents[1]).stdout.strip()
        return {"head": rev, "study_dir_uncommitted_changes": bool(dirty)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
P, S = b.fields[:, 0], b.fields[:, 1]
n_tr = int(b.T * cfg["train_fraction_p1"])
wp = P[:, b.well_rows, :]  # runs, wells, T
# Producer/injector roles are not stored in the data files and are therefore not reported.
late = wp[:, :, -1].mean(0)
init = P[:, :, 0].mean()
persist = []
for r in range(b.n_runs):
    last = P[r, :, n_tr - 1]
    persist.append([float(np.sqrt(np.mean((P[r, :, t] - last) ** 2))) for t in range(n_tr, b.T)])
man = {
    "files": data.manifest(cfg["dataset"], cfg["wells"]),
    "runs": b.n_runs, "grid": list(b.active.shape), "n_active": int(b.n_active),
    "n_inactive": int((~b.active).sum()), "time_steps": b.T,
    "p1_train_snapshots": n_tr, "p1_test_snapshots": b.T - n_tr,
    "n_wells": int(len(b.wells)),
    "pressure_bar_range_active": [float(P.min()), float(P.max())],
    "soil_range_active": [float(S.min()), float(S.max())],
    "mean_pressure_bar_by_time_run1": [float(v) for v in P[0].mean(0)],
    "cross_run_pressure_sd_bar_mean_over_cells_by_time": [float(v) for v in P.std(0).mean(0)],
    "cross_run_soil_sd_mean_over_cells_by_time": [float(v) for v in S.std(0).mean(0)],
    "soil_cells_changed_gt_0p01_run1": int((np.abs(S[0, :, -1] - S[0, :, 0]) > 0.01).sum()),
    "p1_persistence_pressure_rmse_bar_mean_over_runs_by_lead": list(np.mean(persist, axis=0)),
    "well_late_pressure_bar": [float(v) for v in late],
    "initial_mean_pressure_bar": float(init),
    "git_commit": _git(),
    "platform": {"python": sys.version.split()[0], "machine": platform.machine(), "system": platform.platform()},
}
(OUT / "s0_data_manifest.json").write_text(json.dumps(man, indent=2))
print(json.dumps({k: v for k, v in man.items() if not isinstance(v, list) or len(v) < 40}, indent=2))
