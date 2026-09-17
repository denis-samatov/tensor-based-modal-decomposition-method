"""E6 -- Offline and online computational cost (single thread, protocol P2, fold 1).
Median of 5 repeats after one warm-up. Output: outputs/e6_cost.json"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import bases, data, estimators, placement  # noqa: E402


def timeit(fn, rep=5):
    fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
f = data.folds_p2(b)[0]
n = b.n_active
r = bases.pod(f.train, threshold=cfg["energy_threshold"]).extra["rank"]
res = {"fold": f.name, "rank": r, "train_shape": list(f.train.shape), "test_snapshots": f.test.shape[1]}
res["offline_pod_svd_s"] = timeit(lambda: bases.pod(f.train, r=r))
res["offline_tbmd_tucker_s"] = timeit(lambda: bases.tbmd(f.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"], **cfg["tucker"]), rep=3)
tb = bases.tbmd(f.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"], **cfg["tucker"])
pe = bases.pod_energy(f.train, r=r)
res["tbmd_storage_floats"] = tb.extra["storage_floats"]
res["basis_matrix_floats"] = int(pe.B.size)
res["snapshot_tensor_floats_train"] = int(f.train.size)
res["offline_qr_r_s"] = timeit(lambda: placement.qr_dg(tb.B, r))
res["offline_qr_dg_300_s"] = timeit(lambda: placement.qr_dg(tb.B, 300), rep=3)
blocks = [np.array([w, w + n]) for w in b.well_rows]
res["offline_block_dg_30wells_s"] = timeit(lambda: placement.block_dg(tb.B, blocks, 30))
for name, N, order, ur in (("grid30", 30, placement.qr_dg(pe.B, 30), [np.array([q]) for q in range(2 * n)]),
                           ("wells30", 30, placement.block_dg(pe.B, blocks, 30), blocks)):
    rows = np.concatenate([ur[u] for u in order[:N]])
    Y = f.test[rows]
    BS = pe.B[rows]
    res[f"online_ls_{name}_s_per_snapshot"] = timeit(lambda: estimators.least_squares(BS, Y)) / Y.shape[1]
    res[f"online_l1_{name}_s_per_snapshot"] = timeit(lambda: estimators.l1_admm(BS, Y, **{k: cfg["l1"][k] for k in ("epsilon", "delta", "relax", "max_iter", "tol")}), rep=3) / Y.shape[1]
    res["online_synthesis_s_per_snapshot"] = timeit(lambda: pe.B @ estimators.least_squares(BS, Y)) / Y.shape[1]
res["hardware"] = {"machine": platform.machine(), "platform": platform.platform(), "python": sys.version.split()[0],
                   "numpy": np.__version__}
try:
    hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=30).stdout
    res["hardware"]["system_profiler"] = [l.strip() for l in hw.splitlines() if any(k in l for k in ("Chip", "Memory", "Total Number of Cores", "Model Name"))]
except Exception as e:  # noqa: BLE001
    res["hardware"]["system_profiler"] = f"unavailable: {e}"
(OUT / "e6_cost.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
