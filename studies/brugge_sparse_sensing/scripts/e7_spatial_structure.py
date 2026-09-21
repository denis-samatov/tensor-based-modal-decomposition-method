"""E7 -- Spatial structure of the scenario-specific pressure deviation and of the TBMD truncation
error (protocol P2; supports Discussion Sec. 6.1). For each fold and every snapshot:
deviation D = reference - ensemble-mean prior (supplied export unit u_p); spatial mean and SD of D; Pearson correlation
between |D - mean(D)| and distance to the nearest well. Full-observation TBMD (48,48,2) and POD
projection errors per cell (RMSE over snapshots) near wells (<= 2 cells) and far (> 10 cells).
Output: outputs/e7_spatial_structure.csv"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import bases, data  # noqa: E402

cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
n = b.n_active
dwell = np.min(np.linalg.norm(b.ij[:, None, :] - b.wells[None, :, :], axis=2), axis=1)
rows = []
for f in data.folds_p2(b):
    D = f.test_phys[0] - f.scaler.inv(f.prior).reshape(2, n, -1)[0]
    mu = D.mean(axis=0)
    sd = D.std(axis=0)
    corr = [np.corrcoef(np.abs(D[:, t] - mu[t]), dwell)[0, 1] for t in range(1, b.T)]
    r = bases.pod(f.train, threshold=cfg["energy_threshold"]).extra["rank"]
    # ``_bar`` columns are retained only for compatibility with the released
    # result schema; the array unit is unconfirmed and is reported as u_p.
    rec = dict(fold=f.name, dev_abs_spatial_mean_bar=float(np.mean(np.abs(mu[1:]))),
               dev_spatial_sd_bar=float(np.mean(sd[1:])), corr_absdev_distance=float(np.mean(corr)))
    for name, B in (("TBMD", bases.tbmd(f.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"], **cfg["tucker"]).B),
                    ("POD", bases.pod(f.train, r=r).B)):
        X = np.linalg.lstsq(B, f.test, rcond=None)[0]
        E = f.scaler.inv(B @ X).reshape(2, n, -1)[0] - f.test_phys[0]
        cell = np.sqrt((E ** 2).mean(axis=1))
        rec[f"{name}_oracle_rmse_near_wells_bar"] = float(cell[dwell <= 2].mean())
        rec[f"{name}_oracle_rmse_far_wells_bar"] = float(cell[dwell > 10].mean())
    rows.append(rec)
    print(f.name, {k: round(v, 3) for k, v in rec.items() if k != "fold"}, flush=True)
pd.DataFrame(rows).to_csv(OUT / "e7_spatial_structure.csv", index=False)
