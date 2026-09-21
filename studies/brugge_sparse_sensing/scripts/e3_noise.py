"""E3 -- Robustness to measurement noise (protocol P2).

Selected designs: grid QR (per basis) at N = 30 channels, wells-joint DG (per basis) and the
configured order at N = 10 and N = 30 wells. Gaussian noise with standard deviation
(sigma_p [u_p], sigma_So [-]) in {(0,0), (0.1,0.005), (0.5,0.01), (1,0.02), (2,0.05)} is added to
pressure and oil-saturation measurements respectively; 10 draws per level.
Output: outputs/e3_noise.csv
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import data, estimators, experiment, metrics, placement  # noqa: E402

LEVELS = [(0.0, 0.0), (0.1, 0.005), (0.5, 0.01), (1.0, 0.02), (2.0, 0.05)]


def job(k):
    cfg = load_config()
    b = data.load(cfg["dataset"], cfg["wells"])
    f = data.folds_p2(b)[k]
    n = b.n_active
    bd = experiment.build_bases(f, b, cfg)
    xy = b.ij.astype(float)
    blocks = [np.array([w, w + n]) for w in b.well_rows]
    designs = []
    for bn, bs in bd.items():
        designs.append(("grid", f"QR-{bn}", 30, placement.qr_dg(bs.B, 30), [np.array([q]) for q in range(2 * n)], [bn]))
        o = placement.block_dg(bs.B, blocks, 30)
        for N in (10, 30):
            designs.append(("wells-joint", f"DG-{bn}", N, o, blocks, [bn]))
    for N in (10, 30):
        designs.append(("wells-joint", "configured", N, np.arange(30), blocks, list(bd)))
    rows = []
    rng = np.random.default_rng(cfg["master_seed"] + 1000 + k)
    span = f.scaler.hi - f.scaler.lo
    for sensing, place, N, order, unit_rows, bnames in designs:
        R = experiment.rows_for(order, unit_rows, N)
        sd = np.where(R < n, 1.0 / span[0], 1.0 / span[1])
        for sp, ss in LEVELS:
            draws = 1 if sp == 0 else cfg["noise_draws"]
            for d in range(draws):
                noise = np.where(R < n, sp, ss)[:, None] * sd[:, None] * rng.standard_normal((len(R), f.test.shape[1]))
                Y = f.test[R] + noise
                ests = {("none", "IDW"): estimators.idw_residual(f.prior, R, Y, xy, n, cfg["idw_power"])}
                for bn in bnames:
                    B = bd[bn].B
                    ests[(bn, "LS")] = B @ estimators.least_squares(B[R], Y, cfg["ls_rcond"])
                    ests[(bn, "L1")] = B @ estimators.l1_admm(B[R], Y, **{q: cfg["l1"][q] for q in ("epsilon", "delta", "relax", "max_iter", "tol")})[0]
                for (bn, en), E in ests.items():
                    m = metrics.evaluate(E, f, n)
                    rows.append(dict(fold=f.name, sensing=sensing, placement=place, N=N, basis=bn, estimator=en,
                                     sigma_p=sp, sigma_so=ss, draw=d, rmse_p=m["rmse_p"].mean(), rmse_so=m["rmse_so"].mean()))
    mp = metrics.evaluate(f.prior, f, n)
    rows.append(dict(fold=f.name, sensing="none", placement="ensemble-mean", N=0, basis="none", estimator="prior",
                     sigma_p=0, sigma_so=0, draw=0, rmse_p=mp["rmse_p"].mean(), rmse_so=mp["rmse_so"].mean()))
    print("fold", k + 1, "done", flush=True)
    return rows


if __name__ == "__main__":
    from bss import parallel
    if sys.argv[1] == "part":
        k = int(sys.argv[2])
        parallel.save_part("e3", f"{k:02d}", job(k))
    else:
        out = [r for _, rows in parallel.load_parts("e3") for r in rows]
        pd.DataFrame(out).to_csv(OUT / "e3_noise.csv", index=False)
