"""E4 -- Sensitivity of the l1 (TBMD compressive-sensing) estimator to epsilon (protocol P2).
Designs: grid QR per basis at N in {10, 30, 100}; wells-joint DG per basis at N in {5, 10, 30}.
Output: outputs/e4_l1_sensitivity.csv"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import data, estimators, experiment, metrics, placement  # noqa: E402


def job(k):
    cfg = load_config()
    b = data.load(cfg["dataset"], cfg["wells"])
    f = data.folds_p2(b)[k]
    n = b.n_active
    bd = experiment.build_bases(f, b, cfg)
    blocks = [np.array([w, w + n]) for w in b.well_rows]
    rows = []
    for bn, bs in bd.items():
        des = [("grid", N, placement.qr_dg(bs.B, 100), [np.array([q]) for q in range(2 * n)]) for N in (10, 30, 100)]
        o = placement.block_dg(bs.B, blocks, 30)
        des += [("wells-joint", N, o, blocks) for N in (5, 10, 30)]
        for sensing, N, order, ur in des:
            R = experiment.rows_for(order, ur, N)
            for eps in cfg["l1_epsilon_sweep"]:
                X, it = estimators.l1_admm(bs.B[R], f.test[R], epsilon=eps, delta=cfg["l1"]["delta"], relax=cfg["l1"]["relax"],
                                           max_iter=cfg["l1"]["max_iter"], tol=cfg["l1"]["tol"])
                m = metrics.evaluate(bs.B @ X, f, n)
                rows.append(dict(fold=f.name, basis=bn, sensing=sensing, N=N, epsilon=eps, admm_iter=it,
                                 rmse_p=m["rmse_p"].mean(), rmse_so=m["rmse_so"].mean()))
            X = estimators.least_squares(bs.B[R], f.test[R], cfg["ls_rcond"])
            m = metrics.evaluate(bs.B @ X, f, n)
            rows.append(dict(fold=f.name, basis=bn, sensing=sensing, N=N, epsilon=0.0, admm_iter=0,
                             rmse_p=m["rmse_p"].mean(), rmse_so=m["rmse_so"].mean()))
    print("fold", k + 1, "done", flush=True)
    return rows


if __name__ == "__main__":
    from bss import parallel
    if sys.argv[1] == "part":
        k = int(sys.argv[2])
        parallel.save_part("e4", f"{k:02d}", job(k))
    else:
        out = [r for _, rows in parallel.load_parts("e4") for r in rows]
        pd.DataFrame(out).to_csv(OUT / "e4_l1_sensitivity.csv", index=False)
