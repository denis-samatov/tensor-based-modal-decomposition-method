"""V2 -- Convergence check for l1 solves that reach the ADMM iteration cap.
Cap frequency per protocol/basis from E2, and, for the orthonormal POD basis (where the cap is hit
most often), comparison of the capped ADMM solution with a tightly converged ADMM run and with
coordinate-descent LASSO (scikit-learn) on P2 fold 1, QR placements N = 10, 30, 100.
Output: outputs/v2_admm_convergence.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Lasso  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import bases, data, estimators, metrics, placement  # noqa: E402

cfg = load_config()
d = pd.read_csv(OUT / "e2_summary.csv", usecols=["protocol", "basis", "estimator", "admm_iter"])
d = d[d.estimator == "L1"]
res = {"cap": cfg["l1"]["max_iter"],
       "cap_fraction": {f"{p}|{bn}": float((g.admm_iter >= cfg["l1"]["max_iter"]).mean()) for (p, bn), g in d.groupby(["protocol", "basis"])},
       "pod_checks": []}
b = data.load(cfg["dataset"], cfg["wells"])
f = data.folds_p2(b)[0]
n = b.n_active
pod = bases.pod(f.train, threshold=cfg["energy_threshold"])
eps = cfg["l1"]["epsilon"]
for N in (10, 30, 100):
    rows = placement.qr_dg(pod.B, N)
    BS, Y = pod.B[rows], f.test[rows]
    Xc, itc = estimators.l1_admm(BS, Y, epsilon=eps, delta=cfg["l1"]["delta"], relax=cfg["l1"]["relax"], max_iter=cfg["l1"]["max_iter"], tol=cfg["l1"]["tol"])
    Xt, itt = estimators.l1_admm(BS, Y, epsilon=eps, delta=cfg["l1"]["delta"], relax=cfg["l1"]["relax"], max_iter=200000, tol=1e-9)
    las = Lasso(alpha=eps / len(rows), fit_intercept=False, max_iter=200000, tol=1e-12)
    Xcd = np.column_stack([las.fit(BS, Y[:, t]).coef_ for t in range(Y.shape[1])])
    rm = lambda X: float(metrics.evaluate(pod.B @ X, f, n)["rmse_p"].mean())  # noqa: E731
    ob = lambda X: float(estimators.lasso_objective(BS, Y, X, eps).mean())  # noqa: E731
    res["pod_checks"].append(dict(N=N, admm_capped_iter=itc, admm_tight_iter=itt,
                                  rmse_p_capped=rm(Xc), rmse_p_tight=rm(Xt), rmse_p_cd=rm(Xcd),
                                  obj_capped=ob(Xc), obj_tight=ob(Xt), obj_cd=ob(Xcd),
                                  obj_rel_gap_capped_vs_cd=abs(ob(Xc) - ob(Xcd)) / ob(Xcd)))
(OUT / "v2_admm_convergence.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
