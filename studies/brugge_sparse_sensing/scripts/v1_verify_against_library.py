"""V1 -- Implementation cross-check of the study code against the public TBMD library.

(a) pivots of scipy column-pivoted QR on D^T vs TBMD TensorTubeQRDecomposition on the gridded
    dictionary (same fold, same TBMD dictionary, availability mask = active channels);
(b) l1-ADMM solution of this study vs TBMD TensorCompressiveSensing (library defaults of the
    reviewed configuration), compared through the LASSO objective and the reconstructed state.
Output: outputs/v1_verification.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bss import OUT, load_config  # noqa: E402
from bss import bases, data, estimators, placement  # noqa: E402
from TBMD.core.reconstruction.tensor_compressive_sensing import (  # noqa: E402
    CompressiveSensingConfig, ExtensionCompressiveSensingConfig, TensorCompressiveSensing)
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition  # noqa: E402

cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
f = data.folds_p2(b)[0]
r = bases.pod(f.train, threshold=cfg["energy_threshold"]).extra["rank"]
tb = bases.tbmd(f.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"])
n = b.n_active
res = {"fold": f.name, "rank": r}

# (a) QR pivots
N = r
ours = placement.qr_dg(tb.B, N)
grid = np.zeros((139, 48, 2, r), dtype=np.float64)
grid[b.ij[:, 0], b.ij[:, 1], 0, :] = tb.B[:n]
grid[b.ij[:, 0], b.ij[:, 1], 1, :] = tb.B[n:]
reject = np.ones((139, 48, 2), bool)
reject[b.ij[:, 0], b.ij[:, 1], :] = False
qr = TensorTubeQRDecomposition(tensor=grid, N=N, rejection_domain=reject, random_state=0,
                               check_orthogonality=False, uniform_distribution=False,
                               device="cpu", dtype=torch.float64)
P, _, _ = qr.factorize()
lib = np.argwhere(P.numpy() == 1)  # (i, j, k)
lut = -np.ones((139, 48), int); lut[b.ij[:, 0], b.ij[:, 1]] = np.arange(n)
lib_rows = set(int(lut[i, j] + k * n) for i, j, k in lib)
res["qr_pivots_equal_as_sets"] = lib_rows == set(int(x) for x in ours)
res["qr_n"] = N

# (b) ADMM vs library on the QR sensor set, first test snapshot
rows = ours
BS = tb.B[rows]
y = f.test[rows, :1]
Xo, it = estimators.l1_admm(BS, y, **{k: cfg["l1"][k] for k in ("epsilon", "delta", "relax", "max_iter", "tol")})
Pm = np.zeros((139, 48, 2), bool)
Yg = np.zeros((139, 48, 2))
for q, rr in enumerate(rows):
    k = int(rr >= n); c = rr - k * n
    Pm[b.ij[c, 0], b.ij[c, 1], k] = True
    Yg[b.ij[c, 0], b.ij[c, 1], k] = y[q, 0]
core = CompressiveSensingConfig(max_iter=20000, tol=1e-9, epsilon_l1=cfg["l1"]["epsilon"],
                                delta_init=1.0, delta_max=1.0, relax_lambda=0.95, device="cpu",
                                dtype=torch.float64)
ext = ExtensionCompressiveSensingConfig(solver="cholesky", reg=1e-12, delta_policy="boyd",
                                        stop_policy="residual", collect_history=False)
xl, met = TensorCompressiveSensing(grid, Pm, Yg, core, ext).solve()
xl = xl.numpy().reshape(-1, 1)
# library orders the sensing rows by grid raster order; objective is permutation-invariant
obj_ours = float(estimators.lasso_objective(BS, y, Xo, cfg["l1"]["epsilon"])[0])
obj_lib = float(estimators.lasso_objective(BS, y, xl, cfg["l1"]["epsilon"])[0])
state_diff = float(np.linalg.norm(tb.B @ Xo - tb.B @ xl) / np.linalg.norm(tb.B @ xl))
res.update({"admm_iterations_ours": it, "admm_iterations_lib": met.iterations,
            "lasso_objective_ours": obj_ours, "lasso_objective_lib": obj_lib,
            "relative_state_difference": state_diff})
# (c) Proposition 1: TBMD with untruncated spatial ranks equals rotated energy-weighted POD
pe = bases.pod_energy(f.train, r=r)
tf = bases.tbmd(f.train, b.ij, r, spatial_ranks=(139, 48, 2), **cfg["tucker"])
s1 = np.linalg.svd(pe.B, compute_uv=False); s2 = np.linalg.svd(tf.B, compute_uv=False)
Q1 = np.linalg.qr(pe.B)[0]; Q2 = np.linalg.qr(tf.B)[0]
Qrot = np.linalg.lstsq(pe.B, tf.B, rcond=None)[0]
res.update({"prop1_singular_value_max_rel_diff": float(np.max(np.abs(s1 - s2) / s1)),
            "prop1_subspace_distance": float(np.linalg.norm(Q1 @ (Q1.T @ Q2) - Q2) / np.sqrt(r)),
            "prop1_rotation_orthogonality_error": float(np.linalg.norm(Qrot.T @ Qrot - np.eye(r))),
            "prop1_qr_pivots_equal_first_r": bool(np.array_equal(placement.qr_dg(pe.B, r), placement.qr_dg(tf.B, r)))})
(OUT / "v1_verification.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
