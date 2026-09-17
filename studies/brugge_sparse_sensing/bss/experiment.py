"""Shared evaluation loop used by E2 (budget sweep) and E3 (noise)."""
from __future__ import annotations

import time

import numpy as np

from . import bases, estimators, metrics, placement


def build_bases(fold, b, cfg, r=None):
    pod = bases.pod(fold.train, r=r, threshold=cfg["energy_threshold"])
    r = pod.extra["rank"]
    tb = bases.tbmd(fold.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"],
                    **cfg["tucker"])
    return {"POD": pod, "POD-E": bases.pod_energy(fold.train, r=r), "TBMD": tb}


def sensing_designs(b, basis_dict, cfg):
    """Return list of (sensing, placement, seed, unit_order, unit_rows) where unit_rows maps a
    unit index to stacked row indices; budgets count units."""
    n = b.n_active
    ns = cfg["random_seeds"]
    rng_seeds = [cfg["master_seed"] + s for s in range(ns)]
    designs = []
    grid_rows = [np.array([q]) for q in range(2 * n)]
    nmax = max(cfg["grid_budgets"])
    for bn, bs in basis_dict.items():
        designs.append(("grid", f"QR-{bn}", -1, placement.qr_dg(bs.B, nmax), grid_rows))
    for s in rng_seeds:
        designs.append(("grid", "random", s, placement.random_order(2 * n, s)[:nmax], grid_rows))
    for sensing, blocks in (("wells-joint", [np.array([w, w + n]) for w in b.well_rows]),
                            ("wells-pressure", [np.array([w]) for w in b.well_rows])):
        designs.append((sensing, "configured", -1, np.arange(len(blocks)), blocks))
        for bn, bs in basis_dict.items():
            designs.append((sensing, f"DG-{bn}", -1, placement.block_dg(bs.B, blocks, len(blocks)), blocks))
        for s in rng_seeds:
            designs.append((sensing, "random", s, placement.random_order(len(blocks), s), blocks))
    return designs


def rows_for(order, unit_rows, N):
    return np.concatenate([unit_rows[u] for u in order[:N]])


def run_fold(protocol, fold, b, cfg, keep_snapshots=True, noise_sd_bar=0.0, noise_seed=None):
    n = b.n_active
    recs, snaps = [], []
    bd = build_bases(fold, b, cfg)
    r = bd["POD"].extra["rank"]
    xy = b.ij.astype(float)

    def add(sensing, place, seed, N, basis, est, m, extra):
        rec = dict(protocol=protocol, fold=fold.name, test_run=fold.test_run + 1, rank=r,
                   sensing=sensing, placement=place, seed=seed, N=N, basis=basis, estimator=est,
                   rmse_p=float(m["rmse_p"].mean()), rmse_so=float(m["rmse_so"].mean()),
                   relerr=float(m["relerr_norm"].mean()), maxabs_p=float(m["maxabs_p"].mean()),
                   noise_sd_bar=noise_sd_bar, **extra)
        recs.append(rec)
        if keep_snapshots and seed == -1:
            for t, (a, c) in enumerate(zip(m["rmse_p"], m["rmse_so"])):
                snaps.append((protocol, fold.name, sensing, place, N, basis, est, int(fold.test_times[t]), a, c))

    # references without measurements
    mp = metrics.evaluate(fold.prior, fold, n)
    add("none", fold.prior_name, -1, 0, "none", "prior", mp, {})
    for bn, bs in bd.items():
        Xo = np.linalg.lstsq(bs.B, fold.test, rcond=None)[0]
        add("full", "oracle-projection", -1, 2 * n, bn, "projection", metrics.evaluate(bs.B @ Xo, fold, n), {})

    rng = np.random.default_rng(noise_seed) if noise_sd_bar > 0 else None
    for sensing, place, seed, order, unit_rows in sensing_designs(b, bd, cfg):
        budgets = cfg["grid_budgets"] if sensing == "grid" else cfg["well_budgets"]
        for N in budgets:
            rows = rows_for(order, unit_rows, N)
            Y = fold.test[rows]
            if rng is not None:
                sd = np.where(rows < n, noise_sd_bar / (fold.scaler.hi[0] - fold.scaler.lo[0]), 0.0)
                Y = Y + sd[:, None] * rng.standard_normal(Y.shape)
            est_idw = estimators.idw_residual(fold.prior, rows, Y, xy, n, cfg["idw_power"])
            add(sensing, place, seed, N, "none", "IDW", metrics.evaluate(est_idw, fold, n), {"m": len(rows)})
            for bn, bs in bd.items():
                if place.startswith(("QR-", "DG-")) and place.split("-", 1)[1] != bn:
                    continue
                BS = bs.B[rows]
                t0 = time.perf_counter()
                X = estimators.least_squares(BS, Y, cfg["ls_rcond"])
                t_ls = time.perf_counter() - t0
                add(sensing, place, seed, N, bn, "LS", metrics.evaluate(bs.B @ X, fold, n),
                    {"m": len(rows), "cond": float(np.linalg.cond(BS)), "sec_per_snapshot": t_ls / Y.shape[1]})
                t0 = time.perf_counter()
                X, it = estimators.l1_admm(BS, Y, **{k: cfg["l1"][k] for k in ("epsilon", "delta", "relax", "max_iter", "tol")})
                t_l1 = time.perf_counter() - t0
                add(sensing, place, seed, N, bn, "L1", metrics.evaluate(bs.B @ X, fold, n),
                    {"m": len(rows), "admm_iter": it, "sec_per_snapshot": t_l1 / Y.shape[1]})
    info = {bn: {"seconds": bs.seconds, **bs.extra} for bn, bs in bd.items()}
    return recs, snaps, info
