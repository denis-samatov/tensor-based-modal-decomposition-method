"""E5 -- Stability and physical interpretation of selected sensor locations (protocol P2).

For each basis and each of the 10 leave-one-scenario-out bases:
* grid-wide QR-DG selections at N = r and N = 50: pairwise Jaccard similarity across folds;
  share of pressure channels; percentile of the training temporal standard deviation (of the
  measured property) at the selected cells relative to all active cells; distance to the nearest
  existing well (grid cells) relative to the active-cell distribution.
* well ranking by block-DG: Kendall tau between folds and top-10 overlap with the configured order.
Outputs: outputs/e5_stability.csv, outputs/e5_selection_frequency.npz, outputs/e5_well_ranks.csv
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import data, experiment, placement  # noqa: E402

cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
n = b.n_active
folds = data.folds_p2(b)
sel = {}
wellrank = []
tstd = {}
for f in folds:
    bd = experiment.build_bases(f, b, cfg)
    r = bd["POD"].extra["rank"]
    # temporal std of each channel over training data, per training scenario then averaged
    T = f.train.shape[1] // 9
    X = f.train.reshape(2 * n, 9, T)
    tstd[f.name] = X.std(axis=2).mean(axis=1)
    for bn, bs in bd.items():
        o = placement.qr_dg(bs.B, 50)
        sel[(f.name, bn, "N=r")] = o[:r]
        sel[(f.name, bn, "N=50")] = o[:50]
        blocks = [np.array([w, w + n]) for w in b.well_rows]
        wo = placement.block_dg(bs.B, blocks, 30)
        rank = np.empty(30, int); rank[wo] = np.arange(30)
        wellrank.append(dict(fold=f.name, basis=bn, **{f"rank_w{i+1}": int(rank[i]) for i in range(30)},
                             top10_overlap_configured=len(set(wo[:10]) & set(range(10)))))
    print(f.name, "r =", r, flush=True)

dwell = np.min(np.linalg.norm(b.ij[:, None, :] - b.wells[None, :, :], axis=2), axis=1)
rows = []
for bn in ("POD", "POD-E", "TBMD"):
    for lab in ("N=r", "N=50"):
        S = [set(sel[(f.name, bn, lab)].tolist()) for f in folds]
        jac = [len(a & c) / len(a | c) for a, c in itertools.combinations(S, 2)]
        for f in folds:
            o = sel[(f.name, bn, lab)]
            cells = o % n
            ts = tstd[f.name]
            pct = [float((ts[k * n:(k + 1) * n] <= ts[q]).mean()) for q, k in zip(o, o >= n)]
            rows.append(dict(basis=bn, budget=lab, fold=f.name, n_sel=len(o),
                             jaccard_mean=float(np.mean(jac)), jaccard_min=float(np.min(jac)),
                             share_pressure=float(np.mean(o < n)),
                             tstd_percentile_mean=float(np.mean(pct)),
                             dist_to_well_mean=float(dwell[cells].mean()),
                             dist_to_well_active_mean=float(dwell.mean())))
pd.DataFrame(rows).to_csv(OUT / "e5_stability.csv", index=False)
pd.DataFrame(wellrank).to_csv(OUT / "e5_well_ranks.csv", index=False)
# selection frequency maps (per basis, N=50), and Kendall tau of well ranks
freq = {}
for bn in ("POD", "POD-E", "TBMD"):
    F = np.zeros((2, n))
    for f in folds:
        o = sel[(f.name, bn, "N=50")]
        F[(o >= n).astype(int), o % n] += 1
    freq[bn] = F / len(folds)
np.savez_compressed(OUT / "e5_selection_frequency.npz", ij=b.ij, wells=b.wells, dwell=dwell,
                    tstd_p=np.mean([tstd[f.name][:n] for f in folds], axis=0),
                    tstd_s=np.mean([tstd[f.name][n:] for f in folds], axis=0),
                    **{k.replace("-", "_"): v for k, v in freq.items()})
wr = pd.DataFrame(wellrank)
for bn in ("POD", "POD-E", "TBMD"):
    R = wr[wr.basis == bn][[f"rank_w{i+1}" for i in range(30)]].to_numpy()
    taus = [kendalltau(R[i], R[j])[0] for i, j in itertools.combinations(range(len(R)), 2)]
    print(bn, "well-rank Kendall tau mean/min", round(float(np.mean(taus)), 3), round(float(np.min(taus)), 3),
          "top10 overlap with configured", wr[wr.basis == bn].top10_overlap_configured.mean())
print(pd.DataFrame(rows).groupby(["basis", "budget"]).mean(numeric_only=True).round(3).to_string())
