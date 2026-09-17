"""Tables and statistics used in the manuscript (all numbers are read from outputs/*.csv).
Outputs: outputs/table_main_<P>.csv, outputs/table_main_<P>.tex, outputs/stats_wilcoxon.csv,
outputs/table_representation.csv, outputs/table_noise.csv, outputs/table_l1.csv,
outputs/key_numbers.json (single source for every number quoted in the text)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from bss import OUT  # noqa: E402

d = pd.read_csv(OUT / "e2_summary.csv")
key = {}


def fold_values(protocol, sensing, placement, basis, estimator, N, metric="rmse_p"):
    q = d[(d.protocol == protocol) & (d.sensing == sensing) & (d.basis == basis) & (d.estimator == estimator) & (d.N == N)]
    if placement == "random":
        q = q[q.placement == "random"].groupby("test_run")[metric].median()
    else:
        q = q[q.placement == placement].set_index("test_run")[metric]
    return q.sort_index()


def prior(protocol, metric="rmse_p"):
    q = d[(d.protocol == protocol) & (d.estimator == "prior")].set_index("test_run")[metric]
    return q.sort_index()


ROWS = [  # label, sensing-kind, placement template, basis, estimator
    ("No measurements (prior)", None, None, "none", "prior"),
    ("Prior + IDW residual", "same", "same", "none", "IDW"),
    ("POD, LS", "same", "opt:POD", "POD", "LS"),
    ("POD, l1", "same", "opt:POD", "POD", "L1"),
    ("POD-E, LS", "same", "opt:POD-E", "POD-E", "LS"),
    ("POD-E, l1", "same", "opt:POD-E", "POD-E", "L1"),
    ("TBMD (48,48,2), LS", "same", "opt:TBMD", "TBMD", "LS"),
    ("TBMD (48,48,2), l1 [original]", "same", "opt:TBMD", "TBMD", "L1"),
    ("Random placement, POD-E LS (median of 20)", "same", "random", "POD-E", "LS"),
    ("Configured well order, POD-E LS", "wells", "configured", "POD-E", "LS"),
    ("Configured well order, TBMD l1", "wells", "configured", "TBMD", "L1"),
]
COLS = [("grid", 10), ("grid", 30), ("grid", 100), ("wells-joint", 5), ("wells-joint", 10), ("wells-joint", 30), ("wells-pressure", 30)]


def cell_values(protocol, label, kind, ptempl, basis, est, sensing, N, metric):
    if est == "prior":
        return prior(protocol, metric)
    if kind == "wells" and sensing == "grid":
        return None
    if ptempl == "same":  # IDW: use the placement of the best-performing optimised design (POD-E)
        ptempl = "opt:POD-E"
    if ptempl.startswith("opt:"):
        b = ptempl[4:]
        place = f"QR-{b}" if sensing == "grid" else f"DG-{b}"
    else:
        place = ptempl
    return fold_values(protocol, sensing, place, basis, est, N, metric)


for protocol in ("P1", "P2"):
    out = []
    for label, kind, ptempl, basis, est in ROWS:
        row = {"method": label}
        for sensing, N in COLS:
            for metric, scale in (("rmse_p", 1.0), ("rmse_so", 1e3)):
                v = cell_values(protocol, label, kind, ptempl, basis, est, sensing, N, metric)
                col = f"{sensing}|{N}|{metric}"
                if v is None or len(v) == 0:
                    row[col] = ""
                    continue
                v = v.to_numpy() * scale
                row[col] = f"{np.mean(v):.3g} ± {np.std(v, ddof=1):.2g}"
                key[f"{protocol}|{label}|{sensing}|{N}|{metric}|mean"] = float(np.mean(v))
                key[f"{protocol}|{label}|{sensing}|{N}|{metric}|sd"] = float(np.std(v, ddof=1))
                key[f"{protocol}|{label}|{sensing}|{N}|{metric}|median"] = float(np.median(v))
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / f"table_main_{protocol}.csv", index=False)

# paired Wilcoxon signed-rank tests across the 10 control scenarios (P2 and P1)
tests = []
for protocol in ("P1", "P2"):
    def t(name, a, b, N, sensing, metric="rmse_p"):
        a, b = a.align(b, join="inner")
        stat = wilcoxon(a, b, alternative="two-sided")
        tests.append(dict(protocol=protocol, comparison=name, sensing=sensing, N=N, metric=metric,
                          median_a=float(np.median(a)), median_b=float(np.median(b)),
                          n_a_better=int((a < b).sum()), n=len(a), p_value=float(stat.pvalue)))
    for N in (10, 30):
        s = "wells-joint"
        t("POD-E LS vs prior", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N), prior(protocol), N, s)
        t("POD-E LS vs IDW", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "DG-POD-E", "none", "IDW", N), N, s)
        t("POD-E LS vs TBMD l1", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "DG-TBMD", "TBMD", "L1", N), N, s)
        t("DG vs configured (POD-E LS)", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "configured", "POD-E", "LS", N), N, s)
        t("DG vs random median (POD-E LS)", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "random", "POD-E", "LS", N), N, s)
        t("TBMD l1 vs prior", fold_values(protocol, s, "DG-TBMD", "TBMD", "L1", N), prior(protocol), N, s)
        t("POD-E l1 vs prior", fold_values(protocol, s, "DG-POD-E", "POD-E", "L1", N), prior(protocol), N, s)
        t("POD-E l1 vs IDW", fold_values(protocol, s, "DG-POD-E", "POD-E", "L1", N), fold_values(protocol, s, "DG-POD-E", "none", "IDW", N), N, s)
        t("POD-E l1 vs TBMD l1", fold_values(protocol, s, "DG-POD-E", "POD-E", "L1", N), fold_values(protocol, s, "DG-TBMD", "TBMD", "L1", N), N, s)
        t("IDW vs prior", fold_values(protocol, s, "DG-POD-E", "none", "IDW", N), prior(protocol), N, s)
        t("POD-E LS vs prior (So)", fold_values(protocol, s, "DG-POD-E", "POD-E", "LS", N, "rmse_so"), prior(protocol, "rmse_so"), N, s, "rmse_so")
    for N in (10, 30):
        s = "grid"
        t("QR POD-E LS vs random median", fold_values(protocol, s, "QR-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "random", "POD-E", "LS", N), N, s)
        t("POD-E l1 vs POD l1", fold_values(protocol, s, "QR-POD-E", "POD-E", "L1", N), fold_values(protocol, s, "QR-POD", "POD", "L1", N), N, s)
        t("POD-E LS vs POD LS", fold_values(protocol, s, "QR-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "QR-POD", "POD", "LS", N), N, s)
        t("QR POD-E LS vs IDW", fold_values(protocol, s, "QR-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "QR-POD-E", "none", "IDW", N), N, s)
        t("QR POD-E LS vs QR TBMD l1", fold_values(protocol, s, "QR-POD-E", "POD-E", "LS", N), fold_values(protocol, s, "QR-TBMD", "TBMD", "L1", N), N, s)
pd.DataFrame(tests).to_csv(OUT / "stats_wilcoxon.csv", index=False)
for r in tests:
    key[f"wilcoxon|{r['protocol']}|{r['comparison']}|{r['sensing']}|{r['N']}|p"] = r["p_value"]
    key[f"wilcoxon|{r['protocol']}|{r['comparison']}|{r['sensing']}|{r['N']}|n_a_better"] = r["n_a_better"]

# representation (E1)
e1 = pd.read_csv(OUT / "e1_representation.csv")
rep = e1.groupby(["protocol", "basis", "r", "R1"])[["rmse_p", "rmse_so", "relerr", "seconds"]].agg(["mean", "std"]).reset_index()
rep.columns = ["_".join(c).strip("_") for c in rep.columns]
rep.to_csv(OUT / "table_representation.csv", index=False)
for _, row in rep.iterrows():
    key[f"E1|{row.protocol}|{row.basis}|r={row.r}|R1={row.R1}|rmse_p|mean"] = float(row.rmse_p_mean)
    key[f"E1|{row.protocol}|{row.basis}|r={row.r}|R1={row.R1}|rmse_so|mean"] = float(row.rmse_so_mean)
for p in ("P1", "P2"):
    re = e1[(e1.protocol == p) & (e1.basis == "prior")].r_energy
    key[f"E1|{p}|r_energy|min"] = int(re.min()); key[f"E1|{p}|r_energy|max"] = int(re.max())

# noise (E3) and l1 sensitivity (E4)
for name, fn, grp in (("noise", "e3_noise.csv", ["sensing", "placement", "N", "basis", "estimator", "sigma_p"]),
                      ("l1", "e4_l1_sensitivity.csv", ["sensing", "N", "basis", "epsilon"])):
    f = OUT / fn
    if f.exists():
        e = pd.read_csv(f)
        per_fold = e.groupby(["fold"] + grp)[["rmse_p", "rmse_so"]].mean().reset_index()
        tab = per_fold.groupby(grp)[["rmse_p", "rmse_so"]].agg(["mean", "std"]).reset_index()
        tab.columns = ["_".join(c).strip("_") for c in tab.columns]
        tab.to_csv(OUT / f"table_{name}.csv", index=False)
        for _, row in tab.iterrows():
            k = "|".join(str(row[g]) for g in grp)
            key[f"{name}|{k}|rmse_p|mean"] = float(row.rmse_p_mean)
            key[f"{name}|{k}|rmse_so|mean"] = float(row.rmse_so_mean)

(OUT / "key_numbers.json").write_text(json.dumps(key, indent=1, sort_keys=True))
print(pd.read_csv(OUT / "table_main_P2.csv").to_string())
print(pd.read_csv(OUT / "table_main_P1.csv").to_string())
print(pd.DataFrame(tests).round(4).to_string())
