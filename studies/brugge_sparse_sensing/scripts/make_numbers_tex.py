"""Emit LaTeX macros for every number quoted in the manuscript text from outputs/key_numbers.json
and the other result files, so that text, tables and figures share one computational source.
Output: outputs/numbers.tex (copied into the manuscript directory by run_all.sh)."""
import json
import re
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bss import OUT  # noqa: E402

K = json.loads((OUT / "key_numbers.json").read_text())
man = json.loads((OUT / "s0_data_manifest.json").read_text())
cost = json.loads((OUT / "e6_cost.json").read_text())
ver = json.loads((OUT / "v1_verification.json").read_text())
macros = {}


def name(s):
    s = re.sub(r"[^A-Za-z]", "", s.title())
    return s


def put(macro, value, fmt):
    macros[macro] = fmt.format(value)


def k(protocol, label, sensing, N, metric="rmse_p", stat="mean"):
    return K[f"{protocol}|{label}|{sensing}|{N}|{metric}|{stat}"]


# dataset
put("NRuns", man["runs"], "{:d}"); put("NT", man["time_steps"], "{:d}"); put("NActive", man["n_active"], "{:,d}")
put("NInactive", man["n_inactive"], "{:,d}"); put("NWells", man["n_wells"], "{:d}")
put("NTrainPOne", man["p1_train_snapshots"], "{:d}"); put("NTestPOne", man["p1_test_snapshots"], "{:d}")
put("PminBar", man["pressure_bar_range_active"][0], "{:.1f}"); put("PmaxBar", man["pressure_bar_range_active"][1], "{:.1f}")
put("SoMax", man["soil_range_active"][1], "{:.3f}")
sd = man["cross_run_pressure_sd_bar_mean_over_cells_by_time"]
put("CrossSdMax", max(sd), "{:.2f}"); put("CrossSdEnd", sd[-1], "{:.2f}")
pers = man["p1_persistence_pressure_rmse_bar_mean_over_runs_by_lead"]
put("PersistFirst", pers[0], "{:.3f}"); put("PersistLast", pers[-1], "{:.2f}")
put("RankPOneMin", K["E1|P1|r_energy|min"], "{:d}"); put("RankPOneMax", K["E1|P1|r_energy|max"], "{:d}")
put("RankPTwoMin", K["E1|P2|r_energy|min"], "{:d}"); put("RankPTwoMax", K["E1|P2|r_energy|max"], "{:d}")
# verification
macros["VerStateDiff"] = "\\num{{{:.1e}}}".format(ver["relative_state_difference"])
# main results, both protocols
labels = {"Prior": "No measurements (prior)", "Idw": "Prior + IDW residual", "PodLs": "POD, LS", "PodLone": "POD, l1",
          "PodeLs": "POD-E, LS", "PodeLone": "POD-E, l1", "TbmdLs": "TBMD (48,48,2), LS",
          "TbmdLone": "TBMD (48,48,2), l1 [original]", "RandLs": "Random placement, POD-E LS (median of 20)",
          "ConfLs": "Configured well order, POD-E LS", "ConfTbmd": "Configured well order, TBMD l1"}
cols = {"GridTen": ("grid", 10), "GridThirty": ("grid", 30), "GridHundred": ("grid", 100),
        "WellFive": ("wells-joint", 5), "WellTen": ("wells-joint", 10), "WellThirty": ("wells-joint", 30),
        "PWellThirty": ("wells-pressure", 30)}
for P, pn in (("P1", "POne"), ("P2", "PTwo")):
    for lk, lab in labels.items():
        for ck, (s, N) in cols.items():
            for metric, mk, fmtp in (("rmse_p", "P", "{:.3g}"), ("rmse_so", "S", "{:.2g}")):
                key = f"{P}|{lab}|{s}|{N}|{metric}|mean"
                if key in K:
                    v = K[key]
                    macros[f"{pn}{lk}{ck}{mk}"] = fmtp.format(v)
                    sdk = key.replace("|mean", "|sd")
                    macros[f"{pn}{lk}{ck}{mk}sd"] = "{:.2g}".format(K[sdk])
# wilcoxon p-values
for key, v in K.items():
    if key.startswith("wilcoxon|") and key.endswith("|p"):
        _, P, comp, s, N, _ = key.split("|")
        macros["W" + {"P1": "POne", "P2": "PTwo"}[P] + name(comp) + name(s) + name(str({10: 'ten', 30: 'thirty'}[int(N)]))] = f"{v:.3f}"
    if key.startswith("wilcoxon|") and key.endswith("|n_a_better"):
        _, P, comp, s, N, _ = key.split("|")
        macros["W" + {"P1": "POne", "P2": "PTwo"}[P] + name(comp) + name(s) + name(str({10: 'ten', 30: 'thirty'}[int(N)])) + "Wins"] = f"{v:d}"
# cost
put("CostPodSvd", cost["offline_pod_svd_s"], "{:.2f}"); put("CostTucker", cost["offline_tbmd_tucker_s"], "{:.1f}")
put("CostQr", cost["offline_qr_r_s"] * 1e3, "{:.1f}"); put("CostQrDg", cost["offline_qr_dg_300_s"], "{:.1f}")
put("CostBlockDg", cost["offline_block_dg_30wells_s"] * 1e3, "{:.1f}")
put("CostLsWell", cost["online_ls_wells30_s_per_snapshot"] * 1e6, "{:.1f}")
put("CostLoneWell", cost["online_l1_wells30_s_per_snapshot"] * 1e6, "{:.0f}")
put("CostSynth", cost["online_synthesis_s_per_snapshot"] * 1e6, "{:.1f}")
put("TbmdStorage", cost["tbmd_storage_floats"], "{:,d}"); put("BasisStorage", cost["basis_matrix_floats"], "{:,d}")
put("TrainStorage", cost["snapshot_tensor_floats_train"], "{:,d}")

# ---- E1 representation (per fold at the training-selected depth, then mean over folds)
import pandas as pd  # noqa: E402
e1 = pd.read_csv(OUT / "e1_representation.csv")
for P, pn in (("P1", "POne"), ("P2", "PTwo")):
    q = e1[e1.protocol == P]
    pod = q[(q.basis == "POD") & (q.r == q.r_energy)].rmse_p.mean()
    tb = q[(q.basis == "TBMD") & (q.r == q.r_energy)].rmse_p.mean()
    put(f"EOne{pn}PodOracle", pod, "{:.3f}"); put(f"EOne{pn}TbmdOracle", tb, "{:.2f}")
    put(f"EOne{pn}FloorRatio", tb / pod, "{:.0f}")
    put(f"EOne{pn}PriorP", q[q.basis == "prior"].rmse_p.mean(), "{:.3f}")
    sw = q[q.basis == "TBMD-R1sweep"].groupby("R1").rmse_p.mean()
    for R1, v in sw.items():
        put(f"EOne{pn}Rone{name(str({12:'twelve',24:'twentyfour',48:'fortyeight',96:'ninetysix',139:'full'}[int(R1)]))}", v, "{:.3f}")
macros["PropSvDiff"] = "\\num{{1e{:d}}}".format(int(np.ceil(np.log10(ver["prop1_singular_value_max_rel_diff"]))))
macros["PropSubspace"] = "\\num{{1e{:d}}}".format(int(np.ceil(np.log10(ver["prop1_subspace_distance"]))))
macros["PropPivotsEqual"] = "identical" if ver.get("prop1_qr_pivots_equal_first_r") else "not identical"
# ---- gains relative to prior (P2)
for ck, (s_, N) in cols.items():
    kk = f"P2|POD-E, LS|{s_}|{N}|rmse_p|mean"; kp = f"P2|No measurements (prior)|{s_}|{N}|rmse_p|mean"
    if kk in K and kp in K:
        put(f"PTwoGainPrior{ck}", K[kp] / K[kk], "{:.0f}")
        kl = f"P2|POD-E, l1|{s_}|{N}|rmse_p|mean"
        put(f"PTwoGainPriorLone{ck}", K[kp] / K[kl], "{:.0f}")
# ---- E3 noise (wells-joint DG-POD-E and configured, per-fold mean over draws, then mean over folds)
e3 = pd.read_csv(OUT / "e3_noise.csv")
pf = e3.groupby(["fold", "sensing", "placement", "N", "basis", "estimator", "sigma_p"]).rmse_p.mean().reset_index()
t3 = pf.groupby(["sensing", "placement", "N", "basis", "estimator", "sigma_p"]).rmse_p.mean()
def n3(placement, N, basis, est, sig):
    return t3[("wells-joint", placement, N, basis, est, sig)]
for N, nn in ((10, "Ten"), (30, "Thirty")):
    for sig, sn in ((0.0, "Zero"), (1.0, "One"), (2.0, "Two")):
        put(f"NoiseWell{nn}PodeLsSig{sn}", n3("DG-POD-E", N, "POD-E", "LS", sig), "{:.2f}")
        put(f"NoiseWell{nn}PodeLoneSig{sn}", n3("DG-POD-E", N, "POD-E", "L1", sig), "{:.2f}")
        put(f"NoiseWell{nn}IdwSig{sn}", n3("DG-POD-E", N, "none", "IDW", sig), "{:.2f}")
        put(f"NoiseWell{nn}TbmdLoneSig{sn}", n3("DG-TBMD", N, "TBMD", "L1", sig), "{:.2f}")
        put(f"NoiseWell{nn}ConfLsSig{sn}", n3("configured", N, "POD-E", "LS", sig), "{:.1f}")
        put(f"NoiseWell{nn}ConfLoneSig{sn}", n3("configured", N, "POD-E", "L1", sig), "{:.2f}")
# ---- E5 stability
e5 = pd.read_csv(OUT / "e5_stability.csv")
g5 = e5.groupby(["basis", "budget"]).mean(numeric_only=True)
for bn, bk in (("POD", "Pod"), ("POD-E", "Pode"), ("TBMD", "Tbmd")):
    for bud, bb in (("N=r", "R"), ("N=50", "Fifty")):
        put(f"StabJac{bk}{bb}", g5.loc[(bn, bud), "jaccard_mean"], "{:.2f}")
        put(f"StabShareP{bk}{bb}", 100 * g5.loc[(bn, bud), "share_pressure"], "{:.0f}")
        put(f"StabTstd{bk}{bb}", 100 * g5.loc[(bn, bud), "tstd_percentile_mean"], "{:.0f}")
        put(f"StabDist{bk}{bb}", g5.loc[(bn, bud), "dist_to_well_mean"], "{:.1f}")
put("StabDistActive", e5.dist_to_well_active_mean.iloc[0], "{:.1f}")
import itertools  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
wr = pd.read_csv(OUT / "e5_well_ranks.csv")
for bn, bk in (("POD", "Pod"), ("POD-E", "Pode"), ("TBMD", "Tbmd")):
    R = wr[wr.basis == bn][[f"rank_w{i+1}" for i in range(30)]].to_numpy()
    taus = [kendalltau(R[i], R[j])[0] for i, j in itertools.combinations(range(len(R)), 2)]
    put(f"StabTau{bk}", float(np.mean(taus)), "{:.2f}"); put(f"StabTauMin{bk}", float(np.min(taus)), "{:.2f}")
    put(f"StabTopTen{bk}", wr[wr.basis == bn].top10_overlap_configured.mean(), "{:.1f}")

# ---- time-resolved (P2, 10 DG wells, joint), medians over folds per snapshot
sn = pd.read_csv(OUT / "e2_snapshots.csv.gz")
sn = sn[sn.protocol == "P2"]
def tser(mask):
    return sn[mask].groupby("time_index").rmse_p.median()
w10 = (sn.sensing == "wells-joint") & (sn.N == 10)
pri = tser(sn.estimator == "prior")
lone = tser(w10 & (sn.placement == "DG-POD-E") & (sn.estimator == "L1"))
idw = tser(w10 & (sn.placement == "DG-POD-E") & (sn.estimator == "IDW"))
put("TimeLoneFirst", lone.iloc[0], "{:.2f}"); put("TimeIdwFirst", idw.iloc[0], "{:.2f}"); put("TimePriorFirst", pri.iloc[0], "{:.2f}")
put("TimeLoneEarly", lone[lone.index <= 30].mean(), "{:.2f}"); put("TimeLoneLate", lone[lone.index > 30].mean(), "{:.2f}")
put("TimePriorEarly", pri[pri.index <= 30].mean(), "{:.2f}"); put("TimePriorLate", pri[pri.index > 30].mean(), "{:.2f}")
below = (lone < pri).to_numpy()
cross = next(t for t in range(len(below)) if below[t:].all())
put("TimeCrossover", int(lone.index[cross]), "{:d}")
# ---- E7 spatial structure
e7 = pd.read_csv(OUT / "e7_spatial_structure.csv")
put("DevShiftMin", e7.dev_abs_spatial_mean_bar.min(), "{:.2f}"); put("DevShiftMax", e7.dev_abs_spatial_mean_bar.max(), "{:.2f}")
put("DevSdMin", e7.dev_spatial_sd_bar.min(), "{:.2f}"); put("DevSdMax", e7.dev_spatial_sd_bar.max(), "{:.2f}")
put("DevCorrMin", e7.corr_absdev_distance.min(), "{:.2f}"); put("DevCorrMax", e7.corr_absdev_distance.max(), "{:.2f}")
for nm, key in (("TbmdNear", "TBMD_oracle_rmse_near_wells_bar"), ("TbmdFar", "TBMD_oracle_rmse_far_wells_bar"),
                ("PodNear", "POD_oracle_rmse_near_wells_bar"), ("PodFar", "POD_oracle_rmse_far_wells_bar")):
    put(f"Oracle{nm}", e7[key].mean(), "{:.3f}")

# ---- E0 re-execution of the reviewed configuration (single-property wells, case3, slice 10)
e0 = pd.read_csv(OUT / "e0_archived_pressure_wells.csv").set_index("sensors")
put("EZeroErrOne", e0.loc[1, "error_mean"], "{:.3f}"); put("EZeroErrTen", e0.loc[10, "error_mean"], "{:.3f}")
put("EZeroSsimOne", e0.loc[1, "ssim_mean"], "{:.3f}"); put("EZeroSsimTen", e0.loc[10, "ssim_mean"], "{:.3f}")
put("EZeroPsnrOne", e0.loc[1, "psnr_mean"], "{:.1f}"); put("EZeroPsnrTen", e0.loc[10, "psnr_mean"], "{:.1f}")
g0 = pd.read_csv(OUT / "e0_archived_pressure_grid.csv")
ct = pd.read_csv(OUT / "e0_cluster_table.csv")

# ---- condition numbers of POD-E well sensing matrices (P2, LS rows)
dc = pd.read_csv(OUT / "e2_summary.csv", usecols=["protocol", "estimator", "sensing", "basis", "placement", "N", "cond"])
dc = dc[(dc.protocol == "P2") & (dc.estimator == "LS") & (dc.sensing == "wells-joint") & (dc.basis == "POD-E")]
for N, nn in ((10, "Ten"), (30, "Thirty")):
    q = dc[dc.N == N]
    put(f"CondDg{nn}", q[q.placement == "DG-POD-E"].cond.median(), "{:,.0f}")
    put(f"CondConf{nn}", q[q.placement == "configured"].cond.median(), "{:,.0f}")
    put(f"CondRand{nn}", q[q.placement == "random"].cond.median(), "{:,.0f}")

# ---- l1 estimates with alternative well orders (P2, POD-E, joint wells)
d2 = pd.read_csv(OUT / "e2_summary.csv", usecols=["protocol", "sensing", "placement", "basis", "estimator", "N", "test_run", "rmse_p"])
d2 = d2[(d2.protocol == "P2") & (d2.sensing == "wells-joint") & (d2.basis == "POD-E") & (d2.estimator == "L1")]
for N, nn in ((10, "Ten"), (30, "Thirty")):
    q = d2[d2.N == N]
    put(f"LoneConf{nn}", q[q.placement == "configured"].rmse_p.mean(), "{:.3g}")
    put(f"LoneRand{nn}", q[q.placement == "random"].groupby("test_run").rmse_p.median().mean(), "{:.3g}")
    put(f"LoneDg{nn}", q[q.placement == "DG-POD-E"].rmse_p.mean(), "{:.3g}")
# P1: best TBMD configuration vs persistence (all budgets, both estimators)
d1 = pd.read_csv(OUT / "e2_summary.csv", usecols=["protocol", "basis", "placement", "estimator", "N", "rmse_p"])
d1 = d1[(d1.protocol == "P1") & (d1.basis == "TBMD") & (d1.placement != "random")]
put("POneTbmdBest", d1.groupby(["placement", "estimator", "N"]).rmse_p.mean().min(), "{:.2f}")

# ---- V2 ADMM convergence
v2p = OUT / "v2_admm_convergence.json"
if v2p.exists():
    v2 = json.loads(v2p.read_text())
    cf = v2["cap_fraction"]
    put("VTwoCapPodPTwo", 100 * cf["P2|POD"], "{:.0f}"); put("VTwoCapPodPOne", 100 * cf["P1|POD"], "{:.0f}")
    put("VTwoCapOtherMax", 100 * max(v for k, v in cf.items() if not k.endswith("|POD")), "{:.1f}")
    macros["VTwoObjGapMax"] = "\\num{{{:.0e}}}".format(max(c["obj_rel_gap_capped_vs_cd"] for c in v2["pod_checks"]))
    put("VTwoRmseDiffMax", max(abs(c["rmse_p_capped"] - c["rmse_p_cd"]) for c in v2["pod_checks"]), "{:.2f}")

lines = ["% Auto-generated by scripts/make_numbers_tex.py -- do not edit by hand."]
for m, v in sorted(macros.items()):
    lines.append(f"\\newcommand{{\\{m}}}{{{v.replace(',', '{,}')}}}")
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
print(len(macros), "macros")
