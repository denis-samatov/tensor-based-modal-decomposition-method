"""All manuscript and supplementary figures, generated from outputs/*.csv and the raw data.
Palette (validated for CVD separation and contrast): blue #0072B2, vermillion #D55E00,
green #009E73, magenta #AA3377, olive #8C6D00; no-measurement references in dashed grey.
Output: figures/*.pdf (vector) and figures/*.png (300 dpi previews)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bss import FIG, OUT, load_config  # noqa: E402
from bss import data, estimators, experiment, placement  # noqa: E402

FIG.mkdir(exist_ok=True)
C = {"POD-E LS": "#0072B2", "POD LS": "#8C6D00", "POD-E l1": "#009E73", "TBMD l1": "#D55E00",
     "TBMD LS": "#AA3377", "IDW": "#AA3377", "random": "#0072B2"}
GREY = "#6B6B6B"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5",
                     "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402


def logx_ticks(a, ticks):
    a.set_xticks(ticks)
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


cfg = load_config()
b = data.load(cfg["dataset"], cfg["wells"])
man = json.loads((OUT / "s0_data_manifest.json").read_text())
d = pd.read_csv(OUT / "e2_summary.csv")
n = b.n_active


def grid_img(values):
    img = np.full(b.active.shape, np.nan)
    img[b.ij[:, 0], b.ij[:, 1]] = values
    return img.T  # rows = j (48), columns = i (139)


# ---------------------------------------------------------------- Fig. 2 data context
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.5]})
P = b.fields[:, 0]
t = np.arange(b.T)
for r in range(b.n_runs):
    ax[0].plot(t, P[r].mean(0), color="#0072B2", lw=0.8, alpha=0.7)
ax[0].axvline(man["p1_train_snapshots"] - 0.5, color=GREY, ls="--", lw=0.8)
ax[0].text(man["p1_train_snapshots"] - 2, 168.5, "P1 hold-out", fontsize=6.5, color=GREY, ha="right")
ax[0].set(xlabel="Snapshot index", ylabel="Field-mean pressure (bar)", title="(a) Ten control scenarios")
ax[1].plot(t, man["cross_run_pressure_sd_bar_mean_over_cells_by_time"], color="#0072B2", label="Cross-scenario SD")
lead = np.arange(man["p1_train_snapshots"], b.T)
ax[1].plot(lead, man["p1_persistence_pressure_rmse_bar_mean_over_runs_by_lead"], color="#D55E00", ls="-", label="P1 persistence RMSE")
ax[1].set(xlabel="Snapshot index", ylabel="Pressure (bar)", title="(b) Scenario spread vs. persistence")
ax[1].legend(frameon=False, loc="center right")
im = ax[2].imshow(grid_img(P[0].std(axis=1)), origin="lower", cmap="cividis", aspect="auto")
ax[2].scatter(b.wells[:, 0], b.wells[:, 1], s=10, facecolor="white", edgecolor="black", lw=0.5)
for q, (i, j) in enumerate(b.wells):
    ax[2].annotate(str(q + 1), (i, j), xytext=(2, 2), textcoords="offset points", fontsize=5)
ax[2].set(xlabel="Grid index i", ylabel="Grid index j", title="(c) Temporal SD of pressure, scenario 1")
ax[2].grid(False)
fig.colorbar(im, ax=ax[2], label="bar", fraction=0.04, pad=0.02)
fig.tight_layout()
save(fig, "fig2_data_context")

# ---------------------------------------------------------------- Fig. 3 representation
e1 = pd.read_csv(OUT / "e1_representation.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for k, p in enumerate(("P1", "P2")):
    q = e1[e1.protocol == p]
    for basis, col, mk in (("POD", "#0072B2", "o"), ("TBMD", "#D55E00", "s")):
        g = q[q.basis == basis].groupby("r").rmse_p.agg(["mean", "std"])
        ax[k].errorbar(g.index, g["mean"], yerr=g["std"], color=col, marker=mk, capsize=2,
                       label="POD / POD-E" if basis == "POD" else "TBMD, $R_1{=}R_2{=}48$")
    pr = q[q.basis == "prior"].rmse_p.mean()
    ax[k].axhline(pr, color=GREY, ls="--", lw=1, label="persistence" if p == "P1" else "ensemble mean")
    ax[k].set(xscale="log", yscale="log", xlabel="Dictionary depth r", ylabel="Oracle pressure RMSE (bar)",
              title=f"({'ab'[k]}) Full-field projection, {p}")
    logx_ticks(ax[k], [2, 5, 10, 20, 50])
    ax[k].legend(frameon=False, loc="lower left")
for p, col, mk in (("P1", "#009E73", "^"), ("P2", "#AA3377", "D")):
    g = e1[(e1.protocol == p) & (e1.basis == "TBMD-R1sweep")].groupby("R1").rmse_p.agg(["mean", "std"])
    ax[2].errorbar(g.index, g["mean"], yerr=g["std"], color=col, marker=mk, capsize=2, label=f"TBMD, {p}")
    pod = e1[(e1.protocol == p) & (e1.basis == "POD")]
    pod = pod[pod.r == pod.r_energy].rmse_p.mean()
    ax[2].axhline(pod, color=col, ls=":", lw=1)
ax[2].set(xscale="log", yscale="log", xlabel="Spatial Tucker rank $R_1$ ($R_2=\\min(R_1,48)$)",
          ylabel="Oracle pressure RMSE (bar)", title="(c) Effect of spatial truncation, $r=r_E$")
logx_ticks(ax[2], [12, 24, 48, 96, 139])
ax[2].legend(frameon=False, title="dotted: POD at $r_E$", title_fontsize=6.5, loc="upper right")
fig.tight_layout()
save(fig, "fig3_representation")

# ---------------------------------------------------------------- Fig. 4 budget curves
def curve(q):
    return q.groupby("N").rmse_p.mean()


fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
handles = {}
for row, p in enumerate(("P2", "P1")):
    dp = d[d.protocol == p]
    r = int(dp["rank"].median())
    pr = dp[dp.estimator == "prior"].rmse_p.mean()
    for col, sensing in enumerate(("grid", "wells-joint")):
        a = ax[row, col]
        opt = "QR/DG"
        tag = "QR" if sensing == "grid" else "DG"
        ds = dp[dp.sensing == sensing]
        rnd = ds[(ds.placement == "random") & (ds.basis == "POD-E") & (ds.estimator == "LS")]
        band = rnd.groupby(["N", "test_run"]).rmse_p.median().groupby("N").quantile([0.25, 0.5, 0.75]).unstack()
        a.fill_between(band.index, band[0.25], band[0.75], color="#0072B2", alpha=0.15, lw=0)
        h, = a.plot(band.index, band[0.5], color="#0072B2", ls=":", label="random sites, POD-E LS (median, IQR)")
        handles.setdefault(h.get_label(), h)
        lines = [(f"{tag}-POD-E", "POD-E", "LS", "#0072B2", "o", "-", f"{opt} sites, POD-E LS"),
                 (f"{tag}-POD", "POD", "LS", "#8C6D00", "v", "-", f"{opt} sites, POD LS"),
                 (f"{tag}-POD-E", "POD-E", "L1", "#009E73", "^", "-", f"{opt} sites, POD-E $\\ell_1$"),
                 (f"{tag}-TBMD", "TBMD", "L1", "#D55E00", "s", "-", f"{opt} sites, TBMD (48,48,2) $\\ell_1$"),
                 (f"{tag}-POD-E", "none", "IDW", "#AA3377", "D", "--", f"{opt} sites, prior + IDW")]
        if sensing != "grid":
            lines.append(("configured", "POD-E", "LS", "#0072B2", "x", "--", "configured well order, POD-E LS"))
        for place, basis, est, colr, mk, ls, lab in lines:
            c = curve(ds[(ds.placement == place) & (ds.basis == basis) & (ds.estimator == est)])
            h, = a.plot(c.index, c.values, color=colr, marker=mk, ls=ls, label=lab)
            handles.setdefault(lab, h)
        h = a.axhline(pr, color=GREY, ls="--", lw=1, label="no measurements (P2: ensemble mean; P1: persistence)")
        handles.setdefault(h.get_label(), h)
        if sensing == "grid":
            a.axvline(r, color=GREY, lw=0.6, ls=":")
            a.annotate(f"r = {r}", (r, 0.011), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=GREY)
        a.set(xscale="log", yscale="log", ylim=(0.008, 60),
              xlabel="Selected channels N" if sensing == "grid" else "Instrumented wells N (p and $S_o$)",
              ylabel="Pressure RMSE (bar)", title=f"({'abcd'[2*row+col]}) {p}, {'grid-wide channels' if sensing=='grid' else 'existing wells'}")
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=2, frameon=False, fontsize=7)
save(fig, "fig4_budget_curves")

# ---------------------------------------------------------------- Fig. 5 time-resolved error (P2, wells N=10)
s = pd.read_csv(OUT / "e2_snapshots.csv.gz")
s = s[(s.protocol == "P2")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
for k, (metric, lab) in enumerate((("rmse_p", "Pressure RMSE (bar)"), ("rmse_so", "Oil-saturation RMSE (–)"))):
    for sel, colr, ls, name in (((s.estimator == "prior"), GREY, "--", "ensemble mean (no data)"),
                                ((s.sensing == "wells-joint") & (s.N == 10) & (s.placement == "DG-POD-E") & (s.estimator == "L1"), "#009E73", "-", "10 wells, POD-E $\\ell_1$"),
                                ((s.sensing == "wells-joint") & (s.N == 10) & (s.placement == "DG-POD-E") & (s.estimator == "LS") & (s.basis == "POD-E"), "#0072B2", ":", "10 wells, POD-E LS"),
                                ((s.sensing == "wells-joint") & (s.N == 10) & (s.placement == "DG-POD-E") & (s.estimator == "IDW"), "#AA3377", "-.", "10 wells, prior + IDW"),
                                ((s.sensing == "wells-joint") & (s.N == 10) & (s.placement == "DG-TBMD") & (s.estimator == "L1"), "#D55E00", "-", "10 wells, TBMD $\\ell_1$")):
        g = s[sel].groupby("time_index")[metric].median()
        ax[k].plot(g.index, g.values, color=colr, ls=ls, label=name)
    ax[k].set(yscale="log", xlabel="Snapshot index", ylabel=lab, title=f"({'ab'[k]}) P2, median over held-out scenarios",
              ylim=((0.05, 3) if metric == "rmse_p" else (1e-4, 2e-2)), xlim=(-2, 134))
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7)
save(fig, "fig5_time_resolved")

# ---------------------------------------------------------------- Fig. 6 noise
e3 = pd.read_csv(OUT / "e3_noise.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for k, N in enumerate((10, 30)):
    q = e3[(e3.sensing == "wells-joint") & (e3.N == N)]
    for place, basis, est, colr, mk, lab in (("DG-POD-E", "POD-E", "LS", "#0072B2", "o", "POD-E LS"),
                                             ("DG-POD", "POD", "LS", "#8C6D00", "v", "POD LS"),
                                             ("DG-POD-E", "POD-E", "L1", "#009E73", "^", "POD-E $\\ell_1$"),
                                             ("DG-TBMD", "TBMD", "L1", "#D55E00", "s", "TBMD $\\ell_1$"),
                                             ("DG-POD-E", "none", "IDW", "#AA3377", "D", "prior + IDW")):
        g = q[(q.placement == place) & (q.basis == basis) & (q.estimator == est)].groupby(["sigma_p", "fold"]).rmse_p.mean().groupby("sigma_p")
        m, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        x = m.index.to_numpy() + 0.02
        ax[k].errorbar(x, m, yerr=[m - lo, hi - m], color=colr, marker=mk, capsize=2, label=lab)
    ax[k].axhline(e3[e3.estimator == "prior"].rmse_p.median(), color=GREY, ls="--", lw=1, label="ensemble mean")
    ax[k].set(xscale="log", yscale="log", xlabel="Pressure noise SD $\\sigma_p$ (bar), offset +0.02",
              ylabel="Pressure RMSE (bar)", title=f"({'ab'[k]}) {N} DG-selected wells, P2")
ax[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig6_noise")

# ---------------------------------------------------------------- Fig. 7 sensor maps
z = np.load(OUT / "e5_selection_frequency.npz")
wr = pd.read_csv(OUT / "e5_well_ranks.csv")
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0))
tsd_bar = b.fields[:, 0].std(axis=2).mean(axis=0)  # temporal SD of pressure (bar), mean over scenarios
im = ax[0].imshow(grid_img(tsd_bar), origin="lower", cmap="Greys", aspect="auto", alpha=0.8)
F = z["POD_E"]
for k, (mk, colr, lab) in enumerate((("o", "#0072B2", "pressure channel"), ("^", "#D55E00", "$S_o$ channel"))):
    idx = np.flatnonzero(F[k] > 0)
    ax[0].scatter(b.ij[idx, 0], b.ij[idx, 1], s=6 + 30 * F[k, idx], marker=mk, facecolor="none",
                  edgecolor=colr, lw=0.8, label=lab)
ax[0].scatter(b.wells[:, 0], b.wells[:, 1], s=14, marker="+", color="black", lw=0.8, label="existing well")
ax[0].set(xlabel="Grid index i", ylabel="Grid index j",
          title="(a) QR selections (POD-E, N=50); marker size = selection frequency over 10 leave-one-out bases")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
ax[0].grid(False)
fig.colorbar(im, ax=ax[0], label="pressure temporal SD (bar)", fraction=0.03, pad=0.01)
R = wr[wr.basis == "POD-E"][[f"rank_w{i+1}" for i in range(30)]].to_numpy() + 1
med = np.median(R, axis=0)
im2 = ax[1].imshow(grid_img(tsd_bar), origin="lower", cmap="Greys", aspect="auto", alpha=0.8)
sc = ax[1].scatter(b.wells[:, 0], b.wells[:, 1], c=med, cmap="viridis_r", s=40, edgecolor="black", lw=0.4, vmin=1, vmax=30)
for q, (i, j) in enumerate(b.wells):
    ax[1].annotate(f"{q+1}", (i, j), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
ax[1].set(xlabel="Grid index i", ylabel="Grid index j",
          title="(b) Median greedy rank of existing wells (label = configured index); lower rank = selected earlier")
ax[1].grid(False)
fig.colorbar(sc, ax=ax[1], label="median rank", fraction=0.03, pad=0.01)
fig.tight_layout()
save(fig, "fig7_sensor_maps")

# ---------------------------------------------------------------- Fig. S1 l1 sensitivity
e4 = pd.read_csv(OUT / "e4_l1_sensitivity.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for k, sensing in enumerate(("grid", "wells-joint")):
    q = e4[e4.sensing == sensing]
    Ns = sorted(q.N.unique())
    for basis, colr, mk in (("POD", "#8C6D00", "v"), ("POD-E", "#0072B2", "o"), ("TBMD", "#D55E00", "s")):
        for N, ls in zip(Ns, ("-", "--", ":")):
            g = q[(q.basis == basis) & (q.N == N) & (q.epsilon > 0)].groupby("epsilon").rmse_p.mean()
            ax[k].plot(g.index, g.values, color=colr, marker=mk, ls=ls, label=f"{basis}, N={N}")
    ax[k].axvline(cfg["l1"]["epsilon"], color=GREY, lw=0.6)
    ax[k].set(xscale="log", yscale="log", xlabel="$\\ell_1$ weight $\\varepsilon$", ylabel="Pressure RMSE (bar)",
              title=f"({'ab'[k]}) P2, {'grid-wide QR' if sensing=='grid' else 'DG-selected wells'}")
for a in ax:
    a.legend(frameon=False, fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
fig.tight_layout()
save(fig, "figS1_l1_sensitivity")

# ---------------------------------------------------------------- Fig. S2 example reconstruction maps (P2, scenario 1, 10 wells)
f = data.folds_p2(b)[0]
bd = experiment.build_bases(f, b, cfg)
blocks = [np.array([w, w + n]) for w in b.well_rows]
tsel = int(np.argmax(man["cross_run_pressure_sd_bar_mean_over_cells_by_time"]))
panels = []
truth = f.test_phys[0, :, tsel]
prior_p = f.scaler.inv(f.prior).reshape(2, n, -1)[0, :, tsel]
panels.append(("Ensemble-mean prior", prior_p))
for bn, est in (("POD-E", "LS"), ("TBMD", "L1")):
    o = placement.block_dg(bd[bn].B, blocks, 10)
    rows = experiment.rows_for(o, blocks, 10)
    Y = f.test[rows, tsel:tsel + 1]
    X = estimators.least_squares(bd[bn].B[rows], Y) if est == "LS" else estimators.l1_admm(bd[bn].B[rows], Y, **{k: cfg["l1"][k] for k in ("epsilon", "delta", "relax", "max_iter", "tol")})[0]
    E = f.scaler.inv(bd[bn].B @ X).reshape(2, n, -1)[0, :, 0]
    panels.append((f"10 DG wells, {bn} {'LS' if est=='LS' else 'l1'}", E))
fig, ax = plt.subplots(len(panels) + 1, 1, figsize=(7.2, 7.0))
vmin, vmax = np.percentile(truth, [1, 99])
im = ax[0].imshow(grid_img(truth), origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
ax[0].set_title(f"Reference pressure, held-out scenario 1, snapshot {tsel}")
fig.colorbar(im, ax=ax[0], label="pressure (bar)", fraction=0.03, pad=0.01)
lim = 3.0
for k, (name, E) in enumerate(panels, start=1):
    err = E - truth
    im = ax[k].imshow(grid_img(err), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax[k].set_title(f"Error (estimate − reference): {name}; RMSE = {np.sqrt(np.mean(err**2)):.3f} bar")
    fig.colorbar(im, ax=ax[k], label="bar", fraction=0.03, pad=0.01)
for a in ax:
    a.grid(False); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
save(fig, "figS2_example_maps")
