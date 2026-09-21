"""Analyze the nested TBMD study and generate canonical tables and figures."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bss import FIG, OUT, data, load_config  # noqa: E402
from scipy.stats import kendalltau, wilcoxon  # noqa: E402

OPT = OUT / "optimization"
COLOURS = {
    "final_accuracy": "#D55E00",
    "final_pareto": "#009E73",
    "rank_optimized_joint": "#CC79A7",
    "hybrid": "#AA3377",
    "POD": "#8C6D00",
    "POD-E": "#0072B2",
    "original_basis_optimized_recovery": "#7F7F7F",
    "prior+IDW": "#56B4E9",
    "POD-E random": "#999999",
}
DISPLAY = {
    "final_accuracy": "TBMD optimized",
    "final_pareto": "TBMD compression",
    "rank_optimized_joint": "TBMD joint rank-optimized",
    "property_decoupled": "TBMD independent properties",
    "hybrid": "TBMD + POD residual",
    "POD": "POD",
    "POD-E": "POD-E",
    "original_basis_optimized_recovery": "Original TBMD basis + optimized sensing",
    "prior+IDW": "Prior + IDW",
    "POD-E random": "POD-E random (fold median)",
    "prior": "Prior",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=600)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def latex_text(value: object) -> str:
    """Escape plain generated table text without interpreting it as LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def fold_level_sparse(frame: pd.DataFrame) -> pd.DataFrame:
    deterministic = frame[frame["seed"] == -1].copy()
    random = frame[frame["seed"] >= 0].copy()
    if not random.empty:
        keys = ["outer_fold", "test_run", "method", "regime", "budget"]
        numeric = random.select_dtypes(include=np.number).columns.difference(
            ["outer_fold", "test_run", "budget", "seed"]
        )
        collapsed = random.groupby(keys, as_index=False)[list(numeric)].median()
        collapsed["seed"] = -2
        collapsed["candidate"] = "random-median"
        collapsed["placement"] = "random"
        collapsed["solver"] = "fold-median"
        collapsed["measurements"] = random.groupby(keys)["measurements"].first().to_numpy()
        deterministic = pd.concat([deterministic, collapsed], ignore_index=True, sort=False)
    return deterministic


def summarize_sparse(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse_p",
        "rmse_so",
        "mae_p",
        "mae_so",
        "score",
        "condition",
        "online_seconds_per_snapshot",
        "storage_floats",
        "memory_bytes",
        "basis_seconds",
        "measurements",
    ]
    rows = []
    for keys, group in frame.groupby(["method", "regime", "budget"], dropna=False):
        row = dict(zip(("method", "regime", "budget"), keys, strict=True))
        row["n_folds"] = int(group["outer_fold"].nunique())
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                for suffix in ("mean", "median", "sd", "q25", "q75"):
                    row[f"{metric}_{suffix}"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1 and np.isfinite(values).all()
                else (0.0 if len(values) == 1 else np.nan)
            )
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25, method="nearest"))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75, method="nearest"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime", "budget", "method"])


def paired_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = (
        ("final_accuracy", "POD-E"),
        ("final_pareto", "POD-E"),
        ("final_accuracy", "rank_optimized_joint"),
        ("final_accuracy", "original_basis_optimized_recovery"),
        ("final_accuracy", "prior+IDW"),
    )
    for regime in ("well_joint", "well_pressure", "grid"):
        budgets = sorted(frame.loc[frame["regime"] == regime, "budget"].unique())
        for budget in budgets:
            subset = frame[(frame["regime"] == regime) & (frame["budget"] == budget)]
            for left, right in comparisons:
                for metric in ("rmse_p", "rmse_so", "score"):
                    pivot = subset[subset["method"].isin((left, right))].pivot(
                        index="outer_fold", columns="method", values=metric
                    )
                    if len(pivot.dropna()) != 10 or left not in pivot or right not in pivot:
                        continue
                    delta = pivot[left] - pivot[right]
                    try:
                        result = wilcoxon(pivot[left], pivot[right], alternative="two-sided")
                        p_value = float(result.pvalue)
                    except ValueError:
                        p_value = 1.0
                    rows.append(
                        {
                            "regime": regime,
                            "budget": int(budget),
                            "metric": metric,
                            "left": left,
                            "right": right,
                            "left_mean": float(pivot[left].mean()),
                            "right_mean": float(pivot[right].mean()),
                            "mean_delta": float(delta.mean()),
                            "median_delta": float(delta.median()),
                            "left_wins": int((delta < 0).sum()),
                            "ties": int((delta == 0).sum()),
                            "wilcoxon_p": p_value,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["wilcoxon_p_holm"] = np.nan
    for _, family in result.groupby("metric"):
        ordered = family.sort_values("wilcoxon_p")
        count = len(ordered)
        adjusted = np.maximum.accumulate(
            np.minimum(1.0, ordered["wilcoxon_p"].to_numpy() * np.arange(count, 0, -1))
        )
        result.loc[ordered.index, "wilcoxon_p_holm"] = adjusted
    return result


def nondominated(frame: pd.DataFrame, objectives: tuple[str, ...]) -> pd.Series:
    values = frame[list(objectives)].to_numpy(dtype=float)
    result = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(values):
        finite = np.isfinite(values).all(axis=1)
        dominates = finite & np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        dominates[index] = False
        result[index] = not dominates.any()
    return pd.Series(result, index=frame.index)


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    usable = summary[
        summary["method"].isin(
            [
                "final_accuracy",
                "final_pareto",
                "rank_optimized_joint",
                "hybrid",
                "POD",
                "POD-E",
                "original_basis_optimized_recovery",
            ]
        )
    ].copy()
    usable["pareto_accuracy_storage"] = False
    usable["pareto_pressure_storage"] = False
    usable["pareto_accuracy_measurements"] = False
    for _, group in usable.groupby("regime"):
        usable.loc[group.index, "pareto_accuracy_storage"] = nondominated(
            group, ("score_mean", "storage_floats_mean")
        )
        usable.loc[group.index, "pareto_pressure_storage"] = nondominated(
            group, ("rmse_p_mean", "storage_floats_mean")
        )
        usable.loc[group.index, "pareto_accuracy_measurements"] = nondominated(
            group, ("score_mean", "measurements_mean")
        )
    return usable


def ranking_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    rank_columns = [column for column in rankings if column.startswith("well_")]
    rows = []
    for (method, placement), group in rankings.groupby(["method", "placement"]):
        values = group.sort_values("outer_fold")[rank_columns].to_numpy()
        coefficients = [
            kendalltau(values[left], values[right]).statistic
            for left, right in combinations(range(len(values)), 2)
        ]
        rows.append(
            {
                "method": method,
                "placement": placement,
                "mean_kendall_tau": float(np.nanmean(coefficients)),
                "median_kendall_tau": float(np.nanmedian(coefficients)),
                "minimum_kendall_tau": float(np.nanmin(coefficients)),
            }
        )
    return pd.DataFrame(rows)


def write_well_interpretation(rankings: pd.DataFrame) -> None:
    cfg = load_config()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    columns = [f"well_{index + 1}" for index in range(len(brugge.wells))]
    selected = rankings[
        (rankings["method"] == "final_accuracy") & (rankings["placement"] == "condition_pressure")
    ]
    median_rank = selected[columns].median().to_numpy()
    top = np.argsort(median_rank)[:10]
    pressure_variability = brugge.fields[:, 0].std(axis=2).mean(axis=0)[brugge.well_rows]
    saturation_variability = brugge.fields[:, 1].std(axis=2).mean(axis=0)[brugge.well_rows]
    payload = {
        "method": "final_accuracy",
        "placement": "condition_pressure",
        "top_ten_configured_well_indices": (top + 1).tolist(),
        "top_ten_pressure_variability_mean": float(pressure_variability[top].mean()),
        "all_wells_pressure_variability_mean": float(pressure_variability.mean()),
        "top_ten_saturation_variability_mean": float(saturation_variability[top].mean()),
        "all_wells_saturation_variability_mean": float(saturation_variability.mean()),
        "rank_pressure_variability_correlation": float(
            np.corrcoef(median_rank, pressure_variability)[0, 1]
        ),
    }
    (OPT / "well_interpretation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def representation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse_p", "rmse_so", "mae_p", "mae_so", "score", "storage_floats", "fit_seconds"]
    aggregations = {metric: ["mean", "median", "std"] for metric in metrics}
    summary = frame.groupby("label").agg(aggregations)
    summary.columns = ["_".join(column).replace("std", "sd") for column in summary.columns]
    return summary.reset_index()


def placement_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse_p",
        "rmse_so",
        "score",
        "condition",
        "online_seconds_per_snapshot",
        "basis_seconds",
        "basis_clustering_placement_seconds",
    ]
    summary = frame.groupby(["method", "regime", "budget", "placement"])[metrics].agg(
        ["mean", "median", "std"]
    )
    summary.columns = ["_".join(column).replace("std", "sd") for column in summary.columns]
    return summary.reset_index()


def make_tables(summary: pd.DataFrame, representation: pd.DataFrame) -> None:
    main = summary[
        (summary["regime"] == "well_joint")
        & (summary["budget"] == 30)
        & summary["method"].isin(
            [
                "final_accuracy",
                "final_pareto",
                "POD",
                "POD-E",
                "rank_optimized_joint",
                "original_basis_optimized_recovery",
                "prior+IDW",
                "POD-E random",
            ]
        )
    ].copy()
    prior = summary[summary["method"] == "prior"].copy()
    main = pd.concat([main, prior], ignore_index=True, sort=False)
    pod_storage = float(
        representation.loc[representation["label"] == "POD-E", "storage_floats_mean"].iloc[0]
    )
    main["compression_vs_pode"] = pod_storage / main["storage_floats_mean"].replace(0, np.nan)
    main.to_csv(OPT / "table_main.csv", index=False)
    best_p = main["rmse_p_mean"].min()
    best_so = main["rmse_so_mean"].min()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Untouched leave-one-scenario-out reconstruction from all 30 existing wells, with the no-measurement prior shown at zero wells. Values are mean $\pm$ standard deviation across ten outer folds. Bold entries are selected automatically as the smallest column mean.}",
        r"\label{tab:optimization_main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Wells & Pressure RMSE & $S_o$ RMSE & Compression & Storage & Online ms \\",
        r"\midrule",
    ]
    for row in main.sort_values("rmse_p_mean").itertuples():
        p = f"{row.rmse_p_mean:.4f} $\\pm$ {row.rmse_p_sd:.4f}"
        so = f"{row.rmse_so_mean:.6f} $\\pm$ {row.rmse_so_sd:.6f}"
        if np.isclose(row.rmse_p_mean, best_p):
            p = rf"\textbf{{{p}}}"
        if np.isclose(row.rmse_so_mean, best_so):
            so = rf"\textbf{{{so}}}"
        compression = (
            "--"
            if not np.isfinite(row.compression_vs_pode)
            else f"{row.compression_vs_pode:.2f}$\\times$"
        )
        storage = (
            "--"
            if not np.isfinite(row.storage_floats_mean)
            else f"{row.storage_floats_mean / 1000:.1f}k"
        )
        online = row.online_seconds_per_snapshot_mean * 1000
        online_text = "--" if not np.isfinite(online) else f"{online:.3f}"
        lines.append(
            f"{DISPLAY[row.method]} & {int(row.budget)} & {p} & {so} & {compression} & {storage} & {online_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    diagnostic = "\n".join(lines).replace("tab:optimization_main", "tab:optimization_diagnostics")
    diagnostic = diagnostic.replace("Original TBMD basis + optimized sensing", "Original TBMD + tuning").replace("POD-E random (fold median)", "POD-E random")
    (OPT / "tab_optimization_diagnostics.tex").write_text(diagnostic + "\n")
    compact = [r"\begin{table}[htbp]\centering\small",
        r"\caption{Nested P2 reconstruction from all 30 existing joint-property wells (60 scalar channels); the prior uses no observations. Values are mean $\pm$ sample standard deviation over ten scenarios. Pressure uses export unit $u_p$ and saturation is dimensionless. Compression is the POD-E/TBMD factor-storage ratio. Bold marks the automatically determined smallest mean; method labels denote train-selected configurations.}",
        r"\label{tab:optimization_main}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lrrr}\toprule",
        r"Method & Pressure RMSE & $S_o$ RMSE & Compression \\ \midrule"]
    for row in main.sort_values("rmse_p_mean").itertuples():
        name = DISPLAY[row.method].replace("Original TBMD basis + optimized sensing", "Original TBMD + tuning").replace("POD-E random (fold median)", "POD-E random").replace("TBMD joint rank-optimized", "Joint TBMD (tuned ranks)")
        pv = f"{row.rmse_p_mean:.3f} $\\pm$ {row.rmse_p_sd:.3f}"
        sv = f"{row.rmse_so_mean:.5f} $\\pm$ {row.rmse_so_sd:.5f}"
        if np.isclose(row.rmse_p_mean, best_p): pv = "\\textbf{" + pv + "}"
        if np.isclose(row.rmse_so_mean, best_so): sv = "\\textbf{" + sv + "}"
        cv = f"{row.compression_vs_pode:.2f}$\\times$" if np.isfinite(row.compression_vs_pode) else "--"
        compact.append(f"{name} & {pv} & {sv} & {cv} " + r"\\")
    compact += [r"\bottomrule\end{tabular}\end{adjustbox}\end{table}"]
    (OPT / "tab_optimization_main.tex").write_text("\n".join(compact) + "\n")

    old = pd.read_csv(OUT / "e2_summary.csv")
    old = old[
        (old["protocol"] == "P2")
        & (old["sensing"] == "wells-joint")
        & (old["N"] == 30)
        & (old["placement"] == "DG-TBMD")
        & (old["basis"] == "TBMD")
        & (old["estimator"] == "L1")
    ]
    rep_index = representation.set_index("label")
    sparse_index = summary[
        (summary["regime"] == "well_joint") & (summary["budget"] == 30)
    ].set_index("method")
    evolution_rows = [
        {
            "version": "Original TBMD",
            "key_change": "Fixed HOOI ranks (48,48,2), fixed L1",
            "validation_gain": 0.0,
            "outer_pressure_rmse": float(old["rmse_p"].mean()),
            "outer_saturation_rmse": float(old["rmse_so"].mean()),
            "storage_floats": float(rep_index.loc["original_tbmd", "storage_floats_mean"]),
        }
    ]
    evolution_specs = (
        (
            "Original basis + optimized sensing",
            "original_basis_optimized_recovery",
            "original_tbmd",
            "Nested placement/recovery",
        ),
        (
            "Rank-optimized joint TBMD",
            "rank_optimized_joint",
            "rank_optimized_joint",
            "Spatial/modal rank expansion",
        ),
        ("TBMD + residual", "hybrid", "hybrid", "Low-rank POD residual"),
        (
            "Compressed independent TBMD",
            "final_pareto",
            "final_pareto",
            "Independent properties; 5% validation tolerance",
        ),
        (
            "Accuracy-selected independent TBMD",
            "final_accuracy",
            "final_accuracy",
            "Independent properties; minimum validation loss",
        ),
    )
    original_score = float(rep_index.loc["original_tbmd", "score_mean"])
    for version, sparse_label, rep_label, change in evolution_specs:
        outer = sparse_index.loc[sparse_label]
        validation_score = float(rep_index.loc[rep_label, "score_mean"])
        evolution_rows.append(
            {
                "version": version,
                "key_change": change,
                "validation_gain": 1.0 - validation_score / original_score,
                "outer_pressure_rmse": float(outer["rmse_p_mean"]),
                "outer_saturation_rmse": float(outer["rmse_so_mean"]),
                "storage_floats": float(rep_index.loc[rep_label, "storage_floats_mean"]),
            }
        )
    evolution = pd.DataFrame(evolution_rows)
    evolution.to_csv(OPT / "table_tbmd_evolution.csv", index=False)
    evolution_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evolution of TBMD. Validation gain is relative reduction of the train-only representation objective from the original basis; outer errors use 30 joint wells.}",
        r"\label{tab:tbmd_evolution}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}p{0.20\linewidth}p{0.27\linewidth}rrrr@{}}",
        r"\toprule",
        r"TBMD version & Key change & \shortstack{Validation\\gain} & \shortstack{Pressure\\RMSE ($u_p$)} & \shortstack{$S_o$\\RMSE} & \shortstack{Stored\\floats} \\",
        r"\midrule",
    ]
    for row in evolution.itertuples():
        evolution_lines.append(
            f"{latex_text(row.version)} & {latex_text(row.key_change)} & "
            f"{100 * row.validation_gain:.1f}\\% & "
            f"{row.outer_pressure_rmse:.4f} & {row.outer_saturation_rmse:.6f} & "
            f"{row.storage_floats / 1000:.1f}k \\\\"
        )
    evolution_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OPT / "tab_tbmd_evolution.tex").write_text("\n".join(evolution_lines) + "\n")


def make_numbers(
    summary: pd.DataFrame, representation: pd.DataFrame, stability: pd.DataFrame
) -> None:
    macros: dict[str, str] = {}

    def rep(label: str, prefix: str) -> None:
        row = representation[representation["label"] == label].iloc[0]
        macros[f"{prefix}Pressure"] = f"{row.rmse_p_mean:.4f}"
        macros[f"{prefix}PressureSd"] = f"{row.rmse_p_sd:.4f}"
        macros[f"{prefix}Saturation"] = f"{row.rmse_so_mean:.6f}"
        macros[f"{prefix}Joint"] = f"{row.score_mean:.3f}"
        macros[f"{prefix}Storage"] = f"{row.storage_floats_mean / 1000:.1f}"

    def sparse(method: str, regime: str, budget: int, prefix: str) -> None:
        row = summary[
            (summary["method"] == method)
            & (summary["regime"] == regime)
            & (summary["budget"] == budget)
        ].iloc[0]
        macros[f"{prefix}Pressure"] = f"{row.rmse_p_mean:.4f}"
        macros[f"{prefix}PressureSd"] = f"{row.rmse_p_sd:.4f}"
        macros[f"{prefix}Saturation"] = f"{row.rmse_so_mean:.6f}"
        macros[f"{prefix}SaturationSd"] = f"{row.rmse_so_sd:.6f}"
        macros[f"{prefix}Joint"] = f"{row.score_mean:.3f}"

    rep("original_tbmd", "OptRepOriginal")
    rep("rank_optimized_joint", "OptRepJoint")
    rep("final_accuracy", "OptRepAccuracy")
    rep("final_pareto", "OptRepCompression")
    rep("POD-E", "OptRepPode")
    storage = representation.set_index("label")["storage_floats_mean"]
    macros["OptAccuracyCompression"] = f"{storage['POD-E'] / storage['final_accuracy']:.2f}"
    macros["OptCompressionCompression"] = f"{storage['POD-E'] / storage['final_pareto']:.2f}"
    sparse("final_accuracy", "well_joint", 30, "OptWellThirtyAccuracy")
    sparse("final_pareto", "well_joint", 30, "OptWellThirtyCompression")
    sparse("rank_optimized_joint", "well_joint", 30, "OptWellThirtyJoint")
    sparse("POD-E", "well_joint", 30, "OptWellThirtyPode")
    sparse("prior+IDW", "well_joint", 30, "OptWellThirtyIdw")
    sparse("final_pareto", "well_pressure", 30, "OptPressureWellThirtyCompression")
    sparse("POD-E", "well_pressure", 30, "OptPressureWellThirtyPode")
    sparse("final_accuracy", "grid", 100, "OptGridHundredAccuracy")
    sparse("final_pareto", "grid", 100, "OptGridHundredCompression")
    sparse("POD-E", "grid", 100, "OptGridHundredPode")
    sparse("final_accuracy", "grid", 300, "OptGridThreeHundredAccuracy")
    sparse("final_pareto", "grid", 300, "OptGridThreeHundredCompression")
    sparse("POD-E", "grid", 300, "OptGridThreeHundredPode")
    tau = stability[
        (stability["method"] == "final_accuracy") & (stability["placement"] == "condition_pressure")
    ].iloc[0]
    macros["OptWellRankTau"] = f"{tau.mean_kendall_tau:.3f}"
    macros["OptValidationRecords"] = "42000"
    macros["OptOuterRecords"] = "20890"
    macros["OptPlacementRecords"] = "7200"
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    (OPT / "optimization_numbers.tex").write_text("\n".join(lines) + "\n")


def make_figures(
    registry: pd.DataFrame,
    representation: pd.DataFrame,
    fold_sparse: pd.DataFrame,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    # A: spatial rank accuracy, train-validation only.
    configs = registry["config_json"].map(json.loads)
    a = registry[
        (registry["stage"] == "A1-structure")
        & (registry["family"] == "TBMD")
        & (registry["architecture"] == "joint")
        & configs.map(
            lambda item: (
                item.get("modal_rank") == 16
                and item.get("property_rank") == 2
                and item.get("spatial_ranks") is not None
            )
        )
    ].copy()
    a["R1"] = configs[a.index].map(lambda item: item["spatial_ranks"][0])
    a["R2"] = configs[a.index].map(lambda item: item["spatial_ranks"][1])
    ranks = a.groupby(["R1", "R2"])["rmse_p"].agg(["mean", "std"]).reset_index()
    fig, axis = plt.subplots(figsize=(4.6, 2.5))
    for index, (r2, group) in enumerate(ranks.groupby("R2")):
        axis.errorbar(
            group["R1"], group["mean"], yerr=group["std"],
            marker=("o", "s", "^")[index % 3],
            linestyle=("-", "--", ":")[index % 3], label=f"$R_2={r2}$"
        )
    axis.set(
        yscale="log",
        xlabel="$R_1$",
        ylabel="Inner-validation pressure RMSE",
        title="(A) Accuracy versus tensor rank",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig8_rank_accuracy")

    # B: untouched representation accuracy versus compression.
    fig, axis = plt.subplots(figsize=(4.6, 2.5))
    pod_storage = float(
        representation.loc[representation["label"] == "POD-E", "storage_floats_mean"].iloc[0]
    )
    for row in representation.itertuples():
        if row.label not in COLOURS:
            continue
        compression = pod_storage / row.storage_floats_mean
        axis.scatter(
            compression, row.score_mean, color=COLOURS[row.label], s=34,
            marker=("o", "s", "^", "D", "v", "P", "X", "<", ">")[list(COLOURS).index(row.label) % 9],
            label=DISPLAY[row.label]
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Basis compression relative to POD-E",
        ylabel="Normalized joint error",
        title="(B) Accuracy versus compression",
    )
    axis.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "fig9_accuracy_compression")

    # C/D: error and conditioning across well budgets.
    methods = (
        "final_accuracy",
        "final_pareto",
        "POD-E",
        "original_basis_optimized_recovery",
        "prior+IDW",
    )
    fig, axis = plt.subplots(figsize=(3.5, 2.5))
    for index, method in enumerate(methods):
        group = summary[(summary["regime"] == "well_joint") & (summary["method"] == method)]
        if group.empty:
            continue
        axis.plot(
            group["budget"],
            group["rmse_p_median"],
            marker=("o", "s", "^", "D", "v")[index],
            linestyle=("-", "--", "-.", ":", "-")[index],
            ms=3,
            color=COLOURS.get(method),
            label=DISPLAY[method],
        )
        axis.fill_between(
            group["budget"],
            group["rmse_p_q25"],
            group["rmse_p_q75"],
            color=COLOURS.get(method),
            alpha=0.10,
            linewidth=0,
        )
    axis.set(
        yscale="log",
        xlabel="Existing wells",
        ylabel="Pressure RMSE",
        title="(C) Error versus sensor budget",
    )
    axis.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fig10_sensor_budget")

    fig, axis = plt.subplots(figsize=(4.6, 2.5))
    for index, method in enumerate(methods[:-1]):
        group = summary[(summary["regime"] == "well_joint") & (summary["method"] == method)]
        if group.empty:
            continue
        condition = np.minimum(group["condition_median"].to_numpy(), 1e12)
        axis.plot(
            group["budget"],
            condition,
            marker=("o", "s", "^", "D", "v")[index],
            linestyle=("-", "--", "-.", ":", "-")[index],
            ms=3,
            color=COLOURS.get(method),
            label=DISPLAY[method],
        )
    axis.set(
        yscale="log",
        xlabel="Existing wells",
        ylabel=r"Median $\kappa(B_S)$ (clipped at $10^{12}$)",
        title="(D) Conditioning versus sensor budget",
    )
    axis.set_ylim(1, 2e12)
    axis.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "fig11_conditioning")

    # E: final-TBMD well ranking on the active Brugge map.
    cfg = load_config()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    rank_columns = [f"well_{index + 1}" for index in range(len(brugge.wells))]
    chosen = rankings[
        (rankings["method"] == "final_accuracy") & (rankings["placement"] == "condition_joint")
    ]
    median_rank = chosen[rank_columns].median().to_numpy()
    fig, axis = plt.subplots(figsize=(5.5, 2.4))
    active = np.full(brugge.active.shape, np.nan)
    active[brugge.ij[:, 0], brugge.ij[:, 1]] = brugge.fields[:, 0].std(axis=2).mean(axis=0)
    axis.imshow(active.T, origin="lower", cmap="Greys", aspect="auto", alpha=0.75)
    scatter = axis.scatter(
        brugge.wells[:, 0],
        brugge.wells[:, 1],
        c=median_rank,
        cmap="viridis_r",
        s=38,
        edgecolor="black",
        linewidth=0.4,
        vmin=1,
        vmax=30,
    )
    for index, (i, j) in enumerate(brugge.wells):
        axis.annotate(str(index + 1), (i, j), xytext=(2, 2), textcoords="offset points", fontsize=5)
    axis.set(
        xlabel="Grid index i",
        ylabel="Grid index j",
        title="(E) Median conditioning-aware TBMD well rank",
    )
    axis.grid(False)
    fig.colorbar(scatter, ax=axis, label="Median rank")
    fig.tight_layout()
    save(fig, "fig12_well_ranking")

    # F: method-component ablation/evolution at all wells.
    ablation_order = [
        "original_basis_optimized_recovery",
        "rank_optimized_joint",
        "hybrid",
        "final_pareto",
        "final_accuracy",
    ]
    ablation = summary[(summary["regime"] == "well_joint") & (summary["budget"] == 30)].set_index(
        "method"
    )
    ablation = ablation.reindex([item for item in ablation_order if item in ablation.index])
    short_labels = {
        "original_basis_optimized_recovery": "Original basis",
        "rank_optimized_joint": "Rank-optimized joint",
        "hybrid": "Joint + residual",
        "final_pareto": "Independent, compressed",
        "final_accuracy": "Independent, accuracy",
    }
    fig, axis = plt.subplots(figsize=(4.4, 2.5))
    x = np.arange(len(ablation))
    axis.bar(
        x, ablation["score_mean"], color=[COLOURS.get(item, "#777777") for item in ablation.index]
    )
    axis.errorbar(
        x,
        ablation["score_mean"],
        yerr=ablation["score_sd"],
        fmt="none",
        color="black",
        capsize=2,
        lw=0.7,
    )
    axis.set(
        xticks=x,
        xticklabels=[short_labels[item] for item in ablation.index],
        ylabel="Normalized joint error",
        title="(F) TBMD evolution / ablation",
    )
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    save(fig, "fig13_ablation")

    # G: sparse Pareto frontier across methods and budgets.
    fig, axis = plt.subplots(figsize=(4.2, 2.8))
    wells = pareto[pareto["regime"] == "well_joint"]
    for index, (method, group) in enumerate(wells.groupby("method")):
        axis.scatter(
            group["storage_floats_mean"],
            group["score_mean"],
            s=10 + group["budget"],
            alpha=0.75,
            marker=("o", "s", "^", "D", "v", "P", "X", "<", ">")[index % 9],
            color=COLOURS.get(method),
            label=DISPLAY.get(method, method),
        )
    frontier = wells[wells["pareto_accuracy_storage"]].sort_values("storage_floats_mean")
    axis.plot(
        frontier["storage_floats_mean"],
        frontier["score_mean"],
        color="black",
        ls="--",
        lw=0.8,
        label="nondominated envelope",
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Stored floats",
        ylabel="Normalized joint error",
        title="(G) Accuracy--storage--well-budget Pareto map",
    )
    axis.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig14_pareto")


def main() -> None:
    registry = pd.read_csv(OPT / "representation_registry.csv")
    outer_representation = pd.read_csv(OPT / "outer_representation.csv")
    outer_sparse = pd.read_csv(OPT / "outer_sparse.csv")
    outer_placement = pd.read_csv(OPT / "outer_placement_audit.csv")
    rankings = pd.read_csv(OPT / "well_rankings.csv")
    fold_sparse = fold_level_sparse(outer_sparse)
    sparse_summary = summarize_sparse(fold_sparse)
    representation = representation_summary(outer_representation)
    paired = paired_statistics(fold_sparse)
    pareto = pareto_table(sparse_summary)
    stability = ranking_stability(rankings)
    placements = placement_summary(outer_placement)
    sparse_registry = pd.read_csv(OPT / "sparse_registry.csv")
    placement_validation = (
        sparse_registry[sparse_registry["stage"] == "C-placement"]
        .groupby(["method", "regime", "budget", "placement"], as_index=False)
        .agg(score_mean=("score", "mean"), condition_median=("condition", "median"))
    )
    fold_sparse.to_csv(OPT / "outer_sparse_fold_level.csv", index=False)
    sparse_summary.to_csv(OPT / "sparse_summary.csv", index=False)
    representation.to_csv(OPT / "representation_summary.csv", index=False)
    paired.to_csv(OPT / "paired_statistics.csv", index=False)
    pareto.to_csv(OPT / "pareto.csv", index=False)
    stability.to_csv(OPT / "well_ranking_stability.csv", index=False)
    if "--tables-only" not in sys.argv:
        write_well_interpretation(rankings)
    placements.to_csv(OPT / "placement_summary.csv", index=False)
    placement_validation.to_csv(OPT / "placement_validation_summary.csv", index=False)
    make_tables(sparse_summary, representation)
    make_numbers(sparse_summary, representation, stability)
    if "--tables-only" not in sys.argv:
        make_figures(registry, representation, fold_sparse, sparse_summary, pareto, rankings)
    print(
        json.dumps(
            {
                "registry_rows": len(registry),
                "outer_sparse_rows": len(outer_sparse),
                "fold_level_rows": len(fold_sparse),
                "summary_rows": len(sparse_summary),
                "paired_rows": len(paired),
                "pareto_rows": len(pareto),
                "placement_audit_rows": len(outer_placement),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
