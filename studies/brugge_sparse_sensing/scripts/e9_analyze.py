"""Generate E9 summaries, paired statistics, tables, figures, and decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bss import FIG, OUT  # noqa: E402

DEST = OUT / "e9_property_mode"
FIGURES = FIG
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
TABLES = DEST / "tables"
FOUR_D_VARIANTS = {"4D-A", "4D-B", "4D-C", "4D-D", "4D-E"}
METRICS = [
    "anomaly_relative_frobenius_p",
    "anomaly_relative_frobenius_so",
    "anomaly_ssim_p",
    "anomaly_ssim_so",
    "relative_frobenius_p",
    "relative_frobenius_so",
    "ssim_p",
    "ssim_so",
    "rmse_p",
    "rmse_so",
    "train_range_nrmse_p",
    "train_range_nrmse_so",
]


def _representation_choice(selection: dict, capacity: str) -> dict[str, str]:
    rows = [
        row
        for row in selection["representation"]
        if row["capacity_regime"] == capacity
    ]
    optimized_4d = min(
        [row for row in rows if row["variant"] in FOUR_D_VARIANTS],
        key=lambda row: (row["inner_selection_loss"], row["variant"]),
    )
    optimized_3d = next(row for row in rows if row["variant"] == "3D-independent")
    return {
        "3D optimized": optimized_3d["candidate"],
        "4D optimized": optimized_4d["candidate"],
    }


def _sparse_pair_choice(
    selection: dict, capacity: str, geometry: str, budget: int
) -> dict:
    rows = [
        row
        for row in selection["paired_sparse"]
        if row["capacity_regime"] == capacity
        and row["geometry"] == geometry
        and int(row["budget"]) == int(budget)
        and row["four_d_variant"] in FOUR_D_VARIANTS
    ]
    return min(
        rows,
        key=lambda row: (row["pair_inner_loss"], row["four_d_variant"]),
    )


def selected_outer(frame: pd.DataFrame, selections: list[dict]) -> pd.DataFrame:
    records = []
    selection_by_fold = {int(row["outer_fold"]): row for row in selections}
    for _, row in frame.iterrows():
        base = row.to_dict()
        selection = selection_by_fold[int(row.outer_fold)]
        if row.geometry == "full":
            fixed_labels = {
                "3D-current": "3D current",
                "4D-A-current": "4D current",
            }
            if row.variant in fixed_labels:
                records.append({**base, "model_label": fixed_labels[row.variant]})
            choices = _representation_choice(selection, row.capacity_regime)
            for label, candidate in choices.items():
                if row.candidate == candidate:
                    records.append({**base, "model_label": label})
            if row.variant in FOUR_D_VARIANTS:
                records.append({**base, "model_label": row.variant})
            continue

        if row.pair_4d_variant == "4D-A-current":
            label = "3D current" if row.pair_role == "3D" else "4D current"
            records.append({**base, "model_label": label})
        choice = _sparse_pair_choice(
            selection,
            row.capacity_regime,
            row.geometry,
            int(row.budget),
        )
        if row.pair_4d_variant == choice["four_d_variant"]:
            label = "3D optimized" if row.pair_role == "3D" else "4D optimized"
            records.append({**base, "model_label": label})
        if row.pair_role == "4D" and row.variant in FOUR_D_VARIANTS:
            records.append({**base, "model_label": row.variant})
    return pd.DataFrame(records)


def summarise(selected: pd.DataFrame) -> pd.DataFrame:
    groups = ["model_label", "capacity_regime", "geometry", "budget"]
    rows = []
    for keys, group in selected.groupby(groups):
        record = dict(zip(groups, keys, strict=True))
        record["folds"] = int(group.outer_fold.nunique())
        record["storage_floats_mean"] = float(group.storage_floats.mean())
        record["condition_number_median"] = float(group.condition_number.median())
        record["condition_number_mean"] = float(group.condition_number.mean())
        for metric in METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_median"] = float(group[metric].median())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
            record[f"{metric}_iqr"] = float(
                group[metric].quantile(0.75) - group[metric].quantile(0.25)
            )
        rows.append(record)
    return pd.DataFrame(rows)


def paired_statistics(
    selected: pd.DataFrame,
    *,
    three_label: str = "3D optimized",
    four_label: str = "4D optimized",
    comparison: str = "optimized",
) -> pd.DataFrame:
    group_columns = ["capacity_regime", "geometry", "budget"]
    direction = {
        metric: (1 if "ssim" in metric else -1)
        for metric in METRICS
    }
    rows = []
    comparison_rows = selected[selected.model_label.isin([three_label, four_label])]
    for keys, group in comparison_rows.groupby(group_columns):
        pivot = group.pivot(index="outer_fold", columns="model_label", values=METRICS)
        if len(pivot) != 10:
            continue
        for metric in METRICS:
            three = pivot[(metric, three_label)].to_numpy()
            four = pivot[(metric, four_label)].to_numpy()
            difference = direction[metric] * (four - three)
            try:
                p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "comparison": comparison,
                    **dict(zip(group_columns, keys, strict=True)),
                    "metric": metric,
                    "mean_improvement": float(difference.mean()),
                    "median_improvement": float(np.median(difference)),
                    "std_improvement": float(difference.std(ddof=1)),
                    "iqr_improvement": float(
                        np.quantile(difference, 0.75) - np.quantile(difference, 0.25)
                    ),
                    "four_d_wins": int(np.sum(difference > 0)),
                    "ties": int(np.sum(difference == 0)),
                    "wilcoxon_p": p_value,
                    "empirical_advantage": bool(
                        np.median(difference) > 0 and np.sum(difference > 0) >= 7
                    ),
                    "fold_differences": json.dumps(difference.tolist()),
                }
            )
    return pd.DataFrame(rows)


def coupling_associations(paired: pd.DataFrame) -> pd.DataFrame:
    paired = paired[paired.comparison == "optimized"]
    coupling = pd.read_csv(DEST / "cross_property_structure.csv")
    rows = []
    for _, row in paired.iterrows():
        differences = np.asarray(json.loads(row.fold_differences), dtype=float)
        for coupling_metric in (
            "anomaly_pearson",
            "temporal_spatial_mean_pearson",
            "subspace_cosine_mean_16",
            "property_rank1_energy",
        ):
            correlation, p_value = spearmanr(coupling[coupling_metric], differences)
            rows.append(
                {
                    "capacity_regime": row.capacity_regime,
                    "geometry": row.geometry,
                    "budget": row.budget,
                    "metric": row.metric,
                    "coupling_metric": coupling_metric,
                    "spearman_rho": float(correlation),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def metric_ranking_audit(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    optimized = selected[selected.model_label.isin(["3D optimized", "4D optimized"])]
    for keys, group in optimized.groupby(["capacity_regime", "geometry", "budget"]):
        means = group.groupby("model_label")[METRICS].mean()
        for prop in ("p", "so"):
            winners = {
                "rmse": means[f"rmse_{prop}"].idxmin(),
                "reconstruction_error": means[
                    f"anomaly_relative_frobenius_{prop}"
                ].idxmin(),
                "ssim": means[f"anomaly_ssim_{prop}"].idxmax(),
            }
            rows.append(
                {
                    "capacity_regime": keys[0],
                    "geometry": keys[1],
                    "budget": keys[2],
                    "property": prop,
                    **{f"{metric}_winner": winner for metric, winner in winners.items()},
                    "ranking_differs": len(set(winners.values())) > 1,
                }
            )
    return pd.DataFrame(rows)


def configuration_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    optimized = selected[
        selected.model_label.isin(["3D optimized", "4D optimized"])
    ].copy()
    columns = [
        "model_label",
        "capacity_regime",
        "geometry",
        "budget",
        "variant",
        "architecture",
        "preprocessing",
        "spatial_ranks",
        "property_rank",
        "property_weights",
        "solver",
    ]
    return (
        optimized.groupby(columns, dropna=False)
        .size()
        .rename("outer_folds_selected")
        .reset_index()
        .sort_values(
            ["model_label", "capacity_regime", "geometry", "budget", "outer_folds_selected"],
            ascending=[True, True, True, True, False],
        )
    )


def decision_summary(paired: pd.DataFrame, ranking: pd.DataFrame) -> dict:
    headline = paired[
        (paired.comparison == "optimized")
        & paired.metric.isin(
            [
                "anomaly_relative_frobenius_p",
                "anomaly_relative_frobenius_so",
                "anomaly_ssim_p",
                "anomaly_ssim_so",
            ]
        )
    ]
    advantages = headline[headline.empirical_advantage]
    return {
        "decision_rule": "median paired improvement > 0 and at least 7/10 favourable folds",
        "headline_comparisons": int(len(headline)),
        "headline_advantages": int(len(advantages)),
        "advantage_records": advantages[
            ["capacity_regime", "geometry", "budget", "metric", "four_d_wins", "median_improvement"]
        ].to_dict("records"),
        "ranking_difference_regimes": int(ranking.ranking_differs.sum()),
        "ranking_regimes_total": int(len(ranking)),
    }


def _plot_curves(summary: pd.DataFrame, metric_prefix: str, filename: str, ylabel: str) -> None:
    labels = ["3D current", "4D current", "3D optimized", "4D optimized"]
    styles = {"3D current": ("o", ":"), "4D current": ("s", "--"), "3D optimized": ("^", "-."), "4D optimized": ("D", "-")}
    colors = {
        "3D current": "#777777",
        "4D current": "#cc6677",
        "3D optimized": "#4477aa",
        "4D optimized": "#228833",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for column, geometry in enumerate(("wells", "grid")):
        subset = summary[
            (summary.capacity_regime == "equal_total")
            & (summary.geometry == geometry)
        ]
        for row_index, prop in enumerate(("p", "so")):
            axis = axes[row_index, column]
            for label in labels:
                line = subset[subset.model_label == label].sort_values("budget")
                if line.empty:
                    continue
                metric = f"{metric_prefix}_{prop}"
                axis.plot(
                    line.budget,
                    line[f"{metric}_mean"],
                    marker=styles[label][0],
                    linestyle=styles[label][1],
                    label=label,
                    color=colors[label],
                )
                axis.fill_between(
                    line.budget,
                    line[f"{metric}_mean"] - line[f"{metric}_std"],
                    line[f"{metric}_mean"] + line[f"{metric}_std"],
                    color=colors[label],
                    alpha=0.12,
                )
            axis.set_title(f"{geometry.capitalize()} - {'pressure' if prop == 'p' else 'saturation'}")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Number of wells" if geometry == "wells" else "Scalar channels")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_ablation(selected: pd.DataFrame) -> None:
    subset = selected[
        (selected.capacity_regime == "equal_total")
        & (selected.geometry == "wells")
        & (selected.budget == 15)
        & selected.model_label.isin(sorted(FOUR_D_VARIANTS))
    ]
    means = subset.groupby("model_label")[
        [
            "anomaly_relative_frobenius_p",
            "anomaly_relative_frobenius_so",
            "anomaly_ssim_p",
            "anomaly_ssim_so",
        ]
    ].mean()
    loss = 0.25 * (
        means.anomaly_relative_frobenius_p
        + means.anomaly_relative_frobenius_so
        + (1 - means.anomaly_ssim_p) / 2
        + (1 - means.anomaly_ssim_so) / 2
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(loss.index, loss.values, color="#4477aa")
    axis.set_ylabel("Preregistered validation-style loss on outer folds")
    axis.set_xlabel("Fourth-order formulation")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig18_e9_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig18_e9_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_cross_property() -> None:
    frame = pd.read_csv(DEST / "cross_property_structure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(frame.outer_fold, frame.raw_pearson, marker="o", label="raw field")
    axes[0].plot(
        frame.outer_fold,
        frame.anomaly_pearson,
        marker="s",
        label="cellwise anomaly",
    )
    axes[0].plot(
        frame.outer_fold,
        frame.temporal_spatial_mean_pearson,
        marker="^",
        label="spatial-mean anomaly",
    )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Pressure-saturation Pearson correlation")
    axes[0].set_xlabel("Held-out scenario")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        frame.outer_fold,
        frame.subspace_cosine_mean_16,
        marker="o",
        label="mean top-16 alignment",
    )
    axes[1].fill_between(
        frame.outer_fold,
        frame.subspace_cosine_min_16,
        frame.subspace_cosine_max_16,
        alpha=0.18,
        label="top-16 range",
    )
    axes[1].plot(
        frame.outer_fold,
        frame.property_rank1_energy,
        marker="s",
        label="property rank-1 energy",
    )
    axes[1].set_ylabel("Alignment or standardized energy fraction")
    axes[1].set_xlabel("Held-out scenario")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(frame.outer_fold)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig17_e9_cross_property.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig17_e9_cross_property.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_tables(summary: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    subset = summary[
        (summary.capacity_regime == "equal_total")
        & (
            ((summary.geometry == "wells") & summary.budget.isin([5, 15, 30]))
            | ((summary.geometry == "grid") & summary.budget.isin([30, 300]))
        )
        & summary.model_label.isin(["3D optimized", "4D optimized"])
    ].copy()
    subset.to_csv(TABLES / "e9_headline_table.csv", index=False)
    for diagnostic in (False, True):
        stem = "diagnostics" if diagnostic else "headline"
        caption = (
            "E9 equal-total-capacity diagnostic scores and mean factor/core storage (float entries). Pressure RMSE uses the supplied export unit $u_p$; oil-saturation RMSE is dimensionless."
            if diagnostic else
            "E9 equal-total-capacity anomaly reconstruction: 32 coefficients per model, identical observations and common inner-selected solver. $E'$ is anomaly-relative error (lower is better); SSIM$'$ is mask-aware anomaly structural similarity (higher is better)."
        )
        caption += " Budget counts scalar channels for grid sensing and two-property wells for well sensing. Values are mean $\\pm$ sample standard deviation over ten outer scenarios. 3D and 4D denote the independently train-selected models."
        lines = ["% Generated by e9_analyze.py; do not edit.",
                 r"\begin{table*}[htbp]\centering\small",
                 "\\caption{" + caption + "}",
                 "\\label{tab:e9_" + stem + "}",
                 r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{adjustbox}{max width=\textwidth}",
                 r"\begin{tabular}{lrlrrr}\toprule" if diagnostic else r"\begin{tabular}{lrlrrrr}\toprule",
                 (r"Geometry & Budget & Model & RMSE$_p$ & RMSE$_{S_o}$ & Stored floats \\ \midrule" if diagnostic else
                  r"Geometry & Budget & Model & $E^\prime_p$ & $E^\prime_{S_o}$ & SSIM$^\prime_p$ & SSIM$^\prime_{S_o}$ \\ \midrule")]
        for _, row in subset.sort_values(["geometry", "budget", "model_label"]).iterrows():
            columns = [("rmse_p", 3), ("rmse_so", 4)] if diagnostic else [
                ("anomaly_relative_frobenius_p", 3), ("anomaly_relative_frobenius_so", 3),
                ("anomaly_ssim_p", 3), ("anomaly_ssim_so", 3)]
            values = [f"{row[key + '_mean']:.{precision}f} $\\pm$ {row[key + '_std']:.{precision}f}" for key, precision in columns]
            if diagnostic:
                values.append(f"{row.storage_floats_mean:.0f}")
            lines.append(f"{row.geometry} & {int(row.budget)} & {row.model_label.split()[0]} & " + " & ".join(values) + r" \\")
        lines.extend([r"\bottomrule\end{tabular}", r"\end{adjustbox}", r"\end{table*}"])
        (TABLES / f"tab_e9_{stem}.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    frame = pd.read_csv(DEST / "outer_results.csv")
    selections = json.loads((DEST / "selected_configurations.json").read_text())
    selected = selected_outer(frame, selections)
    summary = summarise(selected)
    paired = pd.concat(
        [
            paired_statistics(selected),
            paired_statistics(
                selected,
                three_label="3D current",
                four_label="4D current",
                comparison="current",
            ),
        ],
        ignore_index=True,
    )
    coupling = coupling_associations(paired)
    ranking = metric_ranking_audit(selected)
    configurations = configuration_frequency(selected)
    decision = decision_summary(paired, ranking)
    selected.to_csv(DEST / "outer_selected_results.csv", index=False)
    summary.to_csv(DEST / "outer_summary.csv", index=False)
    paired.to_csv(DEST / "paired_comparisons.csv", index=False)
    coupling.to_csv(DEST / "coupling_associations.csv", index=False)
    ranking.to_csv(DEST / "metric_ranking_audit.csv", index=False)
    configurations.to_csv(DEST / "selected_configuration_frequency.csv", index=False)
    (DEST / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    _plot_curves(
        summary,
        "anomaly_relative_frobenius",
        "fig15_e9_reconstruction_error.png",
        "Anomaly-relative Frobenius error",
    )
    _plot_curves(
        summary,
        "anomaly_ssim",
        "fig16_e9_ssim.png",
        "Mask-aware anomaly SSIM",
    )
    _plot_ablation(selected)
    _plot_cross_property()
    _write_tables(summary)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
