"""Stages B--E: nested recovery/placement tuning and untouched outer evaluation."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bss import OUT, bases, data, estimators, load_config, placement  # noqa: E402
from bss.optimization import fit_active_clusters, normalized_joint_score  # noqa: E402
from bss.study import (  # noqa: E402
    fit_representation,
    physical_error_summary,
    placement_orders,
    solve_coefficients,
    solver_id,
    transform_runs,
)

OPT_OUT = OUT / "optimization"
PARTS = OPT_OUT / "parts"
TUNED_LABELS = ("final_accuracy", "final_pareto", "POD", "POD-E")
ALL_LABELS = (
    "final_accuracy",
    "final_pareto",
    "rank_optimized_joint",
    "property_decoupled",
    "hybrid",
    "POD",
    "POD-E",
    "original_tbmd",
)
OUTPUT_LABELS = {
    "original_tbmd": "original_basis_optimized_recovery",
}
ANCHORS = {
    "well_joint": [1, 3, 5, 10, 20, 30],
    "well_pressure": [5, 10, 30],
    "grid": [5, 10, 30, 100, 300],
}


def load_selection(outer: int) -> dict:
    merged = OPT_OUT / "selected_representations.json"
    if merged.exists():
        return json.loads(merged.read_text())[str(outer + 1)]
    part = PARTS / f"o1_fold{outer + 1:02d}.json"
    if not part.exists():
        raise FileNotFoundError("run Stage A before Stage B")
    payload = json.loads(part.read_text())
    return {
        "train_runs": payload["train_runs"],
        "test_run": payload["test_run"],
        "inner_splits": payload["inner_splits"],
        "selected": payload["selected"],
        "basis_info": payload["basis_info"],
    }


def fit_model(raw_fit: np.ndarray, b: data.Brugge, spec: dict, cfg: dict):
    if spec["family"] != "TBMD-HOOI":
        enriched = {
            **spec,
            "svd_iterations": cfg["optimization"]["randomized_svd_iterations"],
        }
        return fit_representation(raw_fit, b.ij, enriched)
    from bss.optimization import PropertyScaler

    scaler = PropertyScaler.fit(raw_fit, "minmax")
    train = transform_runs(raw_fit, scaler)
    singular = np.linalg.svd(train, compute_uv=False)
    rank = bases.energy_rank(singular, cfg["energy_threshold"])
    model = bases.tbmd(
        train,
        b.ij,
        rank,
        spatial_ranks=tuple(cfg["tbmd_spatial_ranks"]),
        **cfg["tucker"],
    )
    model.extra["spec_id"] = spec["id"]
    model.extra["scaling"] = "minmax"
    return scaler, model, train


def build_context(raw_fit: np.ndarray, b: data.Brugge, spec: dict, cfg: dict) -> dict:
    scaler, model, train = fit_model(raw_fit, b, spec, cfg)
    opt = cfg["optimization"]
    clustering = fit_active_clusters(
        train,
        b.ij,
        k_values=tuple(opt["cluster_k"]),
        seeds=tuple(opt["cluster_seeds"]),
    )
    orders = placement_orders(
        model.B,
        b.well_rows,
        b.ij,
        max_wells=30,
        max_grid=max(opt["grid_budgets"]),
        condition_weight=float(opt["condition_weight"]),
        cluster_labels=clustering["labels"],
    )
    return {
        "scaler": scaler,
        "basis": model,
        "train": train,
        "orders": orders,
        "clustering": clustering,
    }


def rows_for(regime: str, order: np.ndarray, budget: int, b: data.Brugge) -> np.ndarray:
    if regime == "grid":
        return np.asarray(order[:budget], dtype=int)
    selected_wells = np.asarray(order[:budget], dtype=int)
    pressure = b.well_rows[selected_wells]
    if regime == "well_pressure":
        return pressure
    return np.concatenate([pressure, pressure + b.n_active])


def placement_names(context: dict, regime: str) -> list[str]:
    return list(context["orders"]["grid" if regime == "grid" else "well"])


def order_for(context: dict, regime: str, name: str) -> np.ndarray:
    return context["orders"]["grid" if regime == "grid" else "well"][name]


def sparse_validation(
    raw_fit: np.ndarray, raw_validation: np.ndarray, context: dict, rows: np.ndarray, solver: dict
) -> dict[str, float]:
    scaler = context["scaler"]
    basis = context["basis"].B
    prior = raw_fit.mean(axis=0)
    results = []
    elapsed = 0.0
    iterations = []
    for validation_run in raw_validation:
        target = transform_runs(validation_run, scaler)
        start = time.perf_counter()
        coefficients, metadata = solve_coefficients(basis[rows], target[rows], solver)
        elapsed += time.perf_counter() - start
        iterations.append(metadata.get("iterations", 0))
        estimate = scaler.inv((basis @ coefficients).reshape(2, raw_fit.shape[2], -1))
        error = physical_error_summary(estimate, validation_run)
        prior_error = physical_error_summary(prior, validation_run)
        error["prior_rmse_p"] = prior_error["rmse_p"]
        error["prior_rmse_so"] = prior_error["rmse_so"]
        error["score"] = normalized_joint_score(
            error["rmse_p"],
            error["rmse_so"],
            prior_pressure=prior_error["rmse_p"],
            prior_saturation=prior_error["rmse_so"],
        )
        results.append(error)
    summary = {key: float(np.mean([row[key] for row in results])) for key in results[0]}
    summary["online_seconds_per_snapshot"] = elapsed / (
        len(raw_validation) * raw_validation.shape[-1]
    )
    summary["iterations"] = float(np.mean(iterations))
    summary["condition"] = float(np.linalg.cond(basis[rows]))
    return summary


def registry_row(
    stage: str,
    outer: int,
    inner: int,
    label: str,
    regime: str,
    budget: int,
    placement_name: str,
    solver: dict,
    score: dict,
    rows: np.ndarray,
    spec: dict,
) -> dict:
    return {
        "stage": stage,
        "outer_fold": outer + 1,
        "inner_fold": inner + 1,
        "method": label,
        "candidate": spec["id"],
        "regime": regime,
        "budget": budget,
        "measurements": len(rows),
        "placement": placement_name,
        "solver": solver_id(solver),
        "score": score["score"],
        "rmse_p": score["rmse_p"],
        "rmse_so": score["rmse_so"],
        "mae_p": score["mae_p"],
        "mae_so": score["mae_so"],
        "condition": score["condition"],
        "online_seconds_per_snapshot": score["online_seconds_per_snapshot"],
        "iterations": score["iterations"],
        "status": "ok",
        "error": "",
        "solver_json": json.dumps(solver, sort_keys=True),
        "config_json": json.dumps(spec, sort_keys=True),
    }


def best_by(frame: pd.DataFrame, keys: list[str], choice_columns: list[str]) -> dict[tuple, dict]:
    means = frame.groupby(keys + choice_columns, as_index=False).agg(
        score=("score", "mean"),
        condition=("condition", "median"),
    )
    result = {}
    for key, group in means.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        row = group.sort_values(["score", "condition"] + choice_columns).iloc[0]
        result[key] = row.to_dict()
    return result


def nearest_anchor(regime: str, budget: int) -> int:
    return min(ANCHORS[regime], key=lambda anchor: (abs(np.log(anchor) - np.log(budget)), anchor))


def tune_outer(outer: int, selection: dict, b: data.Brugge, cfg: dict):
    opt = cfg["optimization"]
    solvers = opt["solver_candidates"]
    registry = []
    contexts: dict[tuple[int, str], dict] = {}
    inner_payload = selection["inner_splits"]
    for inner, split in enumerate(inner_payload):
        fit_runs = [run - 1 for run in split["fit_runs"]]
        validation_runs = [run - 1 for run in split["validation_runs"]]
        raw_fit = b.fields[fit_runs]
        raw_validation = b.fields[validation_runs]
        for label in TUNED_LABELS:
            spec = selection["selected"][label]
            context = build_context(raw_fit, b, spec, cfg)
            contexts[inner, label] = context
            for regime, anchors in ANCHORS.items():
                base_placement = "qr_joint" if regime == "grid" else "configured"
                order = order_for(context, regime, base_placement)
                for budget in anchors:
                    rows = rows_for(regime, order, budget, b)
                    for solver in solvers:
                        score = sparse_validation(raw_fit, raw_validation, context, rows, solver)
                        registry.append(
                            registry_row(
                                "B-recovery",
                                outer,
                                inner,
                                label,
                                regime,
                                budget,
                                base_placement,
                                solver,
                                score,
                                rows,
                                spec,
                            )
                        )
            print(f"outer {outer + 1} inner {inner + 1}: recovery {label}", flush=True)

    frame_b = pd.DataFrame(registry)
    best_solver = best_by(frame_b, ["method", "regime", "budget"], ["solver"])
    solver_lookup = {solver_id(spec): spec for spec in solvers}

    for inner, split in enumerate(inner_payload):
        fit_runs = [run - 1 for run in split["fit_runs"]]
        validation_runs = [run - 1 for run in split["validation_runs"]]
        raw_fit = b.fields[fit_runs]
        raw_validation = b.fields[validation_runs]
        for label in TUNED_LABELS:
            spec = selection["selected"][label]
            context = contexts[inner, label]
            for regime, anchors in ANCHORS.items():
                for budget in anchors:
                    solver_name = best_solver[label, regime, budget]["solver"]
                    solver = solver_lookup[solver_name]
                    for placement_name in placement_names(context, regime):
                        order = order_for(context, regime, placement_name)
                        rows = rows_for(regime, order, budget, b)
                        score = sparse_validation(raw_fit, raw_validation, context, rows, solver)
                        registry.append(
                            registry_row(
                                "C-placement",
                                outer,
                                inner,
                                label,
                                regime,
                                budget,
                                placement_name,
                                solver,
                                score,
                                rows,
                                spec,
                            )
                        )
            print(f"outer {outer + 1} inner {inner + 1}: placement {label}", flush=True)

    frame_c = pd.DataFrame([row for row in registry if row["stage"] == "C-placement"])
    best_placement = best_by(frame_c, ["method", "regime", "budget"], ["placement"])

    for inner, split in enumerate(inner_payload):
        fit_runs = [run - 1 for run in split["fit_runs"]]
        validation_runs = [run - 1 for run in split["validation_runs"]]
        raw_fit = b.fields[fit_runs]
        raw_validation = b.fields[validation_runs]
        for label in TUNED_LABELS:
            spec = selection["selected"][label]
            context = contexts[inner, label]
            for regime, anchors in ANCHORS.items():
                for budget in anchors:
                    placement_name = best_placement[label, regime, budget]["placement"]
                    order = order_for(context, regime, placement_name)
                    rows = rows_for(regime, order, budget, b)
                    for solver in solvers:
                        score = sparse_validation(raw_fit, raw_validation, context, rows, solver)
                        registry.append(
                            registry_row(
                                "D-joint-refinement",
                                outer,
                                inner,
                                label,
                                regime,
                                budget,
                                placement_name,
                                solver,
                                score,
                                rows,
                                spec,
                            )
                        )
            print(f"outer {outer + 1} inner {inner + 1}: joint {label}", flush=True)

    frame_d = pd.DataFrame([row for row in registry if row["stage"] == "D-joint-refinement"])
    final = best_by(frame_d, ["method", "regime", "budget"], ["placement", "solver"])
    strategies: dict[str, dict] = {label: {} for label in TUNED_LABELS}
    for (label, regime, budget), row in final.items():
        strategies[label].setdefault(regime, {})[str(budget)] = {
            "placement": row["placement"],
            "solver": solver_lookup[row["solver"]],
            "validation_score": row["score"],
            "validation_condition_median": row["condition"],
        }
    return registry, strategies


def evaluate_outer_method(
    outer: int,
    label: str,
    spec: dict,
    context: dict,
    raw_fit: np.ndarray,
    raw_test: np.ndarray,
    b: data.Brugge,
    strategies: dict,
    strategy_label: str,
) -> list[dict]:
    records = []
    for regime, budgets in (
        ("well_joint", load_config()["optimization"]["well_budgets"]),
        ("well_pressure", load_config()["optimization"]["well_budgets"]),
        ("grid", load_config()["optimization"]["grid_budgets"]),
    ):
        for budget in budgets:
            anchor = nearest_anchor(regime, budget)
            strategy = strategies[strategy_label][regime][str(anchor)]
            placement_name = strategy["placement"]
            solver = strategy["solver"]
            order = order_for(context, regime, placement_name)
            rows = rows_for(regime, order, budget, b)
            target = transform_runs(raw_test, context["scaler"])
            start = time.perf_counter()
            coefficients, metadata = solve_coefficients(
                context["basis"].B[rows], target[rows], solver
            )
            online_seconds = time.perf_counter() - start
            estimate = context["scaler"].inv(
                (context["basis"].B @ coefficients).reshape(2, b.n_active, -1)
            )
            error = physical_error_summary(estimate, raw_test)
            prior_error = physical_error_summary(raw_fit.mean(axis=0), raw_test)
            records.append(
                {
                    "outer_fold": outer + 1,
                    "test_run": outer + 1,
                    "method": label,
                    "candidate": spec["id"],
                    "regime": regime,
                    "budget": budget,
                    "measurements": len(rows),
                    "placement": placement_name,
                    "solver": solver_id(solver),
                    **error,
                    "score": normalized_joint_score(
                        error["rmse_p"],
                        error["rmse_so"],
                        prior_pressure=prior_error["rmse_p"],
                        prior_saturation=prior_error["rmse_so"],
                    ),
                    "condition": float(np.linalg.cond(context["basis"].B[rows])),
                    "online_seconds_per_snapshot": online_seconds / raw_test.shape[-1],
                    "iterations": metadata.get("iterations", 0),
                    "storage_floats": context["basis"].extra["storage_floats"],
                    "memory_bytes": 8 * context["basis"].extra["storage_floats"],
                    "basis_seconds": context["basis"].seconds,
                    "config_json": json.dumps(spec, sort_keys=True),
                    "solver_json": json.dumps(solver, sort_keys=True),
                    "seed": -1,
                }
            )
    return records


def evaluate_references_and_random(
    outer: int,
    raw_fit: np.ndarray,
    raw_test: np.ndarray,
    b: data.Brugge,
    pod_context: dict,
    strategies: dict,
    cfg: dict,
) -> list[dict]:
    records = []
    prior = raw_fit.mean(axis=0)
    prior_error = physical_error_summary(prior, raw_test)
    records.append(
        {
            "outer_fold": outer + 1,
            "test_run": outer + 1,
            "method": "prior",
            "candidate": "none",
            "regime": "none",
            "budget": 0,
            "measurements": 0,
            "placement": "none",
            "solver": "prior",
            **prior_error,
            "score": 1.0,
            "condition": np.nan,
            "online_seconds_per_snapshot": 0.0,
            "iterations": 0,
            "storage_floats": 0,
            "memory_bytes": 0,
            "basis_seconds": 0.0,
            "config_json": "{}",
            "solver_json": "{}",
            "seed": -1,
        }
    )
    target = transform_runs(raw_test, pod_context["scaler"])
    prior_normalized = transform_runs(prior, pod_context["scaler"])
    opt = cfg["optimization"]
    for regime, budgets in (
        ("well_joint", opt["well_budgets"]),
        ("well_pressure", opt["well_budgets"]),
        ("grid", opt["grid_budgets"]),
    ):
        n_units = 30 if regime != "grid" else 2 * b.n_active
        for budget in budgets:
            anchor = nearest_anchor(regime, budget)
            strategy = strategies["POD-E"][regime][str(anchor)]
            selected_order = order_for(pod_context, regime, strategy["placement"])
            rows = rows_for(regime, selected_order, budget, b)
            estimate_normalized = estimators.idw_residual(
                prior_normalized,
                rows,
                target[rows],
                b.ij.astype(float),
                b.n_active,
                cfg["idw_power"],
            )
            estimate = pod_context["scaler"].inv(estimate_normalized.reshape(2, b.n_active, -1))
            error = physical_error_summary(estimate, raw_test)
            records.append(
                {
                    "outer_fold": outer + 1,
                    "test_run": outer + 1,
                    "method": "prior+IDW",
                    "candidate": "none",
                    "regime": regime,
                    "budget": budget,
                    "measurements": len(rows),
                    "placement": strategy["placement"],
                    "solver": "IDW",
                    **error,
                    "score": normalized_joint_score(
                        error["rmse_p"],
                        error["rmse_so"],
                        prior_pressure=prior_error["rmse_p"],
                        prior_saturation=prior_error["rmse_so"],
                    ),
                    "condition": np.nan,
                    "online_seconds_per_snapshot": np.nan,
                    "iterations": 0,
                    "storage_floats": 0,
                    "memory_bytes": 0,
                    "basis_seconds": 0.0,
                    "config_json": "{}",
                    "solver_json": "{}",
                    "seed": -1,
                }
            )
            solver = strategy["solver"]
            for seed_offset in range(int(opt["random_placement_seeds"])):
                seed = int(cfg["master_seed"] + 10_000 * (outer + 1) + seed_offset)
                random_order = placement.random_order(n_units, seed)
                random_rows = rows_for(regime, random_order, budget, b)
                coefficients, metadata = solve_coefficients(
                    pod_context["basis"].B[random_rows], target[random_rows], solver
                )
                estimate = pod_context["scaler"].inv(
                    (pod_context["basis"].B @ coefficients).reshape(2, b.n_active, -1)
                )
                error = physical_error_summary(estimate, raw_test)
                records.append(
                    {
                        "outer_fold": outer + 1,
                        "test_run": outer + 1,
                        "method": "POD-E random",
                        "candidate": pod_context["basis"].extra["spec_id"],
                        "regime": regime,
                        "budget": budget,
                        "measurements": len(random_rows),
                        "placement": "random",
                        "solver": solver_id(solver),
                        **error,
                        "score": normalized_joint_score(
                            error["rmse_p"],
                            error["rmse_so"],
                            prior_pressure=prior_error["rmse_p"],
                            prior_saturation=prior_error["rmse_so"],
                        ),
                        "condition": float(np.linalg.cond(pod_context["basis"].B[random_rows])),
                        "online_seconds_per_snapshot": np.nan,
                        "iterations": metadata.get("iterations", 0),
                        "storage_floats": pod_context["basis"].extra["storage_floats"],
                        "memory_bytes": 8 * pod_context["basis"].extra["storage_floats"],
                        "basis_seconds": pod_context["basis"].seconds,
                        "config_json": "{}",
                        "solver_json": json.dumps(solver, sort_keys=True),
                        "seed": seed,
                    }
                )
    return records


def run_outer(outer: int) -> dict:
    cfg = load_config()
    selection = load_selection(outer)
    b = data.load(cfg["dataset"], cfg["wells"])
    registry, strategies = tune_outer(outer, selection, b, cfg)
    train_runs = [run - 1 for run in selection["train_runs"]]
    raw_fit = b.fields[train_runs]
    raw_test = b.fields[outer]
    contexts = {}
    context_by_candidate = {}
    for label in ALL_LABELS:
        spec = selection["selected"][label]
        if spec["id"] not in context_by_candidate:
            context_by_candidate[spec["id"]] = build_context(raw_fit, b, spec, cfg)
        contexts[label] = context_by_candidate[spec["id"]]
    outer_records = []
    for label in ALL_LABELS:
        strategy_label = label if label in TUNED_LABELS else "final_accuracy"
        output_label = OUTPUT_LABELS.get(label, label)
        outer_records.extend(
            evaluate_outer_method(
                outer,
                output_label,
                selection["selected"][label],
                contexts[label],
                raw_fit,
                raw_test,
                b,
                strategies,
                strategy_label,
            )
        )
        print(f"outer {outer + 1}: untouched sparse {output_label}", flush=True)
    outer_records.extend(
        evaluate_references_and_random(
            outer, raw_fit, raw_test, b, contexts["POD-E"], strategies, cfg
        )
    )
    rankings = []
    for label in ALL_LABELS:
        for placement_name, order in contexts[label]["orders"]["well"].items():
            rank = np.empty(len(order), dtype=int)
            rank[order] = np.arange(1, len(order) + 1)
            rankings.append(
                {
                    "outer_fold": outer + 1,
                    "method": label,
                    "placement": placement_name,
                    **{f"well_{index + 1}": int(value) for index, value in enumerate(rank)},
                }
            )
    clustering = {
        label: {
            "selected_k": int(contexts[label]["clustering"]["selected_k"]),
            "diagnostics": contexts[label]["clustering"]["diagnostics"],
        }
        for label in ALL_LABELS
    }
    return {
        "outer_fold": outer + 1,
        "registry": registry,
        "strategies": strategies,
        "outer_records": outer_records,
        "well_rankings": rankings,
        "clustering": clustering,
    }


def merge() -> None:
    paths = sorted(PARTS.glob("o2_fold*.json"))
    if len(paths) != 10:
        raise SystemExit(f"Expected 10 sparse parts, found {len(paths)}")
    parts = [json.loads(path.read_text()) for path in paths]
    registry = [row for part in parts for row in part["registry"]]
    outer = [row for part in parts for row in part["outer_records"]]
    rankings = [row for part in parts for row in part["well_rankings"]]
    pd.DataFrame(registry).to_csv(OPT_OUT / "sparse_registry.csv", index=False)
    pd.DataFrame(outer).to_csv(OPT_OUT / "outer_sparse.csv", index=False)
    pd.DataFrame(rankings).to_csv(OPT_OUT / "well_rankings.csv", index=False)
    (OPT_OUT / "selected_sparse_strategies.json").write_text(
        json.dumps(
            {str(part["outer_fold"]): part["strategies"] for part in parts},
            indent=2,
            sort_keys=True,
        )
    )
    (OPT_OUT / "clustering.json").write_text(
        json.dumps(
            {str(part["outer_fold"]): part["clustering"] for part in parts},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"merged {len(registry)} validation records and {len(outer)} outer records")


if __name__ == "__main__":
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        raise SystemExit("usage: o2_nested_sparse.py part OUTER_INDEX | merge")
    if sys.argv[1] == "part":
        outer_index = int(sys.argv[2])
        payload = run_outer(outer_index)
        (PARTS / f"o2_fold{outer_index + 1:02d}.json").write_text(json.dumps(payload, indent=2))
    elif sys.argv[1] == "merge":
        merge()
    else:
        raise SystemExit("expected part or merge")
