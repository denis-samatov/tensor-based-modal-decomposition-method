"""Nested E9 selection and untouched outer evaluation.

Run inner INDEX for each zero-based outer fold, then merge-inner.
Only after every selection file exists, run outer INDEX and merge-outer.
The script never reads outer metrics during selection.
"""

from __future__ import annotations

import hashlib
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bss import OUT, bases, data, load_config, placement  # noqa: E402
from bss.e9_metrics import MetricReference, evaluate_reconstruction  # noqa: E402
from bss.e9_study import (  # noqa: E402
    E9Transform,
    build_e9_candidate_specs,
    fit_e9_model,
    headline_selection_loss,
)
from bss.optimization import inner_scenario_splits  # noqa: E402
from bss.study import solve_coefficients, solver_id  # noqa: E402

DEST = OUT / "e9_property_mode"
PARTS = DEST / "parts"
PROTOCOL = Path(__file__).resolve().parents[1] / "e9_protocol.json"
PREREGISTRATION = Path(__file__).resolve().parents[3] / "E9_PREREGISTRATION.md"
TIME_INDICES = np.unique(np.rint(np.linspace(0, 132, 12)).astype(int))


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def _protocol_provenance() -> dict[str, str]:
    return {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
    }


def _metric_reference_subset(reference: MetricReference, indices: np.ndarray) -> MetricReference:
    return MetricReference(
        ensemble_mean=reference.ensemble_mean[..., indices],
        raw_range=reference.raw_range,
        anomaly_range=reference.anomaly_range,
    )


def _baseline_subset(transform: E9Transform, indices: np.ndarray) -> np.ndarray:
    if transform.baseline.shape[-1] == 1:
        return transform.baseline
    return transform.baseline[..., indices]


def _transform_subset(
    transform: E9Transform, fields: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    return (fields[..., indices] - _baseline_subset(transform, indices)) / transform.scale


def _inverse_subset(
    transform: E9Transform, fields: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    return fields * transform.scale + _baseline_subset(transform, indices)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _evaluate(
    raw_fit: np.ndarray,
    raw_validation: np.ndarray,
    active_mask: np.ndarray,
    transform: E9Transform,
    model: bases.Basis,
    *,
    indices: np.ndarray,
    rows: np.ndarray | None = None,
    solver: dict | None = None,
) -> dict[str, float]:
    reference = MetricReference.fit(raw_fit)
    subset_reference = _metric_reference_subset(reference, indices)
    metrics = []
    condition = 1.0
    for validation in raw_validation:
        target = _transform_subset(transform, validation, indices).reshape(-1, len(indices))
        if rows is None:
            coefficients = np.linalg.lstsq(model.B, target, rcond=None)[0]
        else:
            coefficients, _ = solve_coefficients(model.B[rows], target[rows], solver)
            condition = float(np.linalg.cond(model.B[rows]))
        estimate_scaled = (model.B @ coefficients).reshape(2, -1, len(indices))
        estimate = _inverse_subset(transform, estimate_scaled, indices)
        scores = evaluate_reconstruction(
            estimate,
            validation[..., indices],
            subset_reference,
            active_mask,
        )
        metrics.append(scores)
    result = _mean_metrics(metrics)
    result["selection_loss"] = headline_selection_loss(result)
    result["condition_number"] = condition
    return result


def _common_grid_order(raw_fit: np.ndarray, ij: np.ndarray, max_budget: int) -> np.ndarray:
    transform = E9Transform.fit(raw_fit, "standardized_anomaly")
    transformed = transform.fwd(raw_fit)
    matrix = np.concatenate(
        [run.reshape(-1, run.shape[-1]) for run in transformed], axis=1
    )
    reference = bases.pod_energy(matrix, r=min(64, min(matrix.shape)))
    return placement.qr_dg(reference.B, max_budget)


def _measurement_rows(
    geometry: str,
    budget: int,
    *,
    n_active: int,
    well_rows: np.ndarray,
    grid_order: np.ndarray,
) -> np.ndarray:
    if geometry == "wells":
        cells = well_rows[:budget]
        return np.concatenate([cells, cells + n_active])
    if geometry == "grid":
        return grid_order[:budget]
    raise ValueError(f"unknown geometry: {geometry}")


def _representation_selection(frame: pd.DataFrame) -> list[dict]:
    selected = []
    valid = frame[frame.status == "ok"]
    for (capacity, variant), group in valid.groupby(["capacity_regime", "variant"]):
        summary = (
            group.groupby("candidate", as_index=False)
            .agg(
                selection_loss=("selection_loss", "mean"),
                storage_floats=("storage_floats", "mean"),
            )
            .sort_values(["selection_loss", "storage_floats", "candidate"])
        )
        row = summary.iloc[0]
        selected.append(
            {
                "capacity_regime": capacity,
                "variant": variant,
                "candidate": row.candidate,
                "inner_selection_loss": float(row.selection_loss),
            }
        )
    return selected


def _sparse_selection(frame: pd.DataFrame) -> list[dict]:
    selected = []
    valid = frame[frame.status == "ok"]
    group_columns = ["capacity_regime", "variant", "geometry", "budget"]
    for keys, group in valid.groupby(group_columns):
        summary = (
            group.groupby(["candidate", "solver"], as_index=False)
            .selection_loss.mean()
            .sort_values(["selection_loss", "candidate", "solver"])
        )
        row = summary.iloc[0]
        selected.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "candidate": row.candidate,
                "solver": row.solver,
                "inner_selection_loss": float(row.selection_loss),
            }
        )
    return selected


def _paired_sparse_selection(
    frame: pd.DataFrame, representation: list[dict]
) -> list[dict]:
    """Choose one common solver for every matched 3D-versus-4D pair."""

    candidate = {
        (row["capacity_regime"], row["variant"]): row["candidate"]
        for row in representation
    }
    pairs = [
        ("3D-independent", variant)
        for variant in ("4D-A", "4D-B", "4D-C", "4D-D", "4D-E")
    ] + [("3D-current", "4D-A-current")]
    selected = []
    valid = frame[frame.status == "ok"]
    for capacity in ("equal_total", "equal_per_property"):
        for geometry, budgets in (
            ("wells", (5, 10, 15, 20, 30)),
            ("grid", (30, 100, 300)),
        ):
            for budget in budgets:
                for three_variant, four_variant in pairs:
                    candidate_3d = candidate[(capacity, three_variant)]
                    candidate_4d = candidate[(capacity, four_variant)]
                    subset = valid[
                        (valid.capacity_regime == capacity)
                        & (valid.geometry == geometry)
                        & (valid.budget == budget)
                        & valid.candidate.isin([candidate_3d, candidate_4d])
                    ]
                    means = (
                        subset.groupby(["candidate", "solver"], as_index=False)
                        .selection_loss.mean()
                    )
                    solver_rows = []
                    for solver in sorted(set(means.solver)):
                        values = means[means.solver == solver].set_index("candidate")
                        if candidate_3d not in values.index or candidate_4d not in values.index:
                            continue
                        solver_rows.append(
                            {
                                "solver": solver,
                                "pair_inner_loss": float(
                                    0.5
                                    * (
                                        values.loc[candidate_3d, "selection_loss"]
                                        + values.loc[candidate_4d, "selection_loss"]
                                    )
                                ),
                            }
                        )
                    winner = min(
                        solver_rows, key=lambda row: (row["pair_inner_loss"], row["solver"])
                    )
                    selected.append(
                        {
                            "capacity_regime": capacity,
                            "geometry": geometry,
                            "budget": budget,
                            "three_d_variant": three_variant,
                            "four_d_variant": four_variant,
                            "candidate_3d": candidate_3d,
                            "candidate_4d": candidate_4d,
                            **winner,
                        }
                    )
    return selected


def _solver_lookup(protocol: dict) -> dict[str, dict]:
    return {solver_id(spec): spec for spec in protocol["recovery_search"]}


def run_inner(outer: int) -> None:
    started = time.perf_counter()
    cfg = load_config()
    protocol = _protocol()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run for run in range(brugge.n_runs) if run != outer]
    specs = build_e9_candidate_specs()
    specs_by_id = {spec["id"]: spec for spec in specs}
    representation_records: list[dict] = []
    split_data = []
    for inner, (fit_runs, validation_runs) in enumerate(inner_scenario_splits(train_runs, 3)):
        raw_fit = brugge.fields[fit_runs]
        raw_validation = brugge.fields[validation_runs]
        split_data.append((fit_runs, validation_runs, raw_fit, raw_validation))
        for position, spec in enumerate(specs):
            fit_started = time.perf_counter()
            try:
                transform, model, _ = fit_e9_model(raw_fit, brugge.ij, spec)
                scores = _evaluate(
                    raw_fit,
                    raw_validation,
                    brugge.active,
                    transform,
                    model,
                    indices=TIME_INDICES,
                )
                representation_records.append(
                    {
                        "outer_fold": outer + 1,
                        "inner_fold": inner + 1,
                        "candidate": spec["id"],
                        "order": spec["order"],
                        "variant": spec["variant"],
                        "capacity_regime": spec["capacity_regime"],
                        "coefficient_budget": spec["coefficient_budget"],
                        "architecture": spec["architecture"],
                        "preprocessing": spec["preprocessing"],
                        "spatial_ranks": json.dumps(spec["spatial_ranks"]),
                        "property_rank": spec.get("property_rank", np.nan),
                        "property_weights": json.dumps(spec["property_weights"]),
                        "storage_floats": model.extra["storage_floats"],
                        "fit_seconds": model.seconds,
                        "elapsed_seconds": time.perf_counter() - fit_started,
                        "status": "ok",
                        "error": "",
                        **scores,
                    }
                )
            except Exception as error:
                representation_records.append(
                    {
                        "outer_fold": outer + 1,
                        "inner_fold": inner + 1,
                        "candidate": spec["id"],
                        "order": spec["order"],
                        "variant": spec["variant"],
                        "capacity_regime": spec["capacity_regime"],
                        "coefficient_budget": spec["coefficient_budget"],
                        "architecture": spec["architecture"],
                        "preprocessing": spec["preprocessing"],
                        "spatial_ranks": json.dumps(spec["spatial_ranks"]),
                        "property_rank": spec.get("property_rank", np.nan),
                        "property_weights": json.dumps(spec["property_weights"]),
                        "storage_floats": np.nan,
                        "fit_seconds": np.nan,
                        "elapsed_seconds": time.perf_counter() - fit_started,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if (position + 1) % 12 == 0:
                print(
                    f"outer {outer + 1} inner {inner + 1} representation "
                    f"{position + 1}/{len(specs)}",
                    flush=True,
                )
    representation = pd.DataFrame(representation_records)
    selected_representation = _representation_selection(representation)

    solvers = protocol["recovery_search"]
    sparse_records: list[dict] = []
    for inner, (_, _, raw_fit, raw_validation) in enumerate(split_data):
        grid_order = _common_grid_order(raw_fit, brugge.ij, 300)
        for rep in selected_representation:
            spec = specs_by_id[rep["candidate"]]
            try:
                transform, model, _ = fit_e9_model(raw_fit, brugge.ij, spec)
            except Exception as error:
                sparse_records.append(
                    {
                        "outer_fold": outer + 1,
                        "inner_fold": inner + 1,
                        "candidate": spec["id"],
                        "variant": spec["variant"],
                        "capacity_regime": spec["capacity_regime"],
                        "geometry": "fit",
                        "budget": 0,
                        "solver": "",
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            for geometry, budgets in (
                ("wells", (5, 10, 15, 20, 30)),
                ("grid", (30, 100, 300)),
            ):
                for budget in budgets:
                    measurement_rows = _measurement_rows(
                        geometry,
                        budget,
                        n_active=brugge.n_active,
                        well_rows=brugge.well_rows,
                        grid_order=grid_order,
                    )
                    for solver in solvers:
                        identifier = solver_id(solver)
                        try:
                            scores = _evaluate(
                                raw_fit,
                                raw_validation,
                                brugge.active,
                                transform,
                                model,
                                indices=TIME_INDICES,
                                rows=measurement_rows,
                                solver=solver,
                            )
                            sparse_records.append(
                                {
                                    "outer_fold": outer + 1,
                                    "inner_fold": inner + 1,
                                    "candidate": spec["id"],
                                    "order": spec["order"],
                                    "variant": spec["variant"],
                                    "capacity_regime": spec["capacity_regime"],
                                    "coefficient_budget": spec["coefficient_budget"],
                                    "geometry": geometry,
                                    "budget": budget,
                                    "channels": len(measurement_rows),
                                    "solver": identifier,
                                    "storage_floats": model.extra["storage_floats"],
                                    "status": "ok",
                                    "error": "",
                                    **scores,
                                }
                            )
                        except Exception as error:
                            sparse_records.append(
                                {
                                    "outer_fold": outer + 1,
                                    "inner_fold": inner + 1,
                                    "candidate": spec["id"],
                                    "order": spec["order"],
                                    "variant": spec["variant"],
                                    "capacity_regime": spec["capacity_regime"],
                                    "coefficient_budget": spec["coefficient_budget"],
                                    "geometry": geometry,
                                    "budget": budget,
                                    "channels": len(measurement_rows),
                                    "solver": identifier,
                                    "storage_floats": model.extra["storage_floats"],
                                    "status": "failed",
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
            print(
                f"outer {outer + 1} inner {inner + 1} sparse {spec['variant']} "
                f"{spec['capacity_regime']}",
                flush=True,
            )
    sparse = pd.DataFrame(sparse_records)
    selected_sparse = _sparse_selection(sparse)
    selected_paired_sparse = _paired_sparse_selection(
        sparse, selected_representation
    )
    PARTS.mkdir(parents=True, exist_ok=True)
    representation.to_csv(PARTS / f"inner_representation_{outer + 1:02d}.csv", index=False)
    sparse.to_csv(PARTS / f"inner_sparse_{outer + 1:02d}.csv", index=False)
    candidates = {row["candidate"] for row in selected_representation}
    selection = {
        "outer_fold": outer + 1,
        "test_run": outer + 1,
        "train_runs": [run + 1 for run in train_runs],
        "time_indices": TIME_INDICES.tolist(),
        "protocol_id": protocol["protocol_id"],
        **_protocol_provenance(),
        "representation": selected_representation,
        "sparse": selected_sparse,
        "paired_sparse": selected_paired_sparse,
        "specs": {candidate: specs_by_id[candidate] for candidate in candidates},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (PARTS / f"selection_{outer + 1:02d}.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(
        f"outer {outer + 1} selection complete: {len(representation)} representation, "
        f"{len(sparse)} sparse records in {selection['elapsed_seconds']:.1f}s",
        flush=True,
    )


def reselect(outer: int) -> None:
    representation_path = PARTS / f"inner_representation_{outer + 1:02d}.csv"
    sparse_path = PARTS / f"inner_sparse_{outer + 1:02d}.csv"
    selection_path = PARTS / f"selection_{outer + 1:02d}.json"
    representation = pd.read_csv(representation_path)
    sparse = pd.read_csv(sparse_path)
    selection = json.loads(selection_path.read_text())
    selected_representation = _representation_selection(representation)
    selection["representation"] = selected_representation
    selection["sparse"] = _sparse_selection(sparse)
    selection["paired_sparse"] = _paired_sparse_selection(
        sparse, selected_representation
    )
    selection.update(_protocol_provenance())
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"outer {outer + 1} selections rebuilt with common paired solvers")


def merge_inner() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    representation_files = [
        PARTS / f"inner_representation_{fold:02d}.csv" for fold in range(1, 11)
    ]
    sparse_files = [PARTS / f"inner_sparse_{fold:02d}.csv" for fold in range(1, 11)]
    selection_files = [PARTS / f"selection_{fold:02d}.json" for fold in range(1, 11)]
    missing = [
        path
        for path in representation_files + sparse_files + selection_files
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing E9 inner parts: {[path.name for path in missing]}")
    pd.concat([pd.read_csv(path) for path in representation_files], ignore_index=True).to_csv(
        DEST / "inner_representation_registry.csv", index=False
    )
    pd.concat([pd.read_csv(path) for path in sparse_files], ignore_index=True).to_csv(
        DEST / "inner_sparse_registry.csv", index=False
    )
    selections = [json.loads(path.read_text()) for path in selection_files]
    (DEST / "selected_configurations.json").write_text(
        json.dumps(selections, indent=2) + "\n"
    )
    print("merged E9 inner registries and frozen selections")


def run_outer(outer: int) -> None:
    selection_path = PARTS / f"selection_{outer + 1:02d}.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"inner selection is not frozen: {selection_path}")
    selection = json.loads(selection_path.read_text())
    cfg = load_config()
    protocol = _protocol()
    solver_lookup = _solver_lookup(protocol)
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run for run in range(brugge.n_runs) if run != outer]
    raw_fit = brugge.fields[train_runs]
    raw_test = brugge.fields[outer]
    specs_by_id = {spec["id"]: spec for spec in build_e9_candidate_specs()}
    reference = MetricReference.fit(raw_fit)
    grid_order = _common_grid_order(raw_fit, brugge.ij, 300)
    model_cache: dict[str, tuple[E9Transform, bases.Basis]] = {}
    evaluation_cache: dict[tuple[str, str, int, str], tuple[dict, dict, float]] = {}

    def fitted(candidate: str):
        if candidate not in model_cache:
            transform, model, _ = fit_e9_model(raw_fit, brugge.ij, specs_by_id[candidate])
            model_cache[candidate] = (transform, model)
        return model_cache[candidate]

    records = []
    for selected in selection["representation"]:
        spec = specs_by_id[selected["candidate"]]
        transform, model = fitted(spec["id"])
        target = transform.fwd(raw_test).reshape(-1, brugge.T)
        coefficients = np.linalg.lstsq(model.B, target, rcond=None)[0]
        estimate = transform.inv(
            (model.B @ coefficients).reshape(2, brugge.n_active, brugge.T)
        )
        scores = evaluate_reconstruction(estimate, raw_test, reference, brugge.active)
        records.append(
            {
                "outer_fold": outer + 1,
                "test_run": outer + 1,
                "candidate": spec["id"],
                "order": spec["order"],
                "variant": spec["variant"],
                "capacity_regime": spec["capacity_regime"],
                "coefficient_budget": spec["coefficient_budget"],
                "architecture": spec["architecture"],
                "preprocessing": spec["preprocessing"],
                "spatial_ranks": json.dumps(spec["spatial_ranks"]),
                "property_rank": spec.get("property_rank", np.nan),
                "property_weights": json.dumps(spec["property_weights"]),
                "geometry": "full",
                "budget": 9900,
                "channels": 9900,
                "solver": "projection",
                "storage_floats": model.extra["storage_floats"],
                "fit_seconds": model.seconds,
                "condition_number": 1.0,
                "selection_loss": headline_selection_loss(scores),
                **scores,
            }
        )
    for selected in selection["paired_sparse"]:
        measurement_rows = _measurement_rows(
            selected["geometry"],
            int(selected["budget"]),
            n_active=brugge.n_active,
            well_rows=brugge.well_rows,
            grid_order=grid_order,
        )
        solver = solver_lookup[selected["solver"]]
        for role, candidate in (
            ("3D", selected["candidate_3d"]),
            ("4D", selected["candidate_4d"]),
        ):
            spec = specs_by_id[candidate]
            transform, model = fitted(candidate)
            cache_key = (
                candidate,
                selected["geometry"],
                int(selected["budget"]),
                selected["solver"],
            )
            if cache_key not in evaluation_cache:
                target = transform.fwd(raw_test).reshape(-1, brugge.T)
                coefficients, metadata = solve_coefficients(
                    model.B[measurement_rows], target[measurement_rows], solver
                )
                estimate = transform.inv(
                    (model.B @ coefficients).reshape(2, brugge.n_active, brugge.T)
                )
                scores = evaluate_reconstruction(
                    estimate, raw_test, reference, brugge.active
                )
                condition = float(np.linalg.cond(model.B[measurement_rows]))
                evaluation_cache[cache_key] = (scores, metadata, condition)
            scores, metadata, condition = evaluation_cache[cache_key]
            records.append(
                {
                    "outer_fold": outer + 1,
                    "test_run": outer + 1,
                    "candidate": spec["id"],
                    "order": spec["order"],
                    "variant": spec["variant"],
                    "capacity_regime": spec["capacity_regime"],
                    "coefficient_budget": spec["coefficient_budget"],
                    "architecture": spec["architecture"],
                    "preprocessing": spec["preprocessing"],
                    "spatial_ranks": json.dumps(spec["spatial_ranks"]),
                    "property_rank": spec.get("property_rank", np.nan),
                    "property_weights": json.dumps(spec["property_weights"]),
                    "geometry": selected["geometry"],
                    "budget": int(selected["budget"]),
                    "channels": len(measurement_rows),
                    "solver": selected["solver"],
                    "pair_role": role,
                    "pair_3d_variant": selected["three_d_variant"],
                    "pair_4d_variant": selected["four_d_variant"],
                    "pair_inner_loss": selected["pair_inner_loss"],
                    "storage_floats": model.extra["storage_floats"],
                    "fit_seconds": model.seconds,
                    "condition_number": condition,
                    "iterations": metadata["iterations"],
                    "selection_loss": headline_selection_loss(scores),
                    **scores,
                }
            )
    frame = pd.DataFrame(records)
    PARTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PARTS / f"outer_{outer + 1:02d}.csv", index=False)
    print(f"outer {outer + 1} evaluated once: {len(frame)} records", flush=True)


def merge_outer() -> None:
    files = [PARTS / f"outer_{fold:02d}.csv" for fold in range(1, 11)]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E9 outer parts: {[path.name for path in missing]}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(DEST / "outer_results.csv", index=False)
    print(f"merged {len(frame)} untouched E9 outer records")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "expected inner INDEX, reselect INDEX, merge-inner, outer INDEX, or merge-outer"
        )
    command = sys.argv[1]
    if command == "inner":
        run_inner(int(sys.argv[2]))
    elif command == "reselect":
        reselect(int(sys.argv[2]))
    elif command == "merge-inner":
        merge_inner()
    elif command == "outer":
        run_outer(int(sys.argv[2]))
    elif command == "merge-outer":
        merge_outer()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
