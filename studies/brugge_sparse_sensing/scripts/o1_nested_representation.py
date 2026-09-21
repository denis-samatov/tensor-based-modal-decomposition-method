"""Stage A: nested, train-only selection of TBMD representation configurations.

Each outer fold holds one scenario untouched. Three deterministic inner folds
partition the remaining nine scenarios. Structural candidates are screened by
full-observation validation, then the best train-only candidates are refined
over scaling and property weights. Frozen configurations are finally refit on
all nine outer-training scenarios and evaluated once on the held-out scenario.
"""

from __future__ import annotations

import copy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bss import OUT, bases, data, load_config  # noqa: E402
from bss.optimization import (  # noqa: E402
    PropertyScaler,
    inner_scenario_splits,
    normalized_joint_score,
    select_candidate,
)
from bss.study import (  # noqa: E402
    build_representation_specs,
    fit_representation,
    physical_error_summary,
    transform_runs,
)

OPT_OUT = OUT / "optimization"
PARTS = OPT_OUT / "parts"


def canonical_id(spec: dict) -> str:
    payload = json.dumps({key: value for key, value in spec.items() if key != "id"}, sort_keys=True)
    label = f"{spec['family']}-{spec['architecture']}"
    return f"{label}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def with_id(spec: dict) -> dict:
    updated = copy.deepcopy(spec)
    updated["id"] = canonical_id(updated)
    return updated


def evaluate_projection(
    raw_fit: np.ndarray, raw_validation: np.ndarray, scaler, basis: bases.Basis
) -> dict[str, float]:
    n = raw_fit.shape[2]
    prior = raw_fit.mean(axis=0)
    metrics = []
    for validation_run in raw_validation:
        target = transform_runs(validation_run, scaler)
        coefficients = np.linalg.lstsq(basis.B, target, rcond=None)[0]
        estimate_normalized = basis.B @ coefficients
        estimate = scaler.inv(estimate_normalized.reshape(2, n, -1))
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
        metrics.append(error)
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def matrix_cache(raw_fit: np.ndarray, specs: list[dict]) -> dict[str, tuple]:
    """Reuse one exact SVD for all POD/POD-E depths on an inner split."""
    cache: dict[str, tuple] = {}
    for scaling in sorted({spec["scaling"] for spec in specs if spec["family"].startswith("POD")}):
        scaler = PropertyScaler.fit(raw_fit, scaling)
        train = transform_runs(raw_fit, scaler)
        start = time.perf_counter()
        left, singular, _ = np.linalg.svd(train, full_matrices=False)
        seconds = time.perf_counter() - start
        cache[scaling] = (scaler, train, left, singular, seconds)
    return cache


def fit_spec(raw_fit: np.ndarray, b: data.Brugge, spec: dict, cache: dict[str, tuple]):
    if spec["family"] in {"POD", "POD-E"}:
        scaler, train, left, singular, seconds = cache[spec["scaling"]]
        if spec["rank_strategy"] == "energy":
            rank = bases.energy_rank(singular, spec["modal_energy"])
        else:
            rank = int(spec["modal_rank"])
        matrix = left[:, :rank].copy()
        if spec["family"] == "POD-E":
            matrix *= singular[:rank]
        model = bases.Basis(
            spec["family"],
            matrix,
            singular,
            seconds,
            {
                "rank": rank,
                "storage_floats": int(matrix.size),
                "spec_id": spec["id"],
                "scaling": spec["scaling"],
            },
        )
        return scaler, model, train
    enriched = {
        **spec,
        "svd_iterations": load_config()["optimization"]["randomized_svd_iterations"],
    }
    return fit_representation(raw_fit, b.ij, enriched)


def evaluate_specs(
    outer: int,
    inner: int,
    raw_fit: np.ndarray,
    raw_validation: np.ndarray,
    b: data.Brugge,
    specs: list[dict],
    stage: str,
) -> list[dict]:
    registry = []
    cache = matrix_cache(raw_fit, specs)
    for number, spec in enumerate(specs):
        started = time.perf_counter()
        try:
            scaler, basis, _ = fit_spec(raw_fit, b, spec, cache)
            score = evaluate_projection(raw_fit, raw_validation, scaler, basis)
            registry.append(
                {
                    "stage": stage,
                    "outer_fold": outer + 1,
                    "inner_fold": inner + 1,
                    "candidate": spec["id"],
                    "family": spec["family"],
                    "architecture": spec["architecture"],
                    "scaling": spec["scaling"],
                    "rank_strategy": spec["rank_strategy"],
                    "score": score["score"],
                    "rmse_p": score["rmse_p"],
                    "rmse_so": score["rmse_so"],
                    "mae_p": score["mae_p"],
                    "mae_so": score["mae_so"],
                    "prior_rmse_p": score["prior_rmse_p"],
                    "prior_rmse_so": score["prior_rmse_so"],
                    "storage_floats": basis.extra["storage_floats"],
                    "fit_seconds": basis.seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "ok",
                    "error": "",
                    "config_json": json.dumps(spec, sort_keys=True),
                }
            )
        except Exception as error:  # keep failed configurations in the registry
            registry.append(
                {
                    "stage": stage,
                    "outer_fold": outer + 1,
                    "inner_fold": inner + 1,
                    "candidate": spec["id"],
                    "family": spec["family"],
                    "architecture": spec["architecture"],
                    "scaling": spec["scaling"],
                    "rank_strategy": spec["rank_strategy"],
                    "score": np.inf,
                    "rmse_p": np.nan,
                    "rmse_so": np.nan,
                    "mae_p": np.nan,
                    "mae_so": np.nan,
                    "prior_rmse_p": np.nan,
                    "prior_rmse_so": np.nan,
                    "storage_floats": np.iinfo(np.int64).max,
                    "fit_seconds": np.nan,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "config_json": json.dumps(spec, sort_keys=True),
                }
            )
        if (number + 1) % 10 == 0:
            print(
                f"outer {outer + 1} inner {inner + 1} {stage}: {number + 1}/{len(specs)}",
                flush=True,
            )
    return registry


def summaries(registry: list[dict]) -> list[dict]:
    frame = pd.DataFrame([row for row in registry if row["status"] == "ok"])
    result = []
    for candidate, group in frame.groupby("candidate"):
        result.append(
            {
                "candidate": candidate,
                "mean_score": float(group.score.mean()),
                "storage_floats": int(round(group.storage_floats.mean())),
            }
        )
    return sorted(result, key=lambda row: (row["mean_score"], row["storage_floats"]))


def selected_from(
    registry: list[dict],
    specs_by_id: dict[str, dict],
    *,
    family: str | None = None,
    architectures: set[str] | None = None,
    tolerance: float = 0.0,
) -> dict:
    filtered = [
        row
        for row in registry
        if row["status"] == "ok"
        and (family is None or row["family"] == family)
        and (architectures is None or row["architecture"] in architectures)
    ]
    selected = select_candidate(filtered, tolerance=tolerance)
    return specs_by_id[selected["candidate"]]


def refinement_specs(
    initial_registry: list[dict], specs_by_id: dict[str, dict], cfg: dict
) -> list[dict]:
    top = [
        row
        for row in summaries(initial_registry)
        if specs_by_id[row["candidate"]]["family"] == "TBMD"
    ][:3]
    variants = []
    for row in top:
        base = specs_by_id[row["candidate"]]
        for scaling in cfg["scaling_methods"]:
            for saturation_weight in cfg["property_weights"]:
                spec = copy.deepcopy(base)
                spec["scaling"] = scaling
                spec["property_weights"] = [1.0, float(saturation_weight)]
                variants.append(with_id(spec))
    return list({spec["id"]: spec for spec in variants}.values())


def outer_projection(
    outer: int,
    b: data.Brugge,
    raw_fit: np.ndarray,
    raw_test: np.ndarray,
    label: str,
    spec: dict,
    cfg: dict,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    if spec["family"] == "TBMD-HOOI":
        scaler = PropertyScaler.fit(raw_fit, "minmax")
        train = transform_runs(raw_fit, scaler)
        singular = np.linalg.svd(train, compute_uv=False)
        rank = bases.energy_rank(singular, cfg["energy_threshold"])
        basis = bases.tbmd(
            train,
            b.ij,
            rank,
            spatial_ranks=tuple(cfg["tbmd_spatial_ranks"]),
            **cfg["tucker"],
        )
        basis.extra["storage_floats"] = int(basis.extra["storage_floats"])
    else:
        scaler, basis, train = fit_spec(raw_fit, b, spec, matrix_cache(raw_fit, [spec]))
        rank = basis.B.shape[1]
    target = transform_runs(raw_test, scaler)
    coefficients = np.linalg.lstsq(basis.B, target, rcond=None)[0]
    estimate = scaler.inv((basis.B @ coefficients).reshape(2, b.n_active, -1))
    error = physical_error_summary(estimate, raw_test)
    prior = raw_fit.mean(axis=0)
    prior_error = physical_error_summary(prior, raw_test)
    record = {
        "outer_fold": outer + 1,
        "test_run": outer + 1,
        "label": label,
        "candidate": spec["id"],
        "family": spec["family"],
        "architecture": spec["architecture"],
        "rank": rank,
        **error,
        "prior_rmse_p": prior_error["rmse_p"],
        "prior_rmse_so": prior_error["rmse_so"],
        "score": normalized_joint_score(
            error["rmse_p"],
            error["rmse_so"],
            prior_pressure=prior_error["rmse_p"],
            prior_saturation=prior_error["rmse_so"],
        ),
        "storage_floats": basis.extra["storage_floats"],
        "storage_bytes": 8 * basis.extra["storage_floats"],
        "explicit_basis_floats": int(basis.B.size),
        "basis_compression": float(basis.B.size / basis.extra["storage_floats"]),
        "training_compression": float(train.size / basis.extra["storage_floats"]),
        "fit_seconds": basis.seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "config_json": json.dumps(spec, sort_keys=True),
    }
    return record, basis.extra


def run_outer(outer: int) -> dict:
    cfg = load_config()
    optimization = cfg["optimization"]
    b = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run for run in range(b.n_runs) if run != outer]
    splits = inner_scenario_splits(train_runs, optimization["inner_splits"])
    structural = [with_id(spec) for spec in build_representation_specs(optimization)]
    specs_by_id = {spec["id"]: spec for spec in structural}
    registry: list[dict] = []
    for inner, (fit_runs, validation_runs) in enumerate(splits):
        registry.extend(
            evaluate_specs(
                outer,
                inner,
                b.fields[fit_runs],
                b.fields[validation_runs],
                b,
                structural,
                "A1-structure",
            )
        )
    refinement = refinement_specs(registry, specs_by_id, optimization)
    specs_by_id.update({spec["id"]: spec for spec in refinement})
    for inner, (fit_runs, validation_runs) in enumerate(splits):
        registry.extend(
            evaluate_specs(
                outer,
                inner,
                b.fields[fit_runs],
                b.fields[validation_runs],
                b,
                refinement,
                "A2-scaling-weight",
            )
        )

    all_tbmd = [row for row in registry if row["family"] == "TBMD"]
    selected = {
        "final_accuracy": selected_from(all_tbmd, specs_by_id),
        "final_pareto": selected_from(
            all_tbmd,
            specs_by_id,
            tolerance=float(optimization["compression_tolerance"]),
        ),
        "rank_optimized_joint": selected_from(
            registry,
            specs_by_id,
            family="TBMD",
            architectures={"joint"},
        ),
        "property_decoupled": selected_from(
            registry,
            specs_by_id,
            family="TBMD",
            architectures={"shared-independent", "independent"},
        ),
        "hybrid": selected_from(
            registry,
            specs_by_id,
            family="TBMD",
            architectures={"joint-residual"},
        ),
        "POD": selected_from(registry, specs_by_id, family="POD"),
        "POD-E": selected_from(registry, specs_by_id, family="POD-E"),
    }
    original = with_id(
        {
            "family": "TBMD-HOOI",
            "architecture": "joint-original",
            "scaling": "minmax",
            "rank_strategy": "energy",
            "spatial_ranks": cfg["tbmd_spatial_ranks"][:2],
            "property_rank": 2,
            "modal_energy": cfg["energy_threshold"],
            "property_weights": [1.0, 1.0],
        }
    )
    selected["original_tbmd"] = original
    raw_fit = b.fields[train_runs]
    raw_test = b.fields[outer]
    outer_records = []
    basis_info = {}
    for label, spec in selected.items():
        record, info = outer_projection(outer, b, raw_fit, raw_test, label, spec, cfg)
        outer_records.append(record)
        basis_info[label] = info
        print(f"outer {outer + 1}: {label} done", flush=True)
    return {
        "outer_fold": outer + 1,
        "train_runs": [run + 1 for run in train_runs],
        "test_run": outer + 1,
        "inner_splits": [
            {
                "fit_runs": [run + 1 for run in fit],
                "validation_runs": [run + 1 for run in validation],
            }
            for fit, validation in splits
        ],
        "selected": selected,
        "registry": registry,
        "outer_records": outer_records,
        "basis_info": basis_info,
    }


def merge() -> None:
    paths = sorted(PARTS.glob("o1_fold*.json"))
    if len(paths) != 10:
        raise SystemExit(f"Expected 10 Stage-A parts, found {len(paths)}")
    parts = [json.loads(path.read_text()) for path in paths]
    registry = [row for part in parts for row in part["registry"]]
    outer = [row for part in parts for row in part["outer_records"]]
    pd.DataFrame(registry).to_csv(OPT_OUT / "representation_registry.csv", index=False)
    pd.DataFrame(outer).to_csv(OPT_OUT / "outer_representation.csv", index=False)
    selections = {
        str(part["outer_fold"]): {
            "train_runs": part["train_runs"],
            "test_run": part["test_run"],
            "inner_splits": part["inner_splits"],
            "selected": part["selected"],
            "basis_info": part["basis_info"],
        }
        for part in parts
    }
    (OPT_OUT / "selected_representations.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True)
    )
    print(f"merged {len(registry)} validation records and {len(outer)} untouched outer records")


if __name__ == "__main__":
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        raise SystemExit("usage: o1_nested_representation.py part OUTER_INDEX | merge")
    if sys.argv[1] == "part":
        outer_index = int(sys.argv[2])
        result = run_outer(outer_index)
        (PARTS / f"o1_fold{outer_index + 1:02d}.json").write_text(json.dumps(result, indent=2))
    elif sys.argv[1] == "merge":
        merge()
    else:
        raise SystemExit("expected part or merge")
