"""Frozen-final LASSO sensitivity declared by the E9 pre-outer amendment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bss import OUT, data, load_config  # noqa: E402
from bss.e9_metrics import MetricReference, evaluate_reconstruction  # noqa: E402
from bss.e9_study import build_e9_candidate_specs, fit_e9_model  # noqa: E402
from bss.study import solve_coefficients, solver_id  # noqa: E402
from e9_nested_property_mode import (  # noqa: E402
    _common_grid_order,
    _measurement_rows,
    _protocol,
)

DEST = OUT / "e9_property_mode"
PARTS = DEST / "lasso_parts"
FOUR_D_VARIANTS = {"4D-A", "4D-B", "4D-C", "4D-D", "4D-E"}


def run_fold(outer: int) -> None:
    protocol = _protocol()
    lasso = {
        key: value
        for key, value in protocol["sensitivity_recovery"][0].items()
        if key != "scope"
    }
    # The default 5,000-iteration diagnostic hit its cap in 140/320 fits.
    # This sensitivity does not select a model, so keep lambda/tolerance fixed
    # and raise only the convergence ceiling.
    lasso["max_iter"] = 20000
    selection = json.loads(
        (DEST / "parts" / f"selection_{outer + 1:02d}.json").read_text()
    )
    specs = {spec["id"]: spec for spec in build_e9_candidate_specs()}
    cfg = load_config()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run for run in range(brugge.n_runs) if run != outer]
    raw_fit = brugge.fields[train_runs]
    raw_test = brugge.fields[outer]
    reference = MetricReference.fit(raw_fit)
    grid_order = _common_grid_order(raw_fit, brugge.ij, 300)
    model_cache = {}
    rows = []
    for capacity in ("equal_total", "equal_per_property"):
        for geometry, budgets in (
            ("wells", (5, 10, 15, 20, 30)),
            ("grid", (30, 100, 300)),
        ):
            for budget in budgets:
                options = [
                    row
                    for row in selection["paired_sparse"]
                    if row["capacity_regime"] == capacity
                    and row["geometry"] == geometry
                    and row["budget"] == budget
                    and row["four_d_variant"] in FOUR_D_VARIANTS
                ]
                pair = min(
                    options,
                    key=lambda row: (row["pair_inner_loss"], row["four_d_variant"]),
                )
                measurement_rows = _measurement_rows(
                    geometry,
                    budget,
                    n_active=brugge.n_active,
                    well_rows=brugge.well_rows,
                    grid_order=grid_order,
                )
                for role, candidate in (
                    ("3D", pair["candidate_3d"]),
                    ("4D", pair["candidate_4d"]),
                ):
                    if candidate not in model_cache:
                        model_cache[candidate] = fit_e9_model(
                            raw_fit, brugge.ij, specs[candidate]
                        )[:2]
                    transform, model = model_cache[candidate]
                    target = transform.fwd(raw_test).reshape(-1, brugge.T)
                    coefficients, metadata = solve_coefficients(
                        model.B[measurement_rows], target[measurement_rows], lasso
                    )
                    estimate = transform.inv(
                        (model.B @ coefficients).reshape(
                            2, brugge.n_active, brugge.T
                        )
                    )
                    scores = evaluate_reconstruction(
                        estimate, raw_test, reference, brugge.active
                    )
                    rows.append(
                        {
                            "outer_fold": outer + 1,
                            "test_run": outer + 1,
                            "capacity_regime": capacity,
                            "geometry": geometry,
                            "budget": budget,
                            "role": role,
                            "candidate": candidate,
                            "variant": specs[candidate]["variant"],
                            "selected_4d_variant": pair["four_d_variant"],
                            "solver": solver_id(lasso),
                            "iterations": metadata["iterations"],
                            "condition_number": float(
                                np.linalg.cond(model.B[measurement_rows])
                            ),
                            **scores,
                        }
                    )
                    print(
                        f"outer {outer + 1} {capacity} {geometry} {budget} {role}",
                        flush=True,
                    )
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PARTS / f"fold_{outer + 1:02d}.csv", index=False)


def merge() -> None:
    paths = [PARTS / f"fold_{fold:02d}.csv" for fold in range(1, 11)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing LASSO sensitivity parts: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(DEST / "lasso_sensitivity.csv", index=False)
    index = ["outer_fold", "capacity_regime", "geometry", "budget"]
    metrics = [
        f"{name}_{prop}"
        for prop in ("p", "so")
        for name in ("anomaly_relative_frobenius", "anomaly_ssim", "rmse")
    ]
    wide = frame.pivot(index=index, columns="role", values=metrics).reset_index()
    paired = wide[index].copy()
    for metric in metrics:
        paired[f"delta_{metric}"] = wide[(metric, "4D")] - wide[(metric, "3D")]
    paired.to_csv(DEST / "lasso_sensitivity_paired.csv", index=False)
    cap_hits = int((frame["iterations"] >= 20000).sum())
    summary = {
        "records": int(len(frame)),
        "paired_comparisons": int(len(paired)),
        "fixed_lambda": 0.001,
        "max_iter": 20000,
        "convergence_cap_hits": cap_hits,
        "median_iterations": float(frame["iterations"].median()),
        "selection_use": "sensitivity_only_not_used_for_model_selection",
    }
    (DEST / "lasso_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"merged {len(frame)} frozen-final LASSO sensitivity records")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: e9_lasso_sensitivity.py part INDEX | merge")
    if sys.argv[1] == "part":
        run_fold(int(sys.argv[2]))
    elif sys.argv[1] == "merge":
        merge()
    else:
        raise SystemExit(f"unknown mode: {sys.argv[1]}")


if __name__ == "__main__":
    main()
