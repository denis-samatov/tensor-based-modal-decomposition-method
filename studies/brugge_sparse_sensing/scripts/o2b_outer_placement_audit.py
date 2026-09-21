"""Prespecified untouched comparison of all placement objectives after nested solver tuning."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bss import data, load_config  # noqa: E402
from bss.optimization import normalized_joint_score  # noqa: E402
from bss.study import physical_error_summary, solve_coefficients, transform_runs  # noqa: E402
from o2_nested_sparse import (  # noqa: E402
    OPT_OUT,
    build_context,
    load_selection,
    nearest_anchor,
    order_for,
    placement_names,
    rows_for,
)

PARTS = OPT_OUT / "placement_audit_parts"
METHODS = ("final_accuracy", "POD-E")


def run_outer(outer: int) -> list[dict]:
    cfg = load_config()
    selection = load_selection(outer)
    strategies = json.loads((OPT_OUT / "selected_sparse_strategies.json").read_text())[
        str(outer + 1)
    ]
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run - 1 for run in selection["train_runs"]]
    raw_fit = brugge.fields[train_runs]
    raw_test = brugge.fields[outer]
    prior_error = physical_error_summary(raw_fit.mean(axis=0), raw_test)
    records: list[dict] = []
    for method in METHODS:
        spec = selection["selected"][method]
        context_start = time.perf_counter()
        context = build_context(raw_fit, brugge, spec, cfg)
        context_seconds = time.perf_counter() - context_start
        target = transform_runs(raw_test, context["scaler"])
        for regime, budgets in (
            ("well_joint", cfg["optimization"]["well_budgets"]),
            ("well_pressure", cfg["optimization"]["well_budgets"]),
            ("grid", cfg["optimization"]["grid_budgets"]),
        ):
            for budget in budgets:
                anchor = nearest_anchor(regime, budget)
                solver = strategies[method][regime][str(anchor)]["solver"]
                for placement_name in placement_names(context, regime):
                    order = order_for(context, regime, placement_name)
                    rows = rows_for(regime, order, budget, brugge)
                    start = time.perf_counter()
                    coefficients, metadata = solve_coefficients(
                        context["basis"].B[rows], target[rows], solver
                    )
                    elapsed = time.perf_counter() - start
                    estimate = context["scaler"].inv(
                        (context["basis"].B @ coefficients).reshape(2, brugge.n_active, -1)
                    )
                    error = physical_error_summary(estimate, raw_test)
                    records.append(
                        {
                            "outer_fold": outer + 1,
                            "method": method,
                            "regime": regime,
                            "budget": budget,
                            "measurements": len(rows),
                            "placement": placement_name,
                            **error,
                            "score": normalized_joint_score(
                                error["rmse_p"],
                                error["rmse_so"],
                                prior_pressure=prior_error["rmse_p"],
                                prior_saturation=prior_error["rmse_so"],
                            ),
                            "condition": float(np.linalg.cond(context["basis"].B[rows])),
                            "online_seconds_per_snapshot": elapsed / raw_test.shape[-1],
                            "basis_seconds": context["basis"].seconds,
                            "basis_clustering_placement_seconds": context_seconds,
                            "iterations": metadata.get("iterations", 0),
                            "solver_json": json.dumps(solver, sort_keys=True),
                            "config_json": json.dumps(spec, sort_keys=True),
                        }
                    )
        print(f"placement audit outer {outer + 1}: {method}", flush=True)
    return records


def merge() -> None:
    paths = sorted(PARTS.glob("fold*.json"))
    if len(paths) != 10:
        raise SystemExit(f"Expected 10 placement-audit parts, found {len(paths)}")
    rows = [row for path in paths for row in json.loads(path.read_text())]
    pd.DataFrame(rows).to_csv(OPT_OUT / "outer_placement_audit.csv", index=False)
    print(f"merged {len(rows)} placement-audit rows")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: o2b_outer_placement_audit.py part <zero-based-fold> | merge")
    PARTS.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "part":
        outer = int(sys.argv[2])
        (PARTS / f"fold{outer + 1:02d}.json").write_text(
            json.dumps(run_outer(outer), indent=2, sort_keys=True)
        )
    elif sys.argv[1] == "merge":
        merge()
    else:
        raise SystemExit(f"unknown mode: {sys.argv[1]}")


if __name__ == "__main__":
    main()
