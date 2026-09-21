"""Independent refit/evaluation of the frozen train-selected configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bss import data, load_config  # noqa: E402
from bss.optimization import compare_reproduction_values  # noqa: E402
from o2_nested_sparse import (  # noqa: E402
    OPT_OUT,
    build_context,
    evaluate_outer_method,
    load_selection,
)

REPRO = OPT_OUT / "reproduction"
REPRO_PARTS = REPRO / "parts"
FROZEN_METHODS = ("final_accuracy", "final_pareto", "POD-E")
IDENTITY_COLUMNS = (
    "outer_fold",
    "method",
    "candidate",
    "regime",
    "budget",
    "measurements",
    "placement",
    "solver",
)
NUMERIC_COLUMNS = (
    "rmse_p",
    "rmse_so",
    "mae_p",
    "mae_so",
    "score",
    "condition",
    "iterations",
    "storage_floats",
    "memory_bytes",
)


def reproduce_outer(outer: int) -> list[dict]:
    cfg = load_config()
    selection = load_selection(outer)
    strategies = json.loads((OPT_OUT / "selected_sparse_strategies.json").read_text())[
        str(outer + 1)
    ]
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run - 1 for run in selection["train_runs"]]
    raw_fit = brugge.fields[train_runs]
    raw_test = brugge.fields[outer]
    records: list[dict] = []
    for label in FROZEN_METHODS:
        spec = selection["selected"][label]
        context = build_context(raw_fit, brugge, spec, cfg)
        records.extend(
            evaluate_outer_method(
                outer,
                label,
                spec,
                context,
                raw_fit,
                raw_test,
                brugge,
                strategies,
                label,
            )
        )
        print(f"reproduced outer {outer + 1}: {label}", flush=True)
    return records


def merge_and_compare() -> dict:
    paths = sorted(REPRO_PARTS.glob("fold*.json"))
    if len(paths) != 10:
        raise SystemExit(f"Expected 10 reproduction parts, found {len(paths)}")
    rows = [row for path in paths for row in json.loads(path.read_text())]
    reproduced = pd.DataFrame(rows)
    reproduced.to_csv(REPRO / "outer_sparse.csv", index=False)
    canonical = pd.read_csv(OPT_OUT / "outer_sparse.csv")
    canonical = canonical[
        canonical["method"].isin(FROZEN_METHODS) & (canonical["seed"] == -1)
    ].copy()
    keys = ["outer_fold", "method", "regime", "budget"]
    canonical = canonical.sort_values(keys).reset_index(drop=True)
    reproduced = reproduced.sort_values(keys).reset_index(drop=True)
    identity_ok = len(canonical) == len(reproduced)
    identity_mismatches: dict[str, int] = {}
    if identity_ok:
        for column in IDENTITY_COLUMNS:
            mismatches = int(
                (canonical[column].astype(str) != reproduced[column].astype(str)).sum()
            )
            identity_mismatches[column] = mismatches
            identity_ok &= mismatches == 0
    numeric_differences: dict[str, dict[str, float | bool]] = {}
    numeric_ok = len(canonical) == len(reproduced)
    if numeric_ok:
        for column in NUMERIC_COLUMNS:
            tolerance = (
                {"atol": 0.0, "rtol": 1e-12}
                if column == "condition"
                else {"atol": 1e-10, "rtol": 0.0}
            )
            comparison = compare_reproduction_values(
                canonical[column].to_numpy(dtype=float),
                reproduced[column].to_numpy(dtype=float),
                **tolerance,
            )
            numeric_differences[column] = {**comparison, **tolerance}
            numeric_ok &= bool(comparison["matches"])
    payload = {
        "status": "pass" if identity_ok and numeric_ok else "fail",
        "canonical_rows": int(len(canonical)),
        "reproduced_rows": int(len(reproduced)),
        "identity_mismatches": identity_mismatches,
        "numeric_comparisons": numeric_differences,
        "runtime_columns_compared": False,
    }
    (REPRO / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: o4_reproduce_frozen.py part <zero-based-fold> | merge")
    REPRO_PARTS.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "part":
        outer = int(sys.argv[2])
        (REPRO_PARTS / f"fold{outer + 1:02d}.json").write_text(
            json.dumps(reproduce_outer(outer), indent=2, sort_keys=True)
        )
    elif sys.argv[1] == "merge":
        print(json.dumps(merge_and_compare(), indent=2, sort_keys=True))
    else:
        raise SystemExit(f"unknown mode: {sys.argv[1]}")


if __name__ == "__main__":
    main()
