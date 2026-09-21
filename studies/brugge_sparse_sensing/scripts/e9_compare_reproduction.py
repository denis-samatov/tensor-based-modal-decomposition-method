"""Compare an independent E9 outer rerun with the canonical frozen output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
CANONICAL = STUDY / "outputs" / "e9_property_mode" / "outer_results.csv"
VALIDATION = STUDY / "outputs" / "e9_property_mode" / "reproduction_validation.json"
RUNTIME_COLUMNS = {"fit_seconds"}


def compare(canonical: pd.DataFrame, reproduced: pd.DataFrame) -> dict:
    keys = [
        "outer_fold",
        "capacity_regime",
        "geometry",
        "budget",
        "pair_4d_variant",
        "pair_role",
        "candidate",
    ]
    canonical = canonical.sort_values(keys, na_position="first").reset_index(drop=True)
    reproduced = reproduced.sort_values(keys, na_position="first").reset_index(drop=True)
    payload: dict = {
        "canonical_rows": int(len(canonical)),
        "reproduced_rows": int(len(reproduced)),
        "runtime_columns_compared": False,
        "identity_mismatches": {},
        "numeric_comparisons": {},
    }
    row_count_ok = len(canonical) == len(reproduced)
    schema_ok = set(canonical.columns) == set(reproduced.columns)
    payload["missing_columns"] = sorted(set(canonical.columns) - set(reproduced.columns))
    payload["extra_columns"] = sorted(set(reproduced.columns) - set(canonical.columns))
    common = sorted(set(canonical.columns) & set(reproduced.columns))
    identity_ok = row_count_ok and schema_ok
    numeric_ok = row_count_ok and schema_ok
    if row_count_ok:
        for column in common:
            if column in RUNTIME_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(canonical[column]):
                left = canonical[column].to_numpy(dtype=float)
                right = reproduced[column].to_numpy(dtype=float)
                difference = np.abs(left - right)
                finite = np.isfinite(left) & np.isfinite(right)
                same_nonfinite = (
                    np.array_equal(np.isnan(left), np.isnan(right))
                    and np.array_equal(np.isposinf(left), np.isposinf(right))
                    and np.array_equal(np.isneginf(left), np.isneginf(right))
                )
                maximum = float(np.max(difference[finite])) if finite.any() else 0.0
                matches = bool(same_nonfinite and maximum <= 1e-10)
                payload["numeric_comparisons"][column] = {
                    "max_absolute_difference": maximum,
                    "atol": 1e-10,
                    "matches": matches,
                }
                numeric_ok &= matches
            else:
                left = canonical[column].fillna("<NA>").astype(str)
                right = reproduced[column].fillna("<NA>").astype(str)
                mismatches = int((left != right).sum())
                payload["identity_mismatches"][column] = mismatches
                identity_ok &= mismatches == 0
    payload["status"] = "pass" if identity_ok and numeric_ok else "fail"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reproduced", type=Path)
    arguments = parser.parse_args()
    payload = compare(pd.read_csv(CANONICAL), pd.read_csv(arguments.reproduced))
    VALIDATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
