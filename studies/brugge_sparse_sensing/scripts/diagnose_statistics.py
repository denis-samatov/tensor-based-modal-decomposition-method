"""Read-only bit-pattern diagnosis of stored-row parsing; no acceptance changes."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from verify_statistics import verify_runtime

INPUTS = ("e2_summary.csv", "e1_representation.csv", "e3_noise.csv", "e4_l1_sensitivity.csv")


def bits_hash(values):
    """Hash canonical little-endian binary64 values in input order."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def inspect_input(path):
    frame = pd.read_csv(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(frame) != len(rows):
        raise ValueError(f"{path.name}: CSV readers disagree on row count")
    columns = {}
    for name in frame.select_dtypes(include="floating").columns:
        parsed = frame[name].to_numpy(dtype=np.float64)
        tokens = [row[name] for row in rows]
        builtin = np.array([float(token) if token else np.nan for token in tokens])
        different = np.flatnonzero(~((parsed == builtin) | (np.isnan(parsed) & np.isnan(builtin))))
        columns[name] = {
            "parsed_sha256": bits_hash(parsed), "builtin_float_sha256": bits_hash(builtin),
            "different_from_builtin_float": len(different),
            "first_differences": [
                {"row": int(i), "token": tokens[i], "parsed_hex": float(parsed[i]).hex(),
                 "builtin_hex": float(builtin[i]).hex()}
                for i in different[:8]
            ],
        }
    return frame, rows, {"raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "shape": list(frame.shape), "float_columns": columns}


def inspect_median(frame, rows):
    """The preselected P2/grid/N10/POD-LS median that differed in Linux output."""
    mask = ((frame.protocol == "P2") & (frame.sensing == "grid") & (frame.N == 10)
            & (frame.basis == "POD") & (frame.estimator == "LS") & (frame.placement == "QR-POD"))
    selected = frame.loc[mask].sort_values("test_run")
    values = selected.rmse_p.to_numpy()
    original = {int(row["test_run"]): row["rmse_p"] for row in rows
                if row["protocol"] == "P2" and row["sensing"] == "grid" and row["N"] == "10"
                and row["basis"] == "POD" and row["estimator"] == "LS" and row["placement"] == "QR-POD"}
    if len(values) != 10 or len(original) != 10:
        raise ValueError("The fixed diagnostic group must contain ten control scenarios")
    builtin = np.array([float(original[int(i)]) for i in selected.test_run])
    ordered = np.sort(values)
    return {
        "scope": "P2/POD-E LS vs POD LS/grid/N10/median_b",
        "parsed_values_hex": [float(v).hex() for v in values],
        "builtin_values_hex": [float(v).hex() for v in builtin],
        "median_hex": float(np.median(values)).hex(),
        "builtin_input_median_hex": float(np.median(builtin)).hex(),
        "middle_pair_hex": [float(v).hex() for v in ordered[4:6]],
        "python_middle_pair_mean_hex": ((float(ordered[4]) + float(ordered[5])) / 2).hex(),
    }


def diagnose(study):
    runtime = verify_runtime(study / "requirements-statistics.txt")
    inputs = {}
    median = None
    for name in INPUTS:
        frame, rows, inputs[name] = inspect_input(study / "outputs" / name)
        if name == "e2_summary.csv":
            median = inspect_median(frame, rows)
    return {
        "scope": "diagnosis only; hashes and selected public numeric tokens, not scientific acceptance",
        "runtime": runtime, "libc": platform.libc_ver(),
        "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "inputs": inputs, "selected_median": median,
        "limits": "No parser, reduction, statistic, canonical artifact or acceptance rule is changed. "
                  "The builtin-float reader is an independent diagnostic reference, not a replacement parser.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    study = Path(__file__).resolve().parents[1]
    output = args.output_file.resolve()
    if output.is_relative_to(study.parents[1]):
        parser.error("Diagnostic output must be outside the source checkout")
    result = diagnose(study)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"READ-ONLY STATISTICS DIAGNOSIS: {len(result['inputs'])} input files")


if __name__ == "__main__":
    main()
