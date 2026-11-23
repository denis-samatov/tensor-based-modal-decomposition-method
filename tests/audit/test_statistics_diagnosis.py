"""Read-only parsing diagnostics preserve bit patterns and reject ambiguous inputs."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "studies/brugge_sparse_sensing/scripts/diagnose_statistics.py")


@pytest.fixture
def diagnosis(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("diagnose_statistics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_is_little_endian_binary64_and_preserves_value_changes(diagnosis):
    values = np.array([1.25, -2.5], dtype=">f8")
    assert diagnosis.bits_hash(values) == hashlib.sha256(np.array([1.25, -2.5], dtype="<f8").tobytes()).hexdigest()
    assert diagnosis.bits_hash(values) != diagnosis.bits_hash([1.25, np.nextafter(-2.5, 0)])


def test_real_csv_probe_is_read_only_and_handles_missing_values(diagnosis, tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("value,label\n1.25,a\n,b\n2.5,c\n")
    before = path.read_bytes()
    frame, rows, result = diagnosis.inspect_input(path)
    assert path.read_bytes() == before
    assert len(frame) == len(rows) == 3
    column = result["float_columns"]["value"]
    assert column["parsed_sha256"] == column["builtin_float_sha256"]
    assert column["different_from_builtin_float"] == 0
    assert column["first_differences"] == []


def test_numeric_drift_is_recorded_as_actual_hex_not_rounded_text(diagnosis, monkeypatch, tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("value\n1.25\n2.5\n")
    changed = np.nextafter(1.25, 2)
    monkeypatch.setattr(diagnosis.pd, "read_csv", lambda _: pd.DataFrame({"value": [changed, 2.5]}))
    _, _, result = diagnosis.inspect_input(path)
    column = result["float_columns"]["value"]
    assert column["different_from_builtin_float"] == 1
    assert column["parsed_sha256"] != column["builtin_float_sha256"]
    assert column["first_differences"] == [{"row": 0, "token": "1.25", "parsed_hex": changed.hex(), "builtin_hex": (1.25).hex()}]


def test_row_count_disagreement_fails(diagnosis, monkeypatch, tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("value\n1.25\n2.5\n")
    monkeypatch.setattr(diagnosis.pd, "read_csv", lambda _: pd.DataFrame({"value": [1.25]}))
    with pytest.raises(ValueError, match="row count"):
        diagnosis.inspect_input(path)


def test_fixed_median_group_records_inputs_and_intermediate_pair(diagnosis):
    rows = [{"protocol": "P2", "sensing": "grid", "N": "10", "basis": "POD",
             "estimator": "LS", "placement": "QR-POD", "test_run": str(i), "rmse_p": str(i)}
            for i in range(1, 11)]
    frame = pd.DataFrame(rows).astype({"N": int, "test_run": int, "rmse_p": float})
    result = diagnosis.inspect_median(frame, rows)
    assert result["parsed_values_hex"] == result["builtin_values_hex"]
    assert result["middle_pair_hex"] == [(5.0).hex(), (6.0).hex()]
    assert result["median_hex"] == result["python_middle_pair_mean_hex"] == (5.5).hex()
    with pytest.raises(ValueError, match="ten control"):
        diagnosis.inspect_median(frame.iloc[:-1], rows[:-1])
