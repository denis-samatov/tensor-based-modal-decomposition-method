"""Regression checks for publication gates, not assertions that a paper is qualified."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "log,raises",
    [
        ("Overfull \\hbox (12.3pt too wide) in paragraph", True),
        ("Overfull \\vbox (4.5pt too high) detected", True),
        ("LaTeX Warning: Reference `missing' undefined", True),
        ("LaTeX Warning: Label `duplicate' multiply defined", True),
        ("Underfull \\hbox (badness 3000) in paragraph", False),
        ("Overfull \\hbox (0.8pt too wide) in paragraph", False),
    ],
)
def test_compile_gate_rejects_layout_and_reference_errors(tmp_path, monkeypatch, log, raises):
    mod = load("docs/paper/build_submission_ready.py", "publication_builder")
    monkeypatch.setattr(mod, "BUILD_LOGS", tmp_path / "evidence")
    (tmp_path / "sample.log").write_text(log)
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    if raises:
        with pytest.raises(RuntimeError, match="LaTeX validation failed"):
            mod.compile_tex(tmp_path, "sample")
    else:
        assert mod.compile_tex(tmp_path, "sample") == []
    assert (tmp_path / "evidence/sample.log").read_text() == log


def fixture():
    return pd.DataFrame(
        [
            {
                "outer_fold": 1,
                "capacity_regime": "equal_total",
                "geometry": "grid",
                "budget": 30,
                "pair_4d_variant": "4D-B",
                "pair_role": "4D",
                "candidate": "fixed",
                "rmse_p": 0.2,
                "fit_seconds": 1.0,
            }
        ]
    )


@pytest.mark.parametrize("change", ["missing", "extra", "positive_inf", "negative_inf"])
def test_e9_comparator_rejects_schema_and_nonfinite_drift(change):
    mod = load("studies/brugge_sparse_sensing/scripts/e9_compare_reproduction.py", "e9_compare")
    original = fixture()
    other = original.copy()
    if change == "missing":
        other = other.drop(columns="rmse_p")
    elif change == "extra":
        other["unexpected"] = 1
    else:
        other.loc[0, "rmse_p"] = float("inf" if change == "positive_inf" else "-inf")
    assert mod.compare(original, other)["status"] == "fail"


def test_e9_comparator_allows_only_recorded_runtime_variation():
    mod = load("studies/brugge_sparse_sensing/scripts/e9_compare_reproduction.py", "e9_compare")
    original = fixture()
    other = original.copy()
    other.loc[0, "fit_seconds"] = 5.0
    assert mod.compare(original, other)["status"] == "pass"
