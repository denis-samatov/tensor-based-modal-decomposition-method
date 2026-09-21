"""Pairing and aggregation contracts for the internal exhaustive E2 audit."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[2] / "docs/paper/audit/final_gate/superiority_audit.py"
    assert path.exists(), "The exhaustive E2 comparison implementation is missing"
    spec = importlib.util.spec_from_file_location("superiority_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    return pd.DataFrame([
        dict(protocol="P2", fold=fold, sensing="grid", regime="random", N=2,
             seed=seed, rank=2, m=2, rmse_p=value)
        for fold, values in [("f1", (1., 3.)), ("f2", (3., 5.))]
        for seed, value in enumerate(values)
    ])


def test_pairs_labels_not_row_positions_and_takes_fold_seed_median(audit):
    left = sample()
    right = left.assign(rmse_p=left.rmse_p + 1).iloc[::-1]
    result = audit.compare_pair(left, right, "rmse_p", expected_folds=2, expected_seeds=2)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.tbmd_mean == 3
    assert row.baseline_mean == 4
    assert row.fold_wins == 2 and row.fold_losses == 0
    assert row.tbmd_sd == pytest.approx(2 ** .5)


def test_missing_pair_is_not_silently_dropped(audit):
    with pytest.raises(ValueError, match="unmatched"):
        audit.compare_pair(sample(), sample().iloc[:-1], "rmse_p", expected_folds=2, expected_seeds=2)


def test_duplicates_and_nonfinite_metrics_rejected(audit):
    with pytest.raises(ValueError, match="duplicate"):
        audit.compare_pair(pd.concat([sample(), sample()]), sample(), "rmse_p")
    bad = sample()
    bad.loc[0, "rmse_p"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit.compare_pair(bad, sample(), "rmse_p")


def test_budget_mismatch_and_tolerance_ties(audit):
    with pytest.raises(ValueError, match="budget"):
        audit.compare_pair(sample(), sample().assign(m=3), "rmse_p")
    result = audit.compare_pair(sample(), sample().assign(rmse_p=sample().rmse_p + 1e-13),
                                "rmse_p", expected_folds=2, expected_seeds=2)
    assert result.iloc[0].fold_ties == 2
    assert result.iloc[0].mean_verdict == "tie"


def test_cost_caption_identifies_timed_operation_without_changing_values(tmp_path):
    root = Path(__file__).resolve().parents[2]
    study = root / "studies/brugge_sparse_sensing"
    for name in ("key_numbers.json", "e6_cost.json"):
        shutil.copyfile(study / "outputs" / name, tmp_path / name)
    subprocess.run([sys.executable, str(study / "scripts/make_latex_tables.py")],
                   env=os.environ | {"BSS_OUT": str(tmp_path)}, check=True)
    generated = (tmp_path / "tab_cost.tex").read_text()
    assert "POD-E coefficient estimation" in generated
    assert "exclude full-state synthesis and inverse scaling" in generated
    assert "Placement timings use TBMD" in generated
    previous = (study / "outputs/tab_cost.tex").read_text()
    assert generated.split("\\begin{tabular}", 1)[1] == previous.split("\\begin{tabular}", 1)[1]
