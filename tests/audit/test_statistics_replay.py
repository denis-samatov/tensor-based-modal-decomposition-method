"""Boundary checks for the separate, exact twelve-file stored-row replay gate."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "studies/brugge_sparse_sensing/scripts/verify_statistics.py")


@pytest.fixture
def replay():
    spec = importlib.util.spec_from_file_location("verify_statistics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_outputs_are_fixed_and_complete(replay):
    assert replay.EXPECTED_OUTPUTS == (
        "table_main_P1.csv", "table_main_P2.csv", "table_representation.csv",
        "table_noise.csv", "table_l1.csv", "stats_wilcoxon.csv", "key_numbers.json",
        "numbers.tex", "tab_main.tex", "tab_cost.tex", "tabS_P1.tex", "tabS_So.tex",
    )
    assert not set(replay.INPUT_FILES) & set(replay.EXPECTED_OUTPUTS)


@pytest.mark.parametrize("change", ["missing", "missing_canonical", "different", "none_generated", "unchanged"])
def test_comparison_requires_every_expected_file(replay, tmp_path, change):
    source, generated = tmp_path / "canonical", tmp_path / "generated"
    source.mkdir()
    generated.mkdir()
    for name in replay.EXPECTED_OUTPUTS:
        (source / name).write_bytes(b"expected\n")
        if change != "none_generated":
            (generated / name).write_bytes(b"expected\n")
    if change == "missing":
        (generated / "numbers.tex").unlink()
    if change == "missing_canonical":
        (source / "numbers.tex").unlink()
    if change == "different":
        (generated / "numbers.tex").write_bytes(b"expected\r\n")
    comparison = replay.compare_outputs(source, generated)
    assert len(comparison) == 12
    assert all(item["identical"] for item in comparison.values()) == (change == "unchanged")
    if change in {"missing", "none_generated"}:
        assert comparison["numbers.tex"]["generated_sha256"] is None


@pytest.mark.parametrize("python_version, pandas_version", [((3, 12), "3.0.5"), ((3, 13), "3.1.0")])
def test_runtime_drift_is_rejected(replay, monkeypatch, tmp_path, python_version, pandas_version):
    lock = tmp_path / "requirements-statistics.txt"
    lock.write_text("numpy==2.5.2\npandas==3.0.5\nscipy==1.18.1\n")
    monkeypatch.setattr(replay.sys, "version_info", python_version)
    versions = {"numpy": "2.5.2", "pandas": pandas_version, "scipy": "1.18.1"}
    monkeypatch.setattr(replay.metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match="Python 3.13|pandas"):
        replay.verify_runtime(lock)


@pytest.mark.parametrize("destination", ["inside_source", "existing"])
def test_output_collision_rejected_without_changes(replay, tmp_path, destination):
    study = tmp_path / "repo/studies/brugge_sparse_sensing"
    study.mkdir(parents=True)
    output = study / "scratch" if destination == "inside_source" else tmp_path / "existing"
    if destination == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    with pytest.raises((ValueError, FileExistsError)):
        replay.verify_statistics(study, output)
    if destination == "existing":
        assert (output / "sentinel").read_text() == "preserve"
        assert sorted(p.name for p in output.iterdir()) == ["sentinel"]
    else:
        assert not output.exists()


@pytest.mark.parametrize("failure", [None, "checksum", "generator", "missing", "mutate_source"])
def test_runner_isolates_inputs_and_preserves_failure_receipts(replay, tmp_path, monkeypatch, failure):
    study = tmp_path / "repo/studies/brugge_sparse_sensing"
    canonical = study / "outputs"
    canonical.mkdir(parents=True)
    (study / "scripts").mkdir()
    (study / "bss").mkdir()
    for name in replay.INPUT_FILES + replay.EXPECTED_OUTPUTS:
        (canonical / name).write_bytes(name.encode())
    for name in replay.GENERATORS + ("check_sha256.py",):
        (study / "scripts" / name).write_text("# unchanged generator\n")
    (study / "bss/__init__.py").write_text("# source\n")
    (study / "requirements-statistics.in").write_text("# pins\n")
    (study / "requirements-statistics.txt").write_text("# lock\n")
    (study / "SHA256SUMS").write_text("".join(
        f"{replay.sha256(canonical / name)}  outputs/{name}\n"
        for name in replay.INPUT_FILES + replay.EXPECTED_OUTPUTS
    ))
    before = {p: p.read_bytes() for p in study.rglob("*") if p.is_file()}
    monkeypatch.setattr(replay, "verify_runtime", lambda _: {"python": "fixture", "packages": {}})
    calls = []

    def run(argv, **kwargs):
        script = Path(argv[1]).name
        calls.append(script)
        if script == "check_sha256.py":
            return subprocess.CompletedProcess(argv, 1 if failure == "checksum" else 0,
                                               "checksum fixture\n", "")
        scratch = Path(kwargs["env"]["BSS_OUT"])
        if script == replay.GENERATORS[0]:
            assert sorted(p.name for p in scratch.iterdir()) == sorted(replay.INPUT_FILES)
            assert not any((scratch / name).exists() for name in replay.EXPECTED_OUTPUTS)
        if failure == "generator":
            return subprocess.CompletedProcess(argv, 7, "partial\n", "generator failed\n")
        if script == replay.GENERATORS[-1]:
            for name in replay.EXPECTED_OUTPUTS:
                if failure != "missing" or name != "numbers.tex":
                    (scratch / name).write_bytes((canonical / name).read_bytes())
            if failure == "mutate_source":
                (canonical / replay.INPUT_FILES[0]).write_bytes(b"unexpected source change")
        return subprocess.CompletedProcess(argv, 0, "generated\n", "")

    monkeypatch.setattr(replay.subprocess, "run", run)
    output = tmp_path / "owned-run"
    receipt = replay.verify_statistics(study, output)
    assert receipt["status"] == ("passed" if failure is None else "failed")
    assert json.loads((output / "receipt.json").read_text()) == receipt
    assert receipt["scope"] == "statistics-only; twelve core artifacts from committed result rows"
    assert receipt["full_historical_study"] is False
    assert receipt["raw_simulation_recomputed"] is False
    if failure == "checksum":
        assert calls == ["check_sha256.py"]
        assert not list((output / "outputs").iterdir())
    elif failure == "generator":
        assert calls == ["check_sha256.py", replay.GENERATORS[0]]
        assert (output / "logs/make_tables.py.stderr.log").read_text() == "generator failed\n"
    elif failure == "mutate_source":
        assert receipt["source_unchanged"] is False
    else:
        assert all(p.read_bytes() == data for p, data in before.items())
