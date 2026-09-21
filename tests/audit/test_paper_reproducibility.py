"""Regression checks for a standalone paper source and disposable verification runs."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/brugge_sparse_sensing"
PAPER = ROOT / "docs/paper"


@pytest.mark.parametrize("change,expected", [("label", 1), ("missing", 1), ("science", 1), ("timing", 0)])
def test_rerun_comparator_catches_non_numeric_and_missing_value_changes(tmp_path, change, expected):
    import pandas as pd

    scripts = tmp_path / "scripts"
    original, rerun = tmp_path / "outputs", tmp_path / "rerun"
    for path in (scripts, original, rerun):
        path.mkdir()
    script = scripts / "compare_rerun.py"
    script.write_text((STUDY / "scripts/compare_rerun.py").read_text())
    names = ("e2_summary.csv", "e1_representation.csv", "e3_noise.csv", "e4_l1_sensitivity.csv",
             "e5_stability.csv", "e7_spatial_structure.csv", "stats_wilcoxon.csv",
             "e2_snapshots.csv.gz", "e5_well_ranks.csv", "e8_property_coupling.csv",
             "e8_property_coupling_summary.csv", "e0_cluster_table.csv", "e0_archived_pressure_wells.csv")
    frame = pd.DataFrame({"fold": ["P2-run1"], "rmse": [0.12], "seconds": [1.0], "sec_per_snapshot": [1.0]})
    for folder in (original, rerun):
        (folder / "numbers.tex").write_text("\\newcommand{\\Example}{0.12}\n")
        for name in names:
            frame.to_csv(folder / name, index=False)
    changed = frame.copy()
    if change == "label":
        changed.loc[0, "fold"] = "P2-run2"
    elif change == "missing":
        changed.loc[0, "rmse"] = float("nan")
    elif change == "science":
        changed.loc[0, "rmse"] = 0.13
    else:
        changed.loc[0, "sec_per_snapshot"] = 2.0
    changed.to_csv(rerun / "e2_summary.csv", index=False)
    result = subprocess.run([sys.executable, str(script), str(rerun)], capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr


def test_environment_verifier_rejects_numerical_dependency_drift(tmp_path):
    """A study run must not silently continue with a different numerical stack."""
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==0.0.0\n")
    result = subprocess.run(
        [sys.executable, str(STUDY / "scripts/verify_environment.py"), str(lock)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "numpy: installed" in result.stderr
    assert "required 0.0.0" in result.stderr


@pytest.mark.parametrize("scientific_value, expected_code", [("1.0", 0), ("1.1", 1)])
def test_claim_checker_allows_only_wall_clock_macro_drift(tmp_path, scientific_value, expected_code):
    """Independent reruns may change timings, never scientific result macros."""
    outputs = tmp_path / "outputs"
    manuscript = tmp_path / "manuscript"
    outputs.mkdir()
    (manuscript / "sections").mkdir(parents=True)
    (manuscript / "main.tex").write_text("")
    (outputs / "numbers.tex").write_text(
        "\\newcommand{\\ScientificResult}{1.0}\n"
        "\\newcommand{\\CostPodSvd}{1.18}\n"
    )
    (manuscript / "numbers.tex").write_text(
        f"\\newcommand{{\\ScientificResult}}{{{scientific_value}}}\n"
        "\\newcommand{\\CostPodSvd}{1.22}\n"
    )
    env = {**os.environ, "BSS_OUT": str(outputs)}
    result = subprocess.run(
        [sys.executable, str(STUDY / "scripts/check_claims.py"), str(manuscript)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr


def builder():
    spec = importlib.util.spec_from_file_location("paper_builder", PAPER / "build_submission_ready.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing_from", ["outputs", "manuscript"])
def test_claim_checker_rejects_missing_timing_definition(tmp_path, missing_from):
    outputs, manuscript = tmp_path / "outputs", tmp_path / "manuscript"
    outputs.mkdir()
    (manuscript / "sections").mkdir(parents=True)
    (manuscript / "main.tex").write_text("")
    for name, directory in (("outputs", outputs), ("manuscript", manuscript)):
        text = "\\newcommand{\\ScientificResult}{1.0}\n"
        if name != missing_from:
            text += "\\newcommand{\\CostPodSvd}{1.18}\n"
        (directory / "numbers.tex").write_text(text)
    result = subprocess.run(
        [sys.executable, str(STUDY / "scripts/check_claims.py"), str(manuscript)],
        env={**os.environ, "BSS_OUT": str(outputs)}, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL macro definitions differ" in result.stdout


def test_paper_flattens_from_canonical_results_without_manuscript_copies():
    module = builder()
    assert module.STUDY == STUDY
    assert not (module.MS / "numbers.tex").exists()
    assert not (module.MS / "figures").exists()
    text = module.flatten(module.MS / "main.tex")
    assert "\\input{" not in text
    assert (STUDY / "outputs/numbers.tex").read_text().strip() in text
    supplement = module.flatten(module.MS / "supplementary/supplementary.tex")
    assert "\\input{" not in supplement
    assert (STUDY / "outputs/tabS_P1.tex").read_text().strip() in supplement
    assert (STUDY / "outputs/tabS_So.tex").read_text().strip() in supplement
    for figure in sorted((STUDY / "figures").glob("fig[0-9]*.pdf")):
        assert f"{{{figure.name}}}" in text or f"{{{figure.name}}}" in supplement
    cover = module.flatten(module.SUB / "cover_letter.tex")
    assert (STUDY / "outputs/numbers.tex").read_text().strip() in cover


def test_submission_builder_includes_standalone_ai_disclosure():
    source = (PAPER / "build_submission_ready.py").read_text()
    verifier = (PAPER / "audit/final_gate/verify_upload_set.py").read_text()
    assert '"ai_use_statement.txt"' in source
    assert '"ai_use_statement.txt"' in verifier


def test_active_submission_sources_have_no_author_input_markers():
    """Resolved author confirmations must not leave active submission placeholders."""
    paths = [
        *sorted((PAPER / "manuscript").rglob("*.tex")),
        *sorted((PAPER / "submission").glob("*.tex")),
        *sorted((PAPER / "submission").glob("*.txt")),
        PAPER / "AUTHOR_FINAL_ACTIONS.md",
        PAPER / "FINAL_PRE_SUBMISSION_GATE.md",
        PAPER / "RELEASE_CANDIDATE.md",
        PAPER / "README.md",
        STUDY / "README.md",
    ]
    marker = "AUTHOR" + " INPUT REQUIRED"
    offenders = [str(path.relative_to(ROOT)) for path in paths if marker in path.read_text()]
    assert offenders == []


def test_confirmed_submission_metadata_is_consistent():
    """Author-confirmed simulator, access, funding, AI and exclusivity facts stay aligned."""
    methods = (PAPER / "manuscript/sections/methods.tex").read_text()
    supplement = (PAPER / "manuscript/supplementary/S1_data.tex").read_text()
    backmatter = (PAPER / "manuscript/sections/backmatter.tex").read_text()
    cover = (PAPER / "submission/cover_letter.tex").read_text()
    data_statement = (PAPER / "submission/data_availability_statement.txt").read_text()
    ai_statement = (PAPER / "submission/ai_use_statement.txt").read_text()
    metadata = (PAPER / "submission/editorial_manager_metadata.md").read_text()

    for text in (methods, supplement, backmatter, data_statement, metadata):
        assert "t-Navigator" in text
    assert "25.1" not in "\n".join((methods, supplement, backmatter, data_statement, metadata))
    assert "available from the corresponding author upon reasonable request" in data_statement
    assert "subject to the applicable data-use and access conditions" in data_statement
    funding = (
        "The research was supported by the program of the National Research Tomsk "
        "Polytechnic University (Prioritet-2030-ISP-032-090-2026)."
    )
    assert funding in backmatter
    assert funding in metadata
    assert "Prioritet -- proekt KIP" not in backmatter
    assert "OpenAI GPT-4o" in ai_statement
    assert "OpenAI GPT-5-Codex" in ai_statement
    assert "reviewed and edited the content" in ai_statement
    assert "not currently under consideration by another journal" in cover


def test_competing_interest_declaration_is_packaged():
    source = (PAPER / "build_submission_ready.py").read_text()
    verifier = (PAPER / "audit/final_gate/verify_upload_set.py").read_text()
    declaration = PAPER / "submission/declaration_of_competing_interest.txt"
    assert declaration.exists()
    assert "no known competing financial interests" in declaration.read_text()
    assert '"declaration_of_competing_interest.docx"' in source
    assert '"declaration_of_competing_interest.docx"' in verifier


def test_data_manifest_records_author_confirmed_simulator_boundary():
    import json

    manifest = json.loads((STUDY / "outputs/s0_data_manifest.json").read_text())
    assert manifest["simulator"] == "t-Navigator"
    assert manifest["simulator_version"] is None
    assert manifest["simulator_to_hdf5_exporter"] is None
    assert manifest["vertical_reduction_rule"] is None
    assert manifest["pressure_unit_status"] == "unconfirmed; HDF5 stores no unit attribute"


def test_submission_builder_compiles_from_the_pinned_tectonic_cache():
    """Submission builds must not depend on a live bundle download."""
    source = (PAPER / "build_submission_ready.py").read_text()
    assert '"--only-cached"' in source


def test_publication_results_do_not_assign_unverified_bar_units():
    """Only the documented BHP control settings may be expressed in bar.

    The supplied HDF5 arrays carry no unit metadata and historical project
    artefacts conflict between bar and psi.  Performance results therefore use
    the explicitly defined supplied-export unit ``u_p``; this gate prevents a
    plotting or prose edit from silently turning that convention into a
    physical unit claim.
    """
    result_surfaces = [
        PAPER / "manuscript/sections/abstract.tex",
        PAPER / "manuscript/sections/methods.tex",
        PAPER / "manuscript/sections/experiments.tex",
        PAPER / "manuscript/sections/results.tex",
        PAPER / "manuscript/sections/discussion.tex",
        PAPER / "manuscript/sections/conclusions.tex",
        PAPER / "manuscript/sections/fig1_workflow.tex",
        PAPER / "manuscript/sections/fig2_data.tex",
        PAPER / "manuscript/sections/fig4_budget.tex",
        PAPER / "manuscript/sections/fig6_noise.tex",
        PAPER / "manuscript/supplementary/S2_audit.tex",
        PAPER / "manuscript/supplementary/S3_algorithms.tex",
        PAPER / "manuscript/supplementary/S4_results.tex",
        PAPER / "submission/cover_letter.tex",
        PAPER / "submission/highlights.txt",
        STUDY / "scripts/make_figures.py",
        STUDY / "scripts/make_graphical_abstract.py",
        STUDY / "scripts/make_latex_tables.py",
    ]
    offenders = [str(path.relative_to(ROOT)) for path in result_surfaces
                 if re.search(r"(?<![\\.])\bbar\b", path.read_text(), flags=re.IGNORECASE)]
    assert offenders == [], (
        "Unverified pressure-array values are labelled as bar in: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("claims_fail", [False, True])
def test_output_verifier_removes_scratch_on_success_and_claim_failure(tmp_path, claims_fail):
    """Mock numerical stages; exercise the real shell's exit and temporary-directory contract."""
    study = tmp_path / "study"
    (study / "outputs").mkdir(parents=True)
    for name in (
        "table_test.csv", "stats_wilcoxon.csv", "key_numbers.json", "numbers.tex",
        "tab_main.tex", "tab_cost.tex", "tabS_P1.tex", "tabS_So.tex", "stage.csv.gz", "stage.npz",
        "e8_property_coupling_summary.csv", "tabS_coupling.tex",
    ):
        (study / "outputs" / name).write_text("fixture\n")
    script = study / "verify_outputs.sh"
    script.write_text((STUDY / "verify_outputs.sh").read_text())
    fake = tmp_path / "python"
    fake.write_text(
        '#!/usr/bin/env python3\n'
        'import os, pathlib, shutil, sys\n'
        'stage = pathlib.Path(sys.argv[1]).name\n'
        'if stage in {"make_tables.py", "make_numbers_tex.py", "make_latex_tables.py"}:\n'
        '    for p in pathlib.Path("outputs").iterdir():\n'
        '        shutil.copy(p, pathlib.Path(os.environ["BSS_OUT"]) / p.name)\n'
        'if stage == "verify_inputs.py": sys.exit(1)\n'
        'if stage == "check_claims.py": sys.exit(int(os.environ["CLAIMS_FAIL"]))\n'
    )
    fake.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {**os.environ, "PYTHON": str(fake), "TMPDIR": str(scratch),
           "CLAIMS_FAIL": str(int(claims_fail))}
    result = subprocess.run(["bash", str(script), "manuscript"], env=env, capture_output=True, text=True)
    assert result.returncode == int(claims_fail), result.stdout + result.stderr
    assert not list(scratch.iterdir()), "Verification left disposable outputs behind"


@pytest.mark.parametrize("claims_fail", [False, True])
def test_full_driver_propagates_claim_failure_and_cleans_successful_parts(tmp_path, claims_fail):
    """Exercise the shell driver with lightweight stages, including its final manuscript gate."""
    script = tmp_path / "run_all.sh"
    script.write_text((STUDY / "run_all.sh").read_text())
    fake = tmp_path / "python"
    fake.write_text(
        '#!/usr/bin/env python3\n'
        'import os, pathlib, sys\n'
        'if len(sys.argv) > 2 and sys.argv[2] == "part":\n'
        '    parts = pathlib.Path(os.environ["BSS_OUT"]) / "parts"\n'
        '    (parts / "fixture.pkl").write_bytes(b"temporary")\n'
        'if pathlib.Path(sys.argv[1]).name == "check_claims.py":\n'
        '    if int(os.environ["CLAIMS_FAIL"]):\n'
        '        print("fixture claim mismatch")\n'
        '        sys.exit(1)\n'
    )
    fake.chmod(0o755)
    outputs = tmp_path / "outputs"
    env = {**os.environ, "PYTHON": str(fake), "BSS_OUT": str(outputs),
           "BSS_FIG": str(tmp_path / "figures"), "BSS_LOG": str(tmp_path / "logs"),
           "MANUSCRIPT_DIR": str(tmp_path / "manuscript"), "CLAIMS_FAIL": str(int(claims_fail))}
    result = subprocess.run(["bash", str(script), "1"], env=env, capture_output=True, text=True)
    assert result.returncode == int(claims_fail), result.stdout + result.stderr
    if claims_fail:
        assert "fixture claim mismatch" in result.stdout
        assert "run_all finished" not in result.stdout
    else:
        assert "run_all finished" in result.stdout
        assert not (outputs / "parts").exists()
