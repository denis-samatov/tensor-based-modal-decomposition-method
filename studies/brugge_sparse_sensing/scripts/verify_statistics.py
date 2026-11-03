"""Replay twelve core artifacts from committed rows; never run the full/raw study."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

EXPECTED_OUTPUTS = (
    "table_main_P1.csv", "table_main_P2.csv", "table_representation.csv",
    "table_noise.csv", "table_l1.csv", "stats_wilcoxon.csv", "key_numbers.json",
    "numbers.tex", "tab_main.tex", "tab_cost.tex", "tabS_P1.tex", "tabS_So.tex",
)
INPUT_FILES = (
    "e1_representation.csv", "e2_summary.csv", "e3_noise.csv", "e4_l1_sensitivity.csv",
    "s0_data_manifest.json", "e6_cost.json", "v1_verification.json", "e5_stability.csv",
    "e5_well_ranks.csv", "e2_snapshots.csv.gz", "e7_spatial_structure.csv",
    "e0_archived_pressure_wells.csv", "e0_archived_pressure_grid.csv", "e0_cluster_table.csv",
    "v2_admm_convergence.json",
)
GENERATORS = ("make_tables.py", "make_numbers_tex.py", "make_latex_tables.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_runtime(lock):
    """Reject a different interpreter minor or any drift from the dedicated lock."""
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Statistics-only replay requires Python 3.13")
    pins = dict(re.findall(r"^([a-zA-Z0-9_-]+)==([^\s\\]+)", lock.read_text(), re.MULTILINE))
    expected = {"numpy": "2.5.2", "pandas": "3.0.5", "scipy": "1.18.1"}
    if not expected.items() <= pins.items():
        raise RuntimeError("Statistics lock does not contain the three qualified numerical pins")
    installed = {name: metadata.version(name) for name in pins}
    for name, version in pins.items():
        if installed[name] != version:
            raise RuntimeError(f"{name}: installed {installed[name]}, required {version}")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": installed}


def compare_outputs(canonical, generated):
    """Enumerate the required set, including outputs a failed generator omitted."""
    comparison = {}
    for name in EXPECTED_OUTPUTS:
        original, rerender = canonical / name, generated / name
        comparison[name] = {
            "source_sha256": sha256(original), "generated_sha256": sha256(rerender),
            "identical": original.is_file() and rerender.is_file()
            and original.read_bytes() == rerender.read_bytes(),
        }
    return comparison


def source_hashes(study):
    paths = {"SHA256SUMS", "requirements-statistics.in", "requirements-statistics.txt",
             "scripts/verify_statistics.py", "bss/__init__.py", "scripts/check_sha256.py"}
    paths.update(f"scripts/{name}" for name in GENERATORS)
    paths.update(line.split(maxsplit=1)[1].strip()
                 for line in (study / "SHA256SUMS").read_text().splitlines())
    return {name: sha256(study / name) for name in sorted(paths)}


def verify_statistics(study, output):
    """Write a receipt and logs only into a new directory outside the checkout."""
    study, output = study.resolve(), output.resolve()
    if output.is_relative_to(study.parents[1]):
        raise ValueError("Output must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    generated, logs = output / "outputs", output / "logs"
    generated.mkdir()
    logs.mkdir()
    receipt = {
        "scope": "statistics-only; twelve core artifacts from committed result rows",
        "full_historical_study": False, "raw_simulation_recomputed": False,
        "status": "failed", "commands": [], "comparison": {},
    }
    before = None
    try:
        receipt["runtime"] = verify_runtime(study / "requirements-statistics.txt")
        before = source_hashes(study)
        receipt["source_sha256"] = before
        env = dict(os.environ, BSS_OUT=str(generated), PYTHONHASHSEED="0",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        for script in ("check_sha256.py",) + GENERATORS:
            if script == GENERATORS[0]:
                for name in INPUT_FILES:
                    shutil.copyfile(study / "outputs" / name, generated / name)
            argv = [sys.executable, str(study / "scripts" / script)]
            proc = subprocess.run(argv, cwd=study, env=env, capture_output=True, text=True)
            stdout, stderr = f"logs/{script}.stdout.log", f"logs/{script}.stderr.log"
            (output / stdout).write_text(proc.stdout, encoding="utf-8")
            (output / stderr).write_text(proc.stderr, encoding="utf-8")
            receipt["commands"].append({"argv": argv, "exit_code": proc.returncode,
                                        "stdout": stdout, "stderr": stderr})
            if proc.returncode:
                raise RuntimeError(f"{script} failed with exit code {proc.returncode}")
        receipt["comparison"] = compare_outputs(study / "outputs", generated)
        if not all(item["identical"] for item in receipt["comparison"].values()):
            raise RuntimeError("Missing or byte-different required statistics outputs")
        receipt["status"] = "passed"
    except (OSError, RuntimeError, metadata.PackageNotFoundError) as exc:
        receipt["error"] = str(exc)
    finally:
        if before is not None:
            receipt["source_sha256_after"] = source_hashes(study)
            receipt["source_unchanged"] = before == receipt["source_sha256_after"]
            if not receipt["source_unchanged"]:
                receipt["status"] = "failed"
                receipt["error"] = "Source files changed during statistics replay"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside the checkout; preserved on failure")
    args = parser.parse_args()
    try:
        receipt = verify_statistics(Path(__file__).resolve().parents[1], args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Statistics-only replay: {exc}\n")
    print(f"STATISTICS-ONLY REPLAY: {receipt['status'].upper()}; "
          f"{sum(item['identical'] for item in receipt['comparison'].values())}/12 exact artifacts")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
