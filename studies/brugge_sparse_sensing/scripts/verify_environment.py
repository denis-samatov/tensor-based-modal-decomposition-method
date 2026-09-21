"""Fail closed when the Brugge study uses a different numerical environment.

The full lock contains notebook and development packages as well as the numerical
stack.  Only packages that can change the study results are enforced here, which
keeps the check portable while preventing a broken virtual environment from
silently falling back to an incompatible system Python.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
NUMERICAL_PACKAGES = {
    "h5py",
    "hdbscan",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "tensorly",
    "torch",
}
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def required_versions(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match and normalise(match.group(1)) in NUMERICAL_PACKAGES:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


def main() -> int:
    lock = Path(sys.argv[1]) if len(sys.argv) > 1 else LOCK
    problems: list[str] = []
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"python: installed {sys.version_info.major}.{sys.version_info.minor}, required 3.12"
        )
    pins = required_versions(lock)
    for name, required in sorted(pins.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != required:
            problems.append(f"{name}: installed {installed}, required {required}")
    if problems:
        print("ENVIRONMENT CHECK FAILED", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    shown = ", ".join(f"{name}={metadata.version(name)}" for name in sorted(pins))
    print(f"ENVIRONMENT VERIFIED: Python {sys.version_info.major}.{sys.version_info.minor}; {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
