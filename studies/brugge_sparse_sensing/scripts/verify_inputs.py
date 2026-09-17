"""Verify that the two input files under $TBMD_DATA_DIR/brugge/ match the checksums in inputs.sha256.
Usage: python scripts/verify_inputs.py   (exit code 1 on a missing file or a checksum mismatch)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bss import ROOT  # noqa: E402
from bss.data import data_dir, sha256  # noqa: E402

ok = True
for line in (ROOT / "inputs.sha256").read_text().splitlines():
    expected, name = line.split()
    path = data_dir() / name
    if not path.exists():
        print(f"MISSING {path}")
        ok = False
    elif sha256(path) != expected:
        print(f"MISMATCH {name}: expected {expected}")
        ok = False
    else:
        print(f"OK {name}")
print("INPUTS VERIFIED" if ok else "INPUT VERIFICATION FAILED (see README, section 2)")
sys.exit(0 if ok else 1)
