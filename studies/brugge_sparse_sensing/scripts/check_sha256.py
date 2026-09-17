"""Check files against a SHA-256 list in `sha256sum` format (paths relative to the study directory).
Usage: python scripts/check_sha256.py [SHA256SUMS]   (exit code 1 on a missing or modified file)"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
listing = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "SHA256SUMS")
n_ok, bad = 0, []
for line in listing.read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = ROOT / name.strip()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        n_ok += 1
    else:
        bad.append(name.strip())
for name in bad:
    print(f"MISSING OR MODIFIED {name}")
print(f"{n_ok} files match {listing.name}; {len(bad)} missing or modified")
sys.exit(1 if bad else 0)
