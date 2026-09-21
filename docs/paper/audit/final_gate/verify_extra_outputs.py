"""Supplementary clean-room checks, beyond the main CSV/macro comparator.

Run with one argument: the independent output directory. This is an internal
verification aid, not a numerical experiment or Editorial Manager upload.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CANONICAL = ROOT / "studies/brugge_sparse_sensing/outputs"
RERUN = Path(sys.argv[1])


def compare(a, b, path="", excluded=frozenset()):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            if key not in excluded:
                compare(a[key], b[key], f"{path}/{key}", excluded)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f"{path}/{i}", excluded)
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        assert type(a) is type(b) and a == b, path
    else:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, err_msg=path)


for name, excluded in (
    ("s0_data_manifest.json", {"platform", "git_commit"}),
    ("e2_bases.json", {"seconds"}),
    ("v1_verification.json", set()),
    ("v2_admm_convergence.json", set()),
    ("v3_shape_contract.json", set()),
):
    compare(json.loads((CANONICAL / name).read_text()),
            json.loads((RERUN / name).read_text()), name, excluded)
    print(f"MATCH {name}; excluded fields: {sorted(excluded)}")

name = "e5_selection_frequency.npz"
with np.load(CANONICAL / name, allow_pickle=False) as a, np.load(RERUN / name, allow_pickle=False) as b:
    assert a.files == b.files
    for key in a.files:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-12, err_msg=key)
    print(f"MATCH {name}; {len(a.files)} arrays, including coordinates and sensor frequencies")

for name in ("e8_property_coupling.csv", "e8_property_coupling_summary.csv", "tabS_coupling.tex", "v3_shape_contract.json"):
    assert (CANONICAL / name).read_bytes() == (RERUN / name).read_bytes(), name
    print(f"BYTE-IDENTICAL {name}")
print("EXTRA OUTPUT CHECKS PASSED")
