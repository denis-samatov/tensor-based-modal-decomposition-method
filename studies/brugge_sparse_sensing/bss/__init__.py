"""Brugge sparse-sensing study (bss): leakage-free benchmark of tensor-based modal
decomposition (TBMD), POD, sensor placement and sparse reconstruction.

All computation used in the revised manuscript is implemented in this package and in
../scripts. Nothing here depends on archived plotting artefacts."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# BSS_OUT / BSS_FIG allow an independent re-run into separate directories
OUT = Path(os.environ.get("BSS_OUT", ROOT / "outputs"))
FIG = Path(os.environ.get("BSS_FIG", ROOT / "figures"))


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text())
