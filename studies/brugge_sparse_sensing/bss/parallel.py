"""Shell-level parallelism helper (process pools are unavailable in some sandboxes).

Usage inside a script:
    python script.py part <tag> <args...>   -> runs one job, pickles result to outputs/parts
    python script.py merge <tag> <n_parts> -> returns list of results in part order
Driver: run_all.sh launches parts with `xargs -P`.
"""
import pickle

from . import OUT

PARTS = OUT / "parts"


def save_part(tag: str, key: str, obj) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    with open(PARTS / f"{tag}__{key}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def load_parts(tag: str) -> list:
    files = sorted(PARTS.glob(f"{tag}__*.pkl"))
    out = []
    for p in files:
        with open(p, "rb") as fh:
            out.append((p.stem.split("__", 1)[1], pickle.load(fh)))
    return out
