"""Data loading, active-cell mask, property-wise normalisation and evaluation folds."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

PROPS = ("pressure", "soil")  # property index k=0 pressure [bar], k=1 oil saturation [-]


def data_dir() -> Path:
    base = Path(os.environ.get("TBMD_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
    return base / "brugge"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Brugge:
    fields: np.ndarray          # (runs, 2, n_active, T) physical units
    active: np.ndarray          # (I, J) bool
    ij: np.ndarray              # (n_active, 2) grid indices of active cells (i along 139, j along 48)
    wells: np.ndarray           # (30, 2) configured well order, grid indices
    well_rows: np.ndarray       # (30,) index into active cells
    names: list

    @property
    def n_runs(self) -> int:
        return self.fields.shape[0]

    @property
    def n_active(self) -> int:
        return self.fields.shape[2]

    @property
    def T(self) -> int:
        return self.fields.shape[3]


def load(dataset: str = "data_exp_4_.h5", wells_file: str = "all_wells_exp_4.json") -> Brugge:
    d = data_dir()
    with h5py.File(d / dataset, "r") as f:
        p = f["pressure"][()]
        s = f["soil"][()]
        names = [n.decode() for n in f["names"][()]]
    # Inactive cells are stored as exact zeros in every run and time step.
    active = (p != 0).all(axis=(0, 3))
    if not np.array_equal(active, (p != 0).any(axis=(0, 3))):
        raise ValueError("pressure zero pattern is not time-invariant")
    ij = np.argwhere(active)
    fields = np.stack([p[:, active, :], s[:, active, :]], axis=1)
    w = json.loads((d / wells_file).read_text())
    cases = [np.asarray(v) for v in w.values()]
    if not all(np.array_equal(cases[0], c) for c in cases):
        raise ValueError("well coordinates differ between cases")
    wells = cases[0]
    # validate + de-duplicate while preserving order
    seen, keep = set(), []
    for i, j in wells:
        if (i, j) not in seen and active[i, j]:
            keep.append((i, j))
            seen.add((i, j))
    wells = np.asarray(keep)
    lut = -np.ones(active.shape, dtype=int)
    lut[active] = np.arange(active.sum())
    well_rows = lut[wells[:, 0], wells[:, 1]]
    return Brugge(fields, active, ij, wells, well_rows, names)


def manifest(dataset: str, wells_file: str) -> dict:
    d = data_dir()
    return {f: {"sha256": sha256(d / f), "bytes": (d / f).stat().st_size} for f in (dataset, wells_file)}


@dataclass
class Scaler:
    lo: np.ndarray  # (2,)
    hi: np.ndarray  # (2,)

    @classmethod
    def fit(cls, fields: np.ndarray) -> "Scaler":
        """fields (..., 2, n_active, T): min/max over active cells of the training data only."""
        f = np.moveaxis(fields, -3, 0).reshape(2, -1)
        return cls(f.min(axis=1), f.max(axis=1))

    def fwd(self, x: np.ndarray) -> np.ndarray:
        return _affine(x, self.lo, self.hi, fwd=True)

    def inv(self, x: np.ndarray) -> np.ndarray:
        return _affine(x, self.lo, self.hi, fwd=False)


def _affine(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, fwd: bool) -> np.ndarray:
    """x has a property axis at position -3 (…, 2, n_active, T) or is a stacked vector
    (…, 2*n_active, T) with pressure block first."""
    x = np.asarray(x, dtype=float)
    if x.ndim >= 3 and x.shape[-3] == 2:
        sh = [1] * x.ndim
        sh[-3] = 2
        l, h = lo.reshape(sh), hi.reshape(sh)
    else:
        n = x.shape[-2] // 2
        sh = [1] * x.ndim
        sh[-2] = 2 * n
        l = np.repeat(lo, n).reshape(sh)
        h = np.repeat(hi, n).reshape(sh)
    return (x - l) / (h - l) if fwd else x * (h - l) + l


def stack(fields_2kt: np.ndarray) -> np.ndarray:
    """(2, n, T) -> (2n, T) with pressure block first."""
    return fields_2kt.reshape(-1, fields_2kt.shape[-1])


@dataclass
class Fold:
    name: str
    test_run: int
    train: np.ndarray      # (2n, T_train) normalised, stacked
    test: np.ndarray       # (2n, T_test) normalised, stacked
    test_phys: np.ndarray  # (2, n, T_test) physical
    prior: np.ndarray      # (2n, T_test) normalised no-measurement reference
    prior_name: str
    scaler: Scaler
    train_runs: list
    train_times: np.ndarray
    test_times: np.ndarray


def folds_p1(b: Brugge, frac: float = 0.8) -> list[Fold]:
    """P1: within-scenario temporal hold-out (original protocol, corrected)."""
    n_tr = int(b.T * frac)
    out = []
    for r in range(b.n_runs):
        tr = b.fields[r, :, :, :n_tr]
        te = b.fields[r, :, :, n_tr:]
        sc = Scaler.fit(tr)
        trn = stack(sc.fwd(tr))
        ten = stack(sc.fwd(te))
        prior = np.repeat(trn[:, -1:], te.shape[-1], axis=1)  # persistence of last training snapshot
        out.append(Fold(f"P1-run{r+1}", r, trn, ten, te, prior, "persistence", sc, [r],
                        np.arange(n_tr), np.arange(n_tr, b.T)))
    return out


def folds_p2(b: Brugge) -> list[Fold]:
    """P2: leave-one-scenario-out; basis from the other nine control scenarios (all times),
    evaluation on every snapshot of the held-out scenario."""
    out = []
    for r in range(b.n_runs):
        tr_runs = [q for q in range(b.n_runs) if q != r]
        tr = b.fields[tr_runs]
        sc = Scaler.fit(tr)
        trn = np.concatenate([stack(sc.fwd(tr[q])) for q in range(len(tr_runs))], axis=1)
        te = b.fields[r]
        ten = stack(sc.fwd(te))
        prior = stack(sc.fwd(tr.mean(axis=0)))  # ensemble mean at the same time index
        out.append(Fold(f"P2-run{r+1}", r, trn, ten, te, prior, "ensemble-mean", sc, tr_runs,
                        np.arange(b.T), np.arange(b.T)))
    return out
