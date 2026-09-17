"""E1 -- Representation capacity (full-observation oracle) of POD, energy-weighted POD and TBMD.

For every fold: least-squares projection of each held-out snapshot onto the basis using *all*
active channels (an upper bound on what any sparse sensing design can achieve with that basis),
as a function of dictionary depth r, and, for TBMD, of the spatial Tucker rank R1 (= R2 capped
at 48). Also records training reconstruction error, decomposition time and storage.
Output: outputs/e1_representation.csv
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import bases, data, metrics  # noqa: E402


def job(args):
    protocol, k = args
    cfg = load_config()
    b = data.load(cfg["dataset"], cfg["wells"])
    f = (data.folds_p1(b, cfg["train_fraction_p1"]) if protocol == "P1" else data.folds_p2(b))[k]
    n = b.n_active
    rows = []
    r_energy = bases.pod(f.train, threshold=cfg["energy_threshold"]).extra["rank"]
    mp = metrics.evaluate(f.prior, f, n)
    rows.append(dict(protocol=protocol, fold=f.name, basis="prior", r=0, R1=0, rmse_p=mp["rmse_p"].mean(),
                     rmse_so=mp["rmse_so"].mean(), relerr=mp["relerr_norm"].mean(), r_energy=r_energy))

    def rec(name, B, r, R1, sec, extra):
        X = np.linalg.lstsq(B, f.test, rcond=None)[0]
        m = metrics.evaluate(B @ X, f, n)
        rows.append(dict(protocol=protocol, fold=f.name, basis=name, r=r, R1=R1, rmse_p=m["rmse_p"].mean(),
                         rmse_so=m["rmse_so"].mean(), relerr=m["relerr_norm"].mean(), seconds=sec,
                         r_energy=r_energy, **extra))

    for r in sorted(set(cfg["rank_sweep"] + [r_energy])):
        if r > min(f.train.shape):
            continue
        p = bases.pod(f.train, r=r)
        rec("POD", p.B, r, 0, p.seconds, {"storage_floats": p.B.size})
        t = bases.tbmd(f.train, b.ij, r, spatial_ranks=cfg["tbmd_spatial_ranks"], **cfg["tucker"])
        rec("TBMD", t.B, r, cfg["tbmd_spatial_ranks"][0], t.seconds,
            {"train_rel_error": t.extra["train_rel_error"], "storage_floats": t.extra["storage_floats"]})
    for R1 in cfg["spatial_rank_sweep"]:
        t = bases.tbmd(f.train, b.ij, r_energy, spatial_ranks=(R1, min(R1, 48), 2), **cfg["tucker"])
        rec("TBMD-R1sweep", t.B, r_energy, R1, t.seconds,
            {"train_rel_error": t.extra["train_rel_error"], "storage_floats": t.extra["storage_floats"],
             "R2": min(R1, 48)})
    print(protocol, k + 1, "done", flush=True)
    return rows


if __name__ == "__main__":
    from bss import parallel
    if sys.argv[1] == "part":
        protocol, k = sys.argv[2], int(sys.argv[3])
        parallel.save_part("e1", f"{protocol}-{k:02d}", job((protocol, k)))
    else:
        out = [r for _, rows in parallel.load_parts("e1") for r in rows]
        pd.DataFrame(out).to_csv(OUT / "e1_representation.csv", index=False)
