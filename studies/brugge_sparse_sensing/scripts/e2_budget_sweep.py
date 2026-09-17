"""E2 -- Main benchmark: bases x placements x budgets x estimators, protocols P1 and P2.
Outputs: outputs/e2_summary.csv (one row per fold/configuration, metrics averaged over test
snapshots), outputs/e2_snapshots.csv.gz (per-snapshot errors, deterministic placements),
outputs/e2_bases.json."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from bss import OUT, load_config  # noqa: E402
from bss import data, experiment  # noqa: E402


def job(args):
    protocol, k = args
    cfg = load_config()
    b = data.load(cfg["dataset"], cfg["wells"])
    folds = data.folds_p1(b, cfg["train_fraction_p1"]) if protocol == "P1" else data.folds_p2(b)
    t0 = time.perf_counter()
    recs, snaps, info = experiment.run_fold(protocol, folds[k], b, cfg)
    print(f"{protocol} fold {k+1} done in {time.perf_counter()-t0:.1f}s", flush=True)
    return recs, snaps, {f"{protocol}-{k+1}": info}


if __name__ == "__main__":
    from bss import parallel
    if sys.argv[1] == "part":
        protocol, k = sys.argv[2], int(sys.argv[3])
        parallel.save_part("e2", f"{protocol}-{k:02d}", job((protocol, k)))
    else:
        allr, alls, infos = [], [], {}
        for _, (recs, snaps, info) in parallel.load_parts("e2"):
            allr += recs; alls += snaps; infos.update(info)
        pd.DataFrame(allr).to_csv(OUT / "e2_summary.csv", index=False)
        pd.DataFrame(alls, columns=["protocol", "fold", "sensing", "placement", "N", "basis", "estimator",
                                    "time_index", "rmse_p", "rmse_so"]).to_csv(OUT / "e2_snapshots.csv.gz", index=False)
        (OUT / "e2_bases.json").write_text(json.dumps(infos, indent=2))
        print(len(allr), "records")
