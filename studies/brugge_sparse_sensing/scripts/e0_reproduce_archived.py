"""E0 -- Provenance check of the archived CAGEO-D-26-01439 numbers.

Re-executes the configuration found in
the archived original notebooks/experiments/exp_tbmd_2.1&2.2.ipynb
(docs/paper/audit/historical_notebooks.zip; hashes in MANIFEST.json) with the public TBMD library
(no modification of library code):

* data_exp_4_.h5, sequential 80/20 split, min-max fitted on the training part
  of all runs pooled (calculate_global_minmax_params over the dict, BG=None);
* TuckerDecomposer(ranks=None, epsilon=1e-2) over the dict of runs, modal tensors
  stacked across runs (pooled dictionary);
* subject = 3rd key ('case3'), single held-out slice index 10;
* ExperimentRunner with noise_level=0.1, num_noise_samples=10, delta_0=0.1,
  convergence_tol=1e-7 (as in the notebook);
* well prefixes N = 1..30 and grid-wide QR budgets N = 1, 11, ..., 351.

Variants: property = 'pressure' (single property) and 'all' (joint).
Outputs: outputs/e0_archived_<property>_<regime>.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorly as tl
import torch

from TBMD.config import SEED
from TBMD.core.data.loaders import DataLoader
from TBMD.core.data.processors import calculate_global_minmax_params, process_data
from TBMD.core.data.splitters import split_data_in_memory_ordered
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.core.modal_processor.modes import (
    BatchModalProcessor,
    ModalProcessorConfig,
    ModalTensorStacker,
    ProcessingStrategy,
)
from TBMD.core.utils.misc import set_seed
from TBMD.experiments import ExperimentConfig, ExperimentRunner

HERE = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(HERE))
from bss import OUT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", choices=["pressure", "soil", "all"], default="pressure")
    ap.add_argument("--regime", choices=["wells", "grid"], default="wells")
    ap.add_argument("--data-dir", default=None, help="directory containing brugge/ (default: TBMD_DATA_DIR or repo data/)")
    args = ap.parse_args()

    set_seed(SEED)
    np.random.seed(SEED)
    tl.set_backend("pytorch")
    from bss.data import data_dir as _dd
    data_dir = Path(args.data_dir) / "brugge" if args.data_dir else _dd()
    tensors = DataLoader.load_h5_tensors(data_dir / "data_exp_4_.h5")
    wells = DataLoader.load_wells_from_json(data_dir / "all_wells_exp_4.json")

    train, test = split_data_in_memory_ordered(tensors[args.property], train_ratio=0.8)
    subject = list(tensors[args.property].keys())[2]
    mn, mx = calculate_global_minmax_params(train, background_value=None)
    params = {"min": mn, "max": mx}
    tr = process_data(train, normalization_method="minmax", global_params=params, verbose=False)
    te = process_data(test, normalization_method="minmax", global_params=params, verbose=False)

    t0 = time.perf_counter()
    dec = TuckerDecomposer(tensors=tr, ranks=None, epsilon=1e-2, device="cpu", random_state=SEED)
    dec.decompose()
    cfg = ModalProcessorConfig(device="cpu", processing_strategy=ProcessingStrategy.BATCH,
                               enable_progress_logging=False, return_numpy=False)
    modal = BatchModalProcessor(cfg).process_multiple_subjects(dec.cores, dec.factors)
    A = ModalTensorStacker(cfg).stack_modal_tensors(modal)
    t_dec = time.perf_counter() - t0
    core_shape = list(dec.cores[subject].shape)

    rcfg = ExperimentConfig(solver_method="triangular", max_iter=1000, epsilon=1e-2, lambd=0.95,
                            delta_0=0.1, delta_max=1.0, noise_level=0.1, num_noise_samples=10,
                            confidence_level=0.95, convergence_tol=1e-7, subject_axis=False,
                            valid_mask=None, wells=wells, seed=SEED, device="cpu", verbose=False)
    runner = ExperimentRunner(rcfg)
    if args.regime == "wells":
        df = runner.run_single_slice_wells_experiments(A, te, subject, 10, list(range(1, 31)))
    else:
        W = A.shape[-1]
        vals = [int(v) for v in np.arange(1, 360, 10) if v <= W]
        df = runner.run_single_slice_experiments(A, te, subject, 10, vals)
    out = OUT / f"e0_archived_{args.property}_{args.regime}.csv"
    df.to_csv(out, index=False)
    meta = {"subject": subject, "train_shape": list(next(iter(tr.values())).shape),
            "test_shape": list(next(iter(te.values())).shape), "minmax": [float(mn), float(mx)],
            "core_shape": core_shape, "A_shape": list(A.shape), "decomp_seconds": t_dec,
            "torch": torch.__version__}
    (OUT / f"e0_archived_{args.property}_{args.regime}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(df[["sensors", "error_mean", "ssim_mean", "psnr_mean", "num_samples"]].to_string())


if __name__ == "__main__":
    main()
