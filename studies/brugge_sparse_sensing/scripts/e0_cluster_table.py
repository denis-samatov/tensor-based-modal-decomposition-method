"""E0b -- Provenance check of Supplementary Table S1 (cluster diagnostic) and of the
joint-tensor Tucker ranks, re-executing exp_tbmd_2.3_probability.ipynb logic with the
public TBMD library. Output: outputs/e0_cluster_table.csv, outputs/e0_joint_ranks.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorly as tl
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from TBMD.config import SEED
from TBMD.core.data.loaders import DataLoader
from TBMD.core.data.processors import calculate_global_minmax_params, process_data
from TBMD.core.data.splitters import split_data_in_memory_ordered
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalProcessorConfig, ModalTensorStacker, ProcessingStrategy
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.core.utils.misc import build_wells_matrix, set_seed

HERE = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(HERE))
from bss import OUT  # noqa: E402
set_seed(SEED); tl.set_backend("pytorch")
from bss.data import data_dir  # noqa: E402
T = DataLoader.load_h5_tensors(data_dir() / "data_exp_4_.h5")
wells = DataLoader.load_wells_from_json(data_dir() / "all_wells_exp_4.json")
cfg = ModalProcessorConfig(device="cpu", processing_strategy=ProcessingStrategy.BATCH, enable_progress_logging=False, return_numpy=False)

def dictionary(prop):
    tr, te = split_data_in_memory_ordered(T[prop], 0.8)
    mn, mx = calculate_global_minmax_params(tr)
    trn = process_data(tr, normalization_method="minmax", global_params={"min": mn, "max": mx}, verbose=False)
    dec = TuckerDecomposer(tensors=trn, ranks=None, epsilon=1e-2, device="cpu", random_state=SEED); dec.decompose()
    A = ModalTensorStacker(cfg).stack_modal_tensors(BatchModalProcessor(cfg).process_multiple_subjects(dec.cores, dec.factors))
    return A, tr, te, dec

# joint ranks actually used by the public default (ranks=None)
Aj, _, _, decj = dictionary("all")
json.dump({"joint_core_shape": list(decj.cores["case1"].shape), "joint_A_shape": list(Aj.shape)},
          open(OUT / "e0_joint_ranks.json", "w"), indent=2)
print("joint", list(decj.cores["case1"].shape), list(Aj.shape))

A, tr, te, _ = dictionary("pressure")
P, Q, R = TensorTubeQRDecomposition(tensor=A, N=300, random_state=SEED, check_orthogonality=False, uniform_distribution=False).factorize()
subject = "case10"
wm = build_wells_matrix(wells, A.shape, device="cpu")[subject]
wp = torch.nonzero(wm).numpy()
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(wp)
bg = np.asarray(te[subject][..., 10]); mask = bg > 0
ys, xs = np.where(mask); lab = cdist(np.c_[ys, xs], km.cluster_centers_).argmin(1)
region = np.full(mask.shape, -1); region[ys, xs] = lab
sp = np.argwhere(P.numpy() == 1); inside = region[sp[:, 0], sp[:, 1]] >= 0
sc = region[sp[inside, 0], sp[inside, 1]]
counts = np.bincount(sc, minlength=5); areas = np.bincount(lab, minlength=5)
df = pd.DataFrame({"cluster": range(5), "area_frac": areas / areas.sum(), "n_sensors": counts,
                   "P_C_given_S": counts / counts.sum(), "density": counts / areas})
df["total_inside"] = counts.sum(); df["total_selected"] = int(P.sum())
df.to_csv(OUT / "e0_cluster_table.csv", index=False)
print(df.round(3).to_string())
