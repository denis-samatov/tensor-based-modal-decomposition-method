"""Independent index/contraction checks; synthetic checks are not field results."""
import json
import sys
from pathlib import Path

import numpy as np
import tensorly as tl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bss import OUT, load_config
from bss import bases, data, estimators, placement
from TBMD.core.modal_processor.modes import ModalProcessorConfig, TimeInsensitiveModeComputer


def main():
    rng = np.random.default_rng(20260919)
    core = rng.normal(size=(2, 3, 2, 4))
    factors = [rng.normal(size=s) for s in ((5, 2), (6, 3), (2, 2), (7, 4))]
    explicit = np.einsum("ia,jb,kc,abcw->ijkw", *factors[:3], core)
    modes = tl.tenalg.multi_mode_dot(core, factors[:3], modes=[0, 1, 2])
    np.testing.assert_allclose(modes, explicit, rtol=1e-12, atol=1e-12)
    reconstruction = np.einsum("ijkw,lw->ijkl", modes, factors[3])
    np.testing.assert_allclose(reconstruction, tl.tucker_to_tensor((core, factors)), atol=1e-12)
    computer = TimeInsensitiveModeComputer(ModalProcessorConfig(numerical_precision=torch.float64))
    library = np.stack([computer.compute_single_mode(
        [torch.from_numpy(f) for f in factors[:3]], torch.from_numpy(core[..., w])
    ).numpy() for w in range(4)], axis=-1)
    np.testing.assert_allclose(library, explicit, rtol=1e-12, atol=1e-12)
    # The public PyTorch processor switches TensorLy's process-wide backend.
    tl.set_backend("numpy")
    # A singleton property factor is exactly the third-order construction.
    core3 = core[:, :, 0, :]
    modes3 = np.einsum("ia,jb,abw->ijw", *factors[:2], core3)
    modes4 = tl.tenalg.multi_mode_dot(core3[:, :, None, :],
                                    factors[:2] + [np.ones((1, 1))], modes=[0, 1, 2])
    np.testing.assert_allclose(modes4[:, :, 0, :], modes3, atol=1e-12)
    # Non-contiguous active cells exercise the property-major/raster permutation.
    ij = np.array([[0, 1], [1, 4], [3, 0], [4, 5]])
    stacked = bases.from_grid(modes, ij)
    expected = np.concatenate([explicit[ij[:, 0], ij[:, 1], k] for k in range(2)])
    np.testing.assert_array_equal(stacked, bases.from_grid(bases.to_grid(stacked, ij, (5, 6)), ij))
    np.testing.assert_allclose(stacked, expected, atol=1e-12)
    rows = np.array([0, 6, 3, 5, 1, 7])
    H = np.eye(len(stacked))[rows]
    coefficients = rng.normal(size=(4, 3))
    truth = stacked @ coefficients
    np.testing.assert_array_equal(H @ truth, truth[rows])
    rec = stacked @ estimators.least_squares(H @ stacked, H @ truth)
    np.testing.assert_allclose(rec, truth, atol=1e-11)
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    np.testing.assert_array_equal(placement.qr_dg(stacked, 4), placement.qr_dg(stacked @ rotation, 4))
    # TBMD/POD-E right-orthogonal rotations preserve the minimum-norm LS state
    # even when the sampled basis is rank deficient: (A Q)^+ = Q.T A^+.
    deficient_rows = np.array([0, 2])
    assert np.linalg.matrix_rank(stacked[deficient_rows]) < stacked.shape[1]
    deficient_y = truth[deficient_rows]
    rec_base = stacked @ estimators.least_squares(stacked[deficient_rows], deficient_y)
    rotated = stacked @ rotation
    rec_rotated = rotated @ estimators.least_squares(rotated[deficient_rows], deficient_y)
    np.testing.assert_allclose(rec_rotated, rec_base, rtol=1e-11, atol=1e-11)
    # POD and POD-E use non-orthogonal diagonal scaling. Their sampled LS
    # states need not agree when the sampled matrix lacks full column rank.
    pod, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    energy = np.diag([3.0, 1.0])
    sample = np.array([0])
    y_sample = np.array([[1.0]])
    rec_pod = pod @ estimators.least_squares(pod[sample], y_sample)
    rec_pode = pod @ energy @ estimators.least_squares((pod @ energy)[sample], y_sample)
    assert not np.allclose(rec_pod, rec_pode, rtol=1e-6, atol=1e-6)
    # Untruncated property factors cancel out of the represented joint modes.
    property_rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    projected = tl.tenalg.mode_dot(modes, property_rotation.T, 2)
    recovered = tl.tenalg.mode_dot(projected, property_rotation, 2)
    np.testing.assert_allclose(recovered, modes, atol=1e-12)
    cfg = load_config()
    b = data.load(cfg["dataset"], cfg["wells"])
    folds = data.folds_p1(b, cfg["train_fraction_p1"]) + data.folds_p2(b)
    shapes = []
    for fold in folds:
        if fold.name.startswith("P1"):
            assert set(fold.train_times).isdisjoint(fold.test_times)
            training = b.fields[fold.test_run][..., fold.train_times]
        else:
            assert fold.test_run not in fold.train_runs
            training = b.fields[fold.train_runs]
        scaler = data.Scaler.fit(training)
        np.testing.assert_array_equal(scaler.lo, fold.scaler.lo)
        np.testing.assert_array_equal(scaler.hi, fold.scaler.hi)
        shapes.append(dict(fold=fold.name, train=list(fold.train.shape), test=list(fold.test.shape),
                           grid=[139, 48, 2, fold.train.shape[1]]))
    result = dict(synthetic_checks="PASS", real_fold_partition_and_scaling_checks="PASS",
                  public_library_is_authors_implementation_not_Zhong_reference_code=True,
                  rank_deficient_tbmd_pode_rotation_invariance="PASS",
                  rank_deficient_pod_pode_equivalence_counterexample="PASS",
                  max_explicit_contraction_difference=float(np.max(np.abs(modes-explicit))),
                  max_public_contraction_difference=float(np.max(np.abs(library-explicit))),
                  active_cells=b.n_active, shapes=shapes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v3_shape_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
