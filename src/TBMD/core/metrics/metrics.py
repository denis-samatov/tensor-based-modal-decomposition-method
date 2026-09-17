"""
metrics.py  ·  TBMD utilities
=============================

Complete, self-contained implementation of the four quality metrics that
accompany the TBMD reconstruction experiments:

    • Normalised Frobenius error  (eq. 40 in the paper)
    • Mean-Squared Error (MSE)
    • Structural Similarity Index (SSIM, eq. 41 with C₁ = 0.012, C₂ = 0.032)
    • Peak-Signal-to-Noise-Ratio (PSNR)

The code supports **NumPy** arrays and **PyTorch** tensors, arbitrary spatial
dimensions (2-D, 3-D, …) and optional foreground masks.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(a: ArrayLike) -> np.ndarray:
    """Detaches a torch tensor and converts it to a NumPy array.

    Args:
        a (ArrayLike): The array-like object to convert.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if torch.is_tensor(a):
        a = a.detach().cpu()
    return np.asarray(a)


def _supports_mask() -> bool:
    """Checks if the installed `skimage.ssims` supports the `mask` keyword.

    Returns:
        bool: True if the `mask` keyword is supported, False otherwise.
    """
    from inspect import signature

    return "mask" in signature(_ssim).parameters


def compute_metrics(
    A_rec: ArrayLike,
    A_ref: ArrayLike,
    *,
    background_value: float | None = None,
    mask: Optional[np.ndarray] = None,
    max_val: float | None = None,
) -> Tuple[float, float, float, float]:
    """Computes quality metrics for reconstructed volumes.

    This function calculates the normalized Frobenius error, mean-squared
    error, structural similarity index (SSIM), and peak signal-to-noise ratio
    (PSNR) for a reconstructed volume, with optional masking of background
    voxels.

    Args:
        A_rec (ArrayLike): The reconstructed volume, as a NumPy array or
            PyTorch tensor.
        A_ref (ArrayLike): The reference volume, as a NumPy array or PyTorch
            tensor.
        background_value (Optional[float]): The intensity value that
            represents the background. Voxels with this value are excluded
            unless an explicit `mask` is supplied. Defaults to None.
        mask (Optional[np.ndarray]): A boolean array selecting the foreground.
            Overrides `background_value`. Defaults to None.
        max_val (Optional[float]): The maximum possible pixel/voxel value,
            used for PSNR. If None, defaults to the data range of `A_ref`.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the normalized
        Frobenius error, mean-squared error, SSIM, and PSNR.
    """
    # -- convert & validate -------------------------------------------------
    A_rec = _to_numpy(A_rec)
    A_ref = _to_numpy(A_ref)

    if A_rec.shape != A_ref.shape:
        raise ValueError("A_rec and A_ref must have identical shapes")

    if mask is None:
        if background_value is None:
            mask = np.ones_like(A_ref, dtype=bool)
        else:
            mask = A_ref != background_value
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != A_ref.shape:
            raise ValueError("mask.shape must match the input volumes")

    if not mask.any():
        raise ValueError("Foreground mask is empty – nothing to evaluate")

    # -- normalised Frobenius error & MSE -----------------------------------
    diff = A_rec.astype(np.float64) - A_ref
    diff_fg = diff[mask]
    ref_fg = A_ref[mask].astype(np.float64)

    mse = float(np.mean(diff_fg**2))
    denom = float(np.sum(ref_fg**2))
    err_norm = np.inf if denom == 0 else float(np.sqrt(np.sum(diff_fg**2)) / np.sqrt(denom))

    # -- SSIM ---------------------------------------------------------------
    C1_paper, C2_paper = 0.012, 0.032  # (K₁L)² and (K₂L)² in eq. 41
    data_range = float(A_ref.max() - A_ref.min())
    if data_range < 1e-12:
        ssim_val = 1.0 if np.allclose(A_rec, A_ref) else 0.0
    else:
        K1 = np.sqrt(C1_paper) / data_range
        K2 = np.sqrt(C2_paper) / data_range

        if A_ref.ndim == 2:  # single-channel
            win_size = min(7, min(A_ref.shape))
            if win_size % 2 == 0:
                win_size -= 1
            ssim_val = _ssim(
                A_ref,
                A_rec,
                data_range=data_range,
                K1=K1,
                K2=K2,
                gaussian_weights=True,
                channel_axis=None,
                mask=mask if _supports_mask() else None,  # falls back gracefully
                win_size=win_size,
            )
            if not _supports_mask() and mask is not None:
                logger.warning("SSIM mask ignored: upgrade scikit-image ≥ 0.20 for masked SSIM")
        else:  # channel-last ≥ 3-D
            ssim_vals = []
            for c in range(A_ref.shape[-1]):
                this_mask = mask[..., c] if mask.ndim == A_ref.ndim else mask
                ssim_vals.append(
                    _ssim(
                        A_ref[..., c],
                        A_rec[..., c],
                        data_range=data_range,
                        K1=K1,
                        K2=K2,
                        gaussian_weights=True,
                        channel_axis=None,
                        mask=this_mask if _supports_mask() else None,
                    )
                )
            ssim_val = float(np.mean(ssim_vals))

    # -- PSNR ---------------------------------------------------------------
    if mse == 0:
        psnr = np.inf
    else:
        max_I = float(max_val) if max_val is not None else data_range
        if max_I < 1e-12:
            psnr = 0.0
        else:
            psnr = float(20 * np.log10(max_I / np.sqrt(mse)))

    return err_norm, mse, ssim_val, psnr
