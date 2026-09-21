"""Train-only pressure-saturation dependence diagnostics for E9."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bss import OUT, data, load_config  # noqa: E402

DEST = OUT / "e9_property_mode"


def _canonical_correlations(pressure: np.ndarray, saturation: np.ndarray, rank: int = 16):
    up, _, _ = randomized_svd(pressure, n_components=rank, n_iter=6, random_state=20260921)
    us, _, _ = randomized_svd(saturation, n_components=rank, n_iter=6, random_state=20260921)
    singular = np.linalg.svd(up.T @ us, compute_uv=False)
    return singular


def _property_mode_spectrum(anomalies: np.ndarray) -> np.ndarray:
    scaled = anomalies / anomalies.std(axis=(0, 2, 3), keepdims=True)
    property_matrix = np.moveaxis(scaled, 1, 0).reshape(2, -1)
    covariance = property_matrix @ property_matrix.T
    values = np.linalg.eigvalsh(covariance)[::-1]
    return values / values.sum()


def main() -> None:
    cfg = load_config()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    rows = []
    for outer in range(brugge.n_runs):
        train_runs = [run for run in range(brugge.n_runs) if run != outer]
        fields = brugge.fields[train_runs]
        ensemble_mean = fields.mean(axis=0)
        anomalies = fields - ensemble_mean[None, ...]
        pressure_raw = fields[:, 0].reshape(-1)
        saturation_raw = fields[:, 1].reshape(-1)
        pressure_anomaly = anomalies[:, 0].reshape(-1)
        saturation_anomaly = anomalies[:, 1].reshape(-1)
        p_matrix = anomalies[:, 0].transpose(1, 0, 2).reshape(brugge.n_active, -1)
        s_matrix = anomalies[:, 1].transpose(1, 0, 2).reshape(brugge.n_active, -1)
        alignment = _canonical_correlations(p_matrix, s_matrix)
        spectrum = _property_mode_spectrum(anomalies)
        spatial_mean_p = anomalies[:, 0].mean(axis=1).reshape(-1)
        spatial_mean_s = anomalies[:, 1].mean(axis=1).reshape(-1)
        rows.append(
            {
                "outer_fold": outer + 1,
                "test_run": outer + 1,
                "raw_pearson": float(np.corrcoef(pressure_raw, saturation_raw)[0, 1]),
                "anomaly_pearson": float(
                    np.corrcoef(pressure_anomaly, saturation_anomaly)[0, 1]
                ),
                "temporal_spatial_mean_pearson": float(
                    np.corrcoef(spatial_mean_p, spatial_mean_s)[0, 1]
                ),
                "subspace_cosine_mean_16": float(alignment.mean()),
                "subspace_cosine_min_16": float(alignment.min()),
                "subspace_cosine_max_16": float(alignment.max()),
                "property_rank1_energy": float(spectrum[0]),
                "property_rank2_energy": float(spectrum[1]),
            }
        )
    DEST.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / "cross_property_structure.csv", index=False)
    summary = {
        column: {
            "mean": float(frame[column].mean()),
            "median": float(frame[column].median()),
            "std": float(frame[column].std(ddof=1)),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
        for column in frame.columns
        if column not in {"outer_fold", "test_run"}
    }
    (DEST / "cross_property_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
