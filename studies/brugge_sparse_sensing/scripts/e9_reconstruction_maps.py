"""Fixed representative E9 reconstruction maps (fold 1, final snapshot, 15 wells)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bss import FIG, OUT, data, load_config  # noqa: E402
from bss.e9_metrics import MetricReference, evaluate_reconstruction  # noqa: E402
from bss.e9_study import build_e9_candidate_specs, fit_e9_model  # noqa: E402
from bss.study import solve_coefficients  # noqa: E402
from e9_nested_property_mode import (  # noqa: E402
    _measurement_rows,
    _protocol,
    _solver_lookup,
)

DEST = OUT / "e9_property_mode"
FIGURES = FIG
plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
OUTER_INDEX = 0
TIME_INDEX = 132
CAPACITY = "equal_total"
GEOMETRY = "wells"
BUDGET = 15
FOUR_D_VARIANTS = {"4D-A", "4D-B", "4D-C", "4D-D", "4D-E"}


def _grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(mask.shape, np.nan)
    result[mask] = values
    return result


def main() -> None:
    selection = json.loads((DEST / "parts" / "selection_01.json").read_text())
    options = [
        row
        for row in selection["paired_sparse"]
        if row["capacity_regime"] == CAPACITY
        and row["geometry"] == GEOMETRY
        and row["budget"] == BUDGET
        and row["four_d_variant"] in FOUR_D_VARIANTS
    ]
    pair = min(options, key=lambda row: (row["pair_inner_loss"], row["four_d_variant"]))
    specs = {spec["id"]: spec for spec in build_e9_candidate_specs()}
    cfg = load_config()
    brugge = data.load(cfg["dataset"], cfg["wells"])
    train_runs = [run for run in range(brugge.n_runs) if run != OUTER_INDEX]
    raw_fit = brugge.fields[train_runs]
    truth = brugge.fields[OUTER_INDEX]
    rows = _measurement_rows(
        GEOMETRY,
        BUDGET,
        n_active=brugge.n_active,
        well_rows=brugge.well_rows,
        grid_order=np.empty(0, dtype=int),
    )
    solver = _solver_lookup(_protocol())[pair["solver"]]
    reconstructions = {}
    for label, candidate in (
        ("Independent 3D", pair["candidate_3d"]),
        (f"Optimized {pair['four_d_variant']}", pair["candidate_4d"]),
    ):
        transform, model, _ = fit_e9_model(raw_fit, brugge.ij, specs[candidate])
        target = transform.fwd(truth).reshape(-1, brugge.T)
        coefficients, _ = solve_coefficients(model.B[rows], target[rows], solver)
        reconstructions[label] = transform.inv(
            (model.B @ coefficients).reshape(2, brugge.n_active, brugge.T)
        )

    reference = MetricReference.fit(raw_fit)
    one_time_reference = MetricReference(
        ensemble_mean=reference.ensemble_mean[..., TIME_INDEX : TIME_INDEX + 1],
        raw_range=reference.raw_range,
        anomaly_range=reference.anomaly_range,
    )
    scores = {
        label: evaluate_reconstruction(
            estimate[..., TIME_INDEX : TIME_INDEX + 1],
            truth[..., TIME_INDEX : TIME_INDEX + 1],
            one_time_reference,
            brugge.active,
        )
        for label, estimate in reconstructions.items()
    }

    fig, axes = plt.subplots(2, 5, figsize=(10, 8.0), constrained_layout=True)
    properties = ((0, "Pressure", "p"), (1, "Oil saturation", "so"))
    labels = list(reconstructions)
    for row_index, (prop, property_name, suffix) in enumerate(properties):
        truth_grid = _grid(truth[prop, :, TIME_INDEX], brugge.active)
        estimate_3d = _grid(
            reconstructions[labels[0]][prop, :, TIME_INDEX], brugge.active
        )
        estimate_4d = _grid(
            reconstructions[labels[1]][prop, :, TIME_INDEX], brugge.active
        )
        lower = float(np.nanmin([truth_grid, estimate_3d, estimate_4d]))
        upper = float(np.nanmax([truth_grid, estimate_3d, estimate_4d]))
        errors = (np.abs(estimate_3d - truth_grid), np.abs(estimate_4d - truth_grid))
        error_max = float(max(np.nanmax(errors[0]), np.nanmax(errors[1])))
        fields = (truth_grid, estimate_3d, estimate_4d, *errors)
        for column, field in enumerate(fields):
            kwargs = (
                {"cmap": "viridis", "vmin": lower, "vmax": upper}
                if column < 3
                else {"cmap": "magma", "vmin": 0.0, "vmax": error_max}
            )
            image = axes[row_index, column].imshow(field, origin="lower", **kwargs)
            axes[row_index, column].set_xticks([0, 24, 47])
            axes[row_index, column].set_yticks([0, 69, 138])
            if column > 0:
                axes[row_index, column].tick_params(labelleft=False)
            axes[row_index, column].set_xlabel("j (grid index)")
            observed = np.unique(rows % brugge.n_active)
            well_ij = brugge.ij[observed]
            axes[row_index, column].scatter(well_ij[:, 1], well_ij[:, 0], s=10, facecolors="none", edgecolors="white", linewidths=0.65)

            if column in {2, 4}:
                fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, label="$u_p$" if prop == 0 else "$S_o$ (-)")
        axes[row_index, 0].set_ylabel(property_name + "\ni (grid index)")
        axes[row_index, 1].set_xlabel(
            f"$E'={scores[labels[0]][f'anomaly_relative_frobenius_{suffix}']:.3f}$\n"
            f"SSIM$'={scores[labels[0]][f'anomaly_ssim_{suffix}']:.3f}$"
        )
        axes[row_index, 2].set_xlabel(
            f"$E'={scores[labels[1]][f'anomaly_relative_frobenius_{suffix}']:.3f}$\n"
            f"SSIM$'={scores[labels[1]][f'anomaly_ssim_{suffix}']:.3f}$"
        )
    for axis, title in zip(
        axes[0],
        ("Reference", "Independent\n3D", f"Optimized\n{pair['four_d_variant']}", "|3D error|", "|4D error|"),
        strict=True,
    ):
        axis.set_title(title)
    fig.suptitle(
        f"Held-out scenario 1, report step {TIME_INDEX}\n"
        f"{BUDGET} configured wells, {CAPACITY.replace('_', ' ')}"
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig19_e9_reconstruction_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig19_e9_reconstruction_maps.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "selection_rule": "fixed before outer evaluation",
        "outer_fold": 1,
        "test_run": 1,
        "time_index": TIME_INDEX,
        "capacity_regime": CAPACITY,
        "geometry": GEOMETRY,
        "budget": BUDGET,
        "selected_pair": pair,
        "snapshot_scores": scores,
    }
    (DEST / "representative_map_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
