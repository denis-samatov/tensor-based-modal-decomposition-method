"""Keep historical plotting entry points compatible with the shared implementation."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from TBMD.experiments import runner
from TBMD.visualization import experiments as plotting


@pytest.mark.parametrize("plot_type", ["individual", "combined", "normalized", "all"])
@pytest.mark.parametrize("statistics", [False, True])
def test_plotting_entry_points_render_identically(tmp_path, plot_type, statistics):
    frame = pd.DataFrame(
        {
            "sensors": [2, 5, 9],
            "error": [0.8, 0.4, 0.2],
            "ssim": [0.2, 0.6, 0.9],
            "psnr": [10.0, 15.0, 20.0],
        }
    )
    if statistics:
        for metric in ("error", "ssim", "psnr"):
            values = frame.pop(metric)
            frame[f"{metric}_mean"] = values
            frame[f"{metric}_std"] = 0.1
            frame[f"{metric}_ci_lower"] = values - 0.1
            frame[f"{metric}_ci_upper"] = values + 0.1

    assert inspect.signature(runner.plot_analytics) == inspect.signature(plotting.plot_analytics)
    for name, function in (
        ("historical", runner.plot_analytics),
        ("shared", plotting.plot_analytics),
    ):
        assert (
            function(frame, plot_type=plot_type, save_path=str(tmp_path / name), show_plots=False)
            is None
        )
        assert not plt.get_fignums()

    historical = sorted(tmp_path.glob("historical*.png"))
    shared = sorted(tmp_path.glob("shared*.png"))
    assert historical
    assert [p.name.removeprefix("historical") for p in historical] == [
        p.name.removeprefix("shared") for p in shared
    ]
    for old, new in zip(historical, shared):
        with Image.open(old) as a, Image.open(new) as b:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("name", ["plot_analytics", "plot_analytics_legacy"])
def test_historical_entry_points_share_the_plotting_implementation(name):
    assert getattr(runner, name) is getattr(plotting, name)
