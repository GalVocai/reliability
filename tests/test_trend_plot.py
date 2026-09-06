"""Unit tests for the trend-plot scale/unit conversion (e.g. V -> mV)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from sensorlab.plotting.trend_plot import (day_legend_label, plot_metric_average,
                                           plot_metric_devices, time_legend_title)


def _agg(values: list[float], avdev: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "day_num": list(range(len(values))),
        "value_mean": values, "value_avdev": [avdev] * len(values),
        "delta_mean": values, "delta_avdev": [avdev] * len(values),
    })


def _tidy(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ESN": ["1"] * len(values), "TNumber": ["11"] * len(values),
        "day_num": list(range(len(values))), "delta": values,
    })


@pytest.fixture
def keep_figure_open(monkeypatch):
    """plot_* closes its figure before returning; intercept that so the test
    can inspect the just-built figure via plt.gcf(), then close it itself."""
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    yield
    plt.close("all")


def test_plot_metric_average_scale_converts_v_to_mv(tmp_path, keep_figure_open):
    out = str(tmp_path / "avg.png")
    plot_metric_average(_agg([-0.003, -0.005]), out, "Delta_Vth_bg0", [],
                        value="delta", scale=1000.0, unit="mV")
    ax = plt.gcf().axes[0]
    y = ax.containers[0].lines[0].get_ydata()
    assert list(y) == pytest.approx([-3.0, -5.0])   # -0.003/-0.005 V -> -3/-5 mV
    assert "mV" in ax.get_ylabel()


def test_plot_metric_average_no_scale_leaves_value_untouched(tmp_path, keep_figure_open):
    out = str(tmp_path / "avg2.png")
    plot_metric_average(_agg([10.0, 12.0]), out, "response_pct", [], value="delta")
    ax = plt.gcf().axes[0]
    y = ax.containers[0].lines[0].get_ydata()
    assert list(y) == pytest.approx([10.0, 12.0])
    assert "mV" not in ax.get_ylabel()


def test_plot_metric_average_can_use_cycle_coordinate(tmp_path, keep_figure_open):
    out = str(tmp_path / "cycles.png")
    agg = _agg([10.0, 12.0])
    agg["cycle_num"] = [0.0, 24.0]
    plot_metric_average(agg, out, "response", [], x_col="cycle_num",
                        xlabel="Stress Cycle #")
    ax = plt.gcf().axes[0]
    assert list(ax.containers[0].lines[0].get_xdata()) == pytest.approx([0.0, 24.0])
    assert ax.get_xlabel() == "Stress Cycle #"


def test_cycle_legend_converts_days_at_two_hours_per_cycle():
    assert day_legend_label(0, "cycles", 2.0) == "Baseline"
    assert day_legend_label(2, "cycles", 2.0) == "24 cycles"
    assert day_legend_label(7, "cycles", 2.0) == "84 cycles"
    assert day_legend_label(13, "cycles", 2.0) == "156 cycles"
    assert time_legend_title("cycles") == "Cycle #"


def test_plot_metric_average_negative_scale_keeps_error_bars_positive(tmp_path, keep_figure_open):
    """delta_Vth is plotted with a NEGATIVE scale to flip the sign convention.
    The mean must flip, but the +/-AVDEV error bar is a magnitude -- passing a
    negative yerr to matplotlib raises 'yerr must not contain negative values'
    (the crash seen in the GUI). Regression: non-zero avdev + negative scale."""
    out = str(tmp_path / "neg.png")
    plot_metric_average(_agg([-0.003, -0.005], avdev=0.001), out, "delta_Vth_bg0", [],
                        value="delta", scale=-1000.0, unit="mV")
    ax = plt.gcf().axes[0]
    y = ax.containers[0].lines[0].get_ydata()
    assert list(y) == pytest.approx([3.0, 5.0])   # sign flipped by the -1000 scale


def test_plot_metric_devices_scale_converts_v_to_mv(tmp_path, keep_figure_open):
    out = str(tmp_path / "dev.png")
    plot_metric_devices(_tidy([-0.001, -0.002]), out, "Vth_bg0", ["ESN", "TNumber"],
                        value_col="delta", scale=1000.0, unit="mV")
    ax = plt.gcf().axes[0]
    y = ax.lines[0].get_ydata()
    assert list(y) == pytest.approx([-1.0, -2.0])
    assert "mV" in ax.get_ylabel()


def test_device_trend_scatter_has_points_and_no_connecting_line(tmp_path, keep_figure_open):
    out = str(tmp_path / "scatter_devices.png")
    plot_metric_devices(_tidy([1.0, 2.0, 3.0]), out, "metric", ["ESN", "TNumber"],
                        value_col="delta", plot_mode="scatter")
    ax = plt.gcf().axes[0]

    points = [item for item in ax.collections if isinstance(item, PathCollection)]
    assert len(points) == 1
    assert len(points[0].get_offsets()) == 3
    assert len(ax.lines) == 0


def test_average_trend_scatter_keeps_errorbars_without_connecting_line(tmp_path, keep_figure_open):
    out = str(tmp_path / "scatter_average.png")
    plot_metric_average(_agg([1.0, 2.0, 3.0], avdev=0.2), out, "metric", [],
                        plot_mode="scatter")
    data_line = plt.gcf().axes[0].containers[0].lines[0]

    assert data_line.get_marker() == "o"
    assert data_line.get_linestyle() == "None"


def _agg_deg(deg_means: list[float], avdev: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "day_num": list(range(len(deg_means))),
        "deg_mean": deg_means, "deg_avdev": [avdev] * len(deg_means),
    })


def test_degradation_retention_view_is_100_minus_deg_with_percent_ylabel(tmp_path, keep_figure_open):
    """The degradation family shows retention: plotted = 100 - degradation%,
    so day 0 (0% degraded) reads 100% and a 30%-degraded day reads 70%. The
    spread is a magnitude (unchanged by the shift) and the y-axis is just '%'."""
    out = str(tmp_path / "deg.png")
    plot_metric_average(_agg_deg([0.0, 30.0], avdev=5.0), out, "Vth_bg0", [],
                        value="deg", scale=-1.0, offset=100.0, ylabel_override="%")
    ax = plt.gcf().axes[0]
    container = ax.containers[0]
    y = container.lines[0].get_ydata()
    assert list(y) == pytest.approx([100.0, 70.0])   # 100 - 0, 100 - 30
    # error bar magnitude is preserved (not negated / not shifted)
    lo = container[1][0].get_ydata()                 # lower cap of the errorbars
    assert lo == pytest.approx([95.0, 65.0])         # 100-5, 70-5
    assert ax.get_ylabel() == "%"
