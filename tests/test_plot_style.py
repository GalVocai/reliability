"""Regression tests for the shared publication plotting defaults."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import LineCollection

from sensorlab.plotting.style import StyleConfig


def test_standard_style_is_line_only_and_hides_jg_h2_recovery():
    style = StyleConfig()

    assert style.linewidth == pytest.approx(2.5)
    assert style.pub_markers_on is False
    assert "marker" not in style.series_kwargs(0, 20, "#123456")
    assert style.show_recovery is False


@pytest.mark.parametrize("mode", ["JG", "JG_H2", "JGH"])
def test_voltage_sweeps_default_to_half_volt_major_ticks(mode):
    style = StyleConfig()
    fig, ax = plt.subplots()
    ax.set_xlim(-2.0, 1.0)

    style.apply_axes(ax, mode, default_log=True)
    ticks = ax.xaxis.get_majorticklocs()

    assert ticks[1] - ticks[0] == pytest.approx(0.5)
    plt.close(fig)


def test_publication_axes_and_tick_numbers_are_large_and_bold():
    style = StyleConfig()
    fig, ax = plt.subplots()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    style.apply_publication(ax)

    assert ax.xaxis.label.get_fontsize() == pytest.approx(16.0)
    assert ax.yaxis.label.get_fontweight() == "bold"
    assert all(label.get_fontsize() == pytest.approx(14.0)
               for label in ax.get_xticklabels() + ax.get_yticklabels())
    assert all(label.get_fontweight() == "bold"
               for label in ax.get_xticklabels() + ax.get_yticklabels())
    plt.close(fig)


def test_errorbar_legend_is_large_and_uses_a_plain_line_handle():
    style = StyleConfig()
    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [1, 2], yerr=[0.2, 0.3], color="red", label="series")

    style.apply_legend(ax, title="Day")
    legend = ax.get_legend()

    assert legend is not None
    assert all(text.get_fontsize() == pytest.approx(11.0) for text in legend.get_texts())
    assert legend.get_title().get_fontsize() == pytest.approx(11.0)
    assert not legend.findobj(match=LineCollection)
    plt.close(fig)


def test_scatter_errorbar_legend_uses_a_point_without_errorbar_caps():
    style = StyleConfig()
    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [1, 2], yerr=[0.2, 0.3], color="red",
                marker="o", linestyle="none", label="series")

    style.apply_legend(ax)
    legend = ax.get_legend()
    proxy = legend.get_lines()[0]

    assert proxy.get_marker() == "o"
    assert proxy.get_linestyle() == "None"
    assert not legend.findobj(match=LineCollection)
    plt.close(fig)
