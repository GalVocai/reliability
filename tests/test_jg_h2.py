"""Unit test for JG_H2 air-vs-gas comparison metrics."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensorlab.measurements.jg_h2 import compare_condition
from sensorlab.model import Channel, MajorKind, Run
from sensorlab.plotting.jg_h2_plot import plot_jg_h2_baseline_vs_end
from sensorlab.plotting.style import StyleConfig
from sensorlab.protocol import JGProtocol, Protocol

PROTO = Protocol(jg=JGProtocol(vth_target_nA=100.0))   # explicit target, default-independent


def test_air_vs_h2_metrics():
    vj = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    air = (vj, np.array([1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]))     # Vth(100nA) = -1.0
    h2 = (vj, np.array([2.0, 20.0, 120.0, 1200.0, 6000.0, 11000.0]))      # Vth(100nA) = -1.1
    m = compare_condition(air, h2, PROTO)
    assert math.isclose(m["Vth_air"], -1.0, abs_tol=1e-3)
    assert math.isclose(m["Vth_gas"], -1.1, abs_tol=1e-3)
    assert math.isclose(m["delta_Vth"], 0.1, abs_tol=1e-3)
    assert math.isclose(m["resp_at_Vth"], 20.0, abs_tol=1.0)              # (120-100)/100
    assert m["max_resp"] > 0 and not math.isnan(m["Vj_max_resp"])


def test_two_checkpoint_overlay_uses_one_bg_and_requested_legend(tmp_path, monkeypatch):
    from sensorlab.plotting import jg_h2_plot

    channel = Channel("Id08", 8, "1216", "11")
    vj = np.linspace(-2.0, 0.3, 6)

    def run(scale):
        data = pd.concat([
            pd.DataFrame({"Part": 2, "Vb": bg, "Vj": vj,
                          "Id08": np.arange(1, 7, dtype=float) * scale * factor})
            for bg, factor in ((0, 1), (4, 100))
        ], ignore_index=True)
        return Run("synthetic.xlsx", 3, MajorKind.JG, "JG", data, [channel], {})

    close = plt.close
    monkeypatch.setattr(jg_h2_plot.plt, "close", lambda *_args: None)
    output = tmp_path / "comparison_Vbg0.png"
    plot_jg_h2_baseline_vs_end(run(1), run(2), run(3), run(4),
                               channel, 0, "28 Days", str(output),
                               style=StyleConfig(dpi=80))
    ax = plt.gcf().axes[0]
    assert output.is_file()
    assert [line.get_label() for line in ax.lines] == [
        "air baseline", "100 ppm H2 baseline", "air 28 Days", "100ppm H2 28 Days"]
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    assert ax.lines[2].get_color() == ax.lines[3].get_color()
    assert ax.lines[0].get_color() != ax.lines[2].get_color()
    assert [line.get_linestyle() for line in ax.lines] == ["-", "--", "-", "--"]
    assert all(max(line.get_ydata()) < 100 for line in ax.lines)
    assert "Vbg" not in ax.get_xlabel() + ax.get_ylabel() + ax.get_title()
    close("all")
