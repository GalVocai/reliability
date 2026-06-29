"""Per-device JG_H2 plot: air vs H2 transfer curves, log scale.

Color encodes the gas condition (air / H2); the back-gate is encoded by line
style. Show one or both Vbg via the style. Colors come from StyleConfig.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..measurements.jg import _clean_curve
from ..measurements.jg_h2 import compare_condition
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def render_jg_h2(ax, air: Run, h2: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
                 protocol: Protocol = DEFAULT, experiment_type: str = "JG_H2", day: str = "") -> None:
    p = protocol.jg
    air_data = air.parts()[p.h2_part]
    h2_data = h2.parts()[p.h2_part]
    for vb in sorted(int(v) for v in air_data["Vb"].dropna().unique()):
        if not style.shows(vb):
            continue
        ls = style.bg_linestyle(vb)
        a = _clean_curve(air_data[air_data["Vb"] == vb]["Vj"].to_numpy(float),
                         air_data[air_data["Vb"] == vb][ch.column].to_numpy(float))
        g = _clean_curve(h2_data[h2_data["Vb"] == vb]["Vj"].to_numpy(float),
                         h2_data[h2_data["Vb"] == vb][ch.column].to_numpy(float))
        m = compare_condition(a, g, protocol)
        ax_x, ax_y = positive_despiked(*a)
        gx, gy = positive_despiked(*g)
        ax.plot(ax_x, ax_y, color=style.air_color, lw=style.linewidth, ls=ls,
                label=f"air BG={vb} (dVth={m['delta_Vth']:.2f}, max resp={m['max_resp']:.0f}%)")
        ax.plot(gx, gy, color=style.h2_color, lw=style.linewidth, ls=ls, label=f"H2 BG={vb}")
    ax.set_yscale("log")
    ax.set_xlim(float(air_data["Vj"].min()), float(air_data["Vj"].max()))
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=7, loc="upper left")


def plot_jg_h2_device(air: Run, h2: Run, ch: Channel, out_path: str,
                      protocol: Protocol = DEFAULT, experiment_type: str = "JG_H2", day: str = "",
                      style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    render_jg_h2(ax, air, h2, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
