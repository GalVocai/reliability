"""Per-device JG transfer-curve plots (Id vs Vj, log scale)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.jg import _clean_curve, jg_curve_metrics, relevant_part
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def render_jg(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
              protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    data = run.parts()[relevant_part(run, protocol)]
    for vb in sorted(int(v) for v in data["Vb"].dropna().unique()):
        if not style.shows(vb):
            continue
        sub = data[data["Vb"] == vb]
        vj, idv = _clean_curve(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float))
        m = jg_curve_metrics(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float), protocol)
        color = style.bg_color(vb)
        px, py = positive_despiked(vj, idv)
        ax.plot(px, py, color=color, lw=style.linewidth, ls=style.linestyle,
                label=f"BG={vb}: Vth={m['Vth']:.2f} SS={m['SS']:.0f} Ion/Ioff={m['Ion_Ioff']:.0f}")
        if not np.isnan(m["Vth"]):
            ax.axvline(m["Vth"], ls="--", color=color, lw=0.8)
    ax.set_xlim(float(data["Vj"].min()), float(data["Vj"].max()))
    ax.set_yscale("log")
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=8, loc="upper left")


def plot_jg_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    render_jg(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
