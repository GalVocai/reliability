"""Per-device JGH hysteresis-loop plots (forward solid, reverse dashed)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..measurements.jgh import jgh_curve_metrics, split_forward_reverse
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def render_jgh(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
               protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    data = run.parts()[sorted(run.parts())[0]]
    for vb in sorted(int(v) for v in data["Vb"].dropna().unique()):
        if not style.shows(vb):
            continue
        sub = data[data["Vb"] == vb]
        vj = sub["Vj"].to_numpy(float)
        idv = sub[ch.column].to_numpy(float)
        (vf, if_), (vr, ir) = split_forward_reverse(vj, idv)
        m = jgh_curve_metrics(vj, idv, protocol)
        color = style.bg_color(vb)
        fx, fy = positive_despiked(vf, if_)
        rx, ry = positive_despiked(vr, ir)
        ax.plot(fx, fy, color=color, lw=style.linewidth,
                label=f"BG={vb} (dVth={m['Delta_Vth']:.2f}, HI={m['HI']:.3f})")
        ax.plot(rx, ry, color=color, lw=style.linewidth, ls="--")
    ax.set_xlim(float(data["Vj"].min()), float(data["Vj"].max()))
    ax.set_yscale("log")
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=8, loc="upper left")


def plot_jgh_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                    experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    render_jgh(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
