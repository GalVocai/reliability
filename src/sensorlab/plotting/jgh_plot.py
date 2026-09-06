"""Per-device JGH hysteresis-loop plots (forward solid, reverse dashed)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..measurements.jg import back_gate_groups
from ..measurements.jgh import jgh_curve_metrics, split_forward_reverse
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import abs_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def render_jgh(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
               protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    data = run.parts()[sorted(run.parts())[0]]
    shown = [(vb, sub) for vb, sub in back_gate_groups(data) if style.shows(vb)]
    for i, (vb, sub) in enumerate(shown):
        vj = sub["Vj"].to_numpy(float)
        idv = sub[ch.column].to_numpy(float)
        (vf, if_), (vr, ir) = split_forward_reverse(vj, idv)
        m = jgh_curve_metrics(vj, idv, protocol)
        fx, fy = abs_despiked(vf, if_)
        rx, ry = abs_despiked(vr, ir)
        if style.legend_metrics:
            lbl = f"BG={vb} (V) (dVth={m['Delta_Vth']:.2f}, HI={m['HI']:.3f})"
        else:
            lbl = f"BG={vb} (V)"
        kw = style.series_kwargs(i, len(fx), style.bg_color(vb))
        ax.plot(fx, fy, lw=style.linewidth, label=lbl, **kw)
        # reverse sweep: same colour, dashed, no marker/label
        ax.plot(rx, ry, color=kw.get("color"), lw=style.linewidth, ls="--")
    ax.set_xlim(float(data["Vj"].min()), float(data["Vj"].max()))
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(style.title_for("JGH", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day))
    style.apply_axes(ax, "JGH", default_log=True)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="upper left")


def plot_jgh_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                    experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    render_jgh(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
