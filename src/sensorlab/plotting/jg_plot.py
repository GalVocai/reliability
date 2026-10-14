"""Per-device JG transfer-curve plots (Id vs Vj, log scale)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.jg import _clean_curve, back_gate_groups, jg_curve_metrics, relevant_part
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import abs_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def render_jg(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
              protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "",
              ss_window_nA: tuple[float, float] | None = None) -> None:
    data = run.parts()[relevant_part(run, protocol)]
    shown = [(vb, sub) for vb, sub in back_gate_groups(data) if style.shows(vb)]
    for i, (vb, sub) in enumerate(shown):
        vj, idv = _clean_curve(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float))
        m = jg_curve_metrics(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float),
                             protocol, ss_window_nA=ss_window_nA)
        px, py = abs_despiked(vj, idv)
        if style.legend_metrics:
            lbl = f"BG={vb} (V): Vth={m['Vth']:.2f} SS={m['SS']:.0f} Ion/Ioff={m['Ion_Ioff']:.0f}"
        else:
            lbl = f"BG={vb} (V)"
        ax.plot(px, py, lw=style.linewidth, ls=style.linestyle, label=lbl,
                **style.series_kwargs(i, len(px), style.bg_color(vb)))
        if style.show_vth_marker and not np.isnan(m["Vth"]):
            ax.plot([m["Vth"]], [protocol.jg.vth_target_nA], "o", color="red",
                    ms=5, zorder=5)
    if style.show_vth_line:
        ax.axhline(protocol.jg.vth_target_nA, ls=":", color="#888888", lw=0.8, zorder=1)
    ax.set_xlim(float(data["Vj"].min()), float(data["Vj"].max()))
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(style.title_for("JG", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day))
    style.apply_axes(ax, "JG", default_log=True)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="upper left")


def plot_jg_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE,
                   ss_window_nA: tuple[float, float] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    render_jg(ax, run, ch, style, protocol, experiment_type, day,
              ss_window_nA=ss_window_nA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
