"""Per-device ZOZO plot: one continuous Id-vs-time trace + optional overlays."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.zozo import build_segments, fit_segment_single, segment_lin_slope
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import despiked_linear
from ._labels import X_TIME, Y_ID, channel_states, device_title, wp_label, wp_label_stack
from .style import DEFAULT_STYLE, StyleConfig

_STACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def render_zozo(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
                protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    segs = sorted(build_segments(run, ch.column), key=lambda s: float(s["time_abs"][0]))
    # this channel's own working points from the script's States sheet; the
    # ZOZO alternates State1 -> State2 -> State1 -> ... in part order
    st = channel_states(getattr(run, "states", None), ch)

    def _seg_label(i: int, seg) -> str:
        name = ("State1", "State2")[i % 2]
        if name in st:
            return f"{name} (Vj={st[name][0]:g}V)"
        return f"Vj={seg['Vj']}"

    # one continuous trace across all parts (no per-part color breaks)
    xs, ys = [], []
    for seg in segs:
        gx, gy = despiked_linear(seg["time_abs"], seg["current"])
        if gx.size:
            xs.append(gx)
            ys.append(gy)
    if xs:
        cx = np.concatenate(xs)
        ax.plot(cx, np.concatenate(ys), lw=style.linewidth, ls=style.linestyle, zorder=2,
                **style.series_kwargs(0, len(cx), style.se_color, with_marker=False))

    seen: set[str] = set()
    for i, seg in enumerate(segs):
        if style.zozo_show_exp_fit:
            f = fit_segment_single(seg)
            if f["fit"] is not None:
                lab = None
                base = _seg_label(i, seg)
                if base not in seen:
                    lab = (f"{base} tau={f['tau']:.1f}s R2={f['R2']:.2f}"
                           if style.legend_metrics else base)
                    seen.add(base)
                ax.plot(seg["time_abs"], f["fit"], color="black", lw=1.2, ls="--", label=lab, zorder=3)
        if style.zozo_show_lin_slope:
            lin = segment_lin_slope(seg)
            if lin["slope"] == lin["slope"]:          # not NaN
                ax.plot([lin["t0"], lin["t1"]], [lin["y0"], lin["y1"]],
                        color="#555555", lw=1.2, ls="-.", zorder=3)
                ax.annotate(f"{lin['slope']:.2f} nA/s",
                            xy=((lin["t0"] + lin["t1"]) / 2, (lin["y0"] + lin["y1"]) / 2),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7, color="#555555")

    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(style.title_for("ZOZO", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day,
                                 wp=wp_label(getattr(run, "states", None), ch,
                                             both_states=True)))
    style.apply_axes(ax, "ZOZO", default_log=False)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def render_zozo_stack(ax, run: Run, channels: list[Channel], style: StyleConfig = DEFAULT_STYLE,
                      protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    """Overlay a card's transistors' continuous ZOZO traces on one figure.
    Fit / slope overlays are drawn without labels or annotations (readability)."""
    if channels:
        hidden = set((style.stack_hidden or {}).get(str(channels[0].esn), []))
        channels = [c for c in channels if c.tnum_label not in hidden]
    if not channels:
        return
    for i, ch in enumerate(channels):
        segs = sorted(build_segments(run, ch.column), key=lambda s: float(s["time_abs"][0]))
        xs, ys = [], []
        for seg in segs:
            gx, gy = despiked_linear(seg["time_abs"], seg["current"])
            if gx.size:
                xs.append(gx)
                ys.append(gy)
        if xs:
            cx = np.concatenate(xs)
            ax.plot(cx, np.concatenate(ys), lw=style.linewidth, ls=style.linestyle,
                    label=ch.tnum_label, zorder=2,
                    **style.series_kwargs(i, len(cx), _STACK_COLORS[i % len(_STACK_COLORS)],
                                          with_marker=False))
        for seg in segs:
            if style.zozo_show_exp_fit:
                f = fit_segment_single(seg)
                if f["fit"] is not None:
                    ax.plot(seg["time_abs"], f["fit"], color="black", lw=0.9, ls="--", zorder=3)
            if style.zozo_show_lin_slope:
                lin = segment_lin_slope(seg)
                if lin["slope"] == lin["slope"]:
                    ax.plot([lin["t0"], lin["t1"]], [lin["y0"], lin["y1"]],
                            color="#555555", lw=0.9, ls="-.", zorder=3)
    esn = channels[0].esn
    title = " ".join(s for s in [experiment_type, day, f"ESN{esn}", "(stacked)"] if s)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(style.title_for("ZOZO", title, esn=esn, t="stacked", day=day,
                                 wp=wp_label_stack(getattr(run, "states", None), channels)))
    style.apply_axes(ax, "ZOZO", default_log=False)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def plot_zozo_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                     experiment_type: str = "", day: str = "",
                     style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    render_zozo(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path


def plot_zozo_esn(run: Run, channels: list[Channel], out_path: str, protocol: Protocol = DEFAULT,
                  experiment_type: str = "", day: str = "",
                  style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    render_zozo_stack(ax, run, channels, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
