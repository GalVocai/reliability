"""Per-device (or per-ESN stacked) ppm-staircase plot, with a background band
per step whose colour intensity scales with the concentration (configurable)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.se import _trailing_level
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import despiked_linear
from ._labels import X_TIME, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig

_STACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _segments(run: Run, col: str, protocol: Protocol):
    """Return (X, Y, spans) for one channel: continuous trace + per-step x-spans."""
    ppm = protocol.se.ppm_steps
    parts = {k: g[["Time", col]].dropna().sort_values("Time") for k, g in run.parts().items()}
    part_ids = sorted(parts)
    xs, ys, spans, prev_end = [], [], [], None
    for i, k in enumerate(part_ids):
        tx, ty = despiked_linear(parts[k]["Time"].to_numpy(float), parts[k][col].to_numpy(float))
        if tx.size == 0:
            continue
        if prev_end is not None:
            tx = tx - (tx[0] - prev_end)
        spans.append((float(tx[0]), float(tx[-1]), ppm[i] if i < len(ppm) else 0, i))
        xs.append(tx)
        ys.append(ty)
        prev_end = float(tx[-1])
    return np.concatenate(xs), np.concatenate(ys), spans


def _draw_bands(ax, spans, style: StyleConfig) -> None:
    if not style.steps_gradient:
        for x0, _x1, _c, _i in spans:
            ax.axvline(x0, ls=":", color="#cccccc", lw=0.8, zorder=1)
        return
    cmap = plt.get_cmap(style.steps_cmap)
    max_ppm = max([c for _0, _1, c, _i in spans if c > 0], default=1.0)
    for x0, x1, conc, _i in spans:
        shade = np.log10(conc + 1) / np.log10(max_ppm + 1) if conc > 0 else 0.0
        ax.axvspan(x0, x1, color=cmap(0.06 + (style.steps_intensity - 0.06) * shade), zorder=0)
        ax.axvline(x0, ls=":", color="#ffffff", lw=0.8, zorder=1)


def _draw_labels(ax, spans) -> None:
    for x0, _x1, conc, i in spans:
        lbl = "baseline" if i == 0 else ("recovery" if conc == 0 else f"{int(conc)} ppm")
        ax.annotate(lbl, xy=(x0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 - 12 * (i % 2)), textcoords="offset points", fontsize=8, color="#333333")


def render_steps(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
                 protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    X, Y, spans = _segments(run, ch.column, protocol)
    _draw_bands(ax, spans, style)
    ax.plot(X, Y, color=style.se_color, lw=style.linewidth, ls=style.linestyle, zorder=2)
    _draw_labels(ax, spans)
    p = protocol.se
    parts = {k: g[["Time", ch.column]].dropna() for k, g in run.parts().items()}
    baseline = _trailing_level(parts[sorted(parts)[0]], ch.column, p.window_s, p.outlier_mad_k)
    ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
    ax.margins(x=0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=8, loc="best")


def render_steps_stack(ax, run: Run, channels: list[Channel], style: StyleConfig = DEFAULT_STYLE,
                       protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    """Overlay several channels' staircases; gradient bands drawn once."""
    if not channels:
        return
    _, _, spans = _segments(run, channels[0].column, protocol)
    _draw_bands(ax, spans, style)
    for idx, ch in enumerate(channels):
        X, Y, _s = _segments(run, ch.column, protocol)
        ax.plot(X, Y, color=_STACK_COLORS[idx % len(_STACK_COLORS)], lw=style.linewidth,
                ls=style.linestyle, zorder=2, label=ch.tnum_label)
    _draw_labels(ax, spans)
    esn = channels[0].esn
    ax.margins(x=0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(" ".join(s for s in [experiment_type, day, f"ESN{esn}", "(stacked)"] if s))
    ax.legend(fontsize=8, loc="best")


def plot_steps_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                      experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_steps(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_steps_esn(run: Run, channels: list[Channel], out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_steps_stack(ax, run, channels, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
