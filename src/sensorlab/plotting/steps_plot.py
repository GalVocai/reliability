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
from ._clean import despiked_linear, stable_min
from ._labels import X_TIME, Y_ID, Y_RESP, device_title, wp_label, wp_label_stack
from .style import DEFAULT_STYLE, StyleConfig

_STACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _norm_baseline(run: Run, col: str, protocol: Protocol) -> float:
    """I0 for the normalized view: mean of the last norm_window_s seconds of
    the FIRST part (MAD-filtered)."""
    p = protocol.se
    first = sorted(run.parts())[0]
    return _trailing_level(run.parts()[first][["Time", col]].dropna(),
                           col, p.norm_window_s, p.outlier_mad_k)


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
    norm, i0 = False, float("nan")
    if style.steps_normalize:
        i0 = _norm_baseline(run, ch.column, protocol)
        if i0 == i0 and i0 != 0:
            Y = (Y - i0) / i0 * 100.0
            norm = True
    _draw_bands(ax, spans, style)
    ax.plot(X, Y, lw=style.linewidth, ls=style.linestyle, zorder=2,
            **style.series_kwargs(0, len(X), style.se_color, with_marker=False))
    _draw_labels(ax, spans)
    p = protocol.se
    parts = {k: g[["Time", ch.column]].dropna() for k, g in run.parts().items()}
    baseline = _trailing_level(parts[sorted(parts)[0]], ch.column, p.window_s, p.outlier_mad_k)
    bv = (baseline - i0) / i0 * 100.0 if norm else baseline
    unit = "%" if norm else " nA"
    base_lbl = f"baseline = {bv:.1f}{unit}" if style.legend_metrics else "baseline"
    ax.axhline(bv, ls=":", color="gray", lw=1.0, label=base_lbl)
    ax.margins(x=0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_RESP if norm else Y_ID)
    ax.set_title(style.title_for("STEPS", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day,
                                 wp=wp_label(getattr(run, "states", None), ch)))
    style.apply_axes(ax, "STEPS", default_log=False)
    if norm and Y.size:                             # normalized: 0-floor, 10% headroom
        top = float(np.nanmax(Y))
        ax.set_ylim(min(0.0, float(np.nanmin(Y))), top * 1.1 if top > 0 else top + 1.0)
    elif style.steps_uniform_y and Y.size:          # center on THIS plot's data ±margin
        m = style.uniform_y_margin_nA
        ax.set_ylim(float(np.min(Y)) - m, float(np.max(Y)) + m)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def render_steps_stack(ax, run: Run, channels: list[Channel], style: StyleConfig = DEFAULT_STYLE,
                       protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    """Overlay several channels' staircases; gradient bands drawn once."""
    if channels:
        hidden = set((style.stack_hidden or {}).get(str(channels[0].esn), []))
        channels = [c for c in channels if c.tnum_label not in hidden]
    if not channels:
        return
    _, _, spans = _segments(run, channels[0].column, protocol)
    _draw_bands(ax, spans, style)
    lo = hi = float("nan")
    norm = style.steps_normalize
    for idx, ch in enumerate(channels):
        X, Y, _s = _segments(run, ch.column, protocol)
        if norm:
            i0 = _norm_baseline(run, ch.column, protocol)
            if i0 == i0 and i0 != 0:
                Y = (Y - i0) / i0 * 100.0                  # normalization supersedes DC shift
            else:
                Y = Y * float("nan")                       # can't normalize -> drop trace
        elif style.steps_stack_dc_shift and Y.size:
            # each trace's lowest point sits at 0; the minimum is searched
            # after the first DC_SHIFT_SKIP_S seconds so the pre-exposure
            # transient/settling doesn't count
            dc = stable_min(X, Y)
            if not np.isnan(dc):
                Y = Y - dc
        if Y.size and not np.all(np.isnan(Y)):
            lo = np.nanmin([lo, float(np.nanmin(Y))])
            hi = np.nanmax([hi, float(np.nanmax(Y))])
        ax.plot(X, Y, lw=style.linewidth, ls=style.linestyle, zorder=2, label=ch.tnum_label,
                **style.series_kwargs(idx, len(X), _STACK_COLORS[idx % len(_STACK_COLORS)],
                                      with_marker=False))
    _draw_labels(ax, spans)
    esn = channels[0].esn
    ax.margins(x=0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_RESP if norm else Y_ID)
    title = " ".join(s for s in [experiment_type, day, f"ESN{esn}", "(stacked)"] if s)
    ax.set_title(style.title_for("STEPS", title, esn=esn, t="stacked", day=day,
                                 wp=wp_label_stack(getattr(run, "states", None), channels)))
    style.apply_axes(ax, "STEPS", default_log=False)
    if norm and not np.isnan(hi):                   # normalized stack: 0 floor, 10% headroom
        ax.set_ylim(0.0, hi * 1.1 if hi > 0 else hi + 1.0)
    elif style.steps_uniform_y and not np.isnan(lo):  # center on the shown traces
        m = style.uniform_y_margin_nA
        if style.steps_stack_dc_shift:
            ax.set_ylim(0.0, hi + m)                # DC at 0 -> hard floor at 0
        else:
            ax.set_ylim(lo - m, hi + m)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def plot_steps_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                      experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_steps(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path


def plot_steps_esn(run: Run, channels: list[Channel], out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_steps_stack(ax, run, channels, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
