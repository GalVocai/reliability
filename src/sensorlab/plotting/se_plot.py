"""SE trace plot: one continuous, corrected sensing curve per device, with an
option to stack a card's transistors (per ESN) on one figure.

Time gaps between parts (JG-sweep interruptions) are removed and the post-sweep
current is corrected (settle -> response), so each device shows as one line.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.se import _trailing_level, is_main_sensing
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import despiked_linear, stable_min
from ._labels import X_TIME, Y_ID, Y_RESP, device_title, wp_label, wp_label_stack
from .style import DEFAULT_STYLE, StyleConfig

_STACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _norm_baseline(run: Run, col: str, protocol: Protocol) -> float:
    """I0 for the normalized view: mean of the last norm_window_s seconds of
    the FIRST part (MAD-filtered). NaN/0 disables normalization for the trace."""
    p = protocol.se
    first = sorted(run.parts())[0]
    return _trailing_level(run.parts()[first][["Time", col]].dropna(),
                           col, p.norm_window_s, p.outlier_mad_k)


def _continuous(run: Run, col: str, protocol: Protocol):
    """Return (X, Y, boundaries, baseline, response) for one channel's corrected trace."""
    p = protocol.se
    parts = {k: g[["Time", col]].dropna().sort_values("Time") for k, g in run.parts().items()}
    part_ids = sorted(parts)
    corrected = is_main_sensing(run, protocol)
    baseline = response = float("nan")
    shift = 0.0
    if corrected:
        lvl = lambda k: _trailing_level(parts[k], col, p.window_s, p.outlier_mad_k)
        baseline, response, settle = lvl(p.baseline_part), lvl(p.response_part), lvl(p.settle_part)
        shift = response - settle
    cont_x, cont_y, bounds, prev_end = [], [], [], None
    for k in part_ids:
        t = parts[k]["Time"].to_numpy(float)
        y = parts[k][col].to_numpy(float)
        if corrected and k in (p.settle_part, p.recovery_part):
            y = y + shift
        tx, ty = despiked_linear(t, y)
        if tx.size == 0:
            continue
        if prev_end is not None:
            tx = tx - (tx[0] - prev_end)
        bounds.append((float(tx[0]), k))
        prev_end = float(tx[-1])
        cont_x.append(tx)
        cont_y.append(ty)
    return np.concatenate(cont_x), np.concatenate(cont_y), bounds, baseline, response


def _draw_exposure_band(ax, X, bounds, run: Run, style: StyleConfig, protocol: Protocol) -> None:
    """Shade the H2-exposure parts (response + settle) like the STEPS bands."""
    if not style.se_gradient or not is_main_sensing(run, protocol):
        return
    p = protocol.se
    cmap = plt.get_cmap(style.se_cmap)
    starts = [bx for bx, _k in bounds] + [float(X[-1])]
    for i, (bx, k) in enumerate(bounds):
        if k in (p.response_part, p.settle_part):
            ax.axvspan(bx, starts[i + 1], color=cmap(style.se_intensity), zorder=0)


def render_se(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
              protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    X, Y, bounds, baseline, response = _continuous(run, ch.column, protocol)
    norm, i0 = False, float("nan")
    if style.se_normalize:
        i0 = _norm_baseline(run, ch.column, protocol)
        if i0 == i0 and i0 != 0:
            Y = (Y - i0) / i0 * 100.0
            norm = True
    _draw_exposure_band(ax, X, bounds, run, style, protocol)
    ax.plot(X, Y, lw=style.linewidth, ls=style.linestyle, zorder=2,
            **style.series_kwargs(0, len(X), style.se_color, with_marker=False))
    if not np.isnan(response):
        rv = (response - i0) / i0 * 100.0 if norm else response
        bv = (baseline - i0) / i0 * 100.0 if norm else baseline
        unit = "%" if norm else " nA"
        resp_lbl = f"response = {rv:.1f}{unit}" if style.legend_metrics else "response"
        base_lbl = f"baseline = {bv:.1f}{unit}" if style.legend_metrics else "baseline"
        ax.axhline(rv, ls="--", color="black", lw=1.0, label=resp_lbl)
        ax.axhline(bv, ls=":", color="gray", lw=1.0, label=base_lbl)
    if not norm and not style.se_uniform_y and not np.isnan(response):
        span = max(response - baseline, 1.0)
        ax.set_ylim(baseline - 0.3 * span, response + 0.5 * span)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_RESP if norm else Y_ID)
    ax.set_title(style.title_for("SE", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day,
                                 wp=wp_label(getattr(run, "states", None), ch)))
    style.apply_axes(ax, "SE", default_log=False)
    if norm and Y.size:                             # normalized: 0-floor, 10% headroom
        top = float(np.nanmax(Y))
        ax.set_ylim(min(0.0, float(np.nanmin(Y))), top * 1.1 if top > 0 else top + 1.0)
    elif style.se_uniform_y and Y.size:             # center on THIS plot's data ±margin
        m = style.uniform_y_margin_nA
        ax.set_ylim(float(np.min(Y)) - m, float(np.max(Y)) + m)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def render_se_stack(ax, run: Run, channels: list[Channel], style: StyleConfig = DEFAULT_STYLE,
                    protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    """Overlay several channels (e.g. a card's transistors) on one figure."""
    if channels:
        hidden = set((style.stack_hidden or {}).get(str(channels[0].esn), []))
        channels = [c for c in channels if c.tnum_label not in hidden]
    if not channels:
        return
    lo = hi = float("nan")
    norm = style.se_normalize
    for i, ch in enumerate(channels):
        X, Y, bounds, _baseline, _r = _continuous(run, ch.column, protocol)
        if i == 0:
            _draw_exposure_band(ax, X, bounds, run, style, protocol)
        if norm:
            i0 = _norm_baseline(run, ch.column, protocol)
            if i0 == i0 and i0 != 0:
                Y = (Y - i0) / i0 * 100.0           # normalization supersedes DC shift
            else:
                Y = Y * float("nan")                # can't normalize -> drop trace
        elif style.se_stack_dc_shift and Y.size:
            # each trace's lowest point sits at 0; the minimum is searched
            # after the first DC_SHIFT_SKIP_S seconds so the pre-exposure
            # transient/settling doesn't count
            dc = stable_min(X, Y)
            if not np.isnan(dc):
                Y = Y - dc
        if Y.size and not np.all(np.isnan(Y)):
            lo = np.nanmin([lo, float(np.nanmin(Y))])
            hi = np.nanmax([hi, float(np.nanmax(Y))])
        ax.plot(X, Y, lw=style.linewidth, label=ch.tnum_label, zorder=2,
                **style.series_kwargs(i, len(X), _STACK_COLORS[i % len(_STACK_COLORS)],
                                      with_marker=False))
    esn = channels[0].esn if channels else ""
    title = " ".join(s for s in [experiment_type, day, f"ESN{esn}", "(stacked)"] if s)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_RESP if norm else Y_ID)
    ax.set_title(style.title_for("SE", title, esn=esn, t="stacked", day=day,
                                 wp=wp_label_stack(getattr(run, "states", None), channels)))
    style.apply_axes(ax, "SE", default_log=False)
    if norm and not np.isnan(hi):                   # normalized stack: 0 floor, 10% headroom
        ax.set_ylim(0.0, hi * 1.1 if hi > 0 else hi + 1.0)
    elif style.se_uniform_y and not np.isnan(lo):   # center on the shown traces
        m = style.uniform_y_margin_nA
        if style.se_stack_dc_shift:
            ax.set_ylim(0.0, hi + m)                # DC at 0 -> hard floor at 0
        else:
            ax.set_ylim(lo - m, hi + m)
    style.apply_publication(ax)
    style.apply_legend(ax, loc="best")


def plot_se_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_se(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path


def plot_se_esn(run: Run, channels: list[Channel], out_path: str, protocol: Protocol = DEFAULT,
                experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_se_stack(ax, run, channels, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
