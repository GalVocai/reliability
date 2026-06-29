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
from ._clean import despiked_linear
from ._labels import X_TIME, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig

_PART_LABEL = {0: "P0 baseline", 1: "P1 exposure", 2: "P2 settle", 3: "P3 recovery"}
_STACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


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


def render_se(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
              protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    X, Y, bounds, baseline, response = _continuous(run, ch.column, protocol)
    ax.plot(X, Y, color=style.se_color, lw=style.linewidth)
    for i, (bx, k) in enumerate(bounds):
        ax.axvline(bx, ls=":", color="#bbbbbb", lw=0.8)
        ax.annotate(_PART_LABEL.get(k, f"P{k}"), xy=(bx, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 - 12 * (i % 2)), textcoords="offset points", fontsize=8, color="#555555")
    if not np.isnan(response):
        ax.axhline(response, ls="--", color="black", lw=1.0, label=f"response = {response:.0f}")
        ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
        span = max(response - baseline, 1.0)
        ax.set_ylim(baseline - 0.3 * span, response + 0.5 * span)
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))


def render_se_stack(ax, run: Run, channels: list[Channel], style: StyleConfig = DEFAULT_STYLE,
                    protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    """Overlay several channels (e.g. a card's transistors) on one figure."""
    for i, ch in enumerate(channels):
        X, Y, _b, _bl, _r = _continuous(run, ch.column, protocol)
        ax.plot(X, Y, color=_STACK_COLORS[i % len(_STACK_COLORS)], lw=style.linewidth, label=ch.tnum_label)
    esn = channels[0].esn if channels else ""
    title = " ".join(s for s in [experiment_type, day, f"ESN{esn}", "(stacked)"] if s)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


def plot_se_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                   experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_se(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_se_esn(run: Run, channels: list[Channel], out_path: str, protocol: Protocol = DEFAULT,
                experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_se_stack(ax, run, channels, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
