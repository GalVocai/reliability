"""Per-device ppm-staircase plot: continuous Id with a background band per step
whose colour intensity scales with the concentration (configurable via style)."""

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


def render_steps(ax, run: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
                 protocol: Protocol = DEFAULT, experiment_type: str = "", day: str = "") -> None:
    col = ch.column
    p = protocol.se
    ppm = p.ppm_steps
    parts = {k: g[["Time", col]].dropna().sort_values("Time") for k, g in run.parts().items()}
    part_ids = sorted(parts)
    baseline = _trailing_level(parts[part_ids[0]], col, p.window_s, p.outlier_mad_k)
    max_ppm = max([c for c in ppm if c > 0], default=1.0)
    cmap = plt.get_cmap(style.steps_cmap)

    prev_end: float | None = None
    for i, k in enumerate(part_ids):
        t = parts[k]["Time"].to_numpy(float)
        y = parts[k][col].to_numpy(float)
        tx, ty = despiked_linear(t, y)
        if tx.size == 0:
            continue
        if prev_end is not None:
            tx = tx - (tx[0] - prev_end)
        x0, x1 = float(tx[0]), float(tx[-1])
        conc = ppm[i] if i < len(ppm) else 0
        if style.steps_gradient:
            shade = np.log10(conc + 1) / np.log10(max_ppm + 1) if conc > 0 else 0.0
            ax.axvspan(x0, x1, color=cmap(0.08 + (style.steps_intensity - 0.08) * shade), zorder=0)
        ax.plot(tx, ty, color=style.se_color, lw=style.linewidth, zorder=2)
        ax.axvline(x0, ls=":", color="#ffffff" if style.steps_gradient else "#cccccc", lw=0.8, zorder=1)
        lbl = "baseline" if i == 0 else ("recovery" if conc == 0 else f"{int(conc)} ppm")
        ax.annotate(lbl, xy=(x0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 - 12 * (i % 2)), textcoords="offset points", fontsize=8, color="#333333")
        prev_end = x1

    ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
    ax.margins(x=0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=8, loc="best")


def plot_steps_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                      experiment_type: str = "", day: str = "", style: StyleConfig = DEFAULT_STYLE) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    render_steps(ax, run, ch, style, protocol, experiment_type, day)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
