"""Per-device SE trace plots (raw vs corrected) for visual verification."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.se import _trailing_level, is_main_sensing
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol

_PART_STYLE = {0: ("#7f7f7f", "P0 baseline"), 1: ("#9467bd", "P1 exposure"),
               2: ("#d62728", "P2 settle (raw)"), 3: ("#1f77b4", "P3 recovery (raw)")}


def _clip_ylim(ax, baseline: float, response: float) -> None:
    span = max(response - baseline, 1.0)
    ax.set_ylim(baseline - 0.3 * span, response + 0.5 * span)


def plot_se_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT) -> str:
    p = protocol.se
    col = ch.column
    parts = {k: g[["Time", col]].dropna() for k, g in run.parts().items()}
    fig, ax = plt.subplots(figsize=(10, 5))

    for k, g in parts.items():
        color, label = _PART_STYLE.get(k, ("#333333", f"P{k}"))
        ax.plot(g["Time"], g[col], color=color, lw=0.8, label=label)

    if is_main_sensing(run, protocol):
        lvl = lambda k: _trailing_level(parts[k], col, p.window_s, p.outlier_mad_k)
        baseline, response, settle = lvl(p.baseline_part), lvl(p.response_part), lvl(p.settle_part)
        shift = response - settle
        for k in (p.settle_part, p.recovery_part):
            ax.plot(parts[k]["Time"], parts[k][col] + shift, color="#2ca02c", lw=1.3,
                    label=(f"corrected (shift {shift:+.0f})" if k == p.settle_part else None))
        ax.axhline(response, ls="--", color="black", lw=1.0, label=f"response = {response:.0f}")
        ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
        _clip_ylim(ax, baseline, response)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{col} (nA)")
    ax.set_title(f"{ch.label} — SE")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
