"""Per-device ppm-staircase plot: continuous Id with ppm-labeled steps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.se import _trailing_level
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import despiked_linear


def plot_steps_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT) -> str:
    col = ch.column
    p = protocol.se
    ppm = p.ppm_steps
    parts = {k: g[["Time", col]].dropna().sort_values("Time") for k, g in run.parts().items()}
    part_ids = sorted(parts)
    baseline = _trailing_level(parts[part_ids[0]], col, p.window_s, p.outlier_mad_k)

    fig, ax = plt.subplots(figsize=(11, 5))
    prev_end: float | None = None
    for i, k in enumerate(part_ids):
        t = parts[k]["Time"].to_numpy(float)
        y = parts[k][col].to_numpy(float)
        tx, ty = despiked_linear(t, y)
        if tx.size == 0:
            continue
        if prev_end is not None:
            tx = tx - (tx[0] - prev_end)
        ax.plot(tx, ty, color="#1f77b4", lw=1.0)
        ax.axvline(tx[0], ls=":", color="#bbbbbb", lw=0.8)
        conc = ppm[i] if i < len(ppm) else None
        lbl = "baseline" if i == 0 else ("recovery" if conc == 0 else f"{int(conc)} ppm")
        ax.annotate(lbl, xy=(tx[0], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 - 12 * (i % 2)), textcoords="offset points",
                    fontsize=8, color="#555555")
        prev_end = float(tx[-1])

    ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
    ax.set_xlabel("Time (s, gaps removed)")
    ax.set_ylabel(f"{col} (nA)")
    ax.set_title(f"{ch.label} — H2 ppm staircase")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
