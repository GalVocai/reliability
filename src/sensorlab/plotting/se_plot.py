"""Per-device SE trace plot: one continuous, corrected sensing curve.

Time gaps between parts (the JG-sweep interruptions, during which SE isn't
recorded) are removed by shifting each part back to butt against the previous
one, and the post-sweep current is corrected (settle -> response), so the whole
baseline -> exposure -> settle -> recovery shows as a single continuous line.
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

_PART_LABEL = {0: "P0 baseline", 1: "P1 exposure", 2: "P2 settle", 3: "P3 recovery"}


def plot_se_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT) -> str:
    p = protocol.se
    col = ch.column
    parts = {k: g[["Time", col]].dropna().sort_values("Time") for k, g in run.parts().items()}
    part_ids = sorted(parts)

    corrected = is_main_sensing(run, protocol)
    baseline = response = shift = float("nan")
    if corrected:
        lvl = lambda k: _trailing_level(parts[k], col, p.window_s, p.outlier_mad_k)
        baseline, response, settle = lvl(p.baseline_part), lvl(p.response_part), lvl(p.settle_part)
        shift = response - settle

    fig, ax = plt.subplots(figsize=(11, 5))
    cont_x: list[np.ndarray] = []
    cont_y: list[np.ndarray] = []
    prev_end_t: float | None = None
    boundaries: list[tuple[float, int]] = []

    for k in part_ids:
        t = parts[k]["Time"].to_numpy(float)
        y = parts[k][col].to_numpy(float)
        if corrected and k in (p.settle_part, p.recovery_part):
            y = y + shift                                   # current correction
        tx, ty = despiked_linear(t, y)
        if tx.size == 0:
            continue
        if prev_end_t is not None:                          # close the time gap
            tx = tx - (tx[0] - prev_end_t)
        boundaries.append((float(tx[0]), k))
        prev_end_t = float(tx[-1])
        cont_x.append(tx)
        cont_y.append(ty)

    X = np.concatenate(cont_x)
    Y = np.concatenate(cont_y)
    ax.plot(X, Y, color="#1f77b4", lw=1.0)                  # single continuous line

    # phase dividers + labels (stagger vertically so short parts don't collide)
    for i, (bx, k) in enumerate(boundaries):
        ax.axvline(bx, ls=":", color="#bbbbbb", lw=0.8)
        ax.annotate(_PART_LABEL.get(k, f"P{k}"), xy=(bx, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 - 12 * (i % 2)), textcoords="offset points",
                    fontsize=8, color="#555555")

    if corrected:
        ax.axhline(response, ls="--", color="black", lw=1.0, label=f"response = {response:.0f}")
        ax.axhline(baseline, ls=":", color="gray", lw=1.0, label=f"baseline = {baseline:.0f}")
        span = max(response - baseline, 1.0)
        ax.set_ylim(baseline - 0.3 * span, response + 0.5 * span)
        ax.legend(fontsize=8, loc="best")

    ax.set_xlabel("Time (s, gaps removed)")
    ax.set_ylabel(f"{col} (nA)")
    ax.set_title(f"{ch.label} — SE (continuous, P2 corrected)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
