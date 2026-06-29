"""Per-device ZOZO plot: Id vs time across the toggled states + single-exp fits."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..measurements.zozo import build_segments, fit_segment_single
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import despiked_linear
from ._labels import X_TIME, Y_ID, device_title

_STATE_COLOR = {}  # assigned per distinct Vj


def plot_zozo_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT,
                     experiment_type: str = "", day: str = "") -> str:
    segs = build_segments(run, ch.column)
    states = sorted({s["Vj"] for s in segs})
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    color = {vj: palette[i % len(palette)] for i, vj in enumerate(states)}

    fig, ax = plt.subplots(figsize=(10, 5))
    seen: set[float] = set()
    for seg in segs:
        f = fit_segment_single(seg)
        c = color[seg["Vj"]]
        lab = None
        if seg["Vj"] not in seen:
            lab = f"Vj={seg['Vj']} (tau={f['tau']:.1f}s, R2={f['R2']:.2f})" if f["fit"] is not None else f"Vj={seg['Vj']}"
            seen.add(seg["Vj"])
        gx, gy = despiked_linear(seg["time_abs"], seg["current"])
        ax.plot(gx, gy, color=c, lw=0.8, label=lab)
        if f["fit"] is not None:
            ax.plot(seg["time_abs"], f["fit"], color="black", lw=1.2, ls="--")

    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch))
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
