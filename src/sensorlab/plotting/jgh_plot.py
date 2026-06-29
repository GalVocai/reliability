"""Per-device JGH hysteresis-loop plots (forward solid, reverse dashed)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.jgh import jgh_curve_metrics, split_forward_reverse
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked

_BG_COLOR = {-4: "#2ca02c", 0: "#1f77b4", 4: "#d62728"}


def plot_jgh_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT) -> str:
    data = run.parts()[sorted(run.parts())[0]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for vb in sorted(int(v) for v in data["Vb"].dropna().unique()):
        sub = data[data["Vb"] == vb]
        vj = sub["Vj"].to_numpy(float)
        idv = sub[ch.column].to_numpy(float)
        (vf, if_), (vr, ir) = split_forward_reverse(vj, idv)
        m = jgh_curve_metrics(vj, idv, protocol)
        color = _BG_COLOR.get(vb, "#333333")
        fx, fy = positive_despiked(vf, if_)        # only positive Id, no spikes
        rx, ry = positive_despiked(vr, ir)
        ax.plot(fx, fy, color=color, lw=1.0,
                label=f"BG={vb} (dVth={m['Delta_Vth']:.2f}, HI={m['HI']:.3f})")
        ax.plot(rx, ry, color=color, lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Vj (V)")
    ax.set_ylabel("Id (nA)")
    ax.set_title(f"{ch.label} — JGH hysteresis (solid=fwd, dashed=rev)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
