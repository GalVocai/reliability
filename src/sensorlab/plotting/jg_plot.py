"""Per-device JG transfer-curve plots (|Id| vs Vj, log scale, both back-gates)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.jg import _clean_curve, jg_curve_metrics, relevant_part
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked

_BG_COLOR = {0: "#1f77b4", 4: "#d62728"}


def plot_jg_device(run: Run, ch: Channel, out_path: str, protocol: Protocol = DEFAULT) -> str:
    data = run.parts()[relevant_part(run, protocol)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for vb in sorted(int(v) for v in data["Vb"].dropna().unique()):
        sub = data[data["Vb"] == vb]
        vj, idv = _clean_curve(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float))
        m = jg_curve_metrics(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float), protocol)
        color = _BG_COLOR.get(vb, "#333333")
        px, py = positive_despiked(vj, idv)        # only positive Id, no spikes
        ax.plot(px, py, color=color, lw=1.0,
                label=f"BG={vb}: Vth={m['Vth']:.2f} SS={m['SS']:.0f} Ion/Ioff={m['Ion_Ioff']:.0f}")
        if not np.isnan(m["Vth"]):
            ax.axvline(m["Vth"], ls="--", color=color, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Vj (V)")
    ax.set_ylabel("Id (nA)")
    ax.set_title(f"{ch.label} — JG transfer (last rep)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
