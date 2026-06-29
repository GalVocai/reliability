"""Per-device JG_H2 plot: air vs H2 transfer curves (per back-gate), log scale."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..measurements.jg import _clean_curve, _vth_constant_current
from ..measurements.jg_h2 import compare_condition
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import positive_despiked
from ._labels import X_VJ, Y_ID, device_title

_BG_COLOR = {0: "#1f77b4", 4: "#d62728"}


def plot_jg_h2_device(air: Run, h2: Run, ch: Channel, out_path: str,
                      protocol: Protocol = DEFAULT, experiment_type: str = "JG_H2", day: str = "") -> str:
    p = protocol.jg
    air_data = air.parts()[p.h2_part]
    h2_data = h2.parts()[p.h2_part]
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for vb in sorted(int(v) for v in air_data["Vb"].dropna().unique()):
        color = _BG_COLOR.get(vb, "#333333")
        a = _clean_curve(air_data[air_data["Vb"] == vb]["Vj"].to_numpy(float),
                         air_data[air_data["Vb"] == vb][ch.column].to_numpy(float))
        g = _clean_curve(h2_data[h2_data["Vb"] == vb]["Vj"].to_numpy(float),
                         h2_data[h2_data["Vb"] == vb][ch.column].to_numpy(float))
        m = compare_condition(a, g, protocol)
        ax_x, ax_y = positive_despiked(*a)
        gx, gy = positive_despiked(*g)
        ax.plot(ax_x, ax_y, color=color, lw=1.0,
                label=f"BG={vb} air (dVth={m['delta_Vth']:.2f}, max resp={m['max_resp']:.0f}%)")
        ax.plot(gx, gy, color=color, lw=1.0, ls="--", label=f"BG={vb} H2")
        for vth, ls in ((m["Vth_air"], "-"), (m["Vth_gas"], "--")):
            if vth == vth:  # not NaN
                ax.axvline(vth, color=color, lw=0.7, ls=ls, alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlim(float(air_data["Vj"].min()), float(air_data["Vj"].max()))
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(device_title(experiment_type, day, ch) + "  (solid=air, dashed=H2)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
