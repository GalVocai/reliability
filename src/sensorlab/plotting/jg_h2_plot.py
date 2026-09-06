"""Per-device JG_H2 plot: air vs H2 transfer curves, log scale.

Each back-gate contributes an air (n03) and a 100 ppm H2 (n05) curve. Recovery
(n06) remains an opt-in compatibility option but is hidden in standard output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..measurements.jg import _clean_curve, back_gate_groups, jg_curve_metrics
from ..measurements.jg_h2 import compare_condition
from ..model import Channel, Run
from ..protocol import DEFAULT, Protocol
from ._clean import abs_despiked
from ._labels import X_VJ, Y_ID, device_title
from .style import DEFAULT_STYLE, StyleConfig


def _draw_vth_inset(ax, curves: list[dict], protocol, style):
    """Zoom inset on the Vth turn-on region. Draws the same curves enlarged
    plus, per back-gate, two coloured <-> arrows: air->H2 (H2 curve colour) and
    air->recovery (recovery colour). No grey ΔVth arrow -- the arrows ARE the
    annotation. Inset bbox + arrow width/head-size/y-position all come from the
    style (draggable / editable in the studio). Returns the inset axes or None."""
    target = protocol.jg.vth_target_nA
    vths = [v for c in curves for v in (c["va"], c["vh"], c["vr"]) if v == v]
    if not vths:
        return None
    lo, hi = min(vths), max(vths)
    pad = max(0.12, (hi - lo) * 0.5)

    axins = ax.inset_axes(list(style.jg_h2_inset_bbox))
    for c in curves:
        for xy, col in ((c["a_xy"], c["c_air"]), (c["g_xy"], c["c_h2"]),
                        (c["r_xy"], c["c_rec"])):
            if xy is not None and col is not None:
                axins.plot(xy[0], xy[1], color=col, lw=style.linewidth)
    axins.set_yscale("log")
    zoom = list(getattr(style, "jg_h2_inset_zoom", []) or [])
    if len(zoom) == 4:                         # explicit zoom region (grey rectangle)
        axins.set_xlim(zoom[0], zoom[1])
        axins.set_ylim(zoom[2], zoom[3])
    else:                                      # auto: fit around the Vth values
        axins.set_xlim(lo - pad, hi + pad)
        axins.set_ylim(target / 6.0, target * 12.0)

    inset_tick_size = max(9.0, float(style.pub_tick_size) * 0.7)
    axins.tick_params(labelsize=inset_tick_size, length=2)
    for lbl in axins.get_xticklabels() + axins.get_yticklabels():
        lbl.set_fontsize(inset_tick_size)
        lbl.set_fontweight("bold" if style.pub_bold_labels else "normal")
    ax.indicate_inset_zoom(axins, edgecolor="#999999", alpha=0.6)
    return axins


def render_jg_h2(ax, air: Run, h2: Run, ch: Channel, style: StyleConfig = DEFAULT_STYLE,
                 protocol: Protocol = DEFAULT, experiment_type: str = "JG_H2", day: str = "",
                 rec: Run | None = None) -> None:
    p = protocol.jg
    air_data = air.parts()[p.h2_part]
    h2_data = h2.parts()[p.h2_part]
    rec_data = rec.parts()[p.h2_part] if rec is not None else None
    h2_groups = dict(back_gate_groups(h2_data))
    rec_groups = dict(back_gate_groups(rec_data)) if rec_data is not None else {}
    shown = [(vb, sub) for vb, sub in back_gate_groups(air_data)
             if style.shows(vb) and vb in h2_groups]

    curves: list[dict] = []
    k = 0
    # pass 1: air + H2 for every back-gate (keeps the original 4-colour order)
    for vb, air_sub in shown:
        ls = style.bg_linestyle(vb)
        a = _clean_curve(air_sub["Vj"].to_numpy(float), air_sub[ch.column].to_numpy(float))
        g = _clean_curve(h2_groups[vb]["Vj"].to_numpy(float), h2_groups[vb][ch.column].to_numpy(float))
        m = compare_condition(a, g, protocol)
        ax_x, ax_y = abs_despiked(*a)
        gx, gy = abs_despiked(*g)
        if style.legend_metrics:
            air_lbl = f"air BG={vb} (V) (dVth={m['delta_Vth']:.2f}V, max={m['max_resp']:.0f}%)"
        else:
            air_lbl = f"air BG={vb} (V)"
        kw_air = style.series_kwargs(k, len(ax_x), style.air_color); k += 1
        ax.plot(ax_x, ax_y, lw=style.linewidth, ls=ls, label=air_lbl, **kw_air)
        kw_h2 = style.series_kwargs(k, len(gx), style.h2_color); k += 1
        ax.plot(gx, gy, lw=style.linewidth, ls=ls,
                label=f"{style.h2_label} BG={vb} (V)", **kw_h2)
        curves.append({
            "vb": vb, "ls": ls,
            "va": jg_curve_metrics(a[0], a[1], protocol)["Vth"],
            "vh": jg_curve_metrics(g[0], g[1], protocol)["Vth"], "vr": float("nan"),
            "a_xy": (ax_x, ax_y), "g_xy": (gx, gy), "r_xy": None,
            "c_air": kw_air.get("color"), "c_h2": kw_h2.get("color"), "c_rec": None,
        })

    # pass 2: recovery (n06) appended AFTER -> its own new colours, no colour shift
    if style.show_recovery:
        for c in curves:
            vb = c["vb"]
            if vb not in rec_groups:
                continue
            r = _clean_curve(rec_groups[vb]["Vj"].to_numpy(float),
                             rec_groups[vb][ch.column].to_numpy(float))
            rx, ry = abs_despiked(*r)
            kw_rec = style.series_kwargs(k, len(rx), style.recovery_color); k += 1
            ax.plot(rx, ry, lw=style.linewidth, ls=c["ls"],
                    label=f"recovery BG={vb} (V)", **kw_rec)
            c["r_xy"] = (rx, ry)
            c["c_rec"] = kw_rec.get("color")
            c["vr"] = jg_curve_metrics(r[0], r[1], protocol)["Vth"]

    if style.show_vth_line:
        ax.axhline(p.vth_target_nA, ls=":", color="#888888", lw=0.8, zorder=1)
    ax.set_xlim(float(air_data["Vj"].min()), float(air_data["Vj"].max()))
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(style.title_for("JG_H2", device_title(experiment_type, day, ch),
                                 esn=ch.esn, t=ch.tnum_label, day=day))
    style.apply_axes(ax, "JG_H2", default_log=True)
    style.apply_publication(ax)
    style.apply_legend(ax, fontsize=7, loc="upper left")
    if style.jg_h2_inset:                       # zoom inset with the coloured arrows
        return _draw_vth_inset(ax, curves, protocol, style)
    return None


def plot_jg_h2_device(air: Run, h2: Run, ch: Channel, out_path: str,
                      protocol: Protocol = DEFAULT, experiment_type: str = "JG_H2", day: str = "",
                      style: StyleConfig = DEFAULT_STYLE, rec: Run | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    render_jg_h2(ax, air, h2, ch, style, protocol, experiment_type, day, rec=rec)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)
    return out_path
