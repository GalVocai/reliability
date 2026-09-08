"""Cross-timepoint raw-data overlays, one figure per device.

The caller independently chooses solid lines or unconnected scatter points.
In either mode, day/cycle checkpoint is distinguished by colour only.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from ..measurements.jg import _clean_curve, _vth_constant_current
from ..plotting._clean import abs_despiked
from ..plotting._labels import X_TIME, X_VJ, Y_ID
from ..plotting.style import load_style
from ..plotting.trend_plot import _savefig, day_legend_label, time_legend_title
from ..protocol import DEFAULT


_DAY_CMAP = "Greens"
# keep off the colormap's very ends: 0.0 is near-white (invisible on a white
# plot background) and >0.95 gets muddy-dark; same guard steps_plot.py's
# ppm-shade band already uses for the same colormap family.
_DAY_SHADE_LO, _DAY_SHADE_HI = 0.35, 0.95
# Matplotlib's scatter ``s`` is marker area in pt^2.  A 4.5-pt diameter is
# therefore stored as 4.5**2 so the configured size matches Origin units.
_RAW_MARKER_SIZE_PT = 4.5
_RAW_SCATTER_SIZE = _RAW_MARKER_SIZE_PT ** 2


def _scatter(plot_mode: str) -> bool:
    """True only for the explicit scatter option; unknown values stay lines."""
    return str(plot_mode).strip().lower() == "scatter"


def _day_style(days: list[float], style):
    """day -> (colour, marker), using colour alone to distinguish checkpoints."""
    if style.publication:
        return {d: (style.pub_color(i), None) for i, d in enumerate(days)}
    cmap = plt.get_cmap(_DAY_CMAP)
    n = max(len(days) - 1, 1)
    return {d: (cmap(_DAY_SHADE_LO + (_DAY_SHADE_HI - _DAY_SHADE_LO) * i / n), None)
            for i, d in enumerate(days)}


def plot_jg_sweep_evolution(device_curves: list[dict], out_path: str, title: str,
                            xlim: tuple[float, float] | None = None,
                            time_axis: str = "days",
                            hours_per_cycle: float = 2.0,
                            linewidth: float = 2.5,
                            plot_mode: str = "line") -> str:
    """One figure for one device.

    `device_curves`: list of dicts, one per (day, back-gate) combination:
    {"day_num": float, "day_label": str, "bg": int|float, "vj": ndarray, "id": ndarray}.
    `xlim`: fixed Vj range (protocol.jg.sweep_start_vj/sweep_close_vj) so every
    day is framed identically -- despiking otherwise clips the off-region
    differently per day and makes the x-range wobble between curves.

    All curves are solid; only the colour (= day) distinguishes them. The two
    back-gates therefore share a day's colour and are told apart by position
    only (no linestyle / back-gate legend).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    style = load_style()
    days = sorted({c["day_num"] for c in device_curves})
    day_style = _day_style(days, style)

    for c in device_curves:
        x, y = abs_despiked(c["vj"], c["id"])
        if x.size == 0:
            continue
        col, _mk = day_style[c["day_num"]]
        if _scatter(plot_mode):
            ax.scatter(x, y, color=col, s=_RAW_SCATTER_SIZE, marker="o", zorder=2)
        else:
            ax.plot(x, y, color=col, lw=linewidth, zorder=2)

    if _scatter(plot_mode):
        day_handles = [plt.Line2D([], [], color=day_style[d][0], linestyle="none",
                                  marker="o", markersize=_RAW_MARKER_SIZE_PT,
                                  label=day_legend_label(d, time_axis, hours_per_cycle))
                       for d in days]
    else:
        day_handles = [plt.Line2D([0], [0], color=day_style[d][0], lw=linewidth,
                                  label=day_legend_label(d, time_axis, hours_per_cycle))
                       for d in days]
    style.apply_axes(ax, "JG", default_log=True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(title)
    style.apply_publication(ax)
    style.apply_legend(ax, handles=day_handles, title=time_legend_title(time_axis),
                       fontsize=8, loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path


def plot_jg_sweep_collapse(device_curves: list[dict], out_path: str, title: str,
                           xlim: tuple[float, float] | None = None,
                           time_axis: str = "days",
                           hours_per_cycle: float = 2.0,
                           linewidth: float = 2.5,
                           plot_mode: str = "line") -> str:
    """Horizontally align each JG curve to the earliest curve's Vth.

    Vth is extracted at the protocol's constant-current threshold from the raw
    rising sweep.  Only the voltage coordinate is shifted; current values and
    curve shape are left untouched.  If a crossing is unavailable, that curve
    is shown unshifted rather than silently discarded.
    """

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    style = load_style()
    curves = sorted(device_curves, key=lambda c: float(c["day_num"]))
    days = sorted({float(c["day_num"]) for c in curves})
    day_style = _day_style(days, style)
    target = DEFAULT.jg.vth_target_nA
    guard = DEFAULT.jg.vth_onset_guard

    vths: dict[int, float] = {}
    for index, curve in enumerate(curves):
        vj, ids = _clean_curve(np.asarray(curve["vj"], dtype=float),
                               np.asarray(curve["id"], dtype=float))
        vths[index] = _vth_constant_current(vj, ids, target, guard)
    baseline_vth = next((vths[i] for i in range(len(curves)) if np.isfinite(vths[i])), float("nan"))

    for index, curve in enumerate(curves):
        vj = np.asarray(curve["vj"], dtype=float)
        ids = np.asarray(curve["id"], dtype=float)
        shift = vths[index] - baseline_vth
        aligned_vj = vj - shift if np.isfinite(shift) else vj
        x, y = abs_despiked(aligned_vj, ids)
        if x.size == 0:
            continue
        day = float(curve["day_num"])
        color = day_style[day][0]
        if _scatter(plot_mode):
            ax.scatter(x, y, color=color, s=_RAW_SCATTER_SIZE, marker="o", zorder=2)
        else:
            ax.plot(x, y, color=color, lw=linewidth, zorder=2)

    if _scatter(plot_mode):
        handles = [plt.Line2D([], [], color=day_style[d][0], linestyle="none",
                              marker="o", markersize=_RAW_MARKER_SIZE_PT,
                              label=day_legend_label(d, time_axis, hours_per_cycle))
                   for d in days]
    else:
        handles = [plt.Line2D([0], [0], color=day_style[d][0], lw=linewidth,
                              label=day_legend_label(d, time_axis, hours_per_cycle))
                   for d in days]
    style.apply_axes(ax, "JG", default_log=True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(X_VJ)
    ax.set_ylabel(Y_ID)
    ax.set_title(title)
    style.apply_publication(ax)
    style.apply_legend(ax, handles=handles, title=time_legend_title(time_axis),
                       fontsize=8, loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path


def _steps_selection(steps_spans: list, selected_ppm) -> tuple:
    """Map a set of ppm concentrations to a time (x) range + the steps to label.

    ``steps_spans`` are ordered ``(x0, x1, ppm, part_index)``. Each ppm is
    matched to its FIRST step (so the trailing recovery -- also 0 ppm -- is not
    pulled in by selecting 0); the recovery tail is appended only when the top
    exposure concentration is selected, so 'show everything' keeps it.
    Returns ``((lo, hi), shown_spans)`` or ``(None, [])`` if nothing matches."""
    sel = {int(round(float(p))) for p in selected_ppm}
    seen: set[int] = set()
    shown = []
    for x0, x1, ppm, i in steps_spans:
        p = int(round(float(ppm)))
        if p in seen:                       # duplicate 0 = recovery; handle below
            continue
        seen.add(p)
        if p in sel:
            shown.append((x0, x1, p, i))
    if not shown:
        return None, []
    lo = min(s[0] for s in shown)
    hi = max(s[1] for s in shown)
    exposures = [int(round(float(s[2]))) for s in steps_spans if float(s[2]) > 0]
    if exposures and max(exposures) in sel:            # keep the recovery tail
        tail = steps_spans[-1]
        if int(round(float(tail[2]))) == 0:
            hi = max(hi, tail[1])
            shown.append((tail[0], tail[1], 0, tail[3]))
    return (float(lo), float(hi)), shown


def plot_sensing_evolution(device_curves: list[dict], out_path: str, title: str,
                           xlim: tuple[float | None, float | None] | None = None,
                           ylim: tuple[float | None, float | None] | None = None,
                           steps_spans: list | None = None,
                           steps_ppm=None, time_axis: str = "days",
                           hours_per_cycle: float = 2.0,
                           plot_mode: str = "line") -> str:
    """One figure for one device: its SE/STEPS continuous Id(t) trace from
    every analyzed timepoint, overlaid.

    `device_curves`: list of dicts, one per day:
    {"day_num": float, "day_label": str, "x": ndarray (seconds), "y": ndarray
    (nA), "baseline": float (that day's stabilized baseline level, nA)}.
    Every day shares the same x-axis (0 .. the longest run among the overlaid
    days), so a shorter day isn't stretched to look like it ran as long.

    DC-shifted: each day's curve is shifted so ITS OWN BASELINE LEVEL sits at
    0 (subtracting `baseline`, the stabilized air-baseline level computed by
    the analyzer). Anchoring on the baseline -- not the curve minimum -- keeps
    the alignment correct even for downward responders, whose minimum lies in
    the middle of the response, not at the baseline. Without any shift, a
    device whose baseline current drifted up between days would show every
    later day sitting entirely above the earlier ones; aligning baselines to a
    common 0 makes the RESPONSE SHAPE comparable across days instead.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    style = load_style()
    days = sorted({c["day_num"] for c in device_curves})
    day_style = _day_style(days, style)
    # Concentration selection controls which raw data is visible; a manual X
    # range controls only the viewport and may intentionally leave a blank tail.
    shown_spans = steps_spans or []
    sel_xlim = None
    filter_spans = None
    if steps_ppm is not None and steps_spans:
        sel_xlim, shown_spans = _steps_selection(steps_spans, steps_ppm)
        filter_spans = shown_spans
    x_max = 0.0

    for c in device_curves:
        x, y = c["x"], c["y"]
        if x.size == 0:
            continue
        base = c.get("baseline", float("nan"))
        y_shifted = y - base if base == base else y      # NaN check: unshifted fallback
        x_max = max(x_max, float(x[-1]))
        if filter_spans is not None:
            visible = np.zeros(x.shape, dtype=bool)
            for x0, x1, *_ in filter_spans:
                visible |= (x >= float(x0)) & (x <= float(x1))
            y_shifted = np.where(visible, y_shifted, np.nan)
        color = day_style[c["day_num"]][0]
        label = day_legend_label(c["day_num"], time_axis, hours_per_cycle)
        if _scatter(plot_mode):
            ax.scatter(x, y_shifted, color=color, s=_RAW_SCATTER_SIZE, marker="o",
                       zorder=2, label=label)
        else:
            ax.plot(x, y_shifted, color=color, lw=1.2, zorder=2, label=label)

    if xlim and len(xlim) == 2:
        lo = 0.0 if xlim[0] is None else float(xlim[0])
        hi = (x_max if x_max > 0 else 1.0) if xlim[1] is None else float(xlim[1])
        ax.set_xlim(lo, hi)
    elif sel_xlim is not None:
        ax.set_xlim(*sel_xlim)
    else:                                             # default: 0 .. longest run
        ax.set_xlim(0.0, x_max if x_max > 0 else 1.0)
    ax.set_xlabel(X_TIME)
    ax.set_ylabel(Y_ID)          # data is DC-shifted; label kept clean (Ids + units)
    ax.set_title(title)
    style.apply_publication(ax)
    if ylim:                                          # optional Ids (y) range; either end may be None
        lo, hi = ylim
        cur = ax.get_ylim()
        ax.set_ylim(cur[0] if lo is None else float(lo),
                    cur[1] if hi is None else float(hi))
    # drop the legend below the top row when the ppm-step labels are drawn there
    leg_anchor = (0.01, 0.88) if shown_spans else None
    style.apply_legend(ax, title=time_legend_title(time_axis), fontsize=8,
                       loc="upper left", bbox_to_anchor=leg_anchor)
    # STEPS: label each shown ppm step with its planned concentration, centred
    # over the step near the top. clip_on -> a step whose centre falls outside
    # the range isn't drawn.
    for x0, x1, ppm, *_ in shown_spans:
        ax.text((float(x0) + float(x1)) / 2, 0.97, f"{int(round(float(ppm)))} ppm",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#333333", clip_on=True, zorder=5)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path
