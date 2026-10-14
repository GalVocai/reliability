"""Degradation trend plots: per-device trends and an average with error bars."""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .style import load_style


def _spread(rows: pd.DataFrame, value: str) -> pd.Series:
    """Use sample SD for new aggregates; accept older AVDEV-only tables."""
    column = f"{value}_sd" if f"{value}_sd" in rows else f"{value}_avdev"
    # SD is undefined with one valid device. Draw its mean without an error bar.
    return pd.to_numeric(rows[column], errors="coerce").fillna(0).abs()


def _savefig(fig, out_path: str, retries: int = 5, delay: float = 0.25, **kwargs) -> None:
    """Save + close, retrying on FileNotFoundError.

    On a cloud-synced folder (OneDrive/Dropbox), a directory created moments
    ago by os.makedirs can still race with the sync client's own filter
    driver -- the save then fails even though the folder "exists". Re-asserting
    the parent dir and retrying briefly rides out that window. Include the
    full extent of long axis labels in the exported image.
    """
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.15)
    try:
        for attempt in range(retries):
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fig.savefig(out_path, **kwargs)
                return
            except FileNotFoundError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
    finally:
        plt.close(fig)


def _device_label(keyvals, dev_cols) -> str:
    parts = []
    d = dict(zip(dev_cols, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
    if "ESN" in d:
        parts.append(f"ESN{d['ESN']}")
    if "TNumber" in d:
        t = str(d["TNumber"])
        parts.append(f"D{t[0]}T{t[1]}" if len(t) == 2 and t.isdigit() else f"T{t}")
    for c in ("Vbg", "segment", "state_Vj"):
        if c in d:
            unit = " [V]" if c == "Vbg" else ""
            parts.append(f"{c}={d[c]}{unit}")
    return " ".join(parts)


_LINESTYLES = ["-", "--", ":", "-."]

# House colour scheme for every trend plot in this module: shades of green.
# Low index -> light, high index -> dark. Stays off the colormap's very ends
# (0.0 is near-white, invisible on a white background; >0.95 gets muddy).
_GROUP_CMAP = "Greens"
_GROUP_SHADE_LO, _GROUP_SHADE_HI = 0.35, 0.95
_SINGLE_LINE_SHADE = 0.65   # one solid, clearly visible green for a lone curve
_SCATTER_SIZE = 36.0


def _group_color(cmap, i: int, n: int):
    return cmap(_GROUP_SHADE_LO + (_GROUP_SHADE_HI - _GROUP_SHADE_LO) * i / n)


def _series(style, cmap, i: int, n: int):
    """(color, marker) for a line-only trend series."""
    if style.publication:
        return style.pub_color(i), None
    return _group_color(cmap, i, n), None


def _scatter(plot_mode: str) -> bool:
    """True only for the explicit scatter option; unknown values stay lines."""
    return str(plot_mode).strip().lower() == "scatter"


def day_legend_label(day_num, time_axis: str = "days",
                     hours_per_cycle: float = 2.0) -> str:
    """Legend label for a degradation checkpoint.

    The stored coordinate is always elapsed days.  Cycle mode only converts
    that coordinate for presentation, preserving the original day metadata
    used by the degradation calculation.
    """
    try:
        d = float(day_num)
    except (TypeError, ValueError):
        return str(day_num)
    if d == 0:
        return "Baseline"
    if str(time_axis).lower() == "cycles":
        cycles = d * 24.0 / float(hours_per_cycle)
        return f"{cycles:g} cycles"
    return f"{d:g} d"


def time_legend_title(time_axis: str = "days") -> str:
    """Title paired with :func:`day_legend_label`."""
    return "Cycle #" if str(time_axis).lower() == "cycles" else "Day"


def plot_metric_devices(tidy_m, out_path: str, metric: str, dev_cols: list[str],
                        title: str = "", include_devices: set | None = None,
                        value_col: str = "value", scale: float = 1.0, unit: str = "",
                        ylabel_override: str | None = None,
                        x_col: str = "day_num", xlabel: str = "Stress Time (days)",
                        plot_mode: str = "line") -> str:
    """`value_col` selects which tidy column to plot (e.g. 'delta' instead of
    'value'); `scale`/`unit` convert it for display (e.g. V -> mV) without
    touching the underlying table. `ylabel_override` sets the y-axis label
    verbatim (else it is built from metric+unit)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    style = load_style()
    keep = {(str(e), str(t)) for e, t in include_devices} if include_devices is not None else None
    groups = list(tidy_m.groupby(dev_cols))

    # colour is keyed by DEVICE (ESN+TNumber): a device keeps one colour across
    # all its sub-curves (segments / states / Vbg). With many sub-curves (ZOZO
    # reaches 8 per device) the legend collapses to one entry per device and
    # the sub-curves cycle line styles instead.
    def dev_of(d: dict) -> tuple[str, str]:
        return (str(d.get("ESN")), str(d.get("TNumber")))

    color_of: dict[tuple[str, str], int] = {}
    for keyvals, _g in groups:
        d = dict(zip(dev_cols, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        color_of.setdefault(dev_of(d), len(color_of))
    collapse = len(groups) > 44
    cmap = plt.get_cmap(_GROUP_CMAP)
    n_dev = max(len(color_of) - 1, 1)
    curve_count: dict[tuple[str, str], int] = {}
    for keyvals, g in groups:
        g = g.sort_values(x_col)
        d = dict(zip(dev_cols, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        dev = dev_of(d)
        n_seen = curve_count.get(dev, 0)
        curve_count[dev] = n_seen + 1
        if collapse:
            label = _device_label(keyvals, ["ESN", "TNumber"]) if n_seen == 0 else "_nolegend_"
        else:
            label = _device_label(keyvals, dev_cols)
        excluded = keep is not None and dev not in keep
        y = g[value_col] * scale
        scatter = _scatter(plot_mode)
        shown_label = f"{label} (excluded)" if excluded and label != "_nolegend_" else label
        if scatter:
            col = "#cccccc" if excluded else _series(style, cmap, color_of[dev], n_dev)[0]
            ax.scatter(g[x_col], y, s=_SCATTER_SIZE, marker="o", color=col,
                       alpha=0.55 if excluded else 0.85,
                       zorder=1 if excluded else 2, label=shown_label)
        elif excluded:
            ax.plot(g[x_col], y, color="#cccccc", lw=0.7, ls=":", alpha=0.7,
                    zorder=1, label=shown_label)
        else:
            col, _mk = _series(style, cmap, color_of[dev], n_dev)
            ax.plot(g[x_col], y, lw=1.0, alpha=0.85, zorder=2,
                    color=col,
                    ls=_LINESTYLES[n_seen % len(_LINESTYLES)] if collapse else "-",
                    label=label)
    ylabel = f"{metric} ({unit})" if unit else metric
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_override if ylabel_override is not None else ylabel)
    ax.set_title(title or f"{ylabel} - per device")
    style.apply_publication(ax)
    n_entries = len(color_of) if collapse else len(groups)
    style.apply_legend(ax, fontsize=6.5, ncol=1 + (n_entries - 1) // 22,
                       loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    _savefig(fig, out_path, dpi=style.dpi, bbox_inches="tight")
    return out_path


def plot_metric_average(agg_m, out_path: str, metric: str, struct_cols: list[str],
                        value: str = "value", title: str = "", scale: float = 1.0,
                        unit: str = "", offset: float = 0.0, ylabel_override: str | None = None,
                        x_col: str = "day_num", xlabel: str = "Stress Time (days)",
                        plot_mode: str = "line", y_floor: float | None = None) -> str:
    """Average +/- sample SD vs day; one line per sub-device group (e.g. Vbg).

    `scale`/`unit` convert the plotted quantity (e.g. V -> mV) for readability;
    the underlying aggregate table is untouched.
    `offset` is added AFTER scaling (plotted = offset + scale*mean); the error
    bar is unaffected (a shift moves location, not spread). The degradation
    family uses scale=-1, offset=100 to show retention -- 100% at the baseline
    day, dropping as the metric degrades -- instead of 0%-at-baseline.
    `ylabel_override` sets the y-axis label verbatim (e.g. just "%"); when None
    it is built from metric+unit.
    `y_floor` fixes only the lower y-axis limit while leaving the automatically
    calculated upper limit (including error bars) unchanged.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    style = load_style()
    groups = list(agg_m.groupby(struct_cols)) if struct_cols else [(None, agg_m)]
    cmap = plt.get_cmap(_GROUP_CMAP)
    n = max(len(groups) - 1, 1)
    for i, (gv, g) in enumerate(groups):
        g = g.sort_values(x_col)
        lbl = None
        if struct_cols:
            lbl = " ".join(f"{c}={v}{' (V)' if c == 'Vbg' else ''}"
                          for c, v in zip(struct_cols, gv if isinstance(gv, tuple) else (gv,)))
        if len(groups) > 1:
            color, marker = _series(style, cmap, i, n)
        else:
            color = style.pub_color(0) if style.publication else cmap(_SINGLE_LINE_SHADE)
            marker = None
        scatter = _scatter(plot_mode)
        # the error bar is a magnitude -> use |scale|; a negative scale (the
        # delta_Vth sign flip) only reflects the mean, never the spread, and
        # matplotlib rejects a negative yerr.
        ax.errorbar(g[x_col], g[f"{value}_mean"] * scale + offset,
                    yerr=_spread(g, value) * abs(scale),
                    marker="o" if scatter else None,
                    markersize=_SCATTER_SIZE ** 0.5,
                    linestyle="none" if scatter else "-",
                    capsize=4, lw=1.4, color=color, label=lbl)
    ylabel = f"{metric} ({unit})" if unit else metric
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_override if ylabel_override is not None else ylabel)
    ax.set_title(title or f"{ylabel} - average")
    style.apply_publication(ax)
    if y_floor is not None:
        ax.set_ylim(bottom=float(y_floor))
    if struct_cols:
        style.apply_legend(ax, fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path


def _ppm_label(ppm: float) -> str:
    return f"{ppm:g} ppm"


def plot_grouped_devices(tidy_m, out_path: str, metrics: list[tuple[float, str]],
                         ylabel: str, include_devices: set | None = None, title: str = "",
                         label_fn=_ppm_label, legend_title: str = "Concentration",
                         value_col: str = "value", scale: float = 1.0, unit: str = "",
                         ylabel_override: str | None = None, abs_value: bool = False,
                         x_col: str = "day_num", xlabel: str = "Stress Time (days)",
                         plot_mode: str = "line") -> str:
    """Per-device trends with several related metrics (e.g. one per ppm step,
    or one per back-gate) overlaid on one axis, colour-coded by `label_fn`'s
    numeric key. `ylabel_override` sets the y-axis label verbatim. `abs_value`
    plots the magnitude |value| (e.g. |ΔVth|, direction-agnostic drift)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    style = load_style()
    keep = {(str(e), str(t)) for e, t in include_devices} if include_devices is not None else None
    cmap = plt.get_cmap(_GROUP_CMAP)
    n = max(len(metrics) - 1, 1)
    for i, (key, metric) in enumerate(metrics):
        color, marker = _series(style, cmap, i, n)
        tm = tidy_m[tidy_m["metric"] == metric]
        for (esn, tnum), g in tm.groupby(["ESN", "TNumber"]):
            g = g.sort_values(x_col)
            dev = (str(esn), str(tnum))
            y = g[value_col] * scale
            if abs_value:
                y = y.abs()
            excluded = keep is not None and dev not in keep
            if _scatter(plot_mode):
                ax.scatter(g[x_col], y, s=_SCATTER_SIZE, marker="o",
                           color="#c8c8c8" if excluded else color,
                           alpha=0.35 if excluded else 0.55,
                           zorder=1 if excluded else 2)
            elif excluded:
                ax.plot(g[x_col], y, color="#c8c8c8", lw=0.7,
                        ls=":", alpha=0.45, zorder=1)
            else:
                ax.plot(g[x_col], y, marker=marker, lw=0.9,
                        color=color, alpha=0.4, zorder=2)
    handles = []
    for i, (key, _metric) in enumerate(metrics):
        color = _series(style, cmap, i, n)[0]
        handles.append(plt.Line2D(
            [0], [0], color=color, label=label_fn(key),
            marker="o" if _scatter(plot_mode) else None,
            markersize=_SCATTER_SIZE ** 0.5,
            linestyle="none" if _scatter(plot_mode) else "-", lw=1.6))
    full_ylabel = f"{ylabel} ({unit})" if unit else ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_override if ylabel_override is not None else full_ylabel)
    ax.set_title(title or f"{full_ylabel} - per device")
    style.apply_publication(ax)
    style.apply_legend(ax, handles=handles, fontsize=8, title=legend_title)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path


def plot_grouped_average(agg_m, out_path: str, metrics: list[tuple[float, str]],
                         ylabel: str, value: str = "value", title: str = "",
                         label_fn=_ppm_label, legend_title: str = "Concentration",
                         scale: float = 1.0, unit: str = "", offset: float = 0.0,
                         ylabel_override: str | None = None, abs_value: bool = False,
                         x_col: str = "day_num", xlabel: str = "Stress Time (days)",
                         plot_mode: str = "line") -> str:
    """Average +/- sample SD trends with several related metrics overlaid on one
    axis (the average-family counterpart of plot_grouped_devices).

    `offset` / `ylabel_override`: see plot_metric_average (used identically by
    the degradation family for the 100%-at-baseline retention view).
    `abs_value` plots the magnitude |mean| (e.g. |ΔVth| drift)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    style = load_style()
    cmap = plt.get_cmap(_GROUP_CMAP)
    n = max(len(metrics) - 1, 1)
    for i, (key, metric) in enumerate(metrics):
        g = agg_m[agg_m["metric"] == metric].sort_values(x_col)
        if g.empty:
            continue
        color, marker = _series(style, cmap, i, n)
        yv = g[f"{value}_mean"] * scale + offset
        if abs_value:
            yv = yv.abs()
        scatter = _scatter(plot_mode)
        # error bar is a magnitude -> |scale| (see plot_metric_average for why)
        ax.errorbar(g[x_col], yv, yerr=_spread(g, value) * abs(scale),
                    marker="o" if scatter else None,
                    markersize=_SCATTER_SIZE ** 0.5,
                    linestyle="none" if scatter else "-",
                    capsize=4, lw=1.4, color=color,
                    label=label_fn(key))
    full_ylabel = f"{ylabel} ({unit})" if unit else ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_override if ylabel_override is not None else full_ylabel)
    ax.set_title(title or f"{full_ylabel} - average")
    style.apply_publication(ax)
    style.apply_legend(ax, fontsize=8, title=legend_title)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path


def plot_response_vs_concentration(agg_m, out_path: str, ppm_items: list,
                                   ylabel: str, xlabel: str = r"H$_2$ Concentration (ppm)",
                                   value: str = "value", title: str = "",
                                   time_axis: str = "days",
                                   hours_per_cycle: float = 2.0,
                                   plot_mode: str = "line") -> str:
    """One error-bar curve PER DAY: y = <metric> mean vs x = ppm concentration.

    ``ppm_items`` = ``[(ppm, metric_name), ...]`` (the per-ppm response metrics,
    e.g. 10ppm_delta / 25ppm_delta / ...). Error bars = sample SD across devices.
    Days are coloured with the same categorical palette / green ramp the trend
    plots use, so the figure reads as one system with the rest of the study."""
    fig, ax = plt.subplots(figsize=(9, 5))
    style = load_style()
    cmap = plt.get_cmap(_GROUP_CMAP)
    ppm_of = {str(m): float(p) for p, m in ppm_items}
    df = agg_m[agg_m["metric"].astype(str).isin(ppm_of)].copy()
    if not df.empty:
        df["ppm"] = df["metric"].astype(str).map(ppm_of)
        days = sorted(df["day_num"].dropna().unique())
        n = max(len(days) - 1, 1)
        for i, day in enumerate(days):
            dd = df[df["day_num"] == day].sort_values("ppm")
            if dd.empty:
                continue
            color, marker = _series(style, cmap, i, n)
            scatter = _scatter(plot_mode)
            ax.errorbar(dd["ppm"], dd[f"{value}_mean"], yerr=_spread(dd, value),
                        marker="o" if scatter else None,
                        markersize=_SCATTER_SIZE ** 0.5,
                        linestyle="none" if scatter else "-",
                        capsize=4, lw=1.4, color=color,
                        label=day_legend_label(day, time_axis, hours_per_cycle))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style.apply_publication(ax)
    style.apply_legend(ax, fontsize=8, title=time_legend_title(time_axis))
    fig.tight_layout()
    _savefig(fig, out_path, dpi=style.dpi)
    return out_path
