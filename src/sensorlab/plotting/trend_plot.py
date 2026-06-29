"""Degradation trend plots: per-device trends and an average with error bars."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
            parts.append(f"{c}={d[c]}")
    return " ".join(parts)


def plot_metric_devices(tidy_m, out_path: str, metric: str, dev_cols: list[str],
                        title: str = "", include_devices: set | None = None) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    keep = {(str(e), str(t)) for e, t in include_devices} if include_devices is not None else None
    for keyvals, g in tidy_m.groupby(dev_cols):
        g = g.sort_values("day_num")
        d = dict(zip(dev_cols, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        excluded = keep is not None and (str(d.get("ESN")), str(d.get("TNumber"))) not in keep
        if excluded:
            ax.plot(g["day_num"], g["value"], color="#cccccc", lw=0.7, ls=":", alpha=0.7, zorder=1)
        else:
            ax.plot(g["day_num"], g["value"], marker="o", lw=0.8, alpha=0.6, zorder=2)
    ax.set_xlabel("Day")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} - per device")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_metric_average(agg_m, out_path: str, metric: str, struct_cols: list[str],
                        value: str = "value", title: str = "") -> str:
    """Average +/- std vs day; one error-bar line per sub-device group (e.g. Vbg)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = agg_m.groupby(struct_cols) if struct_cols else [(None, agg_m)]
    for gv, g in groups:
        g = g.sort_values("day_num")
        lbl = None
        if struct_cols:
            lbl = " ".join(f"{c}={v}" for c, v in zip(struct_cols, gv if isinstance(gv, tuple) else (gv,)))
        ax.errorbar(g["day_num"], g[f"{value}_mean"], yerr=g[f"{value}_std"],
                    marker="o", capsize=4, lw=1.4, label=lbl)
    ax.set_xlabel("Day")
    ax.set_ylabel(f"{metric}  (mean +/- std)")
    ax.set_title(title or f"{metric} - average +/- std")
    if struct_cols:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
