"""Orchestrate a cross-timepoint degradation study and write its outputs."""

from __future__ import annotations

import os
from typing import Callable

from .degradation import _KEY_CANDIDATES, build_degradation
from .io.master_writer import write_master
from .plotting.trend_plot import plot_metric_average, plot_metric_devices
from .util import sanitize_filename


def _cond_folder(cond: object) -> str:
    c = (str(cond) if cond is not None else "").strip()
    return sanitize_filename(c) if c else "unassigned"


def run_degradation(
    timepoints: list[tuple[object, str]],
    out_dir: str,
    baseline_day: object | None = None,
    make_plots: bool = True,
    value: str = "value",
    log: Callable[[str], None] | None = None,
) -> str:
    """Build degradation tables + trend plots under out_dir/degradation/.

    timepoints: list of (day_label, analysis_or_result_folder).
    Layout: degradation/<condition>/<sheet>/{degradation.xlsx, aggregate.xlsx,
            trends/<metric>_devices.png, trends/<metric>_average.png}.
    """
    emit = log or (lambda _m: None)
    results = build_degradation(timepoints, baseline_day)
    deg_root = os.path.join(out_dir, "degradation")
    emit(f"Comparing {len(timepoints)} timepoints; {len(results)} (condition, measurement) groups.")

    for (cond, sheet), (tidy, agg) in results.items():
        d = os.path.join(deg_root, _cond_folder(cond), sheet)
        os.makedirs(d, exist_ok=True)
        write_master(os.path.join(d, "degradation.xlsx"), {"degradation": tidy})
        write_master(os.path.join(d, "aggregate.xlsx"), {"aggregate": agg})
        if not make_plots:
            continue
        dev_cols = [c for c in _KEY_CANDIDATES if c in tidy.columns]
        struct = [c for c in dev_cols if c not in ("ESN", "TNumber")]
        tdir = os.path.join(d, "trends")
        os.makedirs(tdir, exist_ok=True)
        for metric, tm in tidy.groupby("metric"):
            am = agg[agg["metric"] == metric]
            safe = sanitize_filename(str(metric))
            plot_metric_devices(tm, os.path.join(tdir, f"{safe}_devices.png"), str(metric), dev_cols)
            plot_metric_average(am, os.path.join(tdir, f"{safe}_average.png"), str(metric), struct, value=value)
        emit(f"  {cond}/{sheet}: {tidy['metric'].nunique()} metrics")

    emit("Degradation done -> " + deg_root)
    return deg_root
