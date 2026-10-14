"""Orchestrate a cross-timepoint degradation study and write its outputs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from .degradation import (_KEY_CANDIDATES, _sample_sd, active_analysis_root,
                          build_degradation, day_to_num, filter_devices)
from .degradation_labels import load_degradation_labels
from .io import read_run, scan_result_files
from .jg_calibration import JgCalibration, apply_jg_calibration
from .io.master_writer import write_master
from .measurements.jg import _clean_curve, back_gate_groups, relevant_part
from .measurements.se import _trailing_level
from .plotting.evolution_plot import (plot_jg_sweep_collapse,
                                      plot_jg_sweep_evolution,
                                      plot_sensing_evolution)
from .plotting.jg_h2_plot import plot_jg_h2_device
from .plotting.se_plot import _continuous as _se_continuous
from .plotting.style import load_style
from .plotting.steps_plot import _segments as _steps_segments
from .plotting.trend_plot import (
    plot_grouped_average,
    plot_grouped_devices,
    plot_metric_average,
    plot_metric_devices,
    plot_response_vs_concentration,
)
from .protocol import DEFAULT, MAJOR_RE
from .util import sanitize_filename

# JG_n05 (H2 exposure) and JG_n06 (recovery) sweeps only exist to feed the
# JG_H2 gas comparison -- their own transfer-curve metrics duplicate JG_n03
# and are not tracked as a separate degradation study.
_SKIP_SHEETS = {f"JG_n{DEFAULT.jg.h2_exposure_major:02d}", f"JG_n{DEFAULT.jg.h2_recovery_major:02d}"}

# Absolute levels / bookkeeping columns get no trend plots -- each device
# starts at its own current, so only deltas / percentages are comparable.
# (They stay in the tidy/aggregate tables; this only trims the graphs.)
_UNPLOTTED_METRIC = re.compile(
    r"(_level$|_raw$|_corr$|^shift$|^baseline$|^baseline_noise$|^z_ref$|^z_thresh$)")
_PPM_METRIC = re.compile(r"^(\d+(?:\.\d+)?)ppm_(.+)$")

# JG_n03 (air baseline sweep): its constant-current Vth is the one metric we
# also show as an explicit day-vs-baseline shift, in mV (threshold shifts are
# conventionally reported in mV, not volts -- 0.003 V vs 3 mV).
_JG_N03_SHEET = f"JG_n{DEFAULT.jg.h2_air_major:02d}"
_MV_SCALE = 1000.0

# Any sweep-mode metric ending in _bg<N> (Vth_bg0/bg4, SS_bg0/bg4, Delta_Vth_bg0/
# bg4, HI_bg0/bg4, ...): both back-gates are the SAME physical quantity, so
# they belong on one graph, not two separate files.
_BG_METRIC_RE = re.compile(r"^(.+)_bg(-?\d+)$")

# Degradation-family plots show RETENTION, not raw degradation%: plotted =
# 100 - degradation_pct (= value/baseline*100), so the curve starts at 100%
# on the baseline day and drops as the metric degrades -- more intuitive than
# starting at 0. scale=-1, offset=100 does 100 - deg%. The y-axis label, the
# per-metric labels and the titles all come from the user-editable
# degradation_labels config (load_degradation_labels).
_DEG_SCALE, _DEG_OFFSET = -1.0, 100.0


def _time_axis_settings(time_axis: str, hours_per_cycle: float) -> tuple[str, float, str, str]:
    """Validate the presentation axis and return its plot settings.

    Degradation joins and baseline subtraction always remain day-based. In
    cycle mode the elapsed-day coordinate is converted only after those
    calculations are complete: cycle_num = day_num * 24 / hours_per_cycle.
    """
    axis = str(time_axis).strip().lower()
    if axis not in {"days", "cycles"}:
        raise ValueError("time_axis must be 'days' or 'cycles'.")
    try:
        hours = float(hours_per_cycle)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hours per cycle must be a positive number.") from exc
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("Hours per cycle must be a positive number.")
    if axis == "cycles":
        return axis, hours, "cycle_num", "Stress Cycle #"
    return axis, hours, "day_num", "Stress Time (days)"


def _add_cycle_num(df: pd.DataFrame, hours_per_cycle: float) -> pd.DataFrame:
    """Return a copy with the estimated cycle coordinate beside ``day_num``."""
    out = df.copy()
    out["cycle_num"] = pd.to_numeric(out["day_num"], errors="coerce") * 24.0 / hours_per_cycle
    return out


def _display(metric: str) -> str:
    """Metric name for a title's {label}: underscores -> spaces."""
    return str(metric).replace("_", " ")


def _cond_folder(cond: object) -> str:
    c = (str(cond) if cond is not None else "").strip()
    return sanitize_filename(c) if c else "unassigned"


# A timepoint folder is named 'baseline'/'base'/'t0' or like '2D'/'7D'/'14D'.
_TIMEPOINT_NAME_RE = re.compile(r"^(baseline|base|t0|\d+\s*d)$", re.IGNORECASE)


def _find_run_folder(timepoint_dir: str) -> str | None:
    """Find a run folder with active analysis masters."""
    try:
        names = sorted(os.listdir(timepoint_dir))
    except OSError:
        return None
    with_analysis = []
    for name in names:
        folder = Path(timepoint_dir) / name
        if not folder.is_dir():
            continue
        active = active_analysis_root(folder)
        if active != folder and active.is_dir():
            with_analysis.append((name, str(folder)))
    for name, d in with_analysis:
        if name.upper().startswith("E"):
            return d
    return with_analysis[0][1] if with_analysis else None


def discover_timepoints(parent: str) -> list[tuple[str, str]]:
    """Auto-find degradation timepoints under a parent folder.

    Each timepoint is a subfolder named like 'baseline' / '2D' / '7D' / '14D';
    inside it the actual run folder is the 'E...' subfolder with an active
    versioned batch or a legacy 'analysis' dir. Returns
    [(day_label, run_folder), ...] sorted by day. Other folders are skipped."""
    out: list[tuple[str, str]] = []
    if not os.path.isdir(parent):
        return out
    for name in sorted(os.listdir(parent)):
        sub = os.path.join(parent, name)
        if not os.path.isdir(sub) or not _TIMEPOINT_NAME_RE.match(name.strip()):
            continue
        run_folder = _find_run_folder(sub)
        if run_folder is not None:
            out.append((name, run_folder))
    return sorted(out, key=lambda t: day_to_num(t[0]))


def _steps_ppm_groups(metrics) -> dict[str, list[tuple[float, str]]]:
    """Group STEPS metrics like 10ppm_resp_pct by suffix (resp_pct/delta/t90_s)."""
    groups: dict[str, list[tuple[float, str]]] = {}
    for metric in metrics:
        m = _PPM_METRIC.match(str(metric))
        if not m:
            continue
        ppm = float(m.group(1))
        groups.setdefault(m.group(2), []).append((ppm, str(metric)))
    return {suffix: sorted(items) for suffix, items in groups.items() if len(items) > 1}


def _sel_delta_items(ppm_groups: dict, steps_ppm) -> list[tuple[float, str]]:
    """The ``<ppm>_delta`` exposure metrics (ppm > 0) to use, honouring the
    STEPS ppm selection when one is given. Returns sorted ``[(ppm, metric)]``."""
    items = [(p, m) for p, m in ppm_groups.get("delta", []) if p > 0]
    # ``None`` means "all concentrations".  An empty set is deliberately
    # different: the user unticked every concentration in the GUI, so no
    # concentration response/sensitivity graph should be calculated.
    if steps_ppm is not None:
        sel = {int(round(float(x))) for x in steps_ppm}
        items = [(p, m) for p, m in items if int(round(p)) in sel]
    return sorted(items)


def _steps_sensitivity(tidy, delta_items: list[tuple[float, str]]):
    """Per-device linear sensitivity over the GUI-selected concentration steps.

    For each (ESN, TNumber, day_num), fit
    ``delta_Ids = intercept + sensitivity * H2_ppm``.  The returned slope is
    therefore in nA/ppm.  The zero-ppm baseline is not a response point and is
    not included in ``delta_items``.
    """
    metric_ppm = {str(m): float(p) for p, m in delta_items}
    sub = tidy[tidy["metric"].astype(str).isin(metric_ppm)].copy()
    cols = ["ESN", "TNumber", "day_num", "metric", "value"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["ppm"] = sub["metric"].astype(str).map(metric_ppm)
    rows = []
    for (esn, tnum, day), g in sub.groupby(["ESN", "TNumber", "day_num"]):
        g = g.dropna(subset=["value", "ppm"])
        if g["ppm"].nunique() >= 2:                 # need >=2 points for a slope
            slope = float(np.polyfit(g["ppm"].to_numpy(float),
                                     g["value"].to_numpy(float), 1)[0])
            rows.append({"ESN": esn, "TNumber": tnum, "day_num": day,
                         "metric": "sensitivity_nA_per_ppm", "value": slope})
    return pd.DataFrame(rows, columns=cols)


def _bg_groups(metrics) -> dict[str, list[tuple[float, str]]]:
    """Group sweep-mode metrics like Vth_bg0/Vth_bg4 by their base name,
    keyed by back-gate voltage (mirrors _steps_ppm_groups)."""
    groups: dict[str, list[tuple[float, str]]] = {}
    for metric in metrics:
        m = _BG_METRIC_RE.match(str(metric))
        if not m:
            continue
        bg = float(m.group(2))
        groups.setdefault(m.group(1), []).append((bg, str(metric)))
    return {base: sorted(items) for base, items in groups.items() if len(items) > 1}


def _bg_label(bg: float) -> str:
    return f"BG={bg:g} (V)"


def _write_vbg_retention(agg, out_dir: str, stem: str, metric: str,
                         labels, base_label: object, axis_kwargs: dict,
                         ppm_items: list[tuple[float, str]] | None = None,
                         bg_items: list[tuple[float, str]] | None = None) -> None:
    """Add one retention figure per Vbg, with Vbg only in the filename."""
    if bg_items is not None:
        groups = ((vb, agg[agg["metric"] == bg_metric]) for vb, bg_metric in bg_items)
    elif "Vbg" in agg.columns:
        groups = agg.groupby("Vbg", dropna=True)
    else:
        return
    matched_bg = _BG_METRIC_RE.match(metric)
    display_metric = matched_bg.group(1) if matched_bg else metric
    other_struct = [c for c in ("segment", "state_Vj") if c in agg.columns]
    for vb, rows in groups:
        if rows.empty:
            continue
        suffix = f"Vbg{float(vb):g}"
        out_path = os.path.join(out_dir, f"{stem}_{suffix}.png")
        kwargs = dict(value="deg", scale=_DEG_SCALE, offset=_DEG_OFFSET,
                      ylabel_override=labels.deg_ylabel(display_metric),
                      title=labels.title(_display(display_metric), "degradation", base_label),
                      **axis_kwargs)
        if ppm_items is None:
            plot_metric_average(rows, out_path, display_metric, other_struct, **kwargs)
        else:
            plot_grouped_average(rows, out_path, ppm_items, display_metric, **kwargs)


_SHEET_MAJOR_RE = re.compile(r"^(SE|STEPS)_n(\d+)$")


def _find_raw_path_by_major(folder: str, major: int) -> str | None:
    """The raw result file for one major inside a timepoint's folder
    (scan_result_files reads the timepoint root, the same folder passed to
    analyze_folder -- not its 'analysis' output subfolder)."""
    for path in scan_result_files(folder):
        m = MAJOR_RE.search(os.path.basename(path))
        if m and int(m.group(1)) == major:
            return path
    return None


def _dev_condition_map(results: dict, sheet: str) -> dict[tuple[str, str], object]:
    """(ESN, TNumber) -> condition, read off the already-computed tidy table
    for one sheet (so this never has to re-derive device identity itself)."""
    out: dict[tuple[str, str], object] = {}
    for (cond, sh), (tidy, _agg) in results.items():
        if sh != sheet:
            continue
        for esn, tnum in zip(tidy["ESN"], tidy["TNumber"]):
            out[(str(esn), str(tnum))] = cond
    return out


def _write_jg_sweep_evolution(timepoints: list[tuple[object, str]],
                              results: dict, deg_root: str,
                              emit: Callable[[str], None],
                              time_axis: str = "days",
                              hours_per_cycle: float = 2.0,
                              xlim: tuple[float | None, float | None] | None = None,
                              plot_mode: str = "line",
                              jg_calibration: JgCalibration | None = None) -> None:
    """One figure per device AND per back-gate: its JG_n03 transfer curve
    overlaid across every analyzed timepoint (day = colour), so the SHAPE of
    the sweep's drift is visible directly. Vbg=0 and Vbg=4 are drawn on
    SEPARATE figures (one evolution plot each), not overlaid."""
    dev_condition = _dev_condition_map(results, _JG_N03_SHEET)
    if not dev_condition:
        return

    per_device: dict[tuple[str, str], list[dict]] = {}
    dev_label: dict[tuple[str, str], str] = {}
    missing: list[str] = []
    for day_label, folder in timepoints:
        path = _find_raw_path_by_major(folder, DEFAULT.jg.h2_air_major)
        if path is None:
            missing.append(str(day_label))
            continue
        run = read_run(path)
        if jg_calibration is not None:
            run = apply_jg_calibration(run, jg_calibration)
        data = run.parts()[relevant_part(run)]
        day_num = day_to_num(day_label)
        for vb, sub in back_gate_groups(data):
            for ch in run.channels:
                key = (ch.esn, ch.tnumber)
                if key not in dev_condition:
                    continue
                vj, idv = _clean_curve(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float))
                per_device.setdefault(key, []).append({
                    "day_num": day_num, "day_label": str(day_label), "bg": vb, "vj": vj, "id": idv,
                })
                dev_label[key] = ch.label
    if missing:
        emit(f"WARNING: JG_n03 sweep evolution: no raw n{DEFAULT.jg.h2_air_major:02d} file found for "
             f"day(s) {', '.join(missing)} -- those timepoints are skipped in the sweep plots.")
    if not per_device:
        return

    protocol_xlim = (DEFAULT.jg.sweep_start_vj, DEFAULT.jg.sweep_close_vj)
    if xlim is None:
        sweep_xlim = protocol_xlim
    else:
        sweep_xlim = (
            protocol_xlim[0] if xlim[0] is None else float(xlim[0]),
            protocol_xlim[1] if xlim[1] is None else float(xlim[1]),
        )
    n_figs = 0
    n_collapsed = 0
    for key, curves in per_device.items():
        cond = dev_condition[key]
        d = os.path.join(deg_root, _cond_folder(cond), _JG_N03_SHEET, "sweeps")
        collapse_dir = os.path.join(deg_root, _cond_folder(cond), _JG_N03_SHEET, "collapsed")
        os.makedirs(d, exist_ok=True)
        os.makedirs(collapse_dir, exist_ok=True)
        label = dev_label[key]
        # one evolution figure PER back-gate, so Vbg=0 and Vbg=4 are separated
        for vb in sorted({c["bg"] for c in curves}, key=lambda v: float(v)):
            bg_curves = [c for c in curves if c["bg"] == vb]
            fname = sanitize_filename(f"{label}_Vbg{float(vb):g}.png")
            plot_jg_sweep_evolution(
                bg_curves, os.path.join(d, fname),
                f"{label} — JG_n03 sweep evolution (Vbg={float(vb):g} V)", xlim=sweep_xlim,
                time_axis=time_axis, hours_per_cycle=hours_per_cycle,
                plot_mode=plot_mode)
            plot_jg_sweep_collapse(
                bg_curves, os.path.join(collapse_dir, fname),
                f"{label} — JG_n03 aligned to baseline (Vbg={float(vb):g} V)",
                xlim=sweep_xlim, time_axis=time_axis,
                hours_per_cycle=hours_per_cycle, plot_mode=plot_mode)
            n_figs += 1
            n_collapsed += 1
    emit(f"  JG_n03 sweep evolution: {n_figs} plots ({len(per_device)} devices x back-gate)")
    emit(f"  JG_n03 baseline-aligned collapse: {n_collapsed} plots")


def _write_jg_h2_report_overlays(timepoints: list[tuple[object, str]],
                                 results: dict, deg_root: str,
                                 emit: Callable[[str], None],
                                 jg_calibration: JgCalibration | None = None) -> None:
    """Write baseline and final JG air/H2 overlays for every report device."""

    dev_condition = _dev_condition_map(results, "JG_H2")
    if not dev_condition or not timepoints:
        return
    ordered = sorted(timepoints, key=lambda item: day_to_num(item[0]))
    checkpoints = [("baseline", ordered[0]), ("end", ordered[-1])]
    style = load_style()
    written = 0
    for tag, (day_label, folder) in checkpoints:
        air_path = _find_raw_path_by_major(folder, DEFAULT.jg.h2_air_major)
        h2_path = _find_raw_path_by_major(folder, DEFAULT.jg.h2_exposure_major)
        rec_path = _find_raw_path_by_major(folder, DEFAULT.jg.h2_recovery_major)
        if air_path is None or h2_path is None:
            emit(f"WARNING: JG_H2 report overlay: missing n{DEFAULT.jg.h2_air_major:02d}/"
                 f"n{DEFAULT.jg.h2_exposure_major:02d} at {day_label}; {tag} overlays skipped.")
            continue
        air = read_run(air_path)
        h2 = read_run(h2_path)
        rec = read_run(rec_path) if rec_path is not None else None
        if jg_calibration is not None:
            air = apply_jg_calibration(air, jg_calibration)
            h2 = apply_jg_calibration(h2, jg_calibration)
            rec = apply_jg_calibration(rec, jg_calibration) if rec is not None else None
        h2_keys = {(ch.esn, ch.tnumber) for ch in h2.channels}
        for ch in air.channels:
            key = (ch.esn, ch.tnumber)
            if key not in dev_condition or key not in h2_keys:
                continue
            condition = dev_condition[key]
            out_dir = os.path.join(deg_root, _cond_folder(condition), "JG_H2", "overlays")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, sanitize_filename(f"{ch.label}_{tag}.png"))
            plot_jg_h2_device(
                air, h2, ch, out_path, experiment_type="JG_H2",
                day=str(day_label), style=style, rec=rec)
            written += 1
    emit(f"  JG_H2 report overlays: {written} plots (baseline + end)")


def _sensing_curve(run, col: str, kind: str):
    """(x, y, baseline) for one channel's continuous sensing trace.

    baseline = that run's stabilized air-baseline level (the DC-shift anchor).
    SE's continuous builder already returns it; for STEPS it's the trailing
    level of the first (air) part, computed the same robust way the analyzer
    uses. Anchoring the sensing-evolution DC shift on THIS -- not the curve
    minimum -- keeps downward responders aligned at their baseline."""
    p = DEFAULT.se
    if kind == "SE":
        x, y, _bounds, baseline, _resp = _se_continuous(run, col, DEFAULT)
        return x, y, baseline
    x, y, _spans = _steps_segments(run, col, DEFAULT)
    parts = run.parts()
    first = parts[sorted(parts)[0]]
    baseline = _trailing_level(first, col, p.window_s, p.outlier_mad_k)
    return x, y, baseline


def _write_sensing_evolution(timepoints: list[tuple[object, str]],
                             results: dict, deg_root: str,
                             emit: Callable[[str], None],
                             steps_ppm: set | None = None,
                             xlim: tuple[float | None, float | None] | None = None,
                             ylim: tuple[float | None, float | None] | None = None,
                             time_axis: str = "days",
                             hours_per_cycle: float = 2.0,
                             plot_mode: str = "line") -> None:
    """One figure per device per SE/STEPS sheet: its continuous sensing trace
    (Id vs Time) overlaid across every analyzed timepoint (day = colour), each
    day DC-shifted to its own baseline -- the same idea as the JG sweep
    evolution, for the time-domain runs."""
    sheets = sorted({sh for _cond, sh in results if _SHEET_MAJOR_RE.match(sh)})
    for sheet in sheets:
        kind, major_s = _SHEET_MAJOR_RE.match(sheet).groups()
        major = int(major_s)

        dev_condition = _dev_condition_map(results, sheet)
        if not dev_condition:
            continue

        per_device: dict[tuple[str, str], list[dict]] = {}
        dev_label: dict[tuple[str, str], str] = {}
        missing: list[str] = []
        steps_spans = None                 # ppm-step (x0,x1,ppm,i) boundaries, STEPS only
        for day_label, folder in timepoints:
            path = _find_raw_path_by_major(folder, major)
            if path is None:
                missing.append(str(day_label))
                continue
            run = read_run(path)
            day_num = day_to_num(day_label)
            for ch in run.channels:
                key = (ch.esn, ch.tnumber)
                if key not in dev_condition:
                    continue
                x, y, baseline = _sensing_curve(run, ch.column, kind)
                if kind == "STEPS" and steps_spans is None:   # ppm-step timing (from the first day)
                    steps_spans = _steps_segments(run, ch.column, DEFAULT)[2]
                per_device.setdefault(key, []).append({
                    "day_num": day_num, "day_label": str(day_label),
                    "x": x, "y": y, "baseline": baseline,
                })
                dev_label[key] = ch.label
        if missing:
            emit(f"WARNING: {sheet} sensing evolution: no raw n{major:02d} file found for "
                 f"day(s) {', '.join(missing)} -- those timepoints are skipped.")
        if not per_device:
            continue

        for key, curves in per_device.items():
            cond = dev_condition[key]
            d = os.path.join(deg_root, _cond_folder(cond), sheet, "sensing")
            os.makedirs(d, exist_ok=True)
            label = dev_label[key]
            # the Time (s) range control applies to STEPS overlays only, by request;
            # SE keeps its automatic 0..longest-run range
            plot_sensing_evolution(curves, os.path.join(d, sanitize_filename(f"{label}.png")),
                                   f"{label} — {sheet} sensing evolution",
                                   xlim=xlim if kind == "STEPS" else None,
                                   ylim=ylim if kind == "STEPS" else None,
                                   steps_spans=steps_spans if kind == "STEPS" else None,
                                   steps_ppm=steps_ppm if kind == "STEPS" else None,
                                   time_axis=time_axis, hours_per_cycle=hours_per_cycle,
                                   plot_mode=plot_mode)
        emit(f"  {sheet} sensing evolution: {len(per_device)} device plots")


def _calibration_from_masters(timepoints: list[tuple[object, str]]) -> JgCalibration | None:
    """Use the same JG correction for raw overlays as for saved masters."""
    snapshots: list[Path | None] = []
    for _label, folder in timepoints:
        base = Path(folder)
        path = active_analysis_root(base) / "jg_calibration_used.json"
        if not path.is_file():
            path = base / "jg_calibration_used.json"
        snapshots.append(path if path.is_file() else None)
    if not any(snapshots):
        return None
    if not all(snapshots):
        raise ValueError("Some timepoint masters use JG current correction and others do not")
    calibrations = [JgCalibration.from_json(path) for path in snapshots if path is not None]
    ss_windows = [json.loads(path.read_text(encoding="utf-8")).get("ss_window_nA")
                  for path in snapshots if path is not None]
    if any(window != ss_windows[0] for window in ss_windows[1:]):
        raise ValueError("Fixed-current SS window differs between timepoints")
    reference = calibrations[0].as_json_dict()
    reference.pop("source", None)
    for calibration in calibrations[1:]:
        comparable = calibration.as_json_dict()
        comparable.pop("source", None)
        if comparable != reference:
            raise ValueError("JG calibration differs between timepoints")
    return calibrations[0]


def _build_degradation_output(
    timepoints: list[tuple[object, str]],
    out_dir: str,
    baseline_day: object | None = None,
    make_plots: bool = True,
    value: str = "value",
    include_devices: set | None = None,
    steps_ppm: set | None = None,
    jg_sweep_xlim: tuple[float | None, float | None] | None = None,
    sensing_xlim: tuple[float | None, float | None] | None = None,
    sensing_ylim: tuple[float | None, float | None] | None = None,
    trend_plot_mode: str = "line",
    raw_plot_mode: str = "line",
    exclude_cond_days: set | None = None,
    time_axis: str = "days",
    hours_per_cycle: float = 2.0,
    log: Callable[[str], None] | None = None,
    make_raw_plots: bool = True,
    output_name: str = "degradation",
) -> str:
    """Build degradation tables + trend plots under out_dir/output_name/.

    timepoints: list of (day_label, analysis_or_result_folder).
    Layout: degradation/<condition>/<sheet>/{degradation.xlsx, aggregate.xlsx,
            trends/{devices,average,degradation}/<metric>.png}.
    JG_n05/JG_n06 (the JG_H2 gas-condition sweeps) are skipped: their transfer
    curves only exist to feed JG_H2, not to be tracked on their own.
    Any JG-family metric with a _bg<N> suffix (Vth_bg0/bg4, SS_bg0/bg4,
    Delta_Vth_bg0/bg4, ...) gets a combined plot per base name. Retention
    plots also get one file per back-gate, with Vbg only in the filename.
    JG_n03 additionally gets a delta_Vth.png (both back-gates) in devices/ and
    average/: the day-vs-baseline Vth shift, in mV (the master itself keeps
    volts).
    JG_n03 also gets <condition>/JG_n03/sweeps/<device>.png: the RAW transfer
    curve (Id vs Vj, from each timepoint's own result file, not the master)
    overlaid across every timepoint -- day as colour, x-axis fixed to the
    sweep's own Vj range (protocol.jg.sweep_start_vj .. sweep_close_vj) so
    every day is framed identically.
    SE/STEPS sheets get the analogous <condition>/<sheet>/sensing/<device>.png:
    the RAW continuous Id-vs-Time trace overlaid across every timepoint.

    ``time_axis='cycles'`` presents cross-timepoint trends and legends as an
    estimated cycle count while preserving ``day``/``day_num``. Exported
    tables gain ``cycle_num = day_num * 24 / hours_per_cycle``.
    ``trend_plot_mode`` and ``raw_plot_mode`` independently select lines or
    scatter points for calculated trends and raw-data overlays.
    ``make_raw_plots=False`` refreshes the aggregate/trend figures without
    spending time on raw sweep and sensing overlays.
    """
    emit = log or (lambda _m: None)
    jg_calibration = _calibration_from_masters(timepoints)
    time_axis, hours_per_cycle, x_col, xlabel = _time_axis_settings(
        time_axis, hours_per_cycle)
    for label, tp_folder in timepoints:
        if not str(label).strip():
            emit(f"WARNING: timepoint '{tp_folder}' has an empty day label -> "
                 "it is treated as day 0 and will collide with the baseline.")
        elif math.isnan(day_to_num(label)):
            emit(f"WARNING: day label {label!r} ('{tp_folder}') has no number in it -> "
                 "its rows get no day_num and are excluded from trends.")
    # the baseline must be the earliest day, or every trend comes out inverted
    nums = [day_to_num(lbl) for lbl, _f in timepoints]
    finite = [n for n in nums if n == n]
    base_label = baseline_day if baseline_day is not None else (
        timepoints[nums.index(min(finite))][0] if finite else "?")
    if baseline_day is not None and finite and day_to_num(baseline_day) > min(finite):
        emit(f"WARNING: baseline day '{baseline_day}' is NOT the earliest timepoint "
             f"(day {min(finite):g} exists) — degradation will be measured relative "
             f"to day {day_to_num(baseline_day):g} and earlier days will look inverted.")
    results = build_degradation(timepoints, baseline_day, include_devices,
                                exclude_cond_days=exclude_cond_days)
    results = {k: v for k, v in results.items() if k[1] not in _SKIP_SHEETS}
    if time_axis == "cycles":
        # Conversion happens after all day-based matching and calculations.
        results = {k: (_add_cycle_num(tidy, hours_per_cycle),
                       _add_cycle_num(agg, hours_per_cycle))
                   for k, (tidy, agg) in results.items()}
    deg_root = os.path.join(out_dir, output_name)
    Path(deg_root).mkdir(parents=True, exist_ok=False)
    labels = load_degradation_labels()   # user-editable y-axis labels + titles
    inc = "all" if include_devices is None else f"{len(include_devices)} selected"
    axis_note = (f"cycles ({hours_per_cycle:g} h/cycle)" if time_axis == "cycles" else "days")
    emit(f"Comparing {len(timepoints)} timepoints; {len(results)} groups; devices in average: {inc}; "
         f"time axis: {axis_note}.")
    axis_kwargs = {"x_col": x_col, "xlabel": xlabel,
                   "plot_mode": trend_plot_mode}

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
        # one subfolder per plot family, so a folder listing shows three
        # short, sorted groups instead of one flat alphabetical soup.
        devices_dir = os.path.join(tdir, "devices")
        average_dir = os.path.join(tdir, "average")
        degradation_dir = os.path.join(tdir, "degradation")
        for p in (devices_dir, average_dir, degradation_dir):
            os.makedirs(p, exist_ok=True)
        ppm_groups = _steps_ppm_groups(tidy["metric"].unique()) if str(sheet).startswith("STEPS") else {}
        # sweep-mode metrics with a _bg<N> suffix (JG_n03, JGH_n02, ...): both
        # back-gates are the same physical quantity, so they share one graph
        # instead of getting a separate file each.
        bg_groups = _bg_groups(tidy["metric"].unique()) if str(sheet).startswith("JG") else {}
        grouped_metric_names = ({metric for items in ppm_groups.values() for _ppm, metric in items}
                                | {metric for items in bg_groups.values() for _bg, metric in items})
        for metric, tm in tidy.groupby("metric"):
            m = str(metric)
            if _UNPLOTTED_METRIC.search(m) or m in grouped_metric_names or labels.is_hidden(m):
                continue
            am = agg[agg["metric"] == metric]
            safe = sanitize_filename(m)
            yl = labels.ylabel(m)
            plot_metric_devices(tm, os.path.join(devices_dir, f"{safe}.png"), m, dev_cols,
                                include_devices=include_devices, ylabel_override=yl,
                                title=labels.title(_display(m), "devices", base_label), **axis_kwargs)
            plot_metric_average(am, os.path.join(average_dir, f"{safe}.png"), m, struct, value=value,
                                ylabel_override=yl,
                                title=labels.title(_display(m), "average", base_label), **axis_kwargs)
            plot_metric_average(am, os.path.join(degradation_dir, f"{safe}.png"), m, struct, value="deg",
                                scale=_DEG_SCALE, offset=_DEG_OFFSET, ylabel_override=labels.deg_ylabel(m),
                                title=labels.title(_display(m), "degradation", base_label), **axis_kwargs)
            _write_vbg_retention(am, degradation_dir, safe, m, labels, base_label, axis_kwargs)
        for suffix, items in ppm_groups.items():
            key = f"ppm_{suffix}"
            if labels.is_hidden(key):
                continue
            safe = sanitize_filename(key)
            tm = tidy[tidy["metric"].isin([metric for _ppm, metric in items])]
            am = agg[agg["metric"].isin([metric for _ppm, metric in items])]
            yl = labels.ylabel(key)
            plot_grouped_devices(tm, os.path.join(devices_dir, f"{safe}.png"), items, yl,
                                 include_devices=include_devices, ylabel_override=yl,
                                 title=labels.title(yl, "devices", base_label), **axis_kwargs)
            plot_grouped_average(am, os.path.join(average_dir, f"{safe}.png"), items, yl, value=value,
                                 ylabel_override=yl, title=labels.title(yl, "average", base_label),
                                 **axis_kwargs)
            plot_grouped_average(am, os.path.join(degradation_dir, f"{safe}.png"), items, yl, value="deg",
                                 scale=_DEG_SCALE, offset=_DEG_OFFSET, ylabel_override=labels.deg_ylabel(key),
                                 title=labels.title(yl, "degradation", base_label), **axis_kwargs)
            _write_vbg_retention(am, degradation_dir, safe, key, labels, base_label,
                                 axis_kwargs, ppm_items=items)
        # STEPS-only extra graphs: (1) Delta-Response vs H2 concentration, one
        # curve per day; (2) linear sensitivity (nA/ppm) vs day.
        # Both use only the ppm boxes selected in the GUI.
        if str(sheet).startswith("STEPS"):
            delta_items = _sel_delta_items(ppm_groups, steps_ppm)
            if len(delta_items) >= 2:
                am_delta = agg[agg["metric"].isin([m for _p, m in delta_items])]
                r_yl = labels.ylabel("@resp_vs_ppm")
                plot_response_vs_concentration(
                    am_delta, os.path.join(d, "response_vs_ppm.png"), delta_items,
                    ylabel=r_yl, title=labels.title(r_yl, "average", base_label),
                    time_axis=time_axis, hours_per_cycle=hours_per_cycle,
                    plot_mode=trend_plot_mode)
                # sensitivity is ONE curve vs day (mean +/- sample SD of the
                # per-device slopes), dropped at the STEPS folder root.
                # filter_devices FIRST so the slopes honour the device
                # selection -- exactly like every other averaged plot.
                sens = _steps_sensitivity(filter_devices(tidy, include_devices), delta_items)
                if not sens.empty:
                    agg_sens = (sens.groupby("day_num")["value"]
                                    .agg(value_mean="mean", value_sd=_sample_sd)
                                    .reset_index())
                    agg_sens["metric"] = "sensitivity_nA_per_ppm"
                    if time_axis == "cycles":
                        agg_sens = _add_cycle_num(agg_sens, hours_per_cycle)
                    s_yl = labels.ylabel("sensitivity_nA_per_ppm")
                    sensitivity_name = ("sensitivity_vs_cycle.png" if time_axis == "cycles"
                                        else "sensitivity_vs_day.png")
                    plot_metric_average(
                        agg_sens, os.path.join(d, sensitivity_name),
                        "sensitivity_nA_per_ppm", [], value="value", ylabel_override=s_yl,
                        title=labels.title(s_yl, "average", base_label), y_floor=0.0,
                        **axis_kwargs)
        for base, items in bg_groups.items():
            if labels.is_hidden(base):
                continue
            tm = tidy[tidy["metric"].isin([metric for _bg, metric in items])]
            am = agg[agg["metric"].isin([metric for _bg, metric in items])]
            safe = sanitize_filename(base)
            yl = labels.ylabel(base)
            plot_grouped_devices(tm, os.path.join(devices_dir, f"{safe}.png"), items, base,
                                 include_devices=include_devices, label_fn=_bg_label, ylabel_override=yl,
                                 legend_title="Back-gate", title=labels.title(_display(base), "devices", base_label),
                                 **axis_kwargs)
            plot_grouped_average(am, os.path.join(average_dir, f"{safe}.png"), items, base, value=value,
                                 label_fn=_bg_label, legend_title="Back-gate", ylabel_override=yl,
                                 title=labels.title(_display(base), "average", base_label), **axis_kwargs)
            plot_grouped_average(am, os.path.join(degradation_dir, f"{safe}.png"), items, base, value="deg",
                                 label_fn=_bg_label, legend_title="Back-gate",
                                 scale=_DEG_SCALE, offset=_DEG_OFFSET, ylabel_override=labels.deg_ylabel(base),
                                 title=labels.title(_display(base), "degradation", base_label), **axis_kwargs)
            _write_vbg_retention(am, degradation_dir, safe, base, labels, base_label,
                                 axis_kwargs, bg_items=items)
            # JG_n03's Vth: an extra, explicit day-vs-baseline shift in mV,
            # dropped alongside the normal plots (not a separate family).
            # Sign: the pipeline's generic `delta` is baseline-value (positive
            # = "degraded", built for currents); Vth is a signed voltage, so
            # that convention double-negates for a negative Vth. -_MV_SCALE
            # folds the negation back in with the V->mV conversion, so the plot
            # shows the natural value-baseline shift.
            if sheet == _JG_N03_SHEET and base.lower() == "vth" and not labels.is_hidden("delta_Vth"):
                dname = "delta_Vth"
                dyl = labels.ylabel("@delta_Vth")              # "ΔVth (mV)"
                plot_grouped_devices(tm, os.path.join(devices_dir, f"{dname}.png"), items, dname,
                                     include_devices=include_devices, label_fn=_bg_label,
                                     legend_title="Back-gate", value_col="delta", scale=-_MV_SCALE,
                                     ylabel_override=dyl,
                                     title=labels.title(_display(dname), "delta", base_label), **axis_kwargs)
                plot_grouped_average(am, os.path.join(average_dir, f"{dname}.png"), items, dname, value="delta",
                                     label_fn=_bg_label, legend_title="Back-gate", scale=-_MV_SCALE,
                                     ylabel_override=dyl,
                                     title=labels.title(_display(dname), "delta", base_label), **axis_kwargs)
                for vb, bg_metric in items:
                    bg_rows = am[am["metric"] == bg_metric]
                    if bg_rows.empty:
                        continue
                    plot_metric_average(
                        bg_rows,
                        os.path.join(average_dir, f"{dname}_Vbg{vb:g}.png"),
                        dname, [], value="delta", scale=-_MV_SCALE,
                        ylabel_override=dyl,
                        title=labels.title(_display(dname), "delta", base_label),
                        **axis_kwargs)
        emit(f"  {cond}/{sheet}: {tidy['metric'].nunique()} metrics")

    if make_plots and make_raw_plots:
        _write_jg_sweep_evolution(timepoints, results, deg_root, emit,
                                  time_axis=time_axis, hours_per_cycle=hours_per_cycle,
                                  xlim=jg_sweep_xlim, plot_mode=raw_plot_mode,
                                  jg_calibration=jg_calibration)
        _write_sensing_evolution(timepoints, results, deg_root, emit,
                                 steps_ppm=steps_ppm, xlim=sensing_xlim,
                                 ylim=sensing_ylim, time_axis=time_axis,
                                 hours_per_cycle=hours_per_cycle,
                                 plot_mode=raw_plot_mode)
        _write_jg_h2_report_overlays(timepoints, results, deg_root, emit,
                                     jg_calibration=jg_calibration)

    emit("Degradation done -> " + deg_root)
    return deg_root


_GENERATED_MANIFEST = ".sensorlab_generated_files.json"
_ACTIVE_DEGRADATION_PREFIX = ".sensorlab-degradation-batch-"


def active_degradation_root(out_dir: str | Path) -> Path:
    """Return the newest completed degradation run, or the legacy directory."""
    destination = Path(out_dir).resolve()
    for manifest in sorted(destination.glob(f"{_ACTIVE_DEGRADATION_PREFIX}*.json"),
                           reverse=True):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            name = data.get("output_dir")
            if (data.get("version") == 1 and isinstance(name, str)
                    and re.fullmatch(r"degradation_batch_[A-Za-z0-9_]+", name)
                    and (destination / name).is_dir()):
                return destination / name
        except (OSError, ValueError, TypeError):
            continue
    return destination / "degradation"


def _generated_files_from_manifest(root: Path) -> set[str]:
    manifest = root / _GENERATED_MANIFEST
    if not manifest.is_file():
        return set()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), list):
        raise ValueError(f"Invalid degradation generated-file manifest: {manifest}")
    return {str(name) for name in data["files"]}


def run_degradation(
    timepoints: list[tuple[object, str]],
    out_dir: str,
    baseline_day: object | None = None,
    make_plots: bool = True,
    value: str = "value",
    include_devices: set | None = None,
    steps_ppm: set | None = None,
    jg_sweep_xlim: tuple[float | None, float | None] | None = None,
    sensing_xlim: tuple[float | None, float | None] | None = None,
    sensing_ylim: tuple[float | None, float | None] | None = None,
    trend_plot_mode: str = "line",
    raw_plot_mode: str = "line",
    exclude_cond_days: set | None = None,
    time_axis: str = "days",
    hours_per_cycle: float = 2.0,
    log: Callable[[str], None] | None = None,
    make_raw_plots: bool = True,
) -> str:
    """Write a new version and activate it only after a complete build."""
    emit = log or (lambda _message: None)
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    current = active_degradation_root(destination)
    if current.exists() and not current.is_dir():
        raise ValueError(f"Degradation output path is not a directory: {current}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]
    output_name = "degradation_batch_" + run_id
    stage = Path(_build_degradation_output(
        timepoints, str(destination), baseline_day=baseline_day,
        make_plots=make_plots, value=value, include_devices=include_devices,
        steps_ppm=steps_ppm, jg_sweep_xlim=jg_sweep_xlim,
        sensing_xlim=sensing_xlim, sensing_ylim=sensing_ylim,
        trend_plot_mode=trend_plot_mode, raw_plot_mode=raw_plot_mode,
        exclude_cond_days=exclude_cond_days, time_axis=time_axis,
        hours_per_cycle=hours_per_cycle, make_raw_plots=make_raw_plots,
        output_name=output_name,
        log=lambda message: emit(message)
        if not message.startswith("Degradation done -> ") else None))
    expected = destination / output_name
    if stage.resolve() != expected or not stage.is_dir():
        raise RuntimeError("Degradation output path is invalid")
    generated = {path.relative_to(stage).as_posix()
                 for path in stage.rglob("*") if path.is_file()}
    preserved: list[str] = []
    if current.is_dir():
        previously_generated = _generated_files_from_manifest(current)
        for source in current.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(current).as_posix()
            if (relative == _GENERATED_MANIFEST or relative in generated
                    or relative in previously_generated):
                continue
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            preserved.append(relative)
    (stage / _GENERATED_MANIFEST).write_text(
        json.dumps({"version": 1, "files": sorted(generated),
                    "preserved_files": sorted(preserved)}, indent=2),
        encoding="utf-8")
    (destination / f"{_ACTIVE_DEGRADATION_PREFIX}{run_id}.json").write_text(
        json.dumps({"version": 1, "output_dir": output_name}, indent=2),
        encoding="utf-8")
    emit(f"Preserved {len(preserved)} existing files at their original paths.")
    emit("Previous degradation tree remains at " + str(current))
    emit("Degradation done -> " + str(stage))
    return str(stage)
