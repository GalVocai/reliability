"""Folder-level orchestration: load -> group by test type -> analyze -> masters."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable

import pandas as pd

from .io import load_condition_cache, load_states, read_run, scan_result_files
from .jg_calibration import JgCalibration, apply_jg_calibration
from .io.device_map import ConditionCache
from .io.master_writer import write_master
from .measurements import (
    analyze_jg_h2,
    analyze_jg_run,
    analyze_jgh_run,
    analyze_se_run,
    analyze_steps_run,
    analyze_zozo_run,
    classify_se,
    classify_steps,
)
from .model import MajorKind, Run
from .protocol import DEFAULT
from .plotting import (
    load_style,
    plot_jg_device,
    plot_jg_h2_device,
    plot_jgh_device,
    plot_se_device,
    plot_se_esn,
    plot_steps_device,
    plot_steps_esn,
    plot_zozo_device,
    plot_zozo_esn,
)
from .util import sanitize_filename


def load_folder(folder: str, log: Callable[[str], None] | None = None,
                jg_calibration: JgCalibration | None = None) -> list[Run]:
    """Read every result file and classify it by test type. The script file's
    States sheet (per-channel working points) is attached to every run.

    The States sheet numbers channels by setup port; some result files renumber
    their Id columns from 0. So the ports are translated to a stable
    (ESN, TNumber) key via the run whose channel indices match the sheet, and
    every lookup is done by device identity — immune to renumbering."""
    emit = log or (lambda _m: None)
    runs = [read_run(p) for p in scan_result_files(folder)]
    if jg_calibration is not None:
        runs = [apply_jg_calibration(run, jg_calibration) for run in runs]
    by_num = load_states(folder)
    states: dict = {}
    if by_num and runs:
        # The sheet's Number is the setup PORT (2,3,5,6,8,... with gaps); the
        # result files renumber Id columns 0..N-1 in the same order. Sorted
        # ports therefore map positionally onto sorted channel indices.
        ports = sorted(by_num)
        ref = next((r for r in runs if len(r.channels) == len(ports)), None)
        if ref is not None:
            for c, port in zip(sorted(ref.channels, key=lambda c: c.index), ports):
                states[(c.esn, c.tnumber)] = by_num[port]
        else:
            emit(f"WARNING: States sheet has {len(ports)} ports but no run has that "
                 "many channels; working points not attached to plots.")
    for r in runs:
        r.states = states
    return runs


# Absolute-level columns are dropped from the SE/STEPS masters: every device
# starts at its own current, so levels are not comparable and only the deltas
# / percentages matter. They are still computed (the correction math and the
# population grouping need them) — the groups sheet keeps `baseline` on show.
_SE_LEVEL_COLS = ["baseline", "response", "settle_raw", "shift", "settle_corr",
                  "recovery_raw", "recovery_corr"]
_STEPS_LEVEL_COLS = ["baseline", "baseline_noise"]


def group_by_kind(runs: list[Run]) -> dict[MajorKind, list[Run]]:
    grouped: dict[MajorKind, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[run.kind].append(run)
    return grouped


def all_esns(runs: list[Run]) -> list[str]:
    seen: dict[str, None] = {}
    for run in runs:
        for ch in run.channels:
            if ch.esn:
                seen.setdefault(ch.esn, None)
    return list(seen)


def _cond_folder(cond: object) -> str:
    """Folder name for a condition value ('' -> 'unassigned')."""
    c = (str(cond) if cond is not None else "").strip()
    return sanitize_filename(c) if c else "unassigned"


def _write_master_by_condition(out_dir: str, mode: str, sheets: dict) -> list[str]:
    """Write one master per condition: rows split by the 'condition' column,
    into out_dir/<condition>/<mode>/master_<mode>.xlsx."""
    missing = [n for n, d in sheets.items() if "condition" not in d.columns]
    assert not missing, f"{mode}: sheets missing a 'condition' column: {missing}"
    cond_values: set[str] = set()
    for d in sheets.values():
        cond_values |= set(d["condition"].fillna("").astype(str))
    paths: list[str] = []
    for cv in sorted(cond_values):
        fsheets = {n: d[d["condition"].fillna("").astype(str) == cv] for n, d in sheets.items()}
        fsheets = {n: d for n, d in fsheets.items() if not d.empty}
        if not fsheets:
            continue
        cdir = os.path.join(out_dir, _cond_folder(cv), mode)
        os.makedirs(cdir, exist_ok=True)
        paths.append(write_master(os.path.join(cdir, f"master_{mode}.xlsx"), fsheets))
    return paths


def _dev_dir(out_dir: str, conditions: ConditionCache, ch, mode: str) -> str:
    """by_device folder for a channel, under its condition: <cond>/<mode>/by_device."""
    d = os.path.join(out_dir, _cond_folder(conditions.get(ch.esn)), mode, "by_device")
    os.makedirs(d, exist_ok=True)
    return d


def analyze_folder(
    folder: str,
    out_dir: str | None = None,
    conditions: ConditionCache | None = None,
    make_plots: bool = True,
    day: str = "",
    log: Callable[[str], None] | None = None,
    jg_calibration: JgCalibration | None = None,
    ss_window_nA: tuple[float, float] | None = None,
) -> dict[str, list[str]]:
    """Run the analyzers and write outputs split by condition.

    Layout: ``out_dir/<condition>/<MODE>/{master_<MODE>.xlsx, by_device/*.png}``.
    The condition per device comes from the cache (filled once via the GUI);
    devices with no condition land under 'unassigned'.
    Set ``make_plots=False`` for fast master-only runs.

    Returns {mode: [paths of the masters actually written]} — the paths matter
    because a locked master falls back to a timestamped sibling.
    """
    emit = log or (lambda _m: None)
    if ss_window_nA is not None and jg_calibration is None:
        raise ValueError("Fixed-current SS requires the enabled JG calibration")
    runs = load_folder(folder, log=emit, jg_calibration=jg_calibration)
    grouped = group_by_kind(runs)
    conditions = conditions or load_condition_cache()
    style = load_style()
    out_dir = out_dir or os.path.join(folder, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    emit(f"Loaded {len(runs)} majors: " + ", ".join(f"n{r.major:02d}={r.kind.value}" for r in runs))

    written: dict[str, list[str]] = {}

    def finish(mode: str, sheets: dict) -> None:
        for d in sheets.values():           # tag every master row with the timepoint
            d["day"] = day
        paths = _write_master_by_condition(out_dir, mode, sheets)
        if paths:
            written[mode] = paths
        for p in paths:
            if os.path.basename(p) != f"master_{mode}.xlsx":
                emit(f"WARNING: master_{mode}.xlsx is locked (open in Excel?); "
                     f"wrote {p} instead — degradation studies will not find it.")

    # Working points (script States sheet) -> master_WP.xlsx. The WP is
    # retuned per timepoint, so putting it in a master lets the degradation
    # study track its day-to-day drift like any other metric.
    states = runs[0].states if runs else None
    if states:
        ref = max(runs, key=lambda r: len(r.channels))
        wp_rows: list[dict[str, object]] = []
        for ch in ref.channels:
            st = states.get((ch.esn, ch.tnumber))
            if not st:
                continue
            wp_row: dict[str, object] = {
                "ESN": ch.esn, "TNumber": ch.tnumber,
                "condition": conditions.get(ch.esn),
            }
            for state, (vj, vb) in sorted(st.items()):
                wp_row[f"Vj_{state}"] = vj
                wp_row[f"Vb_{state}"] = vb
            wp_rows.append(wp_row)
        if wp_rows:
            emit("WP: writing working points")
            wp_df = pd.DataFrame(wp_rows).sort_values(["ESN", "TNumber"]).reset_index(drop=True)
            finish("WP", {"WP": wp_df})

    # SE sensing (n04)
    se_runs = grouped.get(MajorKind.SE, [])
    if se_runs:
        emit("SE: analyzing" + (" + plotting" if make_plots else ""))
        se_sheets: dict[str, pd.DataFrame] = {}
        for r in se_runs:
            df, se_grp = classify_se(analyze_se_run(r, conditions))
            df = df.drop(columns=[c for c in _SE_LEVEL_COLS if c in df.columns])
            se_sheets[f"SE_n{r.major:02d}"] = df
            if not se_grp.empty:
                se_sheets[f"SE_n{r.major:02d}_groups"] = se_grp
        finish("SE", se_sheets)
        if make_plots:
            for run in se_runs:
                if style.se_stack_per_esn:
                    by_esn: dict[str, list] = {}
                    for ch in run.channels:
                        by_esn.setdefault(ch.esn, []).append(ch)
                    for esn, chs in by_esn.items():
                        d = _dev_dir(out_dir, conditions, chs[0], "SE")
                        plot_se_esn(run, chs, os.path.join(d, sanitize_filename(f"ESN{esn}.png")),
                                    experiment_type=run.kind.value, day=day, style=style)
                else:
                    for ch in run.channels:
                        plot_se_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "SE"),
                                       sanitize_filename(f"{ch.label}.png")),
                                       experiment_type=run.kind.value, day=day, style=style)

    # STEPS (n07 ppm staircase)
    steps_runs = grouped.get(MajorKind.STEPS, [])
    if steps_runs:
        emit("STEPS: analyzing" + (" + plotting" if make_plots else ""))
        steps_sheets: dict[str, pd.DataFrame] = {}
        for r in steps_runs:
            df, st_grp = classify_steps(analyze_steps_run(r, conditions))
            df = df.drop(columns=[c for c in _STEPS_LEVEL_COLS if c in df.columns])
            steps_sheets[f"STEPS_n{r.major:02d}"] = df
            if not st_grp.empty:
                steps_sheets[f"STEPS_n{r.major:02d}_groups"] = st_grp
        finish("STEPS", steps_sheets)
        if make_plots:
            for run in steps_runs:
                if style.steps_stack_per_esn:
                    by_esn: dict[str, list] = {}
                    for ch in run.channels:
                        by_esn.setdefault(ch.esn, []).append(ch)
                    for esn, chs in by_esn.items():
                        d = _dev_dir(out_dir, conditions, chs[0], "STEPS")
                        plot_steps_esn(run, chs, os.path.join(d, sanitize_filename(f"ESN{esn}.png")),
                                       experiment_type=run.kind.value, day=day, style=style)
                else:
                    for ch in run.channels:
                        plot_steps_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "STEPS"),
                                          sanitize_filename(f"{ch.label}.png")),
                                          experiment_type=run.kind.value, day=day, style=style)

    # JG transfer sweeps (n03/n05/n06)
    jg_runs = grouped.get(MajorKind.JG, [])
    if jg_runs:
        emit("JG: analyzing" + (" + plotting" if make_plots else ""))
        finish("JG", {f"JG_n{r.major:02d}": analyze_jg_run(
            r, conditions, ss_window_nA=ss_window_nA) for r in jg_runs})
        if make_plots:
            for run in jg_runs:
                for ch in run.channels:
                    plot_jg_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "JG"),
                                   sanitize_filename(f"n{run.major:02d}_{ch.label}.png")),
                                   experiment_type=run.kind.value, day=day, style=style,
                                   ss_window_nA=ss_window_nA)

    # JG_H2 -> air (n03) vs H2 (n05) vs recovery (n06)
    if jg_runs:
        h2_df = analyze_jg_h2(jg_runs, conditions)
        if not h2_df.empty:
            emit("JG_H2: analyzing" + (" + plotting" if make_plots else ""))
            finish("JG_H2", {"JG_H2": h2_df})
            by_major = {r.major: r for r in jg_runs}
            air = by_major.get(DEFAULT.jg.h2_air_major)
            h2 = by_major.get(DEFAULT.jg.h2_exposure_major)
            rec = by_major.get(DEFAULT.jg.h2_recovery_major)
            if make_plots and air is not None and h2 is not None:
                for ch in air.channels:
                    plot_jg_h2_device(air, h2, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "JG_H2"),
                                      sanitize_filename(f"{ch.label}.png")), day=day, style=style, rec=rec)

    # JGH hysteresis (n02)
    jgh_runs = grouped.get(MajorKind.JGH, [])
    if jgh_runs:
        emit("JGH: analyzing" + (" + plotting" if make_plots else ""))
        finish("JGH", {f"JGH_n{r.major:02d}": analyze_jgh_run(r, conditions) for r in jgh_runs})
        if make_plots:
            for run in jgh_runs:
                for ch in run.channels:
                    plot_jgh_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "JGH"),
                                    sanitize_filename(f"{ch.label}.png")), experiment_type=run.kind.value, day=day, style=style)

    # ZOZO (n01)
    zozo_runs = grouped.get(MajorKind.ZOZO, [])
    if zozo_runs:
        emit("ZOZO: analyzing" + (" + plotting" if make_plots else ""))
        finish("ZOZO", {f"ZOZO_n{r.major:02d}": analyze_zozo_run(r, conditions) for r in zozo_runs})
        if make_plots:
            for run in zozo_runs:
                if style.zozo_stack_per_esn:
                    by_esn: dict[str, list] = {}
                    for ch in run.channels:
                        by_esn.setdefault(ch.esn, []).append(ch)
                    for esn, chs in by_esn.items():
                        d = _dev_dir(out_dir, conditions, chs[0], "ZOZO")
                        plot_zozo_esn(run, chs, os.path.join(d, sanitize_filename(f"ESN{esn}.png")),
                                      experiment_type=run.kind.value, day=day, style=style)
                else:
                    for ch in run.channels:
                        plot_zozo_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "ZOZO"),
                                         sanitize_filename(f"{ch.label}.png")), experiment_type=run.kind.value, day=day, style=style)

    calibration_snapshot = os.path.join(out_dir, "jg_calibration_used.json")
    if jg_calibration is not None and jg_runs:
        snapshot = jg_calibration.as_json_dict()
        snapshot["ss_window_nA"] = list(ss_window_nA) if ss_window_nA is not None else None
        with open(calibration_snapshot, "w", encoding="utf-8") as file:
            json.dump(snapshot, file, indent=2, ensure_ascii=False)
    elif os.path.isfile(calibration_snapshot):
        os.remove(calibration_snapshot)
    emit("Done. Modes: " + ", ".join(written))
    return written
