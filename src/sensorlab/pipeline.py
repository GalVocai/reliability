"""Folder-level orchestration: load -> group by test type -> analyze -> masters."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable

from .io import load_condition_cache, read_run, scan_result_files
from .io.device_map import ConditionCache
from .io.master_writer import write_master
from .measurements import (
    analyze_jg_h2,
    analyze_jg_run,
    analyze_jgh_run,
    analyze_se_run,
    analyze_steps_run,
    analyze_zozo_run,
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
)
from .util import sanitize_filename


def load_folder(folder: str) -> list[Run]:
    """Read every result file and classify it by test type."""
    return [read_run(p) for p in scan_result_files(folder)]


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
    cond_values: set[str] = set()
    for d in sheets.values():
        if "condition" in d.columns:
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
) -> dict[str, str]:
    """Run the analyzers and write outputs split by condition.

    Layout: ``out_dir/<condition>/<MODE>/{master_<MODE>.xlsx, by_device/*.png}``.
    The condition per device comes from the cache (filled once via the GUI);
    devices with no condition land under 'unassigned'.
    Set ``make_plots=False`` for fast master-only runs.
    """
    emit = log or (lambda _m: None)
    runs = load_folder(folder)
    grouped = group_by_kind(runs)
    conditions = conditions or load_condition_cache()
    style = load_style()
    out_dir = out_dir or os.path.join(folder, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    emit(f"Loaded {len(runs)} majors: " + ", ".join(f"n{r.major:02d}={r.kind.value}" for r in runs))

    written: dict[str, str] = {}

    def finish(mode: str, sheets: dict) -> None:
        for d in sheets.values():           # tag every master row with the timepoint
            d["day"] = day
        paths = _write_master_by_condition(out_dir, mode, sheets)
        if paths:
            written[mode] = f"{len(paths)} condition(s)"

    # SE sensing (n04)
    se_runs = grouped.get(MajorKind.SE, [])
    if se_runs:
        emit("SE: analyzing" + (" + plotting" if make_plots else ""))
        finish("SE", {f"SE_n{r.major:02d}": analyze_se_run(r, conditions) for r in se_runs})
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
        finish("STEPS", {f"STEPS_n{r.major:02d}": analyze_steps_run(r, conditions) for r in steps_runs})
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
        finish("JG", {f"JG_n{r.major:02d}": analyze_jg_run(r, conditions) for r in jg_runs})
        if make_plots:
            for run in jg_runs:
                for ch in run.channels:
                    plot_jg_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "JG"),
                                   sanitize_filename(f"n{run.major:02d}_{ch.label}.png")),
                                   experiment_type=run.kind.value, day=day, style=style)

    # JG_H2 -> air (n03) vs H2 (n05) vs recovery (n06)
    if jg_runs:
        h2_df = analyze_jg_h2(jg_runs, conditions)
        if not h2_df.empty:
            emit("JG_H2: analyzing" + (" + plotting" if make_plots else ""))
            finish("JG_H2", {"JG_H2": h2_df})
            by_major = {r.major: r for r in jg_runs}
            air = by_major.get(DEFAULT.jg.h2_air_major)
            h2 = by_major.get(DEFAULT.jg.h2_exposure_major)
            if make_plots and air is not None and h2 is not None:
                for ch in air.channels:
                    plot_jg_h2_device(air, h2, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "JG_H2"),
                                      sanitize_filename(f"{ch.label}.png")), day=day, style=style)

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
                for ch in run.channels:
                    plot_zozo_device(run, ch, os.path.join(_dev_dir(out_dir, conditions, ch, "ZOZO"),
                                     sanitize_filename(f"{ch.label}.png")), experiment_type=run.kind.value, day=day)

    emit("Done. Modes: " + ", ".join(written))
    return written
