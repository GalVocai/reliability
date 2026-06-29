"""Folder-level orchestration: load -> group by test type -> analyze -> masters."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable

from .io import load_condition_cache, read_run, scan_result_files
from .io.device_map import ConditionCache
from .io.master_writer import write_master
from .measurements import (
    analyze_jg_run,
    analyze_jgh_run,
    analyze_se_run,
    analyze_steps_run,
    analyze_zozo_run,
)
from .model import MajorKind, Run
from .plotting import (
    plot_jg_device,
    plot_jgh_device,
    plot_se_device,
    plot_steps_device,
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


def analyze_folder(
    folder: str,
    out_dir: str | None = None,
    conditions: ConditionCache | None = None,
    make_plots: bool = True,
    day: str = "",
    log: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Run the available analyzers and write one master per test type.

    Returns {test_type: master_path}. Conditions are taken from the cache
    (filled once via the GUI); missing ones are simply left blank for now.
    Set ``make_plots=False`` for fast master-only runs (per-device figures are
    the slow part). `log` receives progress messages (used by the GUI).
    """
    emit = log or (lambda _m: None)
    runs = load_folder(folder)
    grouped = group_by_kind(runs)
    conditions = conditions or load_condition_cache()
    out_dir = out_dir or os.path.join(folder, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    emit(f"Loaded {len(runs)} majors: " + ", ".join(f"n{r.major:02d}={r.kind.value}" for r in runs))

    written: dict[str, str] = {}

    # SE sensing -> one master workbook, one sheet per run (n04, n07, ...)
    se_runs = grouped.get(MajorKind.SE, [])
    if se_runs:
        se_dir = os.path.join(out_dir, "SE")
        os.makedirs(se_dir, exist_ok=True)
        emit("SE: analyzing" + (" + plotting" if make_plots else ""))
        sheets = {f"SE_n{run.major:02d}": analyze_se_run(run, conditions) for run in se_runs}
        written["SE"] = write_master(os.path.join(se_dir, "master_SE.xlsx"), sheets)
        if make_plots:
            dev_dir = os.path.join(se_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for run in se_runs:
                for ch in run.channels:
                    plot_se_device(run, ch, os.path.join(dev_dir, sanitize_filename(f"{ch.label}.png")),
                                   experiment_type=run.kind.value, day=day)

    # STEPS (H2 ppm staircase) -> standalone master + plots
    steps_runs = grouped.get(MajorKind.STEPS, [])
    if steps_runs:
        steps_dir = os.path.join(out_dir, "STEPS")
        os.makedirs(steps_dir, exist_ok=True)
        emit("STEPS: analyzing" + (" + plotting" if make_plots else ""))
        sheets = {f"STEPS_n{run.major:02d}": analyze_steps_run(run, conditions) for run in steps_runs}
        written["STEPS"] = write_master(os.path.join(steps_dir, "master_STEPS.xlsx"), sheets)
        if make_plots:
            dev_dir = os.path.join(steps_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for run in steps_runs:
                for ch in run.channels:
                    plot_steps_device(run, ch, os.path.join(dev_dir, sanitize_filename(f"{ch.label}.png")),
                                      experiment_type=run.kind.value, day=day)

    # JG transfer sweeps -> one master, one sheet per major (n03/n05/n06)
    jg_runs = grouped.get(MajorKind.JG, [])
    if jg_runs:
        jg_dir = os.path.join(out_dir, "JG")
        os.makedirs(jg_dir, exist_ok=True)
        emit("JG: analyzing" + (" + plotting" if make_plots else ""))
        sheets = {f"JG_n{run.major:02d}": analyze_jg_run(run, conditions) for run in jg_runs}
        written["JG"] = write_master(os.path.join(jg_dir, "master_JG.xlsx"), sheets)
        if make_plots:
            dev_dir = os.path.join(jg_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for run in jg_runs:
                for ch in run.channels:
                    fname = sanitize_filename(f"n{run.major:02d}_{ch.label}.png")
                    plot_jg_device(run, ch, os.path.join(dev_dir, fname),
                                   experiment_type=run.kind.value, day=day)

    # JGH hysteresis -> one master (single major n02) + per-device loop plots
    jgh_runs = grouped.get(MajorKind.JGH, [])
    if jgh_runs:
        jgh_dir = os.path.join(out_dir, "JGH")
        os.makedirs(jgh_dir, exist_ok=True)
        emit("JGH: analyzing" + (" + plotting" if make_plots else ""))
        sheets = {f"JGH_n{run.major:02d}": analyze_jgh_run(run, conditions) for run in jgh_runs}
        written["JGH"] = write_master(os.path.join(jgh_dir, "master_JGH.xlsx"), sheets)
        if make_plots:
            dev_dir = os.path.join(jgh_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for run in jgh_runs:
                for ch in run.channels:
                    plot_jgh_device(run, ch, os.path.join(dev_dir, sanitize_filename(f"{ch.label}.png")),
                                    experiment_type=run.kind.value, day=day)

    # ZOZO -> one master + per-device plots
    zozo_runs = grouped.get(MajorKind.ZOZO, [])
    if zozo_runs:
        zozo_dir = os.path.join(out_dir, "ZOZO")
        os.makedirs(zozo_dir, exist_ok=True)
        emit("ZOZO: analyzing" + (" + plotting" if make_plots else ""))
        sheets = {f"ZOZO_n{run.major:02d}": analyze_zozo_run(run, conditions) for run in zozo_runs}
        written["ZOZO"] = write_master(os.path.join(zozo_dir, "master_ZOZO.xlsx"), sheets)
        if make_plots:
            dev_dir = os.path.join(zozo_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for run in zozo_runs:
                for ch in run.channels:
                    plot_zozo_device(run, ch, os.path.join(dev_dir, sanitize_filename(f"{ch.label}.png")),
                                     experiment_type=run.kind.value, day=day)

    emit("Done. Masters: " + ", ".join(written))
    return written
