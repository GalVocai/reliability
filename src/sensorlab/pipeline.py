"""Folder-level orchestration: load -> group by test type -> analyze -> masters."""

from __future__ import annotations

import os
from collections import defaultdict

from .io import load_condition_cache, read_run, scan_result_files
from .io.device_map import ConditionCache
from .io.master_writer import write_master
from .measurements import analyze_se_run
from .measurements.se import is_main_sensing
from .model import MajorKind, Run
from .plotting import plot_se_device
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
) -> dict[str, str]:
    """Run the available analyzers and write one master per test type.

    Returns {test_type: master_path}. Conditions are taken from the cache
    (filled once via the GUI); missing ones are simply left blank for now.
    """
    runs = load_folder(folder)
    grouped = group_by_kind(runs)
    conditions = conditions or load_condition_cache()
    out_dir = out_dir or os.path.join(folder, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    written: dict[str, str] = {}

    # SE sensing -> one master workbook, one sheet per run (n04, n07, ...)
    se_runs = grouped.get(MajorKind.SE, [])
    if se_runs:
        se_dir = os.path.join(out_dir, "SE")
        os.makedirs(se_dir, exist_ok=True)
        sheets = {f"SE_n{run.major:02d}": analyze_se_run(run, conditions) for run in se_runs}
        written["SE"] = write_master(os.path.join(se_dir, "master_SE.xlsx"), sheets)
        # per-device verification plots for the corrected main sensing run(s)
        for run in se_runs:
            if not is_main_sensing(run):
                continue
            dev_dir = os.path.join(se_dir, "by_device")
            os.makedirs(dev_dir, exist_ok=True)
            for ch in run.channels:
                fname = sanitize_filename(f"{ch.label}.png")
                plot_se_device(run, ch, os.path.join(dev_dir, fname))

    # ZOZO / JG / JGH analyzers land here next.
    return written
