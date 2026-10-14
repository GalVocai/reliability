"""Discover and analyze all raw reliability timepoints from one parent folder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Callable
from uuid import uuid4

from .degradation import ACTIVE_BATCH_PREFIX, day_to_num
from .io.device_map import ConditionCache
from .io.result_reader import scan_result_files
from .jg_calibration import JgCalibration
from .pipeline import analyze_folder


_TIMEPOINT = re.compile(r"^(baseline|base|t0|\d+\s*d)$", re.IGNORECASE)


def discover_raw_timepoints(parent: str | Path) -> list[tuple[str, Path]]:
    """Find raw E* result folders, including days without prior analysis."""
    root = Path(parent)
    if not root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for day_dir in root.iterdir():
        if not day_dir.is_dir() or not _TIMEPOINT.fullmatch(day_dir.name.strip()):
            continue
        candidates = [folder for folder in day_dir.iterdir()
                      if folder.is_dir() and folder.name.upper().startswith("E")
                      and scan_result_files(str(folder))]
        if len(candidates) > 1:
            raise ValueError(f"More than one raw E* result folder in {day_dir}: {candidates}")
        if candidates:
            found.append((day_dir.name, candidates[0]))
    found.sort(key=lambda item: day_to_num(item[0]))
    return found


def analyze_all_timepoints(
    parent: str | Path,
    *,
    conditions: ConditionCache,
    jg_calibration: JgCalibration | None = None,
    ss_window_nA: tuple[float, float] | None = None,
    make_plots: bool = True,
    overwrite: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[tuple[str, Path]]:
    """Build one versioned analysis for every day, then activate the set.

    OneDrive can deny renaming an existing analysis directory. Each run writes
    to new sibling directories instead. A study-level manifest is created only
    after every day succeeds, so degradation never reads a partial batch.
    Existing analysis directories and raw workbooks are not modified.
    """
    emit = log or (lambda _message: None)
    timepoints = discover_raw_timepoints(parent)
    if not timepoints:
        raise ValueError(f"No raw timepoint result folders found under {parent}")
    study_root = Path(parent).resolve()
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
              + "_" + uuid4().hex[:8])
    analysis_name = f"analysis_batch_{uuid4().hex[:12]}"
    completed: list[tuple[str, Path]] = []
    entries: dict[str, str] = {}
    for index, (day, folder) in enumerate(timepoints, 1):
        emit(f"[{index}/{len(timepoints)}] Analyzing {day}: {folder}")
        folder = folder.resolve()
        legacy = folder / "analysis"
        if legacy.exists() and not overwrite:
            raise FileExistsError(f"Existing analysis directory: {legacy}")
        target = folder / analysis_name
        if target.exists() or target.parent != folder:
            raise RuntimeError(f"Versioned analysis path is unavailable: {target}")
        try:
            written = analyze_folder(str(folder), out_dir=str(target),
                                     conditions=conditions, make_plots=make_plots,
                                     day=day, log=emit,
                                     jg_calibration=jg_calibration,
                                     ss_window_nA=ss_window_nA)
            if not written or not target.is_dir():
                raise RuntimeError(f"No analysis outputs were produced for {day}")
        except Exception:
            emit("Batch failed; the previous active analyses remain unchanged. "
                 f"Incomplete new outputs, if any, are under {analysis_name}.")
            raise
        entries[folder.relative_to(study_root).as_posix()] = analysis_name
        completed.append((day, folder))
        emit(f"[{index}/{len(timepoints)}] Saved new version: {target}")
    manifest = study_root / f"{ACTIVE_BATCH_PREFIX}{run_id}.json"
    manifest.write_text(json.dumps({"version": 1, "run_id": run_id,
                                    "timepoints": entries}, indent=2),
                        encoding="utf-8")
    emit(f"Activated {len(completed)} timepoints via {manifest}")
    return completed
