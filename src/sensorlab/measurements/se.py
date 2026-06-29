"""SE (sensing) metrics -- one row per device.

Anchored on the explicit ``Part`` column and physical ``Time`` windows (never
row indices). For the 4-part main sensing run the post-sweep segment is
*corrected*: the capacitive offset introduced by the in-exposure JG sweep is
removed by shifting the settle (Part 2) back onto the pre-sweep response
(Part 1), and recovery is read off the corrected curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol


def _trailing_level(part_df: pd.DataFrame, col: str, window_s: float, mad_k: float) -> float:
    """Mean of `col` over the last `window_s` seconds of a part.

    Robust: samples more than `mad_k` MADs from the window median are dropped
    first, so a stray spike can't distort the level (values are unchanged when
    the window is clean).
    """
    s = part_df[["Time", col]].dropna()
    if s.empty:
        return float("nan")
    t_end = float(s["Time"].max())
    win = s[s["Time"] >= t_end - window_s][col].to_numpy(dtype=float)
    if win.size == 0:
        return float("nan")
    med = np.median(win)
    mad = np.median(np.abs(win - med))
    if mad > 0:
        win = win[np.abs(win - med) <= mad_k * mad]
    return float(np.mean(win)) if win.size else float(med)


def _safe_pct(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return float("nan")
    return num / den * 100.0


def device_part_levels(run: Run, protocol: Protocol = DEFAULT) -> dict[str, dict[int, float]]:
    """Per device -> {part: stabilized level}."""
    p = protocol.se
    parts = run.parts()
    return {
        ch.column: {k: _trailing_level(g, ch.column, p.window_s, p.outlier_mad_k) for k, g in parts.items()}
        for ch in run.channels
    }


def is_main_sensing(run: Run, protocol: Protocol = DEFAULT) -> bool:
    """The 4-part baseline/exposure/settle/recovery run that gets corrected."""
    p = protocol.se
    return set(run.parts()) == {p.baseline_part, p.response_part, p.settle_part, p.recovery_part}


def analyze_se_run(
    run: Run,
    conditions: ConditionCache | None = None,
    protocol: Protocol = DEFAULT,
) -> pd.DataFrame:
    """Return a per-device metrics table for one SE run."""
    p = protocol.se
    levels = device_part_levels(run, protocol)
    corrected = is_main_sensing(run, protocol)
    part_ids = sorted(next(iter(levels.values()))) if levels else []

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        lv = levels[ch.column]
        row: dict[str, object] = {
            "ESN": ch.esn,
            "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
        }
        if corrected:
            baseline = lv[p.baseline_part]
            response = lv[p.response_part]
            settle = lv[p.settle_part]
            recovery = lv[p.recovery_part]
            shift = response - settle                       # correct settle onto response
            row.update(
                baseline=baseline,
                response=response,
                response_pct=_safe_pct(response - baseline, baseline),
                settle_raw=settle,
                shift=shift,
                settle_corr=settle + shift,                 # == response by construction
                recovery_raw=recovery,
                recovery_corr=recovery + shift,
                recovery_pct=_safe_pct((settle + shift) - (recovery + shift), (settle + shift) - baseline),
            )
        else:
            # generic SE run (e.g. ppm staircase): per-part levels + response vs baseline
            base = lv.get(part_ids[0]) if part_ids else float("nan")
            for k in part_ids:
                row[f"P{k}_level"] = lv[k]
            for k in part_ids[1:]:
                row[f"P{k}_response_pct"] = _safe_pct(lv[k] - base, base)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["ESN", "TNumber"]).reset_index(drop=True)
