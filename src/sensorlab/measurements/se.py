"""SE (sensing) metrics -- one row per device.

Everything is anchored on the explicit ``Part`` column and physical ``Time``
windows. No absolute row indices, so it is robust to cadence changes / dropped
rows, and it self-corrects if the part structure changes.
"""

from __future__ import annotations

import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol


def _trailing_level(part_df: pd.DataFrame, col: str, window_s: float) -> float:
    """Mean of `col` over the last `window_s` seconds of a part."""
    s = part_df[["Time", col]].dropna()
    if s.empty:
        return float("nan")
    t_end = float(s["Time"].max())
    window = s[s["Time"] >= t_end - window_s]
    return float(window[col].mean())


def _safe_pct(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return float("nan")
    return num / den * 100.0


def analyze_se_run(
    run: Run,
    conditions: ConditionCache | None = None,
    protocol: Protocol = DEFAULT,
) -> pd.DataFrame:
    """Return a per-device metrics table for one SE run."""
    p = protocol.se
    parts = run.parts()
    part_ids = sorted(parts)
    has_phases = all(k in parts for k in (p.baseline_part, p.response_part, p.recovery_part))

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        row: dict[str, object] = {
            "ESN": ch.esn,
            "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
        }
        # per-part stabilized level
        levels = {k: _trailing_level(parts[k], ch.column, p.window_s) for k in part_ids}
        for k in part_ids:
            row[f"P{k}_level"] = levels[k]

        if has_phases:
            baseline = levels[p.baseline_part]
            response = levels[p.response_part]
            recovery = levels[p.recovery_part]
            resp_delta = response - baseline
            row["baseline"] = baseline
            row["response"] = response
            row["response_pct"] = _safe_pct(resp_delta, baseline)
            row["recovery"] = recovery
            # how much of the response decayed back toward baseline
            row["recovery_pct"] = _safe_pct(response - recovery, resp_delta)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["ESN", "TNumber"]).reset_index(drop=True)
