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


def _window_mean(part_df: pd.DataFrame, col: str, t_lo: float, t_hi: float, mad_k: float) -> float:
    s = part_df[["Time", col]].dropna()
    win = s[(s["Time"] >= t_lo) & (s["Time"] < t_hi)][col].to_numpy(dtype=float)
    if win.size == 0:
        return float("nan")
    med = np.median(win)
    mad = np.median(np.abs(win - med))
    if mad > 0:
        win = win[np.abs(win - med) <= mad_k * mad]
    return float(np.mean(win)) if win.size else float(med)


def _settle_drift(part_df: pd.DataFrame, col: str, window_s: float, mad_k: float) -> float:
    """How much the settle level moved over the last window vs the window before
    it -- a proxy for 'still rising / not stabilized'."""
    s = part_df[["Time", col]].dropna()
    if s.empty:
        return float("nan")
    t_end = float(s["Time"].max())
    last = _window_mean(part_df, col, t_end - window_s, t_end + 1, mad_k)
    prev = _window_mean(part_df, col, t_end - 2 * window_s, t_end - window_s, mad_k)
    return last - prev


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


def is_ppm_staircase(run: Run, protocol: Protocol = DEFAULT) -> bool:
    """SE run whose parts match the ppm-staircase sequence (n07)."""
    return not is_main_sensing(run, protocol) and len(run.parts()) == len(protocol.se.ppm_steps)


def _se_flags(run: Run, col: str, baseline: float, response: float,
              recovery_pct: float, protocol: Protocol) -> str:
    """Quality flags for one device (semicolon-separated; '' if clean)."""
    p = protocol.se
    flags: list[str] = []
    span = response - baseline
    drift = _settle_drift(run.parts()[p.settle_part], col, p.window_s, p.outlier_mad_k)
    if span and not pd.isna(drift) and abs(drift) > p.settle_drift_frac * abs(span):
        flags.append("settle_not_stable")
    if not pd.isna(recovery_pct) and not (p.recovery_pct_min <= recovery_pct <= p.recovery_pct_max):
        flags.append("recovery_out_of_range")
    if abs(baseline) < p.low_baseline_nA:
        flags.append("low_baseline")
    return ";".join(flags)


def analyze_se_run(
    run: Run,
    conditions: ConditionCache | None = None,
    protocol: Protocol = DEFAULT,
) -> pd.DataFrame:
    """Return a per-device metrics table for one SE run."""
    p = protocol.se
    levels = device_part_levels(run, protocol)
    corrected = is_main_sensing(run, protocol)
    staircase = is_ppm_staircase(run, protocol)
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
            recovery_pct = _safe_pct((settle + shift) - (recovery + shift), (settle + shift) - baseline)
            row.update(
                baseline=baseline,
                response=response,
                response_pct=_safe_pct(response - baseline, baseline),
                settle_raw=settle,
                shift=shift,
                settle_corr=settle + shift,                 # == response by construction
                recovery_raw=recovery,
                recovery_corr=recovery + shift,
                recovery_pct=recovery_pct,
                flags=_se_flags(run, ch.column, baseline, response, recovery_pct, protocol),
            )
        elif staircase:
            # ppm staircase: response of each H2 step relative to the air baseline (P0)
            ppm = p.ppm_steps
            baseline = lv[part_ids[0]]
            row["baseline"] = baseline
            for i, k in enumerate(part_ids):
                if i == 0:
                    continue
                conc = ppm[i] if i < len(ppm) else None
                resp = _safe_pct(lv[k] - baseline, baseline)
                if conc == 0:                                # trailing air step = recovery
                    row["recovery_level"] = lv[k]
                    row["recovery_pct"] = resp
                else:
                    label = f"{int(conc)}ppm" if conc is not None else f"P{k}"
                    row[f"{label}_level"] = lv[k]
                    row[f"{label}_resp_pct"] = resp
        else:
            # fallback: generic per-part levels + response vs baseline
            base = lv.get(part_ids[0]) if part_ids else float("nan")
            for k in part_ids:
                row[f"P{k}_level"] = lv[k]
            for k in part_ids[1:]:
                row[f"P{k}_response_pct"] = _safe_pct(lv[k] - base, base)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["ESN", "TNumber"]).reset_index(drop=True)
