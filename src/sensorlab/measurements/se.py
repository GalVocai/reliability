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


def _t90(part_df: pd.DataFrame, col: str, window_s: float, mad_k: float,
         guard: int = 3, y0: float | None = None) -> float:
    """Seconds from the part start (= the gas switch) until the signal first
    *sustainably* reaches 90% of the step it makes in this part.

    `y0` anchors the step: pass the PREVIOUS part's stabilized level so the
    90% mark is measured on the physical step (previous phase -> this phase's
    stabilized end level), not on whatever the first samples of this part
    happen to read (a fast device is already moving by then). When `y0` is
    omitted/NaN (no previous part) it falls back to the median of the first
    5 samples.

    Direction-agnostic (works for rise and decay). The guard requires the next
    `guard` samples to stay past the 90% mark so a spike can't fake arrival.
    """
    s = part_df[["Time", col]].dropna().sort_values("Time")
    if len(s) < guard + 2:
        return float("nan")
    t = s["Time"].to_numpy(float)
    y = s[col].to_numpy(float)
    if y0 is None or pd.isna(y0):
        y0 = float(np.median(y[: min(5, y.size)]))
    yf = _trailing_level(part_df, col, window_s, mad_k)
    if pd.isna(yf) or yf == y0:
        return float("nan")
    target = y0 + 0.9 * (yf - y0)
    rising = yf > y0
    n = y.size
    for i in range(n):
        arrived = y[i] >= target if rising else y[i] <= target
        if arrived:
            end = min(i + max(guard, 1), n)
            ok = all((y[j] >= target if rising else y[j] <= target) for j in range(i, end))
            if ok:
                return float(t[i] - t[0])
    return float("nan")


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




def _se_flags(run: Run, col: str, baseline: float, response: float,
              recovery_pct: float, protocol: Protocol,
              t90s: tuple[float, ...] = ()) -> str:
    """Quality flags for one device (semicolon-separated; 'ok' if clean)."""
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
    if any(pd.isna(t) for t in t90s):
        flags.append("t90_not_reached")
    # 'ok' (not '') so the master cell isn't blank -> NaN when read back
    return ";".join(flags) or "ok"


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
            recovery_pct = _safe_pct((settle + shift) - (recovery + shift), (settle + shift) - baseline)
            # t90 anchored on the PREVIOUS phase's stabilized level: response
            # vs the air baseline; recovery vs the settle level (both raw --
            # the post-sweep capacitive shift is common to settle and recovery
            # so it cancels out of the 90% geometry).
            t90_resp = _t90(run.parts()[p.response_part], ch.column, p.window_s,
                            p.outlier_mad_k, y0=baseline)
            t90_rec = _t90(run.parts()[p.recovery_part], ch.column, p.window_s,
                           p.outlier_mad_k, y0=settle)
            row.update(
                baseline=baseline,
                response=response,
                response_delta=response - baseline,          # nA, comparable across devices
                response_pct=_safe_pct(response - baseline, baseline),
                t90_response_s=t90_resp,
                settle_raw=settle,
                shift=shift,
                settle_corr=settle + shift,                 # == response by construction
                recovery_raw=recovery,
                recovery_corr=recovery + shift,
                recovery_delta=(recovery + shift) - baseline,   # corrected residual vs baseline, nA
                recovery_pct=recovery_pct,
                t90_recovery_s=t90_rec,
                flags=_se_flags(run, ch.column, baseline, response, recovery_pct, protocol,
                                t90s=(t90_resp, t90_rec)),
            )
        else:
            # fallback: generic per-part deltas + response % vs the first part
            base = lv.get(part_ids[0]) if part_ids else float("nan")
            for k in part_ids[1:]:
                row[f"P{k}_delta"] = lv[k] - base
                row[f"P{k}_response_pct"] = _safe_pct(lv[k] - base, base)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["ESN", "TNumber"]).reset_index(drop=True)
