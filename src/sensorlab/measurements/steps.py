"""Hydrogen ppm-staircase metrics (standalone mode, n07).

Each Part is a held H2 concentration; response is computed relative to the air
baseline (Part 0). The ppm per step is a fixed-protocol value (not recorded in
the data) and lives in ``protocol.se.ppm_steps``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol
from .se import _safe_pct, _t90, device_part_levels


def _baseline_noise(part_df: pd.DataFrame, col: str, window_s: float, mad_k: float) -> float:
    """Std of the baseline over its trailing window, after MAD outlier rejection."""
    s = part_df[["Time", col]].dropna()
    if s.empty:
        return float("nan")
    t_end = float(s["Time"].max())
    win = s[s["Time"] >= t_end - window_s][col].to_numpy(dtype=float)
    if win.size < 3:
        return float("nan")
    med = np.median(win)
    mad = np.median(np.abs(win - med))
    if mad > 0:
        win = win[np.abs(win - med) <= mad_k * mad]
    return float(np.std(win)) if win.size >= 3 else float("nan")


def _sensitivity_and_lod(concs: list[float], resps: list[float], noise_pct: float) -> tuple[float, float]:
    """Fit resp% = a*log10(ppm) + b over the exposure steps.

    sensitivity = a  [%/decade];  LOD = ppm where the fitted line reaches the
    3-sigma baseline noise level (in %).

    CONVENTION (decided 2026-07): the log10 calibration is the primary
    sensitivity/LOD definition — the response is log-linear in ppm, and the
    slope is stable across devices and days (degradation tracking). The
    intuitive engineering KPI is the per-step response itself, reported as
    <ppm>_resp_pct (e.g. 100ppm_resp_pct); a linear %/ppm slope was rejected
    because the 1000 ppm point dominates it on sub-linear devices. sign(a)
    generalizes the 3-sigma crossing to downward-responding devices.
    """
    pts = [(c, r) for c, r in zip(concs, resps) if c > 0 and not pd.isna(r)]
    if len(pts) < 2:
        return float("nan"), float("nan")
    x = np.log10([c for c, _ in pts])
    y = np.array([r for _, r in pts])
    a, b = np.polyfit(x, y, 1)
    lod = float("nan")
    if not pd.isna(noise_pct) and a != 0:
        # response magnitude must exceed 3x noise; sign of a sets the direction
        lod = float(10 ** ((np.sign(a) * 3 * noise_pct - b) / a))
    return float(a), lod


def analyze_steps_run(run: Run, conditions: ConditionCache | None = None,
                      protocol: Protocol = DEFAULT) -> pd.DataFrame:
    """One row per device: baseline + response % / t90 at each H2 step, plus
    sensitivity (%/decade of ppm) and estimated LOD (3-sigma of baseline noise).

    Each step's t90 is anchored on the PREVIOUS step's stabilized level, so it
    reads as 'seconds from the concentration switch until 90% of this step's
    excursion' (responses % stay relative to the air baseline)."""
    p = protocol.se
    ppm = p.ppm_steps
    levels = device_part_levels(run, protocol)
    part_ids = sorted(next(iter(levels.values()))) if levels else []
    parts = run.parts()

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        lv = levels[ch.column]
        baseline = lv[part_ids[0]]
        noise = _baseline_noise(parts[part_ids[0]], ch.column, p.window_s, p.outlier_mad_k)
        noise_pct = _safe_pct(noise, abs(baseline)) if not pd.isna(baseline) else float("nan")
        row: dict[str, object] = {
            "ESN": ch.esn, "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
            "baseline": baseline,
            "baseline_noise": noise,
        }
        step_concs: list[float] = []
        step_resps: list[float] = []
        for i, k in enumerate(part_ids):
            if i == 0:
                continue
            conc = ppm[i] if i < len(ppm) else None
            resp = _safe_pct(lv[k] - baseline, baseline)
            t90 = _t90(parts[k], ch.column, p.window_s, p.outlier_mad_k,
                       y0=lv[part_ids[i - 1]])       # anchored on the previous step

            # absolute levels are not comparable across devices (each starts at
            # its own current), so report the step's delta vs the air baseline
            if conc == 0:                                    # trailing air step = recovery
                row["recovery_delta"] = lv[k] - baseline
                row["recovery_pct"] = resp
                row["recovery_t90_s"] = t90
            else:
                label = f"{int(conc)}ppm" if conc is not None else f"P{k}"
                row[f"{label}_delta"] = lv[k] - baseline
                row[f"{label}_resp_pct"] = resp
                row[f"{label}_t90_s"] = t90
                if conc is not None:
                    step_concs.append(conc)
                    step_resps.append(resp)
        sens, lod = _sensitivity_and_lod(step_concs, step_resps, noise_pct)
        row["sensitivity_pct_per_dec"] = sens
        row["LOD_ppm"] = lod
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["ESN", "TNumber"]).reset_index(drop=True)
