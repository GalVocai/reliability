"""Per-device population grouping: separate devices whose behavior departs
from their condition group, so bad devices can't silently ruin the averages.

Groups (priority order; the first matching one wins):
  dead_baseline      -- |baseline| below ``SEProtocol.low_baseline_nA``:
                        percent metrics are meaningless (tiny denominator).
  opposite_direction -- the response delta's sign opposes the group majority
                        (non-physical: the device moves the wrong way on gas).
  no_recovery        -- non-physical recovery (e.g. current keeps rising after
                        the gas is switched off).
  outlier_high/low   -- robust z of the reference response beyond
                        ``SEProtocol.group_z_thresh``. z = 0.6745*(x-med)/MAD,
                        computed per condition WITHOUT dead-baseline devices
                        so one dead part can't inflate the spread.
  normal             -- everything else.

Classification never drops rows: it adds a ``group`` column to the master
sheet plus a companion summary sheet showing WHY each device was grouped.
Exclusion stays a manual decision (the Degradation window's device list).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..protocol import DEFAULT, Protocol


def _robust_z(x: pd.Series) -> pd.Series:
    """0.6745 * (x - median) / MAD; NaN when MAD is 0 (no usable spread)."""
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(float("nan"), index=x.index)
    return 0.6745 * (x - med) / mad


def _classify(df: pd.DataFrame, ref_col: str, delta_col: str,
              recovery_bad: pd.Series, recovery_col: str,
              protocol: Protocol) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared grouping core. Returns (df + 'group' column, summary table)."""
    p = protocol.se
    dead = df["baseline"].abs() < p.low_baseline_nA
    sign = np.sign(df[delta_col].fillna(0.0))
    z = pd.Series(float("nan"), index=df.index)
    opposite = pd.Series(False, index=df.index)

    conds = df["condition"].fillna("").astype(str) if "condition" in df.columns else pd.Series("", index=df.index)
    for cv in conds.unique():
        live = (conds == cv) & ~dead
        if not live.any():
            continue
        majority = np.sign(sign[live].sum()) or 1.0
        z[live] = _robust_z(df.loc[live, ref_col])
        opposite[live] = (sign[live] == -majority).to_numpy()

    group = np.select(
        [dead.to_numpy(dtype=bool), opposite.to_numpy(dtype=bool),
         recovery_bad.fillna(False).to_numpy(dtype=bool),
         (z > p.group_z_thresh).to_numpy(dtype=bool),
         (z < -p.group_z_thresh).to_numpy(dtype=bool)],
        ["dead_baseline", "opposite_direction", "no_recovery",
         "outlier_high", "outlier_low"],
        default="normal",
    )

    out = df.copy()
    out["group"] = group
    summary = pd.DataFrame({
        "ESN": df["ESN"], "TNumber": df["TNumber"],
        "condition": df.get("condition", ""),
        "group": group,
        "baseline": df["baseline"],
        ref_col: df[ref_col],
        "direction": np.where(sign > 0, "up", np.where(sign < 0, "down", "flat")),
        recovery_col: df[recovery_col],
        "z_ref": z.round(2),
        "z_thresh": p.group_z_thresh,
    })
    return out, summary.reset_index(drop=True)


def classify_se(df: pd.DataFrame, protocol: Protocol = DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group an SE master table. Reference response: response_pct.

    no_recovery: recovery_pct outside [recovery_pct_min, recovery_pct_max] --
    negative means the current kept rising after the gas went off.
    """
    needed = {"baseline", "response_pct", "response_delta", "recovery_pct"}
    if not needed <= set(df.columns):        # fallback (non 4-part) SE table
        return df, pd.DataFrame()
    p = protocol.se
    bad_rec = df["recovery_pct"].notna() & ~df["recovery_pct"].between(
        p.recovery_pct_min, p.recovery_pct_max)
    return _classify(df, "response_pct", "response_delta", bad_rec,
                     "recovery_pct", protocol)


def classify_steps(df: pd.DataFrame, protocol: Protocol = DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group a STEPS master table. Reference response: the top ppm step.

    no_recovery: the recovery residual is negative (overshoot below baseline)
    or larger than the top-step response itself (still rising after gas off).
    """
    top = int(max(protocol.se.ppm_steps))
    ref, delta = f"{top}ppm_resp_pct", f"{top}ppm_delta"
    needed = {"baseline", ref, delta, "recovery_pct"}
    if not needed <= set(df.columns):
        return df, pd.DataFrame()
    bad_rec = df["recovery_pct"].notna() & (
        (df["recovery_pct"] < 0) | (df["recovery_pct"] > df[ref]))
    return _classify(df, ref, delta, bad_rec, "recovery_pct", protocol)
