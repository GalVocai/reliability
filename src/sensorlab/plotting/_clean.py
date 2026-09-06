"""Display-only curve cleaning for plots (does not affect computed metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# DC-shift anchor: the curve minimum is searched only after this many seconds,
# so a start-of-measurement transient can't drag the shifted trace upward.
DC_SHIFT_SKIP_S = 550.0


def stable_min(x, y, skip_s: float = DC_SHIFT_SKIP_S) -> float:
    """Minimum of `y`, ignoring the first `skip_s` seconds of `x`.

    Falls back to the full-curve minimum when the trace is shorter than the
    skip window. NaN for an empty/all-NaN trace.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y.size == 0 or np.all(np.isnan(y)):
        return float("nan")
    m = x >= x[0] + skip_s
    win = y[m]
    if win.size == 0 or np.all(np.isnan(win)):
        win = y
    return float(np.nanmin(win))


def abs_despiked(x, y, win: int = 5, log_thresh: float = 0.7):
    """For log plots: plot |Id| and drop isolated spikes.

    The off-region current swings through zero (it is noise around 0), and a
    log axis cannot show negatives. Taking |Id| keeps those samples as their
    magnitude instead of discarding them, so the off-region noise floor stays
    visible rather than leaving a gap. Exact zeros are still dropped -- log10(0)
    is undefined.

    A point is a spike if its log10|Id| deviates from the local rolling median
    by more than `log_thresh` decades. Note this still removes samples that sit
    very near zero: |Id| there plunges many decades and would otherwise stretch
    the log axis toward -inf.
    """
    x = np.asarray(x, float)
    y = np.abs(np.asarray(y, float))
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[m], y[m]
    if len(y) < max(win, 3):
        return x, y
    s = np.log10(y)
    med = pd.Series(s).rolling(win, center=True, min_periods=1).median().to_numpy()
    keep = np.abs(s - med) <= log_thresh
    return x[keep], y[keep]


def despiked_linear(x, y, win: int = 11, k: float = 6.0):
    """For linear plots: drop isolated spikes via rolling-median / MAD."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < max(win, 3):
        return x, y
    s = pd.Series(y)
    med = s.rolling(win, center=True, min_periods=1).median()
    resid = (s - med).abs()
    mad = resid.rolling(win, center=True, min_periods=1).median()
    fallback = resid.median() if resid.median() > 0 else 1.0
    mad = mad.replace(0, np.nan).fillna(fallback)
    keep = (resid <= k * mad).to_numpy()
    return x[keep], y[keep]
