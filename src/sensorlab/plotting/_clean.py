"""Display-only curve cleaning for plots (does not affect computed metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positive_despiked(x, y, win: int = 5, log_thresh: float = 0.7):
    """For log plots: keep only Id > 0 and drop isolated spikes.

    A point is a spike if its log10(Id) deviates from the local rolling median
    by more than `log_thresh` decades.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
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
