"""ZOZO transient metrics: shared slow/fast bi-exponential fit.

Each held-state segment (a ``Part``) relaxes as
    Id(t) = DC_i + A_slow * exp(-t_abs/tau_slow) + A_fast_i * exp(-t_rel/tau_fast_i)
where A_slow / tau_slow are SHARED across the segments of one state and
DC_i / A_fast_i / tau_fast_i are per-segment. Math ported from the legacy
ZOZOMode; segmentation adapted to the multi-device Part/state layout.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

MIN_TAU = 1e-15
CURVE_FIT_MAXFEV = 20000


def _shared_segment_model(t_abs, t_rel, dc, a_slow, tau_slow, a_fast, tau_fast):
    return dc + a_slow * np.exp(-t_abs / tau_slow) + a_fast * np.exp(-t_rel / tau_fast)


def _shared_concat_model(x_data, *params):
    t_abs, t_rel, seg_ids = x_data
    seg_index = np.asarray(seg_ids, dtype=int)
    out = np.empty_like(t_abs, dtype=float)
    a_slow, tau_slow = float(params[0]), float(params[1])
    for idx in range((len(params) - 2) // 3):
        base = 2 + 3 * idx
        dc_i, af_i, tf_i = float(params[base]), float(params[base + 1]), float(params[base + 2])
        m = seg_index == idx
        out[m] = dc_i + a_slow * np.exp(-t_abs[m] / tau_slow) + af_i * np.exp(-t_rel[m] / tf_i)
    return out


def _r2(y, yp) -> float:
    res = y - yp
    rss = float(np.sum(res * res))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if np.isclose(tss, 0.0):
        return 1.0 if np.isclose(rss, 0.0) else 0.0
    return float(1.0 - rss / tss)


def _fit_shared_model(segments: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if curve_fit is None:
        return None, "scipy not available"
    if not segments:
        return None, "no segments"

    p0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    first = segments[0]["current"]
    last = segments[-1]["current"]
    fw = first[: min(10, first.size)]
    lw = last[-min(10, last.size):]
    a_slow_guess = float(np.median(lw) - np.median(fw)) if fw.size and lw.size else 0.0
    span = max(float(segments[-1]["time_abs"][-1] - segments[0]["time_abs"][0]), 1.0)
    p0 += [a_slow_guess, max(0.3 * span, 1.0)]
    lo += [-np.inf, MIN_TAU]
    hi += [np.inf, np.inf]

    cat_abs, cat_rel, cat_y, cat_ids = [], [], [], []
    for idx, seg in enumerate(segments):
        t_abs, t_rel, y = seg["time_abs"], seg["time_local"], seg["current"]
        if t_abs.size < 5:
            return None, f"segment {seg['part']} has < 5 points"
        tail = y[-min(20, y.size):]
        dc_guess = float(np.median(tail))
        local_span = max(float(t_rel[-1] - t_rel[0]), 1.0)
        p0 += [dc_guess, float(y[0] - dc_guess), max(0.3 * local_span, 1.0)]
        lo += [-np.inf, -np.inf, MIN_TAU]
        hi += [np.inf, np.inf, np.inf]
        cat_abs.append(t_abs); cat_rel.append(t_rel); cat_y.append(y)
        cat_ids.append(np.full(t_abs.shape, idx, dtype=int))

    aa, ar, ay, aid = map(np.concatenate, (cat_abs, cat_rel, cat_y, cat_ids))
    try:
        params, _ = curve_fit(_shared_concat_model, (aa, ar, aid), ay, p0=p0,
                              bounds=(lo, hi), maxfev=CURVE_FIT_MAXFEV)
    except Exception as exc:
        return None, str(exc)
    params = np.asarray(params, float)
    n = len(segments)
    return {
        "A_slow": float(params[0]),
        "tau_slow": float(params[1]),
        "DC_values": [float(params[2 + 3 * i]) for i in range(n)],
        "A_fast_values": [float(params[3 + 3 * i]) for i in range(n)],
        "tau_fast_values": [float(params[4 + 3 * i]) for i in range(n)],
        "global_target": ay,
        "global_prediction": _shared_concat_model((aa, ar, aid), *params),
    }, None


def build_segments(run: Run, col: str) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for k, g in run.parts().items():
        sub = g[["Time", "Vj", col]].dropna(subset=["Time", col])
        if sub.empty:
            continue
        t = sub["Time"].to_numpy(float)
        segs.append({
            "part": int(k),
            "Vj": round(float(sub["Vj"].iloc[0]), 3),
            "time_abs": t,
            "time_local": t - t[0],
            "current": sub[col].to_numpy(float),
        })
    return segs


def fit_device(run: Run, col: str) -> list[dict[str, Any]]:
    """Per device -> list of per-segment fit results (grouped & fit by state)."""
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for seg in build_segments(run, col):
        groups[seg["Vj"]].append(seg)

    out: list[dict[str, Any]] = []
    for vj_state, group in groups.items():
        bundle, _ = _fit_shared_model(group)
        for idx, seg in enumerate(group):
            row = {"state_Vj": vj_state, "segment": seg["part"], "n_points": int(seg["current"].size),
                   "t_start": float(seg["time_abs"][0]), "t_end": float(seg["time_abs"][-1]), "seg": seg}
            if bundle is None:
                row.update(DC=np.nan, A_slow=np.nan, tau_slow=np.nan, A_fast=np.nan,
                           tau_fast=np.nan, R2_seg=np.nan, R2_global=np.nan, fit=None)
            else:
                dc, af, tf = bundle["DC_values"][idx], bundle["A_fast_values"][idx], bundle["tau_fast_values"][idx]
                fit = _shared_segment_model(seg["time_abs"], seg["time_local"], dc,
                                            bundle["A_slow"], bundle["tau_slow"], af, tf)
                row.update(DC=dc, A_slow=bundle["A_slow"], tau_slow=bundle["tau_slow"], A_fast=af,
                           tau_fast=tf, R2_seg=_r2(seg["current"], fit),
                           R2_global=_r2(bundle["global_target"], bundle["global_prediction"]), fit=fit)
            out.append(row)
    return out


def _single_exp(t, dc, a, tau):
    return dc + a * np.exp(-t / tau)


def fit_segment_single(seg: dict[str, Any]) -> dict[str, Any]:
    """Single-exponential settle fit: Id(t) = DC + A*exp(-t/tau), local time."""
    t, y = seg["time_local"], seg["current"]
    nan = {"level": np.nan, "A": np.nan, "tau": np.nan, "R2": np.nan, "fit": None}
    if curve_fit is None or t.size < 4:
        return nan
    dc0 = float(np.median(y[-min(20, y.size):]))
    try:
        p, _ = curve_fit(_single_exp, t, y, p0=[dc0, float(y[0] - dc0), 5.0],
                         bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e4]),
                         maxfev=CURVE_FIT_MAXFEV)
    except Exception:
        return nan
    fit = _single_exp(t, *p)
    return {"level": float(p[0]), "A": float(p[1]), "tau": float(p[2]), "R2": _r2(y, fit), "fit": fit}


def analyze_zozo_run(run: Run, conditions: ConditionCache | None = None,
                     protocol: Protocol = DEFAULT) -> pd.DataFrame:
    """One row per device per segment: single-exp settle (level/A/tau/R2)."""
    rows: list[dict[str, object]] = []
    for ch in run.channels:
        for seg in build_segments(run, ch.column):
            f = fit_segment_single(seg)
            rows.append({
                "ESN": ch.esn, "TNumber": ch.tnumber,
                "condition": conditions.get(ch.esn) if conditions else "",
                "state_Vj": seg["Vj"], "segment": seg["part"], "n_points": int(seg["current"].size),
                "level": f["level"], "A": f["A"], "tau": f["tau"], "R2": f["R2"],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["ESN", "TNumber", "state_Vj", "segment"]).reset_index(drop=True)
