"""JG transfer-sweep metrics (Vth / SS / gm / Ion / Ioff).

Each JG major has 3 reps (Parts 0,1,2); the first two are for stabilization, so
only the **last** rep is used. Within that rep the two back-gates (Vb = 0, 4)
are separate transfer curves (Vj = -2.8 -> 0.3). One row per device, with each
metric reported per back-gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol


def _clean_curve(vj: np.ndarray, idv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"Vj": vj, "Id": idv}).dropna()
    df = df.groupby("Vj", as_index=False)["Id"].mean().sort_values("Vj")
    return df["Vj"].to_numpy(float), df["Id"].to_numpy(float)


def back_gate_groups(data: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    """Split a sweep table by back-gate voltage, tolerant of recorded noise.

    Vb is grouped on the value rounded to 10 mV: an exact-setpoint file (0, 4)
    behaves exactly as before, while a measured 3.9997 V still lands in the
    4 V group instead of silently producing an empty subset (-> all-NaN
    metrics). Labels are ints when the voltage is integral ('bg4'), else the
    rounded float.
    """
    vb_r = data["Vb"].round(2)
    out: list[tuple[object, pd.DataFrame]] = []
    for vb in sorted(vb_r.dropna().unique()):
        label = int(vb) if float(vb).is_integer() else float(vb)
        out.append((label, data[vb_r == vb]))
    return out


def _vth_constant_current(vj: np.ndarray, idv: np.ndarray, target: float, guard: int = 1) -> float:
    """Vj at the first *sustained* upward crossing of `target`.

    Interpolates between the two adjacent measured points (no regression). The
    onset guard requires the next `guard` points to stay >= target, so an
    isolated noise spike in the off region cannot trigger a false Vth.
    """
    n = len(idv)
    for i in range(1, n):
        if idv[i - 1] < target <= idv[i]:
            end = min(i + max(guard, 1), n)
            if all(idv[j] >= target for j in range(i, end)):
                x0, x1, y0, y1 = vj[i - 1], vj[i], idv[i - 1], idv[i]
                return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def _subthreshold_swing(vj: np.ndarray, idv: np.ndarray, noise: float, ss_upper: float) -> float:
    """mV/decade from a fit of Vj vs log10(Id) over the log-linear turn-on
    (Id from just above the noise floor up to `ss_upper` = a fraction of Ion)."""
    mask = (idv > noise) & (idv < ss_upper)
    if mask.sum() >= 3:
        x = vj[mask]
        y = np.log10(idv[mask])
        slope = np.polyfit(y, x, 1)[0]      # dVj / d log10(Id)
        return abs(slope) * 1000.0
    return float("nan")


def _fixed_current_ss(vj: np.ndarray, idv: np.ndarray,
                      window_nA: tuple[float, float]) -> tuple[float, int, float, float, str]:
    """Fit log10(Id) against Vj within one common positive-current window.

    Return SS, point count, R², current span in decades, and a QC status.
    A missing SS is deliberate when the measured points cannot support the
    proposed subthreshold fit; interpolation never invents extra points.
    """
    low, high = map(float, window_nA)
    if not 0 < low < high:
        raise ValueError("SS current window must have 0 < lower < upper nA")
    use = np.isfinite(vj) & np.isfinite(idv) & (idv >= low) & (idv <= high)
    n = int(np.count_nonzero(use))
    if n < 4:
        return float("nan"), n, float("nan"), float("nan"), "too_few_points"
    x = vj[use]
    y = np.log10(idv[use])
    span = float(np.ptp(y))
    if np.ptp(x) <= 0 or span < 0.3:
        return float("nan"), n, float("nan"), span, "insufficient_current_span"
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r2 = float(1 - np.sum(residual ** 2) / np.sum((y - np.mean(y)) ** 2))
    if slope <= 0:
        return float("nan"), n, r2, span, "nonpositive_slope"
    if r2 < 0.95:
        return float("nan"), n, r2, span, "low_linearity"
    return float(1000 / slope), n, r2, span, "ok"


def _gm_max(vj: np.ndarray, idv: np.ndarray) -> float:
    if len(vj) < 3:
        return float("nan")
    return float(np.nanmax(np.gradient(idv, vj)))


def _snr_off(idv: np.ndarray) -> float:
    """SNR of the off-region current: |mean| / std over the low-Vj quarter."""
    off = idv[: max(5, len(idv) // 4)]
    sd = float(np.std(off))
    if sd == 0 or off.size < 3:
        return float("nan")
    return float(abs(np.mean(off)) / sd)


def _vth_second_derivative(vj: np.ndarray, idv: np.ndarray) -> float:
    """Vth at the inflection of the transfer curve: the zero crossing of
    d2Id/dVj2 at the max-gm point (interpolated between adjacent samples)."""
    if len(vj) < 5:
        return float("nan")
    gm = np.gradient(idv, vj)
    d2 = np.gradient(gm, vj)
    i = int(np.nanargmax(gm))
    # walk rightward from the gm peak to the next sign change of d2 (the
    # curve inflects past the peak; left of it d2 is positive by construction)
    for j in range(max(i, 1), len(d2)):
        if d2[j - 1] > 0 >= d2[j] or d2[j - 1] >= 0 > d2[j]:
            y0, y1 = d2[j - 1], d2[j]
            if y0 == y1:
                return float(vj[j])
            return float(vj[j - 1] + y0 * (vj[j] - vj[j - 1]) / (y0 - y1))
    return float(vj[i])


def jg_curve_metrics(vj: np.ndarray, idv: np.ndarray, protocol: Protocol = DEFAULT,
                     ss_window_nA: tuple[float, float] | None = None) -> dict[str, float]:
    p = protocol.jg
    vj, idv = _clean_curve(vj, idv)
    if len(vj) < 3:
        return {k: float("nan") for k in ("Vth", "Vth_d2", "SS", "gm", "Ion", "Ioff", "Ion_Ioff", "SNR_off")}
    ion = float(idv[-1])                      # on end (Vj = 0.3)
    ioff = float(idv[0])                      # off end (Vj = -2.8)
    ratio = ion / max(ioff, p.noise_floor_nA)
    metrics = {
        "Vth": _vth_constant_current(vj, idv, p.vth_target_nA, p.vth_onset_guard),
        "Vth_d2": _vth_second_derivative(vj, idv),
        "SS": _subthreshold_swing(vj, idv, p.noise_floor_nA, p.ss_upper_frac * ion),
        "gm": _gm_max(vj, idv),
        "Ion": ion,
        "Ioff": ioff,
        "Ion_Ioff": ratio,
        "SNR_off": _snr_off(idv),
    }
    if ss_window_nA is not None:
        ss, n, r2, span, status = _fixed_current_ss(vj, idv, ss_window_nA)
        metrics.update(SS=ss, SS_fit_n=n, SS_fit_R2=r2,
                       SS_fit_span_dec=span, SS_fit_status=status)
    return metrics


def _jg_flags(m: dict[str, float], protocol: Protocol) -> str:
    flags: list[str] = []
    if pd.isna(m["Vth"]):
        flags.append("vth_undefined")
    if pd.isna(m["SS"]):
        flags.append("ss_unreliable")
    if not pd.isna(m["Ion_Ioff"]) and m["Ion_Ioff"] < protocol.jg.min_on_off:
        flags.append("low_on_off")
    # 'ok' (not '') so the master cell isn't blank -> NaN when read back
    return ";".join(flags) or "ok"


def relevant_part(run: Run, protocol: Protocol = DEFAULT) -> int:
    parts = sorted(run.parts())
    return parts[-1] if protocol.jg.relevant_rep_is_last else parts[0]


def analyze_jg_run(
    run: Run,
    conditions: ConditionCache | None = None,
    protocol: Protocol = DEFAULT,
    ss_window_nA: tuple[float, float] | None = None,
) -> pd.DataFrame:
    data = run.parts()[relevant_part(run, protocol)]
    gate_groups = back_gate_groups(data)

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        row: dict[str, object] = {
            "ESN": ch.esn,
            "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
        }
        for vb, sub in gate_groups:
            m = jg_curve_metrics(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float),
                                 protocol, ss_window_nA=ss_window_nA)
            for k, v in m.items():
                row[f"{k}_bg{vb}"] = v
            row[f"flags_bg{vb}"] = _jg_flags(m, protocol)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["ESN", "TNumber"]).reset_index(drop=True)
