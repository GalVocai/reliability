"""JGH hysteresis metrics.

Each back-gate (Vb = -4, 0, 4) is a forward sweep (Vj -2.8 -> 0.3) followed by a
reverse sweep (0.3 -> -2.8). Hysteresis is the Vth shift between the two
directions (dVth) plus the area enclosed by the loop. One row per device, per
back-gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol
from .jg import _clean_curve, _vth_constant_current


def split_forward_reverse(vj: np.ndarray, idv: np.ndarray):
    """Split a loop at its Vj peak into (forward, reverse) halves."""
    imax = int(np.argmax(vj))
    return (vj[: imax + 1], idv[: imax + 1]), (vj[imax:], idv[imax:])


def _loop_area_and_hi(vf: np.ndarray, if_: np.ndarray, vr: np.ndarray, ir: np.ndarray) -> tuple[float, float]:
    """Return (loop_area, HI) over the overlapping Vj range.

    loop_area = integral |Id_fwd - Id_rev| dVj            (area-based hysteresis, nA*V)
    HI        = loop_area / integral max(Id_fwd, Id_rev)  (normalized, legacy definition)
    """
    mask = (vf >= float(vr.min())) & (vf <= float(vr.max()))
    xg, yf = vf[mask], if_[mask]
    if xg.size < 2:
        return float("nan"), float("nan")
    yb = np.interp(xg, vr, ir)
    _trapz = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz
    area = float(_trapz(np.abs(yf - yb), xg))
    den = float(_trapz(np.maximum(yf, yb), xg))
    hi = area / den if abs(den) > 1e-30 else float("nan")
    return area, hi


def jgh_curve_metrics(vj: np.ndarray, idv: np.ndarray, protocol: Protocol = DEFAULT) -> dict[str, float]:
    p = protocol.jg
    (vf, if_), (vr, ir) = split_forward_reverse(vj, idv)
    vf, if_ = _clean_curve(vf, if_)
    vr, ir = _clean_curve(vr, ir)
    if len(vf) < 3 or len(vr) < 3:
        return {k: float("nan") for k in ("Vth_FWD", "Vth_BWD", "Delta_Vth", "loop_area", "HI", "Ion")}

    vth_f = _vth_constant_current(vf, if_, p.vth_target_nA)
    vth_r = _vth_constant_current(vr, ir, p.vth_target_nA)
    area, hi = _loop_area_and_hi(vf, if_, vr, ir)

    return {
        "Vth_FWD": vth_f,
        "Vth_BWD": vth_r,
        "Delta_Vth": vth_f - vth_r,            # (1) hysteresis by Vth shift (fwd - rev)
        "loop_area": area,                     # (2) hysteresis by loop area (nA*V)
        "HI": hi,                              # (3) hysteresis by normalized index
        "Ion": float(max(if_.max(), ir.max())),
    }


def _jgh_flags(m: dict[str, float]) -> str:
    flags: list[str] = []
    if pd.isna(m["Vth_FWD"]) or pd.isna(m["Vth_BWD"]):
        flags.append("vth_undefined")
    return ";".join(flags)


def analyze_jgh_run(
    run: Run,
    conditions: ConditionCache | None = None,
    protocol: Protocol = DEFAULT,
) -> pd.DataFrame:
    data = run.parts()[sorted(run.parts())[0]]   # JGH is a single part
    back_gates = sorted(int(v) for v in data["Vb"].dropna().unique())

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        row: dict[str, object] = {
            "ESN": ch.esn,
            "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
        }
        for vb in back_gates:
            sub = data[data["Vb"] == vb]
            m = jgh_curve_metrics(sub["Vj"].to_numpy(float), sub[ch.column].to_numpy(float), protocol)
            for k, v in m.items():
                row[f"{k}_bg{vb}"] = v
            row[f"flags_bg{vb}"] = _jgh_flags(m)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["ESN", "TNumber"]).reset_index(drop=True)
