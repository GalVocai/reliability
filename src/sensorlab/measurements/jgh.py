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


def jgh_curve_metrics(vj: np.ndarray, idv: np.ndarray, protocol: Protocol = DEFAULT) -> dict[str, float]:
    p = protocol.jg
    (vf, if_), (vr, ir) = split_forward_reverse(vj, idv)
    vf, if_ = _clean_curve(vf, if_)
    vr, ir = _clean_curve(vr, ir)
    if len(vf) < 3 or len(vr) < 3:
        return {k: float("nan") for k in ("Vth_fwd", "Vth_rev", "dVth", "loop_area", "Ion")}

    vth_f = _vth_constant_current(vf, if_, p.vth_target_nA)
    vth_r = _vth_constant_current(vr, ir, p.vth_target_nA)

    # loop area on the common Vj range: integral of (reverse - forward) dVj
    lo, hi = max(vf.min(), vr.min()), min(vf.max(), vr.max())
    grid = np.linspace(lo, hi, 200)
    _trapz = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz
    area = float(_trapz(np.interp(grid, vr, ir) - np.interp(grid, vf, if_), grid))

    return {
        "Vth_fwd": vth_f,
        "Vth_rev": vth_r,
        "dVth": vth_r - vth_f,                 # hysteresis voltage shift
        "loop_area": area,                     # nA*V (signed)
        "Ion": float(max(if_.max(), ir.max())),
    }


def _jgh_flags(m: dict[str, float]) -> str:
    flags: list[str] = []
    if pd.isna(m["Vth_fwd"]) or pd.isna(m["Vth_rev"]):
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
