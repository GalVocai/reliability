"""JG_H2: sensing via JG transfer sweeps compared across gas conditions.

Compares the part-2 transfer sweep in air (n03) against H2 (n05) and recovery
(n06), per back-gate. Per device per Vbg:
  * delta_Vth  = Vth(air) - Vth(gas)                         [V(100nA,air) - V(100nA,gas)]
  * resp_at_Vth = (I_gas - I_air)/I_air * 100  at the air Vth
  * max_resp   = max over Vj of (I_gas - I_air)/I_air * 100, and the Vj where it occurs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol
from .jg import _clean_curve, _vth_constant_current


def _interp_at(vj: np.ndarray, idv: np.ndarray, x: float) -> float:
    if np.isnan(x) or vj.size < 2:
        return float("nan")
    return float(np.interp(x, vj, idv))


def compare_condition(air, gas, protocol: Protocol = DEFAULT) -> dict[str, float]:
    """air, gas = (vj, idv) cleaned curves. Returns the comparison metrics."""
    p = protocol.jg
    av, ai = air
    gv, gi = gas
    if av.size < 3 or gv.size < 3:
        return {k: float("nan") for k in ("Vth_air", "Vth_gas", "delta_Vth", "resp_at_Vth", "max_resp", "Vj_max_resp")}

    vth_air = _vth_constant_current(av, ai, p.vth_target_nA, p.vth_onset_guard)
    vth_gas = _vth_constant_current(gv, gi, p.vth_target_nA, p.vth_onset_guard)

    # response at the air threshold voltage
    i_air_vth = _interp_at(av, ai, vth_air)
    i_gas_vth = _interp_at(gv, gi, vth_air)
    resp_at_vth = (i_gas_vth - i_air_vth) / i_air_vth * 100.0 if i_air_vth not in (0.0,) and not np.isnan(i_air_vth) else float("nan")

    # max response over Vj, where air current is reliably above the noise
    lo, hi = max(av.min(), gv.min()), min(av.max(), gv.max())
    max_resp = vj_max = float("nan")
    if hi > lo:
        grid = np.linspace(lo, hi, 400)
        i_air = np.interp(grid, av, ai)
        i_gas = np.interp(grid, gv, gi)
        mask = i_air > p.h2_response_floor_nA
        if mask.any():
            resp = (i_gas[mask] - i_air[mask]) / i_air[mask] * 100.0
            j = int(np.argmax(resp))
            max_resp = float(resp[j])
            vj_max = float(grid[mask][j])

    return {
        "Vth_air": vth_air, "Vth_gas": vth_gas, "delta_Vth": vth_air - vth_gas,
        "resp_at_Vth": resp_at_vth, "max_resp": max_resp, "Vj_max_resp": vj_max,
    }


def _part2_by_vb(run: Run, protocol: Protocol):
    data = run.parts()[protocol.jg.h2_part]
    return data, sorted(int(v) for v in data["Vb"].dropna().unique())


def analyze_jg_h2(jg_runs: list[Run], conditions: ConditionCache | None = None,
                  protocol: Protocol = DEFAULT) -> pd.DataFrame:
    """One row per device per back-gate, comparing H2 (and recovery) vs air."""
    p = protocol.jg
    by_major = {r.major: r for r in jg_runs}
    air = by_major.get(p.h2_air_major)
    h2 = by_major.get(p.h2_exposure_major)
    rec = by_major.get(p.h2_recovery_major)
    if air is None or h2 is None:
        return pd.DataFrame()

    air_data, back_gates = _part2_by_vb(air, protocol)
    h2_data, _ = _part2_by_vb(h2, protocol)
    rec_data = _part2_by_vb(rec, protocol)[0] if rec is not None else None

    def curve(data, col, vb):
        sub = data[data["Vb"] == vb]
        return _clean_curve(sub["Vj"].to_numpy(float), sub[col].to_numpy(float))

    rows: list[dict[str, object]] = []
    for ch in air.channels:
        for vb in back_gates:
            air_c = curve(air_data, ch.column, vb)
            h2_c = curve(h2_data, ch.column, vb)
            m = compare_condition(air_c, h2_c, protocol)
            row: dict[str, object] = {
                "ESN": ch.esn, "TNumber": ch.tnumber,
                "condition": conditions.get(ch.esn) if conditions else "",
                "Vbg": vb,
                "Vth_air": m["Vth_air"], "Vth_H2": m["Vth_gas"], "delta_Vth_H2": m["delta_Vth"],
                "resp_at_Vth_H2_pct": m["resp_at_Vth"],
                "max_resp_H2_pct": m["max_resp"], "Vj_max_resp_H2": m["Vj_max_resp"],
            }
            if rec_data is not None:
                mr = compare_condition(air_c, curve(rec_data, ch.column, vb), protocol)
                row.update(
                    Vth_recovery=mr["Vth_gas"], delta_Vth_recovery=mr["delta_Vth"],
                    resp_at_Vth_recovery_pct=mr["resp_at_Vth"],
                    max_resp_recovery_pct=mr["max_resp"], Vj_max_resp_recovery=mr["Vj_max_resp"],
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["ESN", "TNumber", "Vbg"]).reset_index(drop=True)
