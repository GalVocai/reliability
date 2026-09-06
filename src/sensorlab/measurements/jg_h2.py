"""JG_H2: sensing via JG transfer sweeps compared across gas conditions.

Compares the part-2 transfer sweep in air (n03) against H2 (n05) and recovery
(n06), per back-gate. Per device per Vbg:
  * delta_Vth  = Vth(air) - Vth(gas)     [V(vth_target_nA,air) - V(vth_target_nA,gas)]
  * resp_at_Vth = (I_gas - I_air)/I_air * 100  at the air Vth
  * max_resp   = max over Vj of (I_gas - I_air)/I_air * 100, and the Vj where it occurs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol
from .jg import _clean_curve, _vth_constant_current, back_gate_groups


def _interp_at(vj: np.ndarray, idv: np.ndarray, x: float) -> float:
    if np.isnan(x) or vj.size < 2:
        return float("nan")
    return float(np.interp(x, vj, idv))


def _turnon_onset_vj(av: np.ndarray, ai: np.ndarray, guard: int) -> float:
    """Vj where the air current sustainably rises out of its own off-noise band.

    The off-noise level is estimated from the low-Vj end of the sweep; the onset
    is the first Vj past which the current stays above it (so the response search
    starts at the turn-on, not in the deep-off region where it diverges)."""
    n = ai.size
    if n < 5:
        return float("nan")
    off = ai[: max(5, n // 4)]                         # low-Vj end = off region
    noise = float(np.percentile(np.abs(off), 95))      # noise band amplitude
    for i in range(n):
        if ai[i] > noise and all(ai[j] > noise for j in range(i, min(i + max(guard, 1), n))):
            return float(av[i])
    return float("nan")


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

    # max response over Vj, searched only from the air turn-on onset rightward
    # (so it never re-enters the deep-off region where the ratio diverges).
    onset = _turnon_onset_vj(av, ai, p.vth_onset_guard)
    lo = max(av.min(), gv.min(), onset if not np.isnan(onset) else -np.inf)
    hi = min(av.max(), gv.max())
    max_resp = vj_max = float("nan")
    if hi > lo:
        grid = np.linspace(lo, hi, 800)
        i_air = np.interp(grid, av, ai)
        i_gas = np.interp(grid, gv, gi)
        resp = (i_gas - i_air) / i_air * 100.0
        valid = i_air > 0
        if valid.any():
            j = int(np.argmax(resp[valid]))
            max_resp = float(resp[valid][j])
            vj_max = float(grid[valid][j])
    # the Vth point lies inside the onset region; keep it as a candidate so a
    # sharp peak there is never missed by grid discretization.
    if not np.isnan(resp_at_vth) and (np.isnan(max_resp) or resp_at_vth > max_resp):
        max_resp, vj_max = resp_at_vth, vth_air

    return {
        "Vth_air": vth_air, "Vth_gas": vth_gas, "delta_Vth": vth_air - vth_gas,
        "resp_at_Vth": resp_at_vth, "max_resp": max_resp, "Vj_max_resp": vj_max,
    }


def _part2_by_vb(run: Run, protocol: Protocol):
    data = run.parts()[protocol.jg.h2_part]
    return data, [vb for vb, _sub in back_gate_groups(data)]


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
        # rounded match pairs with back_gate_groups (tolerates recorded noise)
        sub = data[data["Vb"].round(2) == round(float(vb), 2)]
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
