"""Hydrogen ppm-staircase metrics (standalone mode, n07).

Each Part is a held H2 concentration; response is computed relative to the air
baseline (Part 0). The ppm per step is a fixed-protocol value (not recorded in
the data) and lives in ``protocol.se.ppm_steps``.
"""

from __future__ import annotations

import pandas as pd

from ..io.device_map import ConditionCache
from ..model import Run
from ..protocol import DEFAULT, Protocol
from .se import _safe_pct, device_part_levels


def analyze_steps_run(run: Run, conditions: ConditionCache | None = None,
                      protocol: Protocol = DEFAULT) -> pd.DataFrame:
    """One row per device: baseline + response % at each H2 step vs baseline."""
    ppm = protocol.se.ppm_steps
    levels = device_part_levels(run, protocol)
    part_ids = sorted(next(iter(levels.values()))) if levels else []

    rows: list[dict[str, object]] = []
    for ch in run.channels:
        lv = levels[ch.column]
        baseline = lv[part_ids[0]]
        row: dict[str, object] = {
            "ESN": ch.esn, "TNumber": ch.tnumber,
            "condition": conditions.get(ch.esn) if conditions else "",
            "baseline": baseline,
        }
        for i, k in enumerate(part_ids):
            if i == 0:
                continue
            conc = ppm[i] if i < len(ppm) else None
            resp = _safe_pct(lv[k] - baseline, baseline)
            if conc == 0:                                    # trailing air step = recovery
                row["recovery_level"] = lv[k]
                row["recovery_pct"] = resp
            else:
                label = f"{int(conc)}ppm" if conc is not None else f"P{k}"
                row[f"{label}_level"] = lv[k]
                row[f"{label}_resp_pct"] = resp
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["ESN", "TNumber"]).reset_index(drop=True)
