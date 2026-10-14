"""Collapse selected Hot-storage JG_n03 sweeps to each device's day-0 Vth.

Uses the active seven-timepoint corrected analysis data and the existing
SensorLab line-only JG collapse style. Writes a new Desktop folder only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import (
    _clean_curve, _vth_constant_current, back_gate_groups, relevant_part,
)
from sensorlab.plotting.evolution_plot import plot_jg_sweep_collapse
from sensorlab.plotting.style import load_style
from sensorlab.protocol import DEFAULT

from export_hot_storage_ss_window import _active_timepoints, _selected_devices


OUTPUT = Path.home() / "Desktop" / "Hot_storage_JGn03_Vbg0_collapsed_to_day0"
BG_V = 0
ALIGN_CURRENT_NA = float(DEFAULT.jg.vth_target_nA)
CHECK_CURRENTS_NA = (500.0, 5000.0)
GUARD = int(DEFAULT.jg.vth_onset_guard)


def _crossing(vjg: np.ndarray, current: np.ndarray, target: float) -> float:
    return _vth_constant_current(vjg, current, target, GUARD)


def main() -> None:
    selected, selection_source = _selected_devices()
    timepoints = _active_timepoints()
    expected_days = {day for day, *_ in timepoints}
    curves: dict[tuple[str, str], list[dict]] = defaultdict(list)
    records: list[dict] = []

    for day_num, day_label, raw_folder, analysis_folder in timepoints:
        files = sorted(raw_folder.glob("*_n03_JG_.xlsx"))
        if len(files) != 1:
            raise ValueError(f"Expected one JG_n03 workbook at {raw_folder}")
        calibration_path = analysis_folder / "jg_calibration_used.json"
        calibration = JgCalibration.from_json(calibration_path)
        run = apply_jg_calibration(read_run(str(files[0])), calibration)
        sweep = dict(back_gate_groups(run.parts()[relevant_part(run)]))[BG_V]
        found = set()
        for channel in run.channels:
            key = (channel.esn, channel.tnumber)
            if key not in selected:
                continue
            found.add(key)
            vjg, current = _clean_curve(sweep.Vj.to_numpy(float),
                                        sweep[channel.column].to_numpy(float))
            vth = _crossing(vjg, current, ALIGN_CURRENT_NA)
            if not np.isfinite(vth):
                raise ValueError(f"No {ALIGN_CURRENT_NA:g} nA crossing: {key}, {day_label}")
            checks = {f"V_at_{int(target)}nA_V": _crossing(vjg, current, target)
                      for target in CHECK_CURRENTS_NA}
            curves[key].append({"day_num": day_num, "day_label": day_label,
                                "bg": BG_V, "vj": vjg, "id": current})
            records.append({
                "day": day_label, "day_num": day_num, "ESN": channel.esn,
                "TNumber": channel.tnumber, "Vbg_V": BG_V,
                "Vth_150nA_V": vth, **checks,
                "raw_workbook": str(files[0]),
                "calibration_json": str(calibration_path),
            })
        if found != selected:
            raise ValueError(f"Missing selected devices at {day_label}: "
                             f"{sorted(selected - found)}")

    for key, device_curves in curves.items():
        if {c["day_num"] for c in device_curves} != expected_days:
            raise ValueError(f"Incomplete timepoints for {key}")

    table = pd.DataFrame(records).sort_values(["ESN", "TNumber", "day_num"])
    baseline = table[table.day_num.eq(0)].set_index(["ESN", "TNumber"])
    if len(baseline) != len(selected):
        raise ValueError("Incomplete day-0 baseline")
    base150 = np.array([baseline.loc[(e, t), "Vth_150nA_V"]
                        for e, t in zip(table.ESN, table.TNumber)], dtype=float)
    table["horizontal_shift_mV"] = 1000 * (table.Vth_150nA_V - base150)
    for target in CHECK_CURRENTS_NA:
        column = f"V_at_{int(target)}nA_V"
        base_target = np.array([baseline.loc[(e, t), column]
                                for e, t in zip(table.ESN, table.TNumber)],
                               dtype=float)
        table[f"shape_residual_at_{int(target)}nA_mV"] = (
            1000 * (table[column] - base_target) - table.horizontal_shift_mV)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    style = load_style()
    for key in sorted(selected):
        esn, tnumber = key
        basename = f"ESN{esn}_D{tnumber[0]}T{tnumber[1]}_JGn03_Vbg0_collapsed_to_day0"
        for suffix in ("png", "pdf"):
            plot_jg_sweep_collapse(
                curves[key], str(OUTPUT / f"{basename}.{suffix}"),
                title=f"ESN{esn} D{tnumber[0]}T{tnumber[1]}",
                xlim=(DEFAULT.jg.sweep_start_vj, DEFAULT.jg.sweep_close_vj),
                time_axis="days", linewidth=style.linewidth, plot_mode="line",
            )
    table.to_csv(OUTPUT / "alignment_shifts_and_shape_check.csv", index=False)
    (OUTPUT / "README.txt").write_text(
        "Hot storage JG_n03, last sweep repetition, Vbg=0 V.\n"
        f"Devices: {len(selected)} from the active degradation selection.\n"
        f"Checkpoints: {', '.join(str(int(x)) for x in sorted(expected_days))} days.\n"
        "Currents use each timepoint's active JG blank-current correction snapshot.\n"
        "Each day's Vth is the guarded, linearly interpolated upward crossing "
        f"of {ALIGN_CURRENT_NA:g} nA in the measured corrected sweep.\n"
        "For each device, the day-0 Vth is the anchor. Every other voltage "
        "coordinate is plotted as VJG - (Vth_day - Vth_0D). Current is not "
        "normalized or vertically shifted.\n"
        "The same SensorLab collapse plot style is used: solid lines only, "
        "one colour per day, log-current axis. The display helper uses |Id| "
        "and drops isolated off-region spikes; Vth shifts are computed from "
        "the original signed corrected values.\n"
        "The CSV includes crossing-voltage shift residuals at 500 and 5000 nA: "
        "zero means perfect horizontal translation at that current level.\n"
        f"Selection source: {selection_source}\n",
        encoding="utf-8",
    )
    print(f"Saved {len(selected)} device plots (PNG and PDF) to {OUTPUT}")
    for target in CHECK_CURRENTS_NA:
        column = f"shape_residual_at_{int(target)}nA_mV"
        values = table.loc[table.day_num.ne(0), column].dropna().abs()
        print(f"{target:g} nA residual: median={values.median():.1f} mV, "
              f"90th percentile={values.quantile(0.9):.1f} mV, n={len(values)}")


if __name__ == "__main__":
    main()
