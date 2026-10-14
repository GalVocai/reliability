"""Audit the number of directly measured points in each modeled SS interval."""

from pathlib import Path

import numpy as np
import pandas as pd

from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import _clean_curve, back_gate_groups, relevant_part


ROOT = Path.home() / "Desktop" / "Hot_storage_SS_full_sweep_Vbg0"
fits = pd.read_csv(ROOT / "Hot_storage_SS_full_sweep_Vbg0_device_fits.csv",
                   dtype={"ESN": str, "TNumber": str})
regions = pd.read_csv(
    ROOT / "Hot_storage_SS_full_curve_region_3sigma_to_150nA_Vbg0_devices.csv",
    dtype={"ESN": str, "TNumber": str})
rows = []
for day, group in fits.groupby("day_num"):
    source = read_run(group.raw_workbook.iloc[0])
    calibration = JgCalibration.from_json(group.calibration_json.iloc[0])
    corrected = apply_jg_calibration(source, calibration)
    sweep = dict(back_gate_groups(corrected.parts()[relevant_part(corrected)]))[0]
    channels = {(ch.esn, ch.tnumber): ch for ch in corrected.channels}
    for _, row in group.iterrows():
        channel = channels[(row.ESN, row.TNumber)]
        vjg, current = _clean_curve(sweep.Vj.to_numpy(float),
                                    sweep[channel.column].to_numpy(float))
        lower = float(row["3sigma_onset_nA"])
        use = (current >= lower) & (current <= 150)
        count = int(use.sum())
        direct_ss = np.nan
        direct_r2 = np.nan
        if count >= 3 and np.ptp(vjg[use]) > 0:
            logi = np.log10(current[use])
            slope, intercept = np.polyfit(vjg[use], logi, 1)
            if slope > 0:
                direct_ss = float(1000 / slope)
                fit_log = slope * vjg[use] + intercept
                direct_r2 = float(1 - np.sum((logi - fit_log) ** 2)
                                  / np.sum((logi - logi.mean()) ** 2))
        rows.append({"day_num": day, "ESN": row.ESN,
                     "TNumber": row.TNumber, "n_measured_region": count,
                     "SS_direct_region_mV_per_dec": direct_ss,
                     "direct_log_linear_R2": direct_r2})
audit = pd.DataFrame(rows)
audit = audit.merge(regions[["day_num", "ESN", "TNumber",
                             "SS_region_mV_per_dec"]],
                    on=["day_num", "ESN", "TNumber"], validate="one_to_one")
audit["direct_minus_model_mV_per_dec"] = (
    audit.SS_direct_region_mV_per_dec - audit.SS_region_mV_per_dec)
audit.to_csv(ROOT / "Hot_storage_SS_region_reliability_audit.csv", index=False)
print("Measured points within each device-specific noise-to-150 nA interval:")
print(audit.n_measured_region.value_counts().sort_index().to_string())
print("By checkpoint:")
print(audit.groupby("day_num").n_measured_region.agg(
    ["min", "median", "max"]).to_string())
valid = audit.dropna(subset=["SS_direct_region_mV_per_dec"])
print(f"Direct log-linear fits possible: {len(valid)}/{len(audit)}")
print("Absolute model-vs-direct SS difference, median/90th percentile:",
      np.percentile(abs(valid.direct_minus_model_mV_per_dec), [50, 90]))
print("Direct fit R2 median/min:", valid.direct_log_linear_R2.median(),
      valid.direct_log_linear_R2.min())
