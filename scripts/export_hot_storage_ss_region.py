"""Export full-curve-model SS across the above-noise subthreshold region.

Uses the per-sweep full-curve parameters already saved by
export_hot_storage_ss_curve_fit.py. The quadratic sensitivity model is not used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from export_hot_storage_ss_curve_fit import transfer_model


OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_full_sweep_Vbg0"
SOURCE = OUTPUT / "Hot_storage_SS_full_sweep_Vbg0_device_fits.csv"
UPPER_CURRENT_NA = 150.0
BASE_NAME = "Hot_storage_SS_full_curve_region_3sigma_to_150nA_Vbg0"


def _current_at(vjg: float, params: np.ndarray) -> float:
    return float(transfer_model(np.array([vjg]), params)[0])


def _crossing(params: np.ndarray, current_na: float) -> float:
    low = -5.0
    high = 2.0
    if not _current_at(low, params) < current_na < _current_at(high, params):
        raise ValueError(f"Fitted model does not cross {current_na:g} nA")
    return float(brentq(lambda v: _current_at(v, params) - current_na,
                        low, high))


def _region_ss(row: pd.Series) -> pd.Series:
    lower_current = float(row["3sigma_onset_nA"])
    if not 0 < lower_current < UPPER_CURRENT_NA:
        raise ValueError(f"Invalid current range: {row.device}, {row.day}")
    params = np.array([
        float(row.fit_offset_nA),
        np.log(float(row.fit_amplitude_nA)),
        float(row.fit_V0_V),
        np.log(float(row.fit_width_V)),
        float(row.fit_exponent),
    ])
    low_v = _crossing(params, lower_current)
    high_v = _crossing(params, UPPER_CURRENT_NA)
    vjg = np.linspace(low_v, high_v, 201)
    modeled_current = transfer_model(vjg, params)
    slope, intercept = np.polyfit(vjg, np.log10(modeled_current), 1)
    predicted_log = slope * vjg + intercept
    observed_log = np.log10(modeled_current)
    r2 = 1 - np.sum((observed_log - predicted_log) ** 2) / np.sum(
        (observed_log - observed_log.mean()) ** 2)
    if slope <= 0 or not np.isfinite(r2):
        raise ValueError(f"Invalid subthreshold slope: {row.device}, {row.day}")
    return pd.Series({
        "SS_region_mV_per_dec": 1000 / slope,
        "region_log_linear_R2": r2,
        "region_lower_VJG_V": low_v,
        "region_upper_VJG_V": high_v,
    })


def main() -> None:
    table = pd.read_csv(SOURCE, dtype={"ESN": str, "TNumber": str})
    if not table.fit_status.eq("ok").all():
        raise ValueError("Some full-curve fits require review")
    region = table.apply(_region_ss, axis=1)
    table = pd.concat([table, region], axis=1)
    if table.groupby("day_num").size().nunique() != 1:
        raise ValueError("Checkpoint populations are unequal")
    summary = table.groupby("day_num", as_index=False).agg(
        n_devices=("device", "size"),
        SS_mean_mV_per_dec=("SS_region_mV_per_dec", "mean"),
        SS_SD_mV_per_dec=("SS_region_mV_per_dec", "std"),
        minimum_region_R2=("region_log_linear_R2", "min"),
    ).sort_values("day_num")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    x = summary.day_num.to_numpy(float)
    ax.errorbar(
        x, summary.SS_mean_mV_per_dec, yerr=summary.SS_SD_mV_per_dec,
        color="#1f77b4", ecolor="#555555", marker="o", markersize=6,
        linewidth=2.2, elinewidth=1.5, capsize=4,
        label=f"Mean ± SD ({int(summary.n_devices.iloc[0])} devices)",
    )
    ax.set_xticks(x)
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Subthreshold Slope (mV/dec)", fontsize=15,
                  fontweight="bold")
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.margins(x=0.05, y=0.16)
    ax.set_ylim(bottom=0)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{BASE_NAME}.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    table[["day_num", "day", "ESN", "TNumber", "device",
           "3sigma_onset_nA", "SS_region_mV_per_dec",
           "region_log_linear_R2", "region_lower_VJG_V",
           "region_upper_VJG_V"]].to_csv(
        OUTPUT / f"{BASE_NAME}_devices.csv", index=False)
    summary.to_csv(OUTPUT / f"{BASE_NAME}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Minimum region linearity R2: {table.region_log_linear_R2.min():.4f}")
    print(f"Output: {OUTPUT / (BASE_NAME + '.png')}")


if __name__ == "__main__":
    main()
