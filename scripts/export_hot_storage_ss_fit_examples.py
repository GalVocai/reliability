"""Plot measured JG sweeps and the stored full-curve SS model fits."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import _clean_curve, back_gate_groups, relevant_part

from export_hot_storage_ss_curve_fit import transfer_model


OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_full_sweep_Vbg0"
FIT_TABLE = OUTPUT / "Hot_storage_SS_full_sweep_Vbg0_device_fits.csv"
REGION_TABLE = OUTPUT / "Hot_storage_SS_full_curve_region_3sigma_to_150nA_Vbg0_devices.csv"


def _model_params(row: pd.Series) -> np.ndarray:
    return np.array([
        float(row.fit_offset_nA), np.log(float(row.fit_amplitude_nA)),
        float(row.fit_V0_V), np.log(float(row.fit_width_V)),
        float(row.fit_exponent),
    ])


def _model_crossing(params: np.ndarray, current_na: float) -> float:
    return float(brentq(
        lambda v: float(transfer_model(np.array([v]), params)[0] - current_na),
        -5.0, 2.0,
    ))


def _load_sweep(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source = read_run(str(row.raw_workbook))
    calibration = JgCalibration.from_json(str(row.calibration_json))
    corrected = apply_jg_calibration(source, calibration)
    channel = next(ch for ch in source.channels
                   if ch.esn == row.ESN and ch.tnumber == row.TNumber)
    raw_group = dict(back_gate_groups(source.parts()[relevant_part(source)]))[0]
    corrected_group = dict(back_gate_groups(
        corrected.parts()[relevant_part(corrected)]))[0]
    raw_v, raw_i = _clean_curve(raw_group.Vj.to_numpy(float),
                                raw_group[channel.column].to_numpy(float))
    vjg, measured_i = _clean_curve(corrected_group.Vj.to_numpy(float),
                                   corrected_group[channel.column].to_numpy(float))
    if not np.array_equal(raw_v, vjg):
        raise ValueError("Raw and corrected sweep voltages differ")
    blank_offset = calibration.offset_for(row.ESN, row.TNumber, 0)
    return vjg, raw_i, measured_i, blank_offset


def _plot_case(ax_full: plt.Axes, ax_near: plt.Axes,
               row: pd.Series, region: pd.Series) -> None:
    vjg, raw_i, measured_i, blank_offset = _load_sweep(row)
    params = _model_params(row)
    smooth_v = np.linspace(float(vjg.min()), float(vjg.max()), 900)
    fit_i = transfer_model(smooth_v, params)
    day_label = "0D" if float(row.day_num) == 0 else f"{int(row.day_num)}D"
    title = f"ESN{row.ESN} D{row.TNumber[0]}T{row.TNumber[1]} · {day_label}"

    ax_full.plot(vjg, raw_i, "o", color="#8b8b8b", markersize=2.5,
                 label="Raw measured")
    ax_full.plot(vjg, measured_i, "o", color="#171717", markersize=2.2,
                 label="Blank-current corrected")
    ax_full.plot(smooth_v, fit_i + blank_offset, color="#8b8b8b",
                 linestyle="--", linewidth=1.5, label="Fit in raw coordinates")
    ax_full.plot(smooth_v, fit_i, color="#1976b8", linewidth=2,
                 label="Fit to corrected current")
    ax_full.set_yscale("symlog", linthresh=20)
    ax_full.set_xlim(float(vjg.min()), float(vjg.max()))
    ax_full.set_title(f"{title} · blank offset {blank_offset:+.1f} nA",
                      fontsize=11, fontweight="bold")
    ax_full.set_xlabel(r"$V_{JG}$ (V)")
    ax_full.set_ylabel(r"$I_D$ (nA), symmetric log")
    ax_full.grid(alpha=0.18)
    ax_full.legend(frameon=False, fontsize=8, loc="upper left")

    low_current = float(row["3sigma_onset_nA"])
    high_current = 150.0
    low_v = float(region.region_lower_VJG_V)
    high_v = float(region.region_upper_VJG_V)
    n_measured_in_interval = int(np.count_nonzero(
        (measured_i >= low_current) & (measured_i <= high_current)))
    zoom_max_v = _model_crossing(params, 600.0)
    ax_near.semilogy(vjg[measured_i > 0], measured_i[measured_i > 0],
                     "o", color="#171717", markersize=3,
                     label="Corrected measured")
    ax_near.semilogy(smooth_v[fit_i > 0], fit_i[fit_i > 0],
                     color="#1976b8", linewidth=2, label="Full-curve fit")
    interval_v = np.linspace(low_v, high_v, 201)
    interval_i = transfer_model(interval_v, params)
    slope, intercept = np.polyfit(interval_v, np.log10(interval_i), 1)
    ax_near.semilogy(interval_v, 10 ** (slope * interval_v + intercept),
                     color="#d07d24", linestyle="--", linewidth=2.2,
                     label="SS log-linear fit")
    ax_near.axvspan(low_v, high_v, color="#63ad70", alpha=0.15,
                    label="SS extraction interval")
    ax_near.axhline(low_current, color="#4a9557", linestyle=":", linewidth=1)
    ax_near.axhline(high_current, color="#d25a32", linestyle=":", linewidth=1)
    ax_near.set_xlim(low_v - 0.12, zoom_max_v + 0.06)
    ax_near.set_ylim(max(1, low_current / 3), 900)
    ax_near.set_title(
        f"{title} · SS={float(region.SS_region_mV_per_dec):.1f} mV/dec "
        f"({n_measured_in_interval} measured points in interval)",
        fontsize=11, fontweight="bold",
    )
    ax_near.set_xlabel(r"$V_{JG}$ (V)")
    ax_near.set_ylabel(r"Corrected $I_D$ (nA), log scale")
    ax_near.grid(alpha=0.18, which="both")
    ax_near.legend(frameon=False, fontsize=8, loc="lower right")
    print(f"{title}: {n_measured_in_interval} measured points in "
          f"{low_current:.1f}–150 nA interval")


def main() -> None:
    fits = pd.read_csv(FIT_TABLE, dtype={"ESN": str, "TNumber": str})
    regions = pd.read_csv(REGION_TABLE, dtype={"ESN": str, "TNumber": str})
    for esn, tnumber in (("1216", "11"), ("1286", "31")):
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8),
                                 layout="constrained")
        for plot_row, day in enumerate((0.0, 28.0)):
            fit = fits[(fits.ESN == esn) & (fits.TNumber == tnumber)
                       & fits.day_num.eq(day)].iloc[0]
            region = regions[(regions.ESN == esn) &
                             (regions.TNumber == tnumber) &
                             regions.day_num.eq(day)].iloc[0]
            _plot_case(axes[plot_row, 0], axes[plot_row, 1], fit, region)
        basename = f"Hot_storage_SS_full_curve_fit_ESN{esn}_D{tnumber[0]}T{tnumber[1]}_0D_28D"
        for extension in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{basename}.{extension}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(OUTPUT / f"{basename}.png")


if __name__ == "__main__":
    main()
