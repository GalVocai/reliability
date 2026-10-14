"""Model each full JG transfer sweep, then estimate local SS at 100 nA.

Exploratory, model-derived analysis for the selected Hot storage population.
Reads the active calibrated data; writes new files on Desktop only.
Run: PYTHONPATH=src python scripts/export_hot_storage_ss_curve_fit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import expit

from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import _clean_curve, back_gate_groups, relevant_part

from export_hot_storage_ss_window import _active_timepoints, _selected_devices


OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_full_sweep_Vbg0"
OFF_END_VJG = -1.8
EVAL_CURRENT_NA = 100.0
NEAR_MAX_NA = 500.0
QUADRATIC_MAX_NA = 1000.0
RELATIVE_FIT_WEIGHT = 0.08


def transfer_model(vjg: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Smooth off-to-on FET curve with exponential tail and power-law on-state."""
    offset, log_amplitude, v0, log_width, exponent = params
    width = np.exp(log_width)
    softplus = np.logaddexp(0.0, (vjg - v0) / width)
    return offset + np.exp(log_amplitude) * softplus ** exponent


def _noise(current: np.ndarray, vjg: np.ndarray) -> tuple[float, float, float]:
    off = current[np.isfinite(current) & np.isfinite(vjg) & (vjg <= OFF_END_VJG)]
    if off.size < 10:
        raise ValueError("Too few off-state samples for a noise estimate")
    center = float(np.median(off))
    sigma = float(max(1.4826 * np.median(np.abs(off - center)), 1.0))
    return center, sigma, float(max(0.0, center) + 3 * sigma)


def _sustained_vth(vjg: np.ndarray, current: np.ndarray) -> float:
    for i in range(1, len(current) - 2):
        if (current[i - 1] < 150
                and np.all(np.isfinite(current[i:i + 3]))
                and np.all(current[i:i + 3] >= 150)):
            return float(vjg[i])
    return float(vjg[np.nanargmin(np.abs(current - 150))])


def fit_full_curve(vjg: np.ndarray, current: np.ndarray) -> tuple[dict, np.ndarray]:
    center, sigma, threshold = _noise(current, vjg)
    weights = np.sqrt(sigma ** 2
                      + (RELATIVE_FIT_WEIGHT * np.clip(current, 0, None)) ** 2)
    vth = _sustained_vth(vjg, current)
    lower = np.array([center - 3 * sigma, np.log(0.01), -2.8,
                      np.log(0.005), 0.5])
    upper = np.array([center + 3 * sigma, np.log(1e7), 0.5,
                      np.log(1.0), 5.0])
    attempts = []
    for exponent in (1.5, 2.5):
        start = np.array([center, np.log(300.0), vth, np.log(0.06), exponent])
        result = least_squares(
            lambda params: (transfer_model(vjg, params) - current) / weights,
            start, bounds=(lower, upper), loss="soft_l1", max_nfev=3000,
        )
        attempts.append(result)
    best = min(attempts, key=lambda result: result.cost)
    params = best.x
    prediction = transfer_model(vjg, params)
    if not prediction[0] < EVAL_CURRENT_NA < prediction[-1]:
        raise ValueError("Fitted curve does not cross the evaluation current")
    v100 = brentq(
        lambda v: float(transfer_model(np.array([v]), params)[0] - EVAL_CURRENT_NA),
        float(vjg[0]), float(vjg[-1]),
    )
    offset, log_amplitude, v0, log_width, exponent = params
    width = np.exp(log_width)
    z = (v100 - v0) / width
    softplus = np.logaddexp(0.0, z)
    derivative_nA_per_V = (np.exp(log_amplitude) * exponent
                           * softplus ** (exponent - 1) * expit(z) / width)
    ss = float(1000 * np.log(10) * EVAL_CURRENT_NA / derivative_nA_per_V)

    near = ((current > threshold) & (current <= NEAR_MAX_NA)
            & np.isfinite(current) & np.isfinite(prediction))
    if near.sum() < 3:
        raise ValueError("Insufficient measured points for near-threshold QC")
    near_weights = np.sqrt(sigma ** 2
                           + (RELATIVE_FIT_WEIGHT * current[near]) ** 2)
    normalized_rmse = float(np.sqrt(np.mean(
        ((prediction[near] - current[near]) / near_weights) ** 2)))
    asinh_measured = np.arcsinh(current / sigma)
    asinh_predicted = np.arcsinh(prediction / sigma)
    asinh_r2 = float(1 - np.sum((asinh_measured - asinh_predicted) ** 2)
                     / np.sum((asinh_measured - asinh_measured.mean()) ** 2))
    at_bound = bool(np.any(np.isclose(params, lower, rtol=0, atol=1e-3))
                    or np.any(np.isclose(params, upper, rtol=0, atol=1e-3)))
    status = ("ok" if best.success and normalized_rmse <= 2 and not at_bound
              and derivative_nA_per_V > 0 else "review")
    details = {
        "off_median_nA": center, "off_sigma_MAD_nA": sigma,
        "3sigma_onset_nA": threshold, "n_near_threshold": int(near.sum()),
        "near_threshold_normalized_RMSE": normalized_rmse,
        "R2_asinh_full_curve": asinh_r2,
        "fit_offset_nA": float(offset),
        "fit_amplitude_nA": float(np.exp(log_amplitude)),
        "fit_V0_V": float(v0), "fit_width_V": float(width),
        "fit_exponent": float(exponent), "VJG_at_100nA_V": float(v100),
        "SS_model_100nA_mV_per_dec": ss,
        "fit_status": status,
    }
    return details, prediction


def fit_quadratic_turnon(vjg: np.ndarray, current: np.ndarray,
                         threshold: float) -> dict:
    """Independent empirical sensitivity check around the turn-on region."""
    use = (np.isfinite(vjg) & np.isfinite(current)
           & (current > threshold) & (current <= QUADRATIC_MAX_NA))
    x = vjg[use]
    y = np.log10(current[use])
    result = {"quad_n_points": int(x.size), "quad_R2_log": np.nan,
              "SS_quadratic_100nA_mV_per_dec": np.nan}
    if x.size < 5 or np.ptp(x) <= 0:
        return result
    x_center = float(x.mean())
    coeff = np.polyfit(x - x_center, y, 2)
    fitted = np.polyval(coeff, x - x_center)
    result["quad_R2_log"] = float(1 - np.sum((y - fitted) ** 2)
                                   / np.sum((y - y.mean()) ** 2))
    roots = np.roots([coeff[0], coeff[1], coeff[2] - np.log10(EVAL_CURRENT_NA)])
    candidates = [float(root.real) for root in roots if abs(root.imag) < 1e-8
                  and x.min() - 0.1 <= root.real + x_center <= x.max() + 0.1
                  and coeff[1] + 2 * coeff[0] * root.real > 0]
    if candidates:
        root = min(candidates, key=abs)
        result["SS_quadratic_100nA_mV_per_dec"] = float(
            1000 / (coeff[1] + 2 * coeff[0] * root))
    return result


def _save(fig: plt.Figure, basename: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_average(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    x = summary.day_num.to_numpy(float)
    ax.errorbar(x, summary.delta_model_mean, yerr=summary.delta_model_SD,
                color="#1f77b4", ecolor="#555555", marker="o", markersize=6,
                linewidth=2.2, elinewidth=1.5, capsize=4,
                label="Full-curve model: mean ± SD (13 devices)")
    ax.plot(x, summary.delta_quadratic_mean, color="#d62728", linestyle="--",
            marker="s", markersize=5, linewidth=1.5,
            label="Quadratic turn-on fit: model check")
    ax.axhline(0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"Model-derived $\Delta SS_{100\,\mathrm{nA}}$ (mV/dec)",
                  fontsize=15, fontweight="bold")
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.margins(x=0.05, y=0.18)
    _save(fig, "Hot_storage_Delta_SS_full_sweep_Vbg0_mean_SD")


def _plot_absolute(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    x = summary.day_num.to_numpy(float)
    ax.errorbar(x, summary.SS_model_mean, yerr=summary.SS_model_SD,
                color="#1f77b4", ecolor="#555555", marker="o", markersize=6,
                linewidth=2.2, elinewidth=1.5, capsize=4,
                label="Full-curve model: mean ± SD (13 devices)")
    ax.plot(x, summary.SS_quadratic_mean, color="#d62728", linestyle="--",
            marker="s", markersize=5, linewidth=1.5,
            label="Quadratic turn-on fit: model check")
    ax.set_xticks(x)
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"Model-derived $SS_{100\,\mathrm{nA}}$ (mV/dec)",
                  fontsize=15, fontweight="bold")
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.margins(x=0.05, y=0.16)
    ax.set_ylim(bottom=0)
    _save(fig, "Hot_storage_SS_full_sweep_Vbg0_absolute_mean_SD")


def _plot_devices(table: pd.DataFrame) -> None:
    days = sorted(table.day_num.unique())
    devices = sorted(set(zip(table.ESN, table.TNumber)))
    matrix = np.array([
        [float(table[(table.ESN == esn) & (table.TNumber == tnum)
                     & table.day_num.eq(day)].delta_SS_model_mV_per_dec.iloc[0])
         for day in days] for esn, tnum in devices
    ])
    extent = float(np.nanmax(np.abs(matrix[:, 1:])))
    fig, ax = plt.subplots(figsize=(10.0, 7.0), layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-extent, vmax=extent,
                      aspect="auto")
    ax.set_xticks(range(len(days)), ["0D" if day == 0 else f"{int(day)}D" for day in days])
    ax.set_yticks(range(len(devices)),
                  [f"ESN{esn} D{tnum[0]}T{tnum[1]}" for esn, tnum in devices])
    ax.set_xlabel("Stress time (days)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Selected Hot storage devices", fontsize=13, fontweight="bold")
    ax.set_title("Full-curve model: paired device ΔSS at 100 nA (Vbg = 0 V)",
                 fontsize=14, fontweight="bold", pad=16)
    for i in range(len(devices)):
        for j in range(len(days)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(value) > extent * 0.58 else "#17212d")
    bar = fig.colorbar(image, ax=ax, shrink=0.78)
    bar.set_label("ΔSS (mV/dec)")
    _save(fig, "Hot_storage_Delta_SS_full_sweep_Vbg0_devices")


def _plot_examples(curves: dict, table: pd.DataFrame) -> None:
    examples = [("1216", "11"), ("1286", "11")]
    days = [0.0, 28.0]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), layout="constrained")
    for row, key in enumerate(examples):
        for col, day in enumerate(days):
            ax = axes[row, col]
            vjg, current, fitted = curves[(day, key)]
            entry = table[(table.ESN == key[0]) & (table.TNumber == key[1])
                          & table.day_num.eq(day)].iloc[0]
            ax.semilogy(vjg, np.where(current > 0, current, np.nan),
                        "o", color="#777777", markersize=3, label="Measured")
            ax.semilogy(vjg, np.where(fitted > 0, fitted, np.nan),
                        color="#1f77b4", linewidth=2, label="Full-curve fit")
            ax.axhline(EVAL_CURRENT_NA, color="#d62728", linestyle=":",
                       linewidth=1.4)
            ax.axhline(entry["3sigma_onset_nA"], color="#2ca02c",
                       linestyle="--", linewidth=1)
            center = float(entry.VJG_at_100nA_V)
            ax.set_xlim(center - 0.25, center + 0.25)
            ax.set_ylim(5, 1200)
            ax.set_title(f"ESN{key[0]} D{key[1][0]}T{key[1][1]}, "
                         f"{int(day)}D: SS={entry.SS_model_100nA_mV_per_dec:.0f} mV/dec",
                         fontsize=11)
            ax.tick_params(direction="in", labelsize=9)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.supxlabel(r"$V_{JG}$ (V)", fontsize=14, fontweight="bold")
    fig.supylabel(r"Corrected $I_D$ (nA)", fontsize=14, fontweight="bold")
    _save(fig, "Hot_storage_SS_full_sweep_Vbg0_fit_examples")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, selection_source = _selected_devices()
    rows: list[dict] = []
    curves: dict = {}
    for day_num, day_label, raw_folder, analysis_folder in _active_timepoints():
        files = sorted(raw_folder.glob("*_n03_JG_.xlsx"))
        if len(files) != 1:
            raise ValueError(f"Expected one JG_n03 workbook in {raw_folder}")
        calibration = JgCalibration.from_json(analysis_folder / "jg_calibration_used.json")
        run = apply_jg_calibration(read_run(str(files[0])), calibration)
        sweep = dict(back_gate_groups(run.parts()[relevant_part(run)]))[0]
        found: set[tuple[str, str]] = set()
        for channel in run.channels:
            key = (channel.esn, channel.tnumber)
            if key not in selected:
                continue
            found.add(key)
            vjg, current = _clean_curve(sweep.Vj.to_numpy(float),
                                        sweep[channel.column].to_numpy(float))
            full, fitted = fit_full_curve(vjg, current)
            quad = fit_quadratic_turnon(vjg, current, full["3sigma_onset_nA"])
            curves[(day_num, key)] = (vjg, current, fitted)
            rows.append({"day": day_label, "day_num": day_num,
                         "ESN": channel.esn, "TNumber": channel.tnumber,
                         "device": channel.label, "Vbg_V": 0,
                         **full, **quad,
                         "raw_workbook": str(files[0]),
                         "calibration_json": str(analysis_folder / "jg_calibration_used.json")})
        if found != selected:
            raise ValueError(f"Missing selected devices at {day_label}: {sorted(selected - found)}")
    table = pd.DataFrame(rows).sort_values(["day_num", "ESN", "TNumber"])
    base_model = table[table.day_num.eq(0)].set_index(["ESN", "TNumber"])
    table["SS_0D_model_mV_per_dec"] = [base_model.loc[(e, t), "SS_model_100nA_mV_per_dec"]
                                       for e, t in zip(table.ESN, table.TNumber)]
    table["SS_0D_quadratic_mV_per_dec"] = [
        base_model.loc[(e, t), "SS_quadratic_100nA_mV_per_dec"]
        for e, t in zip(table.ESN, table.TNumber)]
    table["delta_SS_model_mV_per_dec"] = (table.SS_model_100nA_mV_per_dec
                                           - table.SS_0D_model_mV_per_dec)
    table["delta_SS_quadratic_mV_per_dec"] = (table.SS_quadratic_100nA_mV_per_dec
                                               - table.SS_0D_quadratic_mV_per_dec)
    table.to_csv(OUTPUT / "Hot_storage_SS_full_sweep_Vbg0_device_fits.csv", index=False)
    summary = table.groupby(["day_num", "day"], as_index=False).agg(
        n_selected=("device", "size"),
        n_model_QC_pass=("fit_status", lambda s: int((s == "ok").sum())),
        SS_model_mean=("SS_model_100nA_mV_per_dec", "mean"),
        SS_model_SD=("SS_model_100nA_mV_per_dec", "std"),
        SS_quadratic_mean=("SS_quadratic_100nA_mV_per_dec", "mean"),
        SS_quadratic_SD=("SS_quadratic_100nA_mV_per_dec", "std"),
        delta_model_mean=("delta_SS_model_mV_per_dec", "mean"),
        delta_model_SD=("delta_SS_model_mV_per_dec", "std"),
        delta_quadratic_mean=("delta_SS_quadratic_mV_per_dec", "mean"),
        delta_quadratic_SD=("delta_SS_quadratic_mV_per_dec", "std"),
        n_quad_R2_below_0_95=("quad_R2_log", lambda s: int((s < 0.95).sum())),
    ).sort_values("day_num")
    summary.to_csv(OUTPUT / "Hot_storage_SS_full_sweep_Vbg0_checkpoint_summary.csv", index=False)
    _plot_average(summary)
    _plot_absolute(summary)
    _plot_devices(table)
    _plot_examples(curves, table)
    (OUTPUT / "README.txt").write_text(
        "Hot storage JG_n03, last repetition, Vbg=0 V.\n"
        f"Population: all {len(selected)} devices selected in degradation.\n"
        "Every sweep is fitted separately after the active JG current correction.\n"
        "Full curve model: I(V) = Ioff + A*[ln(1 + exp((V - V0)/s))]^p.\n"
        "The model has an exponential tail and power-law high-current region.\n"
        "Weighted robust fit: residual divided by sqrt(noise_sigma^2 + (0.08*Ipositive)^2).\n"
        "Noise_sigma = 1.4826*MAD of corrected Id for VJG <= -1.8 V.\n"
        "Local SS = 1000/(dlog10(I_model)/dVJG), evaluated where I_model=100 nA.\n"
        "Full-model QC uses normalized RMSE of measured points from the 3-sigma "
        "noise threshold to 500 nA; <=2 passes.\n"
        "An independent quadratic fit to log10(Id) from noise threshold to 1000 nA "
        "is plotted as a model sensitivity check.\n"
        "This is MODEL-DERIVED local SS, not the original 60-130 nA linear-regression SS.\n"
        "Model choice can change individual SS estimates. Interpret trends with this limit.\n"
        f"Selection source: {selection_source}\n",
        encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
