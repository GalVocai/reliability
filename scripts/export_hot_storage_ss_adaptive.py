"""Fit Hot storage local SS above each sweep's own noise, through 140 nA.

This is an exploratory export. It reads raw JG_n03 sweeps and the active
calibration snapshot; analysis and degradation directories are not changed.
Run with: PYTHONPATH=src python scripts/export_hot_storage_ss_adaptive.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import _clean_curve, back_gate_groups, relevant_part

from export_hot_storage_ss_window import _active_timepoints, _selected_devices


OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_adaptive_Vbg0"
OFF_END_VJG = -1.8
NOISE_MULTIPLIER = 3.0
UPPER_NA = 140.0
ONSET_GUARD = 3
MIN_POINTS = 3
R2_PASS = 0.95


def fit_above_noise(vj: np.ndarray, current: np.ndarray) -> tuple[dict, np.ndarray]:
    """Find the sustained positive turn-on and fit its measured points."""
    result = {
        "off_median_nA": np.nan, "off_sigma_MAD_nA": np.nan,
        "noise_threshold_nA": np.nan, "onset_VJG_V": np.nan,
        "fit_VJG_min_V": np.nan, "fit_VJG_max_V": np.nan,
        "fit_current_min_nA": np.nan, "fit_current_max_nA": np.nan,
        "n_fit_points": 0, "slope_dec_per_V": np.nan,
        "intercept_log10_nA": np.nan, "R2": np.nan,
        "SS_fit_mV_per_dec": np.nan, "SS_used_mV_per_dec": np.nan,
        "fit_status": "no_sustained_onset",
    }
    empty = np.array([], dtype=int)
    off = current[np.isfinite(vj) & np.isfinite(current) & (vj <= OFF_END_VJG)]
    if off.size < 10:
        result["fit_status"] = "insufficient_off_samples"
        return result, empty
    median = float(np.median(off))
    sigma = float(1.4826 * np.median(np.abs(off - median)))
    threshold = float(max(0.0, median) + NOISE_MULTIPLIER * sigma)
    result.update(off_median_nA=median, off_sigma_MAD_nA=sigma,
                  noise_threshold_nA=threshold)
    if not np.isfinite(threshold) or threshold >= UPPER_NA:
        result["fit_status"] = "noise_at_or_above_140_nA"
        return result, empty

    onset = next((i for i in range(len(current) - ONSET_GUARD + 1)
                  if np.all(np.isfinite(current[i:i + ONSET_GUARD]))
                  and np.all(current[i:i + ONSET_GUARD] > threshold)), None)
    if onset is None:
        return result, empty
    result["onset_VJG_V"] = float(vj[onset])
    upper = next((i for i in range(onset, len(current) - ONSET_GUARD + 1)
                  if np.all(np.isfinite(current[i:i + ONSET_GUARD]))
                  and np.all(current[i:i + ONSET_GUARD] > UPPER_NA)), len(current))
    indices = np.arange(onset, upper)
    indices = indices[(np.isfinite(vj[indices]) & np.isfinite(current[indices])
                       & (current[indices] > threshold) & (current[indices] <= UPPER_NA))]
    result["n_fit_points"] = int(indices.size)
    if indices.size:
        result.update(
            fit_VJG_min_V=float(np.min(vj[indices])),
            fit_VJG_max_V=float(np.max(vj[indices])),
            fit_current_min_nA=float(np.min(current[indices])),
            fit_current_max_nA=float(np.max(current[indices])),
        )
    if indices.size < MIN_POINTS or np.ptp(vj[indices]) <= 0:
        result["fit_status"] = "too_few_points"
        return result, indices
    x = vj[indices]
    y = np.log10(current[indices])  # onset guarantees strictly positive current
    if np.ptp(y) <= 0:
        result["fit_status"] = "no_current_variation"
        return result, indices
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(1 - np.sum((y - (slope * x + intercept)) ** 2)
               / np.sum((y - y.mean()) ** 2))
    result.update(slope_dec_per_V=float(slope),
                  intercept_log10_nA=float(intercept), R2=r2)
    if slope <= 0:
        result["fit_status"] = "nonpositive_slope"
        return result, indices
    result["SS_fit_mV_per_dec"] = float(1000 / slope)
    if r2 < R2_PASS:
        result["fit_status"] = "review_low_R2"
        return result, indices
    result["SS_used_mV_per_dec"] = result["SS_fit_mV_per_dec"]
    result["fit_status"] = "ok"
    return result, indices


def _save(fig: plt.Figure, basename: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mean(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    x = summary["day_num"].to_numpy(float)
    y = summary["delta_SS_mean"].to_numpy(float)
    sd = summary["delta_SS_SD"].fillna(0).to_numpy(float)
    ax.errorbar(x, y, yerr=sd, color="#1f77b4", ecolor="#555555",
                linewidth=2.2, elinewidth=1.5, capsize=4, marker="o",
                markersize=6, label="QC-passing paired subset: mean ± SD")
    ax.axhline(0, color="#888888", linewidth=1, linestyle=":")
    span = max(float(np.nanmax(y + sd) - np.nanmin(y - sd)), 1.0)
    for row in summary.itertuples(index=False):
        ax.text(row.day_num, row.delta_SS_mean + (0 if pd.isna(row.delta_SS_SD)
                else row.delta_SS_SD) + span * 0.045,
                f"N={int(row.n_paired)}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["0" if day == 0 else str(int(day)) for day in x])
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\Delta SS$ (mV/dec)", fontsize=15, fontweight="bold")
    ax.set_title("Exploratory subset: 6 of 13 selected devices have a valid 0D fit",
                 fontsize=13, pad=14)
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.margins(x=0.05, y=0.25)
    _save(fig, "Hot_storage_Delta_SS_adaptive_Vbg0_mean_SD")


def _plot_devices(table: pd.DataFrame) -> None:
    paired = table[table["delta_SS_mV_per_dec"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    palette = plt.get_cmap("tab10")
    days = sorted(table["day_num"].unique())
    for k, ((esn, tnum), group) in enumerate(paired.groupby(["ESN", "TNumber"])):
        # Missing QC-failed days remain gaps, rather than a continuous line.
        values = group.set_index("day_num")["delta_SS_mV_per_dec"].reindex(days)
        ax.plot(days, values,
                marker="o", linewidth=1.8, markersize=5, color=palette(k),
                label=f"ESN{esn} D{tnum[0]}T{tnum[1]}")
    ax.axhline(0, color="#888888", linewidth=1, linestyle=":")
    ax.set_xticks(sorted(table["day_num"].unique()))
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"Device $\Delta SS$ (mV/dec)", fontsize=15, fontweight="bold")
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              fontsize=10, title="0D-valid subset (6/13)")
    _save(fig, "Hot_storage_Delta_SS_adaptive_Vbg0_devices")


def _plot_coverage(table: pd.DataFrame) -> None:
    devices = sorted(set(zip(table.ESN, table.TNumber)))
    days = sorted(table.day_num.unique())
    baseline_ok = {
        (row.ESN, row.TNumber): row.fit_status == "ok"
        for row in table[table.day_num.eq(0)].itertuples(index=False)
    }
    cells = np.zeros((len(devices), len(days)), dtype=int)
    counts = np.zeros_like(cells)
    for i, key in enumerate(devices):
        for j, day in enumerate(days):
            row = table[(table.ESN == key[0]) & (table.TNumber == key[1])
                        & table.day_num.eq(day)].iloc[0]
            counts[i, j] = int(row.n_fit_points)
            if row.fit_status == "ok":
                cells[i, j] = 0 if baseline_ok[key] else 1
            elif row.fit_status == "review_low_R2":
                cells[i, j] = 2
            else:
                cells[i, j] = 3
    colors = ["#2b8c68", "#79b9da", "#f2bb53", "#d97779"]
    fig, ax = plt.subplots(figsize=(10.2, 7.0), layout="constrained")
    ax.imshow(cells, cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5,
              aspect="auto")
    ax.set_xticks(range(len(days)), ["0D" if x == 0 else f"{int(x)}D" for x in days])
    ax.set_yticks(range(len(devices)),
                  [f"ESN{e} D{t[0]}T{t[1]}" for e, t in devices])
    ax.set_xlabel("Stress time (days)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Selected Hot storage devices", fontsize=13, fontweight="bold")
    ax.set_title("Which measurements enter the adaptive ΔSS graph (Vbg = 0 V)",
                 fontsize=14, fontweight="bold", pad=15)
    for i in range(len(devices)):
        for j in range(len(days)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#17212d")
    ax.legend(handles=[
        Patch(color=colors[0], label="Included in paired ΔSS"),
        Patch(color=colors[1], label="Current fit passes; 0D fit unavailable"),
        Patch(color=colors[2], label="R² below 0.95"),
        Patch(color=colors[3], label="Fewer than 3 fit points"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
        frameon=False, fontsize=10)
    fig.text(0.5, -0.055, "Numbers in cells = measured points used in the fit",
             ha="center", fontsize=10)
    _save(fig, "Hot_storage_SS_adaptive_Vbg0_inclusion_QC")


def _plot_baseline_fit_panels(
        table: pd.DataFrame,
        curves: dict[tuple[float, tuple[str, str]], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    baseline = table[table["day_num"] == 0].sort_values(["ESN", "TNumber"])
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), layout="constrained")
    for ax, row in zip(axes.flat, baseline.itertuples(index=False)):
        vj, current, indices = curves[(0.0, (row.ESN, row.TNumber))]
        positive = np.where(current > 0, current, np.nan)
        ax.semilogy(vj, positive, color="#767676", linewidth=1.2)
        if indices.size:
            ax.scatter(vj[indices], current[indices], color="#1f77b4", s=29,
                       zorder=4)
        if row.fit_status == "ok":
            xx = np.linspace(float(vj[indices].min()), float(vj[indices].max()), 40)
            yy = 10 ** (row.slope_dec_per_V * xx + row.intercept_log10_nA)
            ax.semilogy(xx, yy, color="#d62728", linewidth=2)
        ax.axhline(row.noise_threshold_nA, color="#2ca02c", linestyle="--",
                   linewidth=1)
        ax.axhline(UPPER_NA, color="#d62728", linestyle=":", linewidth=1)
        if indices.size:
            center = float(np.median(vj[indices]))
        else:
            rising = np.flatnonzero(current > row.noise_threshold_nA)
            center = float(vj[rising[0]]) if rising.size else float(np.median(vj))
        ax.set_xlim(center - 0.20, center + 0.20)
        ax.set_ylim(5, 300)
        ax.set_title(f"ESN{row.ESN} D{row.TNumber[0]}T{row.TNumber[1]}: "
                     f"N={row.n_fit_points}, "
                     f"R²={row.R2:.2f}" if pd.notna(row.R2)
                     else f"ESN{row.ESN} D{row.TNumber[0]}T{row.TNumber[1]}: "
                          f"N={row.n_fit_points}", fontsize=10,
                     color="black" if row.fit_status == "ok" else "#b22222")
        ax.tick_params(labelsize=8, direction="in")
    for ax in axes.flat[len(baseline):]:
        ax.axis("off")
    fig.supxlabel(r"$V_{JG}$ (V)", fontsize=14, fontweight="bold")
    fig.supylabel(r"Corrected $I_D$ (nA)", fontsize=14, fontweight="bold")
    fig.suptitle("0D local SS fits: blue points, red fit, green noise threshold",
                 fontsize=15, fontweight="bold")
    _save(fig, "Hot_storage_SS_adaptive_Vbg0_baseline_fit_QC")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, selection_source = _selected_devices()
    timepoints = _active_timepoints()
    rows: list[dict] = []
    curves: dict[tuple[float, tuple[str, str]], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for day_num, day_label, raw_folder, analysis_folder in timepoints:
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
            vj, current = _clean_curve(sweep["Vj"].to_numpy(float),
                                       sweep[channel.column].to_numpy(float))
            fit, indices = fit_above_noise(vj, current)
            curves[(day_num, key)] = (vj, current, indices)
            rows.append({"day": day_label, "day_num": day_num,
                         "ESN": channel.esn, "TNumber": channel.tnumber,
                         "device": channel.label, "Vbg_V": 0, **fit,
                         "raw_workbook": str(files[0]),
                         "calibration_json": str(analysis_folder / "jg_calibration_used.json")})
        if found != selected:
            raise ValueError(f"Missing devices at {day_label}: {sorted(selected - found)}")

    table = pd.DataFrame(rows).sort_values(["day_num", "ESN", "TNumber"])
    baseline_ss = table[table.day_num.eq(0)].set_index(["ESN", "TNumber"])["SS_used_mV_per_dec"]
    table["SS_0D_mV_per_dec"] = [baseline_ss.loc[(e, t)]
                                  for e, t in zip(table.ESN, table.TNumber)]
    table["delta_SS_mV_per_dec"] = table["SS_used_mV_per_dec"] - table["SS_0D_mV_per_dec"]
    table.to_csv(OUTPUT / "Hot_storage_SS_adaptive_Vbg0_device_fits.csv", index=False)

    summary = table.groupby(["day_num", "day"], as_index=False).agg(
        n_selected=("device", "size"),
        n_with_3_points=("n_fit_points", lambda s: int((s >= MIN_POINTS).sum())),
        n_fit_pass=("fit_status", lambda s: int((s == "ok").sum())),
        n_paired=("delta_SS_mV_per_dec", "count"),
        delta_SS_mean=("delta_SS_mV_per_dec", "mean"),
        delta_SS_SD=("delta_SS_mV_per_dec", "std"),
    ).sort_values("day_num")
    summary.to_csv(OUTPUT / "Hot_storage_SS_adaptive_Vbg0_checkpoint_summary.csv", index=False)
    _plot_mean(summary)
    _plot_devices(table)
    _plot_coverage(table)
    _plot_baseline_fit_panels(table, curves)

    valid_baseline = table[(table.day_num.eq(0)) & (table.fit_status.eq("ok"))]
    common = [key for key in selected if
              table[(table.ESN == key[0]) & (table.TNumber == key[1])]
              ["fit_status"].eq("ok").all()]
    (OUTPUT / "README.txt").write_text(
        "Hot storage JG_n03, last repetition, Vbg=0 V, selected devices.\n"
        "Input current was corrected with each active analysis batch calibration snapshot.\n"
        f"Off-state noise region: VJG <= {OFF_END_VJG:g} V.\n"
        "Noise center = median(Id); robust sigma = 1.4826 * median(|Id - median|).\n"
        f"Onset threshold = max(0, noise center) + {NOISE_MULTIPLIER:g} * robust sigma.\n"
        f"Onset needs {ONSET_GUARD} consecutive measured points above threshold.\n"
        f"Fit positive Id points above threshold through {UPPER_NA:g} nA, "
        "ending before the sustained upper crossing.\n"
        f"Fit log10(Id) = m*VJG + b; SS = 1000/m. QC: >= {MIN_POINTS} points, "
        f"m > 0, R2 >= {R2_PASS:g}.\n"
        "Delta SS is paired to the same device's 0D SS, then mean +/- sample SD.\n"
        f"Selected devices: {len(selected)}; valid 0D fits: {len(valid_baseline)}.\n"
        f"Valid across all seven checkpoints: {len(common)} device(s): {sorted(common)}.\n"
        "The plotted population varies across days; N is printed at every point.\n"
        "The mean is an exploratory QC-passing subset, not an estimate for all 13 devices.\n"
        "Three-point fits have limited R2 degrees of freedom; treat this as exploratory.\n"
        f"Selection source: {selection_source}\n",
        encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Valid baseline fits: {len(valid_baseline)}/{len(selected)}; "
          f"complete-case devices: {len(common)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
