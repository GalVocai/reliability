"""Audit a fixed-current JG SS window for the selected Hot storage devices.

Run with: PYTHONPATH=src python scripts/export_hot_storage_ss_window.py
The current-window audit never modifies analysis or degradation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensorlab.compare import active_degradation_root
from sensorlab.degradation import day_to_num
from sensorlab.io import read_run
from sensorlab.jg_calibration import JgCalibration, apply_jg_calibration
from sensorlab.measurements.jg import _clean_curve, back_gate_groups, relevant_part
from sensorlab.plotting.overview_plot import _infer_selected_devices


DATA_ROOT = Path(__file__).resolve().parents[2] / "reliability"
OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_60-130nA_Vbg0"
LOW_NA, HIGH_NA = 60.0, 130.0
MIN_POINTS = 3
R2_REVIEW = 0.95
EXAMPLE = ("1216", "11")


def _selected_devices() -> tuple[set[tuple[str, str]], Path]:
    table_root = active_degradation_root(DATA_ROOT) / "Hot storage" / "JG_n03"
    tidy = pd.read_excel(table_root / "degradation.xlsx")
    aggregate = pd.read_excel(table_root / "aggregate.xlsx")
    selected = _infer_selected_devices(
        tidy, aggregate, ("Vth_bg0", "Vth_bg4", "Ion_bg0", "Ion_bg4")
    )
    return selected, table_root


def _active_timepoints() -> list[tuple[float, str, Path, Path]]:
    for manifest in sorted(DATA_ROOT.glob(".sensorlab-analysis-batch-*.json"), reverse=True):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        entries = info.get("timepoints", {})
        if (info.get("version") == 1 and len(entries) == 7
                and all((DATA_ROOT / rel / name).is_dir() for rel, name in entries.items())):
            out = []
            for rel, name in entries.items():
                day = rel.split("/")[0]
                out.append((day_to_num(day), day, DATA_ROOT / rel, DATA_ROOT / rel / name))
            return sorted(out)
    raise FileNotFoundError("No complete active seven-timepoint analysis batch found")


def _fit(vj: np.ndarray, current: np.ndarray) -> dict:
    mask = (np.isfinite(vj) & np.isfinite(current)
            & (np.abs(current) >= LOW_NA) & (np.abs(current) <= HIGH_NA))
    n = int(mask.sum())
    result = {"n_points": n, "slope_dec_per_V": np.nan,
              "SS_mV_per_dec": np.nan, "R2": np.nan,
              "negative_current_points": int((current[mask] < 0).sum()),
              "fit_status": "too_few_points"}
    if n < MIN_POINTS or np.ptp(vj[mask]) <= 0:
        return result
    x = vj[mask]
    y = np.log10(np.abs(current[mask]))
    if np.ptp(y) <= 0:
        result["fit_status"] = "no_current_variation"
        return result
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(1.0 - np.sum((y - slope * x - intercept) ** 2)
               / np.sum((y - y.mean()) ** 2))
    result["slope_dec_per_V"] = float(slope)
    result["R2"] = r2
    if slope <= 0:
        result["fit_status"] = "nonpositive_slope"
    elif r2 < R2_REVIEW:
        result["fit_status"] = "review_low_R2"
    else:
        result["SS_mV_per_dec"] = float(1000.0 / slope)
        result["fit_status"] = "ok"
    return result


def _plot_counts(table: pd.DataFrame, devices: list[tuple[str, str]], days: list[float]) -> None:
    matrix = np.array([
        [int(table.loc[(table.ESN == esn) & (table.TNumber == tnum)
                       & (table.day_num == day), "n_points"].iloc[0]) for day in days]
        for esn, tnum in devices
    ])
    labels = [f"ESN{esn} D{tnum[0]}T{tnum[1]}" for esn, tnum in devices]
    fig, ax = plt.subplots(figsize=(9.6, 7.4), layout="constrained")
    shown = np.minimum(matrix, 4)
    image = ax.imshow(shown, cmap="YlGnBu", vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(range(len(days)), ["0D" if d == 0 else f"{int(d)}D" for d in days])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Stress time (days)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Selected Hot storage devices", fontsize=13, fontweight="bold")
    ax.set_title("Measured points in the 60–130 nA window (Vbg = 0 V)",
                 fontsize=14, fontweight="bold", pad=18)
    for row in range(len(devices)):
        for col in range(len(days)):
            count = matrix[row, col]
            ax.text(col, row, str(count), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if shown[row, col] >= 3 else "#182333")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(devices), fill=False,
                               edgecolor="#d62728", linewidth=2.4))
    cb = fig.colorbar(image, ax=ax, shrink=0.8)
    cb.set_label("Independent measured points (4 means 4 or more)")
    fig.text(0.50, -0.015,
             "At 0D, all 13 devices have fewer than 3 points; baseline SS and ΔSS are unavailable.",
             ha="center", color="#b22222", fontsize=11, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"SS_60-130nA_Vbg0_point_count_QC.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_example(curves: dict[tuple[float, tuple[str, str]], tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6), layout="constrained")
    for day, color, label in ((0.0, "#1f77b4", "0D"), (28.0, "#d62728", "28D")):
        vj, current = curves[(day, EXAMPLE)]
        magnitude = np.abs(current)
        window = np.isfinite(magnitude) & (magnitude >= LOW_NA) & (magnitude <= HIGH_NA)
        ax.semilogy(vj, magnitude, color=color, linewidth=2,
                    label=f"{label} sweep ({int(window.sum())} point in window)")
        ax.scatter(vj[window], magnitude[window], s=74, zorder=5,
                   facecolors="white", edgecolors=color, linewidths=2)
    ax.axhspan(LOW_NA, HIGH_NA, color="#ffcc66", alpha=0.28,
               label="60–130 nA fit window")
    ax.axhline(150, color="#777777", linestyle=":", linewidth=1.4,
               label="150 nA Vth criterion")
    ax.set_xlim(-0.85, 0.05)
    ax.set_ylim(8, 1000)
    ax.set_xlabel(r"$V_{JG}$ (V)", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$|I_D|$ (nA)", fontsize=14, fontweight="bold")
    ax.set_title("ESN1216 D1T1: corrected JG sweeps at Vbg = 0 V",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.tick_params(direction="in", labelsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"ESN1216_D1T1_0D_28D_SS_window_example.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, selection_source = _selected_devices()
    timepoints = _active_timepoints()
    rows: list[dict] = []
    curves: dict[tuple[float, tuple[str, str]], tuple[np.ndarray, np.ndarray]] = {}
    for day_num, day_label, raw_folder, analysis_folder in timepoints:
        files = sorted(raw_folder.glob("*_n03_JG_.xlsx"))
        if len(files) != 1:
            raise ValueError(f"Expected one JG_n03 workbook in {raw_folder}; found {len(files)}")
        calibration_path = analysis_folder / "jg_calibration_used.json"
        run = apply_jg_calibration(read_run(str(files[0])),
                                   JgCalibration.from_json(calibration_path))
        sweep = dict(back_gate_groups(run.parts()[relevant_part(run)]))[0]
        found = set()
        for channel in run.channels:
            key = (channel.esn, channel.tnumber)
            if key not in selected:
                continue
            found.add(key)
            vj, current = _clean_curve(sweep["Vj"].to_numpy(float),
                                       sweep[channel.column].to_numpy(float))
            curves[(day_num, key)] = (vj, current)
            rows.append({"day": day_label, "day_num": day_num,
                         "ESN": channel.esn, "TNumber": channel.tnumber,
                         "device": channel.label, "Vbg_V": 0,
                         **_fit(vj, current), "raw_workbook": str(files[0]),
                         "calibration_json": str(calibration_path)})
        if found != selected:
            raise ValueError(f"Missing selected devices at {day_label}: {sorted(selected - found)}")
    table = pd.DataFrame(rows).sort_values(["day_num", "ESN", "TNumber"])
    table.to_csv(OUTPUT / "SS_60-130nA_Vbg0_device_fit_QC.csv", index=False)
    summary = table.groupby(["day_num", "day"], as_index=False).agg(
        devices=("device", "size"),
        min_points=("n_points", "min"), median_points=("n_points", "median"),
        max_points=("n_points", "max"),
        with_at_least_3_points=("n_points", lambda s: int((s >= 3).sum())),
        valid_fits=("fit_status", lambda s: int((s == "ok").sum())),
    ).sort_values("day_num")
    summary.to_csv(OUTPUT / "SS_60-130nA_Vbg0_checkpoint_QC.csv", index=False)
    _plot_counts(table, sorted(selected), list(summary.day_num))
    _plot_example(curves)
    baseline_valid = table[(table.day_num == 0) & (table.fit_status == "ok")]
    (OUTPUT / "README.txt").write_text(
        "Hot storage JG_n03, Vbg=0 V, 13 devices from the existing degradation selection.\n"
        "Currents are corrected using each active analysis batch calibration JSON.\n"
        "For every last-repetition sweep, take measured points with 60 <= |Id| <= 130 nA.\n"
        "Fit log10(|Id|) = m*VJG + b only when at least 3 independent points exist.\n"
        "A positive slope and R2 >= 0.95 are used as a QC pass; lower R2 requires review.\n"
        f"Valid 0D baseline fits: {len(baseline_valid)}/{len(selected)}.\n"
        "With no valid 0D fits, device-level delta SS and mean +/- SD are undefined.\n"
        "No delta SS curve was drawn because it would imply unsupported estimates.\n"
        f"Selection source: {selection_source}\n",
        encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Selected devices: {len(selected)}; valid baseline fits: {len(baseline_valid)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
