"""Cross-test publication plots built from degradation tables.

The Results Outline must not depend on manually exported, potentially stale
images.  This module reads the population aggregates already produced by the
degradation study and regenerates the regular/normalised comparison figures.
The selected device population is therefore inherited from the degradation
calculation rather than recomputed from raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .style import load_style


_CONDITION_ORDER = ("hot_storage", "85c_85rh", "cycles")
_CONDITION_STYLE = {
    "hot_storage": ("125 °C Hot Storage", "#111111", "o"),
    "85c_85rh": ("85 °C/85% RH", "#e41a1c", "s"),
    "cycles": ("Cycles", "#1f77b4", "^"),
}
_SENSITIVITY_METRICS = tuple((float(ppm), f"{ppm}ppm_delta") for ppm in (10, 25, 50, 100))


@dataclass(frozen=True)
class _MetricSource:
    key: str
    mode: str
    metric: str
    mean_column: str = "value_mean"
    avdev_column: str = "value_avdev"
    vbg: float | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class _OutputPlot:
    data_key: str
    relative_path: str
    ylabel: str
    normalized: bool = False
    zero_line: bool = False
    headroom: float = 0.14


_METRIC_SOURCES = (
    _MetricSource(
        "stress_delta_vth", "JG_n03", "Vth_bg4",
        mean_column="delta_mean", avdev_column="delta_avdev", scale=-1000.0,
    ),
    _MetricSource("h2_delta_vth", "JG_H2", "delta_Vth_H2", vbg=4.0, scale=1000.0),
    _MetricSource("maximum_h2_response", "JG_H2", "max_resp_H2_pct", vbg=4.0),
    _MetricSource("recovery_delta_vth", "JG_H2", "delta_Vth_recovery", vbg=4.0, scale=1000.0),
    _MetricSource("h2_response_time", "SE_n04", "t90_response_s"),
)

_ENDPOINT_METRICS = (
    ("stress_delta_vth", "Stress-induced delta Vth", "mV", "signed_shift"),
    ("h2_delta_vth", "H2-induced delta Vth", "mV", "higher_is_better"),
    ("maximum_h2_response", "H2 response", "%", "higher_is_better"),
    ("recovery_delta_vth", "Recovery delta Vth", "mV", "signed_shift"),
    ("sensitivity", "Sensitivity", "nA/ppm", "higher_is_better"),
    ("h2_response_time", "H2 response time t90", "s", "lower_is_better"),
)

_OUTPUT_PLOTS = (
    _OutputPlot(
        "stress_delta_vth", "regular/stress_induced_deltaVTh.png",
        r"Stress Induced $\mathbf{\Delta V_{Th}}$ (mV)",
    ),
    _OutputPlot(
        "h2_delta_vth", "regular/H2_induced_deltaVTh.png",
        r"$\mathbf{H_2}$-induced $\mathbf{|\Delta V_{Th}|}$ (mV)",
    ),
    _OutputPlot(
        "h2_delta_vth", "normalized/normalized_H2_induced_deltaVTh.png",
        r"Normalized $\mathbf{H_2}$ Induced $\mathbf{\Delta V_{Th}}$", normalized=True,
    ),
    _OutputPlot(
        "maximum_h2_response", "regular/maximum_H2_response_vs_stress_time.png",
        r"$\mathbf{H_2}$ Response (%)",
    ),
    _OutputPlot(
        "maximum_h2_response", "normalized/normalized_maximum_H2_response_vs_stress_time.png",
        r"Normalized $\mathbf{H_2}$ Response", normalized=True,
    ),
    _OutputPlot(
        "recovery_delta_vth", "regular/recovery_deltaVTh_vs_stress_time.png",
        r"Recovery $\mathbf{\Delta V_{Th}}$ (mV)", zero_line=True,
    ),
    _OutputPlot(
        "recovery_delta_vth", "normalized/normalized_recovery_deltaVTh_vs_stress_time.png",
        r"Normalized Recovery $\mathbf{\Delta V_{Th}}$", normalized=True, zero_line=True,
    ),
    _OutputPlot(
        "sensitivity", "regular/sensitivity_vs_stress_time.png",
        r"Sensitivity (nA/ppm)",
    ),
    _OutputPlot(
        "sensitivity", "normalized/normalized_sensitivity_vs_stress_time.png",
        r"Normalized Sensitivity", normalized=True,
    ),
    _OutputPlot(
        "h2_response_time", "regular/H2_response_time_vs_stress_time.png",
        r"$\mathbf{H_2}$ Response Time, $\mathbf{t_{90}}$ (s)",
    ),
    _OutputPlot(
        "h2_response_time", "normalized/normalized_H2_response_time_vs_stress_time.png",
        r"Normalized $\mathbf{H_2}$ Response Time", normalized=True, headroom=0.30,
    ),
)


def _publication_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.name.lower() in {"regular", "normalized"} and root.parent.name.lower() == "publication_plots":
        return root.parent
    if root.name.lower() == "publication_plots":
        return root
    return root / "publication_plots"


def _read_table(condition_dir: Path, mode: str, filename: str) -> pd.DataFrame:
    path = condition_dir / mode / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot regenerate Results overview; degradation table is missing: {path}"
        )
    sheet = "aggregate" if filename == "aggregate.xlsx" else "degradation"
    return pd.read_excel(path, sheet_name=sheet)


def _series_from_aggregate(condition_dir: Path, source: _MetricSource) -> pd.DataFrame:
    table = _read_table(condition_dir, source.mode, "aggregate.xlsx")
    required = {"metric", "day_num", source.mean_column, source.avdev_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{condition_dir / source.mode / 'aggregate.xlsx'} is missing columns: "
            + ", ".join(sorted(missing))
        )
    rows = table[table["metric"].astype(str) == source.metric].copy()
    if source.vbg is not None:
        if "Vbg" not in rows.columns:
            raise ValueError(f"Metric {source.metric} has no Vbg column in {condition_dir}.")
        rows = rows[np.isclose(pd.to_numeric(rows["Vbg"], errors="coerce"), source.vbg)]
    if rows.empty:
        suffix = f" at Vbg={source.vbg:g} V" if source.vbg is not None else ""
        raise ValueError(f"Metric {source.metric}{suffix} is missing under {condition_dir}.")
    if rows["day_num"].duplicated().any():
        raise ValueError(f"Metric {source.metric} has duplicate stress times under {condition_dir}.")
    result = pd.DataFrame({
        "day_num": pd.to_numeric(rows["day_num"], errors="coerce"),
        "value_mean": pd.to_numeric(rows[source.mean_column], errors="coerce") * source.scale,
        "value_avdev": pd.to_numeric(rows[source.avdev_column], errors="coerce") * abs(source.scale),
        "n": pd.to_numeric(rows["n"], errors="coerce") if "n" in rows else np.nan,
    }).dropna(subset=["day_num", "value_mean", "value_avdev"])
    if result.empty:
        raise ValueError(f"Metric {source.metric} has no finite values under {condition_dir}.")
    return result.sort_values("day_num").reset_index(drop=True)


def _identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _infer_selected_devices(tidy: pd.DataFrame, aggregate: pd.DataFrame,
                            metrics: tuple[str, ...]) -> set[tuple[str, str]]:
    """Recover the population already used for an aggregate table.

    Older degradation folders did not persist their checkbox selection.  The
    aggregate means and counts nevertheless define it: find the binary subset
    of devices whose per-device values reproduce those means.  New reports can
    therefore preserve the original population instead of silently averaging
    every device present in the tidy workbook.
    """

    for column in ("ESN", "TNumber", "metric", "day_num", "value"):
        if column not in tidy:
            raise ValueError(f"STEPS degradation table is missing column {column!r}.")
    for column in ("metric", "day_num", "value_mean", "n"):
        if column not in aggregate:
            raise ValueError(f"STEPS aggregate table is missing column {column!r}.")

    work = tidy[tidy["metric"].astype(str).isin(metrics)].copy()
    work["_esn"] = work["ESN"].map(_identifier)
    work["_tnum"] = work["TNumber"].map(_identifier)
    devices = sorted(set(zip(work["_esn"], work["_tnum"])))
    targets = aggregate[aggregate["metric"].astype(str).isin(metrics)].copy()
    targets = targets.dropna(subset=["day_num", "value_mean", "n"])
    if not devices or targets.empty:
        raise ValueError("Cannot recover the selected STEPS population from the degradation tables.")

    selected_count = int(pd.to_numeric(targets["n"], errors="coerce").max())
    if selected_count <= 0 or selected_count > len(devices):
        raise ValueError("The STEPS aggregate contains an invalid population count.")
    if selected_count == len(devices):
        return set(devices)

    rows: list[np.ndarray] = [np.ones(len(devices), dtype=float)]
    target_values: list[float] = [float(selected_count)]
    for target in targets.itertuples(index=False):
        metric = str(target.metric)
        day = float(target.day_num)
        sample = work[
            (work["metric"].astype(str) == metric)
            & np.isclose(pd.to_numeric(work["day_num"], errors="coerce"), day)
        ]
        values = sample.groupby(["_esn", "_tnum"])["value"].mean()
        vector = np.asarray([
            pd.to_numeric(values.get(device, np.nan), errors="coerce") for device in devices
        ], dtype=float)
        available = np.isfinite(vector).astype(float)
        row_n = int(target.n)
        rows.append(available)
        target_values.append(float(row_n))
        rows.append(np.nan_to_num(vector, nan=0.0))
        target_values.append(float(target.value_mean) * row_n)

    matrix = np.vstack(rows)
    expected = np.asarray(target_values, dtype=float)
    scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
    scaled_matrix = matrix / scale[:, None]
    scaled_expected = expected / scale

    # SciPy is already a SensorLab dependency.  Import MILP lazily so opening
    # the GUI does not pay its startup cost; it is needed only for legacy
    # studies that predate persisted population metadata.
    from scipy.optimize import Bounds, LinearConstraint, milp

    result = milp(
        c=np.zeros(len(devices)),
        integrality=np.ones(len(devices)),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            scaled_matrix, scaled_expected - 1e-8, scaled_expected + 1e-8
        ),
        options={"time_limit": 20.0},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "Could not recover the selected STEPS devices from this legacy degradation study. "
            "Run Degradation once with the current SensorLab version and rebuild the report."
        )
    chosen_vector = np.rint(result.x)
    residual = np.max(np.abs(matrix @ chosen_vector - expected))
    tolerance = 1e-5 * max(1.0, float(np.max(np.abs(expected))))
    if residual > tolerance or int(chosen_vector.sum()) != selected_count:
        raise ValueError("Recovered STEPS population does not reproduce the saved aggregate.")
    return {device for device, chosen in zip(devices, chosen_vector) if chosen > 0.5}


def _sensitivity_series(condition_dir: Path) -> pd.DataFrame:
    tidy = _read_table(condition_dir, "STEPS_n07", "degradation.xlsx")
    aggregate = _read_table(condition_dir, "STEPS_n07", "aggregate.xlsx")
    metric_names = tuple(metric for _ppm, metric in _SENSITIVITY_METRICS)
    selected = _infer_selected_devices(tidy, aggregate, metric_names)

    work = tidy[tidy["metric"].astype(str).isin(metric_names)].copy()
    work["_esn"] = work["ESN"].map(_identifier)
    work["_tnum"] = work["TNumber"].map(_identifier)
    work = work[[device in selected for device in zip(work["_esn"], work["_tnum"])]]
    ppm_by_metric = {metric: ppm for ppm, metric in _SENSITIVITY_METRICS}
    work["ppm"] = work["metric"].astype(str).map(ppm_by_metric)

    slopes: list[dict[str, float]] = []
    for (_esn, _tnum, day), group in work.groupby(["_esn", "_tnum", "day_num"]):
        group = group.dropna(subset=["value", "ppm"])
        if group["ppm"].nunique() >= 2:
            slope = float(np.polyfit(
                group["ppm"].to_numpy(float), group["value"].to_numpy(float), 1
            )[0])
            slopes.append({"day_num": float(day), "value": slope})
    frame = pd.DataFrame(slopes)
    if frame.empty:
        raise ValueError(f"No 10/25/50/100 ppm sensitivity values were found under {condition_dir}.")

    def avdev(values: pd.Series) -> float:
        return float((values - values.mean()).abs().mean())

    return (
        frame.groupby("day_num")["value"]
        .agg(value_mean="mean", value_avdev=avdev, n="count")
        .reset_index()
        .sort_values("day_num")
        .reset_index(drop=True)
    )


def _collect_overview_data(
    condition_directories: Mapping[str, str | Path],
) -> dict[str, dict[str, pd.DataFrame]]:
    missing = [key for key in _CONDITION_ORDER if key not in condition_directories]
    if missing:
        raise ValueError("Missing reliability conditions for overview: " + ", ".join(missing))
    data: dict[str, dict[str, pd.DataFrame]] = {
        source.key: {} for source in _METRIC_SOURCES
    }
    data["sensitivity"] = {}
    for condition in _CONDITION_ORDER:
        directory = Path(condition_directories[condition]).expanduser().resolve()
        for source in _METRIC_SOURCES:
            data[source.key][condition] = _series_from_aggregate(directory, source)
        data["sensitivity"][condition] = _sensitivity_series(directory)
    return data


def _save_figure_pair(fig, png_path: Path, dpi: float) -> None:
    try:
        for target in (png_path, png_path.with_suffix(".pdf")):
            for attempt in range(5):
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(target, dpi=dpi)
                    break
                except FileNotFoundError:
                    if attempt == 4:
                        raise
                    time.sleep(0.25)
    finally:
        plt.close(fig)


def summarize_overview_endpoints(
    data: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Last plotted population mean versus the same reference used by plots.

    This is a ratio of aggregate means, not a mean of per-device percentage
    changes. Each metric/condition uses its own last available checkpoint.
    Signed voltage shifts have no scalar performance-loss interpretation;
    their absolute change is reported even when the reference is zero.
    """
    rows = []
    for condition in _CONDITION_ORDER:
        for key, label, unit, direction in _ENDPOINT_METRICS:
            frame = data[key][condition].sort_values("day_num")
            if frame.empty or frame["day_num"].duplicated().any():
                raise ValueError(f"Invalid overview checkpoints for {condition}/{key}.")
            values = frame[["day_num", "value_mean", "value_avdev"]].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite overview values for {condition}/{key}.")
            first, last = frame.iloc[0], frame.iloc[-1]
            baseline, final = float(first.value_mean), float(last.value_mean)
            delta = final - baseline
            ratio = final / baseline if abs(baseline) >= 1e-15 else None
            change_pct = 100.0 * (ratio - 1.0) if ratio is not None else None
            loss = None
            if len(frame) < 2:
                interpretation = "Only one checkpoint; change over time is unavailable."
                ratio = change_pct = None
            elif direction == "higher_is_better" and baseline > 0:
                loss = -change_pct
                interpretation = (f"{loss:.1f}% decrease" if loss >= 0
                                  else f"{-loss:.1f}% increase")
            elif direction == "lower_is_better" and baseline > 0:
                interpretation = (f"{change_pct:.1f}% longer response time" if change_pct >= 0
                                  else f"{-change_pct:.1f}% shorter response time")
            elif direction == "signed_shift":
                interpretation = f"Signed shift change: {delta:+.3g} {unit}"
            else:
                interpretation = "No performance-loss percentage for a nonpositive reference."
            source = next((s for s in _METRIC_SOURCES if s.key == key), None)
            rows.append({
                "condition": condition, "condition_label": _CONDITION_STYLE[condition][0],
                "metric": key, "metric_label": label, "unit": unit, "direction": direction,
                "vbg_v": 4.0 if key == "stress_delta_vth" else (source.vbg if source else None),
                "baseline_day": float(first.day_num), "last_day": float(last.day_num),
                "baseline_value": baseline, "baseline_avdev": float(first.value_avdev),
                "baseline_n": float(first.get("n", np.nan)),
                "last_value": final, "last_avdev": float(last.value_avdev),
                "last_n": float(last.get("n", np.nan)),
                "absolute_change": delta, "normalized_final": ratio,
                "relative_change_pct": change_pct, "performance_loss_pct": loss,
                "interpretation": interpretation,
            })
    return pd.DataFrame(rows)


def _plot_comparison(series: Mapping[str, pd.DataFrame], output: Path,
                     spec: _OutputPlot) -> None:
    style = load_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    legend_handles: list[Line2D] = []
    all_x: list[float] = []
    for condition in _CONDITION_ORDER:
        label, color, marker = _CONDITION_STYLE[condition]
        frame = series[condition].sort_values("day_num")
        x = frame["day_num"].to_numpy(float)
        mean = frame["value_mean"].to_numpy(float)
        error = np.abs(frame["value_avdev"].to_numpy(float))
        if spec.normalized:
            baseline_index = int(np.argmin(x))
            baseline = float(mean[baseline_index])
            if not np.isfinite(baseline) or abs(baseline) < 1e-15:
                raise ValueError(
                    f"Cannot normalize {spec.data_key} for {label}; its baseline is zero or invalid."
                )
            mean = mean / baseline
            error = error / abs(baseline)
        ax.errorbar(
            x, mean, yerr=error, color=color, marker=marker,
            markersize=7.0, markeredgewidth=1.0,
            linestyle="-", linewidth=float(style.linewidth),
            elinewidth=1.4, capsize=4.0, zorder=2,
        )
        legend_handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle="-",
            linewidth=float(style.linewidth), markersize=7.0, label=label,
        ))
        all_x.extend(x.tolist())

    if spec.zero_line:
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlabel("Stress Time (days)")
    ax.set_ylabel(spec.ylabel)
    if all_x:
        low_x, high_x = min(all_x), max(all_x)
        pad = max(0.4, (high_x - low_x) * 0.015)
        ax.set_xlim(low_x - pad, high_x + pad)
    style.apply_publication(ax)
    style.apply_legend(ax, handles=legend_handles, loc="upper right")
    fig.tight_layout()
    # Preserve extra space above the data so an upper-right legend never
    # obscures the late-time series (notably normalized response time).
    low_y, high_y = ax.get_ylim()
    span = high_y - low_y
    if np.isfinite(span) and span > 0:
        ax.set_ylim(low_y, high_y + spec.headroom * span)
    _save_figure_pair(fig, output, float(style.dpi))


def generate_overview_plots(
    condition_directories: Mapping[str, str | Path],
    output_root: str | Path,
    *,
    data: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> tuple[Path, ...]:
    """Regenerate all cross-test regular/normalised publication plots.

    The returned paths are the PNG files.  A vector PDF is written beside each
    PNG.  Every series uses elapsed days, including the temperature-cycle
    study; its 21-day checkpoint therefore appears at x=21 (252 cycles at the
    standard two-hour cycle duration).
    """

    if data is None:
        data = _collect_overview_data(condition_directories)
    publication_root = _publication_root(output_root)
    written: list[Path] = []
    for spec in _OUTPUT_PLOTS:
        path = publication_root / spec.relative_path
        _plot_comparison(data[spec.data_key], path, spec)
        written.append(path)
    return tuple(written)
