"""Prepare the Hot-storage Vbg=0 correlation data for JMP.

The numeric inputs are the SensorLab degradation workbooks, not plotted points.
The old correlation CSV is used only to recover the exact selected devices.
The companion .mjs file writes the Excel workbook with typed numeric cells.

Run from anywhere with the SensorLab Python environment:
    python scripts/prepare_jmp_retention_data_vbg0.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "reliability"
HOT = ROOT / "degradation" / "Hot storage"
DESKTOP = Path.home() / "Desktop"
STEM = "JMP_retention_data_Vbg0"
SELECTED_CSV = HOT / "correlation" / "Vth_vs_H2_sensing_retention_Vbg0_paired_devices.csv"
DAY_SET = (0, 2, 7, 14, 20, 25, 28)
H2_PPM = 100
VBG = 0

DATA_COLUMNS = [
    "ChipID", "DeviceID", "Day", "Vbg", "H2_ppm", "AirVth_n03",
    "AirVth_H2", "H2Vth", "DeltaVth_H2", "BaseRetention",
    "H2Retention", "AirVth_Difference",
]
SUMMARY_COLUMNS = [
    "Day", "ValidPairs", "BaseRetention_Mean", "BaseRetention_SD",
    "H2Retention_Mean", "H2Retention_SD",
]


def _id(value: object) -> str:
    """Use the same integer-like ESN/TNumber canonicalization as the plot."""
    if pd.isna(value):
        raise ValueError("Missing ESN or TNumber in a selected source row")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Non-integer ESN/TNumber: {value!r}")
    return str(int(numeric))


def _selection() -> set[tuple[str, str]]:
    previous = pd.read_csv(SELECTED_CSV)
    selected = {tuple(ast.literal_eval(raw)) for raw in previous["device"].unique()}
    if any(len(key) != 2 for key in selected):
        raise ValueError("Invalid device key in the correlation selection")
    return selected


def _metric(source: pd.DataFrame, metric: str, selected: set[tuple[str, str]],
            value_name: str, *, vbg: int | None = None) -> pd.DataFrame:
    frame = source.loc[source["metric"].eq(metric)].copy()
    if vbg is not None:
        frame = frame.loc[frame["Vbg"].eq(vbg)].copy()
    frame["ESN_key"] = frame["ESN"].map(_id)
    frame["TNumber_key"] = frame["TNumber"].map(_id)
    frame = frame.loc[[key in selected for key in zip(frame.ESN_key, frame.TNumber_key)]].copy()
    wrong_condition = frame.loc[~frame["condition"].eq("Hot storage")]
    if len(wrong_condition):
        raise ValueError(f"{metric}: selected rows include a non-Hot-storage condition")
    frame["Day"] = pd.to_numeric(frame["day_num"], errors="raise")
    if frame["Day"].isna().any() or not (frame["Day"] % 1).eq(0).all():
        raise ValueError(f"{metric}: missing/noninteger Day")
    frame["Day"] = frame["Day"].astype(int)
    frame["ChipID"] = "ESN" + frame["ESN_key"]
    frame["DeviceID"] = frame["ChipID"] + "_T" + frame["TNumber_key"]
    duplicates = frame.loc[frame.duplicated(["DeviceID", "Day"], keep=False),
                           ["DeviceID", "Day"]]
    if len(duplicates):
        raise ValueError(f"{metric}: duplicate selected DeviceID/Day keys:\n"
                         + duplicates.to_string(index=False))
    # Keep the original Excel numeric value; no rounding or absolute value here.
    frame[value_name] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["value"].notna().sum() != frame[value_name].notna().sum():
        raise ValueError(f"{metric}: a nonnumeric source value was found")
    keep = ["ChipID", "DeviceID", "Day", value_name]
    if "degradation_pct" in frame:
        keep.append("degradation_pct")
    return frame[keep].rename(columns={"degradation_pct": f"source_deg_{value_name}"})


def _r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _quality_and_summary(data: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    n_devices = data.DeviceID.nunique()
    n_chips = data.ChipID.nunique()
    observed_days = sorted(data.Day.unique().tolist())
    notes.append(f"Selected population: {n_devices} devices on {n_chips} chips.")
    notes.append(f"Observed days: {observed_days}.")
    notes.append(f"All_Data rows: {len(data)}; PostBaseline rows: {int(data.Day.gt(0).sum())}.")
    notes.append(f"Expected comparison: 13 devices, 4 chips, days {list(DAY_SET)}, 91/78 rows.")
    if (n_devices, n_chips, len(data), int(data.Day.gt(0).sum())) != (13, 4, 91, 78):
        notes.append("DEVIATION: observed population/row counts differ from expected.")
    if observed_days != list(DAY_SET):
        notes.append("DEVIATION: observed checkpoint days differ from expected.")
    duplicates = int(data.duplicated(["DeviceID", "Day"]).sum())
    notes.append(f"Duplicate DeviceID+Day rows: {duplicates}.")
    counts = data.groupby("DeviceID").Day.nunique()
    incomplete = counts.loc[counts.ne(len(DAY_SET))]
    notes.append(f"Devices missing one or more checkpoint rows: {len(incomplete)}.")
    if len(incomplete):
        notes.append("Affected DeviceIDs: " + ", ".join(incomplete.index.tolist()))

    for column in DATA_COLUMNS:
        missing = int(data[column].isna().sum())
        notes.append(f"Missing {column}: {missing}.")
    reference = data.loc[data.Day.eq(0), ["DeviceID", "AirVth_n03", "DeltaVth_H2"]]
    notes.append(f"0D reference rows: {len(reference)}; missing AirVth_n03 references: "
                 f"{int(reference.AirVth_n03.isna().sum())}; zero AirVth_n03 denominators: "
                 f"{int(reference.AirVth_n03.eq(0).sum())}.")
    notes.append(f"Missing 0D H2 response references: {int(reference.DeltaVth_H2.isna().sum())}; "
                 f"zero H2 response denominators: {int(reference.DeltaVth_H2.eq(0).sum())}.")
    max_diff = data.AirVth_Difference.abs().max()
    notes.append(f"Maximum |AirVth_n03 - AirVth_H2|: {max_diff:.17g} V.")
    notes.append("Both air VTH columns are from the same n03 air sweep, reused by JG_H2.")
    notes.append("DeltaVth_H2 = H2Vth - AirVth_H2; negative values are preserved.")
    notes.append("H2_ppm=100 follows the documented JG-H2 exposure condition; the "
                 "degradation tables do not contain a measured ppm column.")

    comparable = data.loc[data.Day.gt(0) & data.BaseRetention.notna()
                          & data.H2Retention.notna()].copy()
    grouped = comparable.groupby("Day")
    summary = grouped.agg(
        ValidPairs=("DeviceID", "size"),
        BaseRetention_Mean=("BaseRetention", "mean"),
        BaseRetention_SD=("BaseRetention", lambda s: s.std(ddof=1)),
        H2Retention_Mean=("H2Retention", "mean"),
        H2Retention_SD=("H2Retention", lambda s: s.std(ddof=1)),
    ).reset_index().reindex(columns=SUMMARY_COLUMNS)
    # Every summary row uses the same complete device pairs for X and Y.
    if len(summary) >= 3:
        trend = _r(summary.BaseRetention_Mean.to_numpy(float),
                   summary.H2Retention_Mean.to_numpy(float))
    else:
        trend = float("nan")
    if len(comparable) >= 3:
        xc = comparable.BaseRetention - comparable.groupby("Day").BaseRetention.transform("mean")
        yc = comparable.H2Retention - comparable.groupby("Day").H2Retention.transform("mean")
        within = _r(xc.to_numpy(float), yc.to_numpy(float))
    else:
        within = float("nan")
    notes.append(f"Pearson r across post-0D checkpoint means: {trend:.12f} (prior plot about 0.94).")
    notes.append(f"Pearson r pooled after day centering: {within:.12f} (prior plot about 0.52).")
    if not (np.isfinite(trend) and abs(trend - 0.94) < 0.02
            and np.isfinite(within) and abs(within - 0.52) < 0.02):
        notes.append("DEVIATION: correlation check differs from the previous plot.")

    old = old.copy()
    old["DeviceID"] = old.device.map(lambda s: "ESN" + ast.literal_eval(s)[0]
                                     + "_T" + ast.literal_eval(s)[1])
    check = data.merge(old[["DeviceID", "day_num", "x_retention_pct", "y_retention_pct"]],
                       left_on=["DeviceID", "Day"], right_on=["DeviceID", "day_num"],
                       how="left", validate="one_to_one")
    both = check[["BaseRetention", "H2Retention", "x_retention_pct",
                  "y_retention_pct"]].notna().all(axis=1)
    notes.append(f"Rows comparable to prior plot CSV: {int(both.sum())}/{len(data)}.")
    if both.any():
        bx = float((check.loc[both, "BaseRetention"]
                    - check.loc[both, "x_retention_pct"]).abs().max())
        hy = float((check.loc[both, "H2Retention"]
                    - check.loc[both, "y_retention_pct"]).abs().max())
        notes.append(f"Maximum absolute retention difference vs prior plot: "
                     f"Base={bx:.17g} pp, H2={hy:.17g} pp.")
    omitted = data.loc[data.Day.eq(0)]
    notes.append(f"PostBaseline excludes exactly {len(omitted)} 0D reference rows.")
    notes.append("No Day>0 row is omitted from PostBaseline; missing values stay blank.")
    notes.append("CheckpointSummary excludes only rows without a valid pair of retentions.")
    for day in DAY_SET[1:]:
        day_rows = data.loc[data.Day.eq(day)]
        valid = day_rows.BaseRetention.notna() & day_rows.H2Retention.notna()
        notes.append(f"Day {day}: {len(day_rows)} observed rows, {int(valid.sum())} valid pairs, "
                     f"{int((~valid).sum())} rows omitted from summary.")
    return summary, notes


def _column_info() -> pd.DataFrame:
    entries = [
        ("All_Data / PostBaseline", "ChipID", "Chip identifier from ESN.", "identifier",
         "JG_n03/JG_H2 degradation.xlsx: ESN; prefixed with ESN.", "Nominal"),
        ("All_Data / PostBaseline", "DeviceID", "Stable transistor identifier containing ChipID and TNumber.", "identifier",
         "ESN{ESN}_T{TNumber}; same identifier at every checkpoint.", "Nominal"),
        ("All_Data / PostBaseline", "Day", "Elapsed stress time, stored as a number.", "days",
         "day_num in degradation.xlsx; use as a categorical checkpoint in adjusted models.",
         "Nominal (numeric storage)"),
        ("All_Data / PostBaseline", "Vbg", "Back-gate voltage; Vth_bg0 and Vbg=0 H2 rows.", "V",
         "Fixed at 0 for this export.", "Continuous"),
        ("All_Data / PostBaseline", "H2_ppm", "H2 exposure concentration for the JG-H2 sweep.", "ppm",
         "100 ppm from experiment figures/README.md and SensorLab H2 plot label; "
         "not recorded as a numeric column in degradation.xlsx.", "Continuous"),
        ("All_Data / PostBaseline", "AirVth_n03", "Signed air VTH used for baseline analysis.", "V",
         "Hot storage/JG_n03/degradation.xlsx, metric Vth_bg0, value.", "Continuous"),
        ("All_Data / PostBaseline", "AirVth_H2", "Signed air VTH reused in H2 comparison.", "V",
         "Hot storage/JG_H2/degradation.xlsx, metric Vth_air, Vbg=0, value. "
         "The code reuses the same n03 air sweep; equality was checked row by row.", "Continuous"),
        ("All_Data / PostBaseline", "H2Vth", "Signed VTH during H2 exposure.", "V",
         "Hot storage/JG_H2/degradation.xlsx, metric Vth_H2, Vbg=0, value.", "Continuous"),
        ("All_Data / PostBaseline", "DeltaVth_H2", "Signed H2-induced threshold change.", "V",
         "H2Vth - AirVth_H2. This is the negative of source delta_Vth_H2 "
         "(source uses air minus H2); no absolute value applied.", "Continuous"),
        ("All_Data / PostBaseline", "BaseRetention", "Baseline VTH retention from the prior correlation plot.", "% (numeric)",
         "100 * AirVth_n03(t) / AirVth_n03(0), using signed voltages for the "
         "same DeviceID/Vbg/H2_ppm. Equivalent to 100 - degradation_pct in "
         "SensorLab compare.py. Blank if the 0D denominator is missing or zero.", "Continuous"),
        ("All_Data / PostBaseline", "H2Retention", "Retention of the H2-induced threshold-shift magnitude.", "% (numeric)",
         "100 * abs(DeltaVth_H2(t)) / abs(DeltaVth_H2(0)) for the same "
         "DeviceID/Vbg/H2_ppm. Blank if the 0D denominator is missing or zero.", "Continuous"),
        ("All_Data / PostBaseline", "AirVth_Difference", "Air-threshold identity audit; expected zero.", "V",
         "AirVth_n03 - AirVth_H2; do not treat these as independent measurements.", "Continuous"),
        ("CheckpointSummary", "Day", "Post-baseline checkpoint.", "days",
         "Day>0 from PostBaseline.", "Nominal (numeric storage)"),
        ("CheckpointSummary", "ValidPairs", "Devices with both retention values present.", "devices",
         "Count of complete BaseRetention/H2Retention pairs in PostBaseline at this Day.", "Continuous"),
        ("CheckpointSummary", "BaseRetention_Mean", "Mean baseline retention for valid pairs.", "% (numeric)",
         "Arithmetic mean of BaseRetention over the same complete device pairs.", "Continuous"),
        ("CheckpointSummary", "BaseRetention_SD", "Sample SD of baseline retention for valid pairs.", "percentage points",
         "Sample standard deviation of BaseRetention, denominator n-1; blank if n<2.", "Continuous"),
        ("CheckpointSummary", "H2Retention_Mean", "Mean H2 response retention for valid pairs.", "% (numeric)",
         "Arithmetic mean of H2Retention over the same complete device pairs.", "Continuous"),
        ("CheckpointSummary", "H2Retention_SD", "Sample SD of H2 response retention for valid pairs.", "percentage points",
         "Sample standard deviation of H2Retention, denominator n-1; blank if n<2.", "Continuous"),
    ]
    return pd.DataFrame(entries, columns=["Sheet", "Column", "Description", "Units",
                                          "Formula_or_Source", "JMP_Modeling_Type"])


def _json_rows(frame: pd.DataFrame) -> list[list[object]]:
    rows = [list(frame.columns)]
    for values in frame.itertuples(index=False, name=None):
        cells: list[object] = []
        for value in values:
            if pd.isna(value):
                cells.append(None)
            elif isinstance(value, np.generic):
                cells.append(value.item())
            else:
                cells.append(value)
        rows.append(cells)
    return rows


def _export_workbook(sheets: dict[str, pd.DataFrame]) -> None:
    builder = Path(__file__).with_suffix(".mjs")
    if not builder.exists():
        raise FileNotFoundError(f"Missing companion workbook builder: {builder}")
    node = shutil.which("node")
    if not node:
        raise FileNotFoundError("Node.js is required for the companion .mjs builder")
    dependencies = (Path.home() / ".cache" / "codex-runtimes" /
                    "codex-primary-runtime" / "dependencies" / "node" /
                    "node_modules")
    if not dependencies.is_dir():
        raise FileNotFoundError(f"Bundled artifact-tool modules not found: {dependencies}")

    def ps_quote(value: Path) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    with tempfile.TemporaryDirectory(prefix="jmp_retention_") as tmp:
        temporary = Path(tmp).resolve()
        payload_path = temporary / "payload.json"
        payload = {name: _json_rows(frame) for name, frame in sheets.items()}
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8")
        runtime_builder = temporary / builder.name
        shutil.copy2(builder, runtime_builder)
        junction = temporary / "node_modules"
        create = (f"New-Item -ItemType Junction -Path {ps_quote(junction)} "
                  f"-Target {ps_quote(dependencies)} | Out-Null")
        subprocess.run(["powershell", "-NoProfile", "-Command", create], check=True)
        try:
            if not junction.is_junction():
                raise RuntimeError("Expected a temporary node_modules junction")
            env = dict(os.environ)
            env["JMP_PREVIEW_DIR"] = str(Path(__file__).resolve().parent /
                                         ".jmp_retention_previews")
            env["JMP_RENDER_PREVIEWS"] = "0"
            subprocess.run([node, str(runtime_builder), str(payload_path),
                            str(DESKTOP / f"{STEM}.xlsx")], check=True, env=env)
        finally:
            # Remove only the temporary junction, never the dependency target.
            if junction.is_junction() and junction.parent.resolve() == temporary:
                os.rmdir(junction)
    # The authoring engine can leave an inspect sidecar beside the xlsx.
    sidecar = DESKTOP / f"{STEM}.xlsx.inspect.ndjson"
    if sidecar.exists():
        sidecar.unlink()


def main() -> None:
    selected = _selection()
    n03_source = pd.read_excel(HOT / "JG_n03" / "degradation.xlsx",
                               sheet_name="degradation")
    h2_source = pd.read_excel(HOT / "JG_H2" / "degradation.xlsx",
                              sheet_name="degradation")
    n03 = _metric(n03_source, "Vth_bg0", selected, "AirVth_n03")
    air = _metric(h2_source, "Vth_air", selected, "AirVth_H2", vbg=VBG)
    gas = _metric(h2_source, "Vth_H2", selected, "H2Vth", vbg=VBG)
    source_delta = _metric(h2_source, "delta_Vth_H2", selected,
                           "source_delta_air_minus_h2", vbg=VBG)
    keys = ["ChipID", "DeviceID", "Day"]
    data = n03.merge(air, on=keys, how="outer", validate="one_to_one")
    data = data.merge(gas, on=keys, how="outer", validate="one_to_one")
    data = data.merge(source_delta, on=keys, how="outer", validate="one_to_one")
    data["Vbg"] = VBG
    data["H2_ppm"] = H2_PPM
    data["DeltaVth_H2"] = data.H2Vth - data.AirVth_H2
    data["AirVth_Difference"] = data.AirVth_n03 - data.AirVth_H2

    ref = data.loc[data.Day.eq(0), ["DeviceID", "Vbg", "H2_ppm",
                                      "AirVth_n03", "DeltaVth_H2"]].copy()
    if ref.duplicated(["DeviceID", "Vbg", "H2_ppm"]).any():
        raise ValueError("Duplicate 0D reference for a DeviceID/Vbg/H2_ppm")
    ref = ref.rename(columns={"AirVth_n03": "AirVth_n03_0D",
                              "DeltaVth_H2": "DeltaVth_H2_0D"})
    data = data.merge(ref, on=["DeviceID", "Vbg", "H2_ppm"], how="left",
                      validate="many_to_one")
    base_valid = data.AirVth_n03.notna() & data.AirVth_n03_0D.notna() & data.AirVth_n03_0D.ne(0)
    h2_valid = data.DeltaVth_H2.notna() & data.DeltaVth_H2_0D.notna() & data.DeltaVth_H2_0D.ne(0)
    data["BaseRetention"] = np.nan
    data.loc[base_valid, "BaseRetention"] = (
        100 * data.loc[base_valid, "AirVth_n03"] /
        data.loc[base_valid, "AirVth_n03_0D"])
    data["H2Retention"] = np.nan
    data.loc[h2_valid, "H2Retention"] = (
        100 * data.loc[h2_valid, "DeltaVth_H2"].abs() /
        data.loc[h2_valid, "DeltaVth_H2_0D"].abs())

    # Independent numerical checks against the saved source computations.
    delta_comparable = data.DeltaVth_H2.notna() & data.source_delta_air_minus_h2.notna()
    delta_error = ((data.loc[delta_comparable, "DeltaVth_H2"]
                    + data.loc[delta_comparable, "source_delta_air_minus_h2"]).abs())
    if len(delta_error) and delta_error.max() > 1e-12:
        raise ValueError("Signed H2 delta does not reproduce the source metric")
    base_comparable = data.BaseRetention.notna() & data.source_deg_AirVth_n03.notna()
    base_error = ((data.loc[base_comparable, "BaseRetention"]
                   - (100 - data.loc[base_comparable, "source_deg_AirVth_n03"])).abs())
    if len(base_error) and base_error.max() > 1e-9:
        raise ValueError("BaseRetention disagrees with the original degradation formula")
    h2_comparable = data.H2Retention.notna() & data.source_deg_source_delta_air_minus_h2.notna()
    h2_error = ((data.loc[h2_comparable, "H2Retention"]
                 - (100 - data.loc[h2_comparable,
                                   "source_deg_source_delta_air_minus_h2"])).abs())
    # A sign change in the source response would make the signed old-plot formula
    # differ from requested magnitude retention. Report it, never force equality.

    data = data.sort_values(["ChipID", "DeviceID", "Day"]).reset_index(drop=True)
    all_data = data[DATA_COLUMNS].copy()
    post = all_data.loc[all_data.Day.gt(0)].copy()
    old = pd.read_csv(SELECTED_CSV)
    summary, notes = _quality_and_summary(all_data, old)
    notes.append(f"Maximum |DeltaVth_H2 + source delta_Vth_H2|: "
                 f"{float(delta_error.max()) if len(delta_error) else float('nan'):.17g} V.")
    notes.append(f"Maximum |BaseRetention - (100 - source degradation_pct)|: "
                 f"{float(base_error.max()) if len(base_error) else float('nan'):.17g} pp.")
    notes.append(f"Maximum |H2Retention - (100 - source degradation_pct)|: "
                 f"{float(h2_error.max()) if len(h2_error) else float('nan'):.17g} pp.")
    if len(h2_error) and h2_error.max() > 1e-9:
        notes.append("The requested magnitude retention differs from the old signed-source "
                     "retention for one or more rows; no values were changed to match it.")

    DESKTOP.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(DESKTOP / f"{STEM}_All_Data.csv", index=False,
                    encoding="utf-8-sig", float_format="%.17g")
    post.to_csv(DESKTOP / f"{STEM}_PostBaseline.csv", index=False,
                encoding="utf-8-sig", float_format="%.17g")
    info = _column_info()
    _export_workbook({"All_Data": all_data, "PostBaseline": post,
                      "CheckpointSummary": summary, "ColumnInfo": info})
    report = DESKTOP / f"{STEM}_QC.txt"
    report.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"Workbook: {DESKTOP / (STEM + '.xlsx')}")
    print(f"Quality report: {report}")


if __name__ == "__main__":
    main()
