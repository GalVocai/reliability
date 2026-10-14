"""JMP export: row identity, stacking, retention and workbook fidelity."""

from __future__ import annotations

import pandas as pd
import pytest

from sensorlab.jmp_export import (ExportSource, MetricOption, make_export_tables,
                                  write_export)


def _source() -> ExportSource:
    n03 = MetricOption("JG_n03", "Vth_bg0")
    h2 = MetricOption("JG_H2", "delta_Vth_H2", (("Vbg", "0"),))
    rows = []
    for esn, v0, vt, h0, ht in ((101, -0.5, -0.4, -0.2, 0.1),
                                (102, -0.6, -0.3, -0.3, -0.15)):
        for day, v, h in ((0, v0, h0), (7, vt, ht)):
            rows.append((esn, 11, "Hot storage", day, "Vth_bg0", v, v0))
    base = pd.DataFrame(rows, columns=["ESN", "TNumber", "condition", "day_num",
                                       "metric", "value", "baseline_value"])
    gas = base.drop(columns=["metric", "value", "baseline_value"]).copy()
    gas["Vbg"] = 0
    gas["metric"] = "delta_Vth_H2"
    gas["value"] = [-0.2, 0.1, -0.3, -0.15]
    gas["baseline_value"] = [-0.2, -0.2, -0.3, -0.3]
    return ExportSource({("Hot storage", "JG_n03"): base,
                         ("Hot storage", "JG_H2"): gas}, [n03, h2], 0)


def test_stacking_changes_only_order_and_h2_retention_uses_magnitude():
    source = _source()
    day = make_export_tables(source, source.options, stacking="day")
    device = make_export_tables(source, source.options, stacking="device")
    assert day.all_data[["DeviceID", "Day"]].values.tolist() == [
        ["ESN101_T11", 0], ["ESN102_T11", 0],
        ["ESN101_T11", 7], ["ESN102_T11", 7]]
    assert device.all_data[["DeviceID", "Day"]].values.tolist() == [
        ["ESN101_T11", 0], ["ESN101_T11", 7],
        ["ESN102_T11", 0], ["ESN102_T11", 7]]
    sorted_day = day.all_data.sort_values(["DeviceID", "Day"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_day, device.all_data)
    row = day.all_data.query("DeviceID == 'ESN101_T11' and Day == 7").iloc[0]
    assert row["JG_n03__Vth_bg0"] == pytest.approx(-0.4)
    assert row["JG_n03__Vth_bg0__Retention_pct"] == pytest.approx(80)
    assert row["JG_H2__delta_Vth_H2__Vbg0"] == pytest.approx(0.1)
    assert row["JG_H2__delta_Vth_H2__Vbg0__Retention_pct"] == pytest.approx(50)
    assert row["Vbg"] == 0
    assert row["H2_ppm"] == 100
    assert len(day.post_baseline) == 2
    assert set(day.checkpoint_summary.Measure) == {"RawValue", "Retention_pct"}


def test_missing_or_zero_reference_is_blank_and_duplicate_source_fails(tmp_path):
    source = _source()
    gas = source.tables[("Hot storage", "JG_H2")].copy()
    gas.loc[gas.ESN.eq(102), "baseline_value"] = 0
    source.tables[("Hot storage", "JG_H2")] = gas
    tables = make_export_tables(source, source.options)
    missing = tables.all_data.loc[tables.all_data.DeviceID.eq("ESN102_T11"),
                                  "JG_H2__delta_Vth_H2__Vbg0__Retention_pct"]
    assert missing.isna().all()
    files = write_export(tmp_path / "jmp.xlsx", tables)
    assert len(files) == 4 and all(path.exists() for path in files)
    reloaded = pd.read_excel(files[0], sheet_name="All_Data")
    assert reloaded["JG_n03__Vth_bg0"].iloc[0] == pytest.approx(-0.5)
    assert reloaded["JG_H2__delta_Vth_H2__Vbg0__Retention_pct"].isna().sum() == 2
    assert list(pd.ExcelFile(files[0]).sheet_names) == [
        "All_Data", "PostBaseline", "CheckpointSummary", "ColumnInfo"]
    duplicate = pd.concat([gas, gas.iloc[[0]]], ignore_index=True)
    source.tables[("Hot storage", "JG_H2")] = duplicate
    with pytest.raises(ValueError, match="Duplicate device/day/metric"):
        make_export_tables(source, source.options)
