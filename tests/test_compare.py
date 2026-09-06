"""Unit tests for degradation-study orchestration helpers."""

import os

import pandas as pd

from sensorlab.compare import (_bg_groups, _sel_delta_items, _steps_ppm_groups,
                               _steps_sensitivity, discover_timepoints)


def test_sel_delta_items_honours_ppm_selection():
    ppm_groups = {"delta": [(10.0, "10ppm_delta"), (25.0, "25ppm_delta"),
                            (100.0, "100ppm_delta"), (1000.0, "1000ppm_delta")]}
    # no selection -> all exposure deltas (ppm > 0)
    assert _sel_delta_items(ppm_groups, None) == [
        (10.0, "10ppm_delta"), (25.0, "25ppm_delta"),
        (100.0, "100ppm_delta"), (1000.0, "1000ppm_delta")]
    # selection excludes 1000 -> stays in the linear regime
    assert _sel_delta_items(ppm_groups, {10, 25, 100}) == [
        (10.0, "10ppm_delta"), (25.0, "25ppm_delta"), (100.0, "100ppm_delta")]


def test_steps_sensitivity_is_the_linear_slope_nA_per_ppm():
    # one device, one day: delta = 5*ppm exactly -> slope 5 nA/ppm
    tidy = pd.DataFrame([
        {"ESN": "1", "TNumber": 11, "day_num": 0.0, "metric": "10ppm_delta", "value": 50.0},
        {"ESN": "1", "TNumber": 11, "day_num": 0.0, "metric": "50ppm_delta", "value": 250.0},
        {"ESN": "1", "TNumber": 11, "day_num": 0.0, "metric": "100ppm_delta", "value": 500.0},
    ])
    items = [(10.0, "10ppm_delta"), (50.0, "50ppm_delta"), (100.0, "100ppm_delta")]
    out = _steps_sensitivity(tidy, items)
    assert len(out) == 1
    assert out.iloc[0]["metric"] == "sensitivity_nA_per_ppm"
    assert abs(out.iloc[0]["value"] - 5.0) < 1e-9


def test_steps_sensitivity_honours_the_device_filter():
    from sensorlab.degradation import filter_devices
    tidy = pd.DataFrame([
        {"ESN": "1", "TNumber": "11", "day_num": 0.0, "metric": "10ppm_delta", "value": 50.0},
        {"ESN": "1", "TNumber": "11", "day_num": 0.0, "metric": "100ppm_delta", "value": 500.0},
        # a QC outlier device with a much steeper slope
        {"ESN": "2", "TNumber": "11", "day_num": 0.0, "metric": "10ppm_delta", "value": 500.0},
        {"ESN": "2", "TNumber": "11", "day_num": 0.0, "metric": "100ppm_delta", "value": 5000.0},
    ])
    items = [(10.0, "10ppm_delta"), (100.0, "100ppm_delta")]
    # keep only device 1 -> only its slope (5), the excluded device never counts
    out = _steps_sensitivity(filter_devices(tidy, {("1", "11")}), items)
    assert set(zip(out["ESN"].astype(str), out["TNumber"].astype(str))) == {("1", "11")}
    assert abs(out.iloc[0]["value"] - 5.0) < 1e-9


def test_steps_ppm_groups_by_metric_suffix():
    groups = _steps_ppm_groups([
        "10ppm_resp_pct",
        "25ppm_resp_pct",
        "100ppm_delta",
        "10ppm_delta",
        "LOD_ppm",
        "recovery_pct",
    ])

    assert groups["resp_pct"] == [(10.0, "10ppm_resp_pct"), (25.0, "25ppm_resp_pct")]
    assert groups["delta"] == [(10.0, "10ppm_delta"), (100.0, "100ppm_delta")]
    assert "LOD_ppm" not in groups


def test_bg_groups_by_metric_base_name():
    groups = _bg_groups([
        "Vth_bg0", "Vth_bg4",
        "Vth_d2_bg0", "Vth_d2_bg4",
        "Delta_Vth_bg0", "Delta_Vth_bg4",
        "Ion_Ioff_bg0", "Ion_Ioff_bg4",
        "Vth_air", "delta_Vth_H2",       # JG_H2: no _bg suffix, must not match
        "LOD_ppm",
    ])

    assert groups["Vth"] == [(0.0, "Vth_bg0"), (4.0, "Vth_bg4")]
    assert groups["Vth_d2"] == [(0.0, "Vth_d2_bg0"), (4.0, "Vth_d2_bg4")]
    assert groups["Delta_Vth"] == [(0.0, "Delta_Vth_bg0"), (4.0, "Delta_Vth_bg4")]
    assert groups["Ion_Ioff"] == [(0.0, "Ion_Ioff_bg0"), (4.0, "Ion_Ioff_bg4")]
    assert "Vth_air" not in groups
    assert "delta_Vth_H2" not in groups
    assert "LOD_ppm" not in groups


def test_bg_groups_ignores_a_lone_metric_with_no_pair():
    # a single Vth_bg0 (no matching Vth_bg4) has nothing to overlay -- stays
    # out of the grouped plots and falls through to the normal per-metric one
    assert _bg_groups(["Vth_bg0", "Ion_bg4"]) == {}


def _mk(*parts):
    os.makedirs(os.path.join(*parts), exist_ok=True)


def test_discover_timepoints_finds_analysed_runs_and_skips_the_rest(tmp_path):
    root = str(tmp_path)
    # valid timepoints: an 'E...' run folder with an 'analysis' dir
    _mk(root, "baseline", "E1_baseline", "analysis")
    _mk(root, "2D", "E1_2D", "analysis")
    _mk(root, "2D", "2D resistors calibration")          # sibling, no analysis
    _mk(root, "14D", "E1_14D", "analysis")
    # skipped: right name but the run folder was never analysed (no 'analysis')
    _mk(root, "7D", "E1_7D")
    # skipped: not timepoint-looking names (even though one has a digit)
    _mk(root, "degradation", "85C_85RH")
    _mk(root, "degradation-DESKTOP-KJI3CLJ", "85C_85RH")
    _mk(root, "preliminery work", "whatever")

    found = discover_timepoints(root)

    assert [d for d, _f in found] == ["baseline", "2D", "14D"]   # sorted by day, 7D dropped
    assert found[0][1].endswith(os.path.join("baseline", "E1_baseline"))
    assert found[1][1].endswith(os.path.join("2D", "E1_2D"))


def test_discover_timepoints_prefers_the_E_folder(tmp_path):
    root = str(tmp_path)
    # two analysed subfolders -- the 'E...' one wins over an alphabetically
    # earlier sibling that also happens to carry an analysis dir
    _mk(root, "2D", "Aux results", "analysis")
    _mk(root, "2D", "E1_2D", "analysis")
    found = discover_timepoints(root)
    assert len(found) == 1
    assert found[0][1].endswith(os.path.join("2D", "E1_2D"))


def test_discover_timepoints_empty_for_missing_parent(tmp_path):
    assert discover_timepoints(str(tmp_path / "does_not_exist")) == []
