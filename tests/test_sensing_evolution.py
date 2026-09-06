"""Unit tests for the SE/STEPS sensing-evolution plots (raw continuous trace
overlay across timepoints, same idea as the JG sweep evolution)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PathCollection

import sensorlab.compare as compare_mod
from sensorlab.compare import run_degradation
from sensorlab.io.master_writer import write_master
from sensorlab.plotting.evolution_plot import plot_sensing_evolution
from sensorlab.protocol import DEFAULT


def _raw_se_n04(path: str, esn: str, tnum: str, response_level: float) -> None:
    """4-part main sensing run (baseline/response/settle/recovery); the
    response level varies per timepoint so the overlaid curves differ."""
    levels = {0: 10000.0, 1: response_level, 2: response_level - 500.0, 3: 10500.0}
    rows = []
    for part, level in levels.items():
        for i in range(60):
            rows.append({"Time": float(part * 100 + i), "Part": part, "Vb": 4.0, "Vj": -0.5, "Id00": level})
    data = pd.DataFrame(rows)
    desc = pd.DataFrame([("TestType", "SE_STATE"), ("ESN00", esn), ("TNumber00", tnum)])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="Data", index=False)
        desc.to_excel(w, sheet_name="Description", index=False, header=False)


def _se_master(path: str, day: str, esn: str, tnum: str, response_pct: float) -> None:
    df = pd.DataFrame({"ESN": [esn], "TNumber": [tnum], "condition": ["A"],
                       "response_pct": [response_pct], "day": [day]})
    write_master(path, {"SE_n04": df})


def _timepoint(root: str, day: str, esn: str, tnum: str, response_level: float) -> str:
    folder = os.path.join(root, f"tp_{day}")
    os.makedirs(folder, exist_ok=True)
    _raw_se_n04(os.path.join(folder, f"E0000_2026_01_0{day}_N01_n04_SE_STATE_.xlsx"), esn, tnum, response_level)
    mdir = os.path.join(folder, "analysis", "A", "SE")
    os.makedirs(mdir, exist_ok=True)
    _se_master(os.path.join(mdir, "master_SE.xlsx"), day, esn, tnum, response_level / 100.0)
    return folder


def test_se_sensing_evolution_writes_one_file_per_device_with_all_days(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", response_level=13000.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", response_level=11000.0)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_sensing_evolution
    compare_mod.plot_sensing_evolution = lambda dc, p, t, **kw: (calls.append(dc), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)
    finally:
        compare_mod.plot_sensing_evolution = orig

    assert len(calls) == 1   # one device
    curves = calls[0]
    assert {c["day_num"] for c in curves} == {0.0, 2.0}
    assert len(curves) == 2   # one curve per day, no back-gate dimension

    png = os.path.join(out, "degradation", "A", "SE_n04", "sensing", "ESN111_D1T1.png")
    assert os.path.isfile(png)


def test_se_sensing_evolution_curve_differs_between_timepoints(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", response_level=13000.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", response_level=11000.0)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_sensing_evolution
    compare_mod.plot_sensing_evolution = lambda dc, p, t, **kw: (calls.append(dc), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)
    finally:
        compare_mod.plot_sensing_evolution = orig

    curves = calls[0]
    day0 = next(c for c in curves if c["day_num"] == 0.0)
    day2 = next(c for c in curves if c["day_num"] == 2.0)
    assert not np.allclose(day0["y"], day2["y"])


def test_n04_raw_overlay_mode_is_independent_and_propagated(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", response_level=13000.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", response_level=11000.0)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_sensing_evolution
    compare_mod.plot_sensing_evolution = lambda dc, p, t, **kw: (calls.append(kw), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0",
                        make_plots=True, trend_plot_mode="line",
                        raw_plot_mode="scatter")
    finally:
        compare_mod.plot_sensing_evolution = orig

    assert calls
    assert all(call["plot_mode"] == "scatter" for call in calls)


def test_sensing_raw_overlay_can_be_scatter_without_connecting_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    t = np.linspace(0.0, 10.0, 11)
    curves = [{"day_num": 0.0, "day_label": "0", "x": t,
               "y": 1000.0 + t, "baseline": 1000.0}]

    plot_sensing_evolution(curves, str(tmp_path / "scatter.png"), "title",
                           plot_mode="scatter")
    ax = plt.gcf().axes[0]

    assert not ax.lines
    assert len([c for c in ax.collections if isinstance(c, PathCollection)]) == 1
    plt.close("all")


def test_sensing_evolution_dc_shifts_each_day_to_its_own_baseline(tmp_path, monkeypatch):
    """Two days with DIFFERENT absolute baselines but the SAME response shape,
    for a DOWNWARD responder (current dips BELOW baseline on gas). Shifting by
    each day's BASELINE level -- not its minimum -- lands both on the same
    values AND keeps the dip below 0. A minimum-anchor would instead pin the
    dip at 0 and leave the two baselines floating apart (the bug this fixes)."""
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    t = np.linspace(0, 3000, 60)
    shape = np.where(t < 600, 0.0, -3.0)   # flat baseline at 0, then dips to -3
    day0 = {"day_num": 0.0, "day_label": "0", "x": t, "y": 1000.0 + shape, "baseline": 1000.0}
    day2 = {"day_num": 2.0, "day_label": "2", "x": t, "y": 4000.0 + shape, "baseline": 4000.0}

    plot_sensing_evolution([day0, day2], str(tmp_path / "p.png"), "title")
    ax = plt.gcf().axes[0]
    lines = {ln.get_label(): ln.get_ydata() for ln in ax.get_lines()}
    # +3000 baseline drift is cancelled -> both collapse onto `shape`
    assert np.allclose(lines["Baseline"], shape)
    assert np.allclose(lines["2 d"], shape)
    assert lines["Baseline"][0] == pytest.approx(0.0)    # baseline sits at 0
    assert lines["Baseline"][-1] == pytest.approx(-3.0)  # dip stays BELOW 0 (not pinned to 0)
    assert "DC-shifted" not in ax.get_ylabel()    # label kept clean: just Ids + units
    plt.close("all")


def test_steps_selection_filters_data_but_keeps_manual_x_limits(tmp_path, monkeypatch):
    """Omitted ppm data stays blank even when the manual viewport includes it."""
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    x = np.arange(0.0, 701.0, 10.0)
    curve = {"day_num": 0.0, "day_label": "0", "x": x,
             "y": 1000.0 + x, "baseline": 1000.0}
    spans = [(0.0, 100.0, 0, 0), (100.0, 200.0, 10, 1),
             (200.0, 300.0, 25, 2), (300.0, 400.0, 50, 3),
             (400.0, 500.0, 100, 4), (500.0, 600.0, 1000, 5),
             (600.0, 700.0, 0, 6)]

    plot_sensing_evolution(
        [curve], str(tmp_path / "steps.png"), "title",
        xlim=(0.0, 700.0), steps_spans=spans,
        steps_ppm={0, 10, 25, 50, 100},
    )
    ax = plt.gcf().axes[0]
    line = next(ln for ln in ax.get_lines() if ln.get_label() == "Baseline")
    y = np.asarray(line.get_ydata(), dtype=float)

    assert ax.get_xlim() == pytest.approx((0.0, 700.0))
    assert np.isfinite(y[x <= 500.0]).all()
    assert np.isnan(y[x > 500.0]).all()
    assert "1000 ppm" not in {text.get_text() for text in ax.texts}
    plt.close("all")


def test_sensing_evolution_skips_missing_timepoint_with_warning(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", response_level=13000.0)
    empty = tmp_path / "tp_empty"
    empty.mkdir()
    out = str(tmp_path / "out")

    logs = []
    run_degradation([("0", t0), ("5", str(empty))], out_dir=out, baseline_day="0",
                    make_plots=True, log=logs.append)
    assert any("SE_n04 sensing evolution" in m and "no raw n04 file found" in m for m in logs)
