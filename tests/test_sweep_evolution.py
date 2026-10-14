"""Unit tests for the JG_n03 sweep-evolution plots (raw curve overlay across
timepoints, not a derived scalar trend)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

import sensorlab.compare as compare_mod
from sensorlab.compare import run_degradation
from sensorlab.io.master_writer import write_master
from sensorlab.plotting.evolution_plot import plot_jg_sweep_evolution
from sensorlab.protocol import DEFAULT

VJ = np.linspace(-2.8, 0.3, 10)


def _raw_jg_n03(path: str, esn: str, tnum: str, vth_shift: float) -> None:
    """A minimal raw JG_n03 result file: 3 reps (only the last is used), two
    back-gates, one channel. `vth_shift` moves the turn-on along Vj so
    different timepoints produce visibly different curves."""
    rows = []
    for part in range(3):
        for vb in (0.0, 4.0):
            # simple monotonic turn-on, shifted by vth_shift on the last rep only
            shift = vth_shift if part == 2 else 0.0
            idv = 1.0 + 50000.0 / (1.0 + np.exp(-8.0 * (VJ - (-1.0 + shift))))
            for vj, i in zip(VJ, idv):
                rows.append({"Time": float(part * 10), "Part": part, "Vb": vb, "Vj": float(vj), "Id00": float(i)})
    data = pd.DataFrame(rows)
    desc = pd.DataFrame([("TestType", "JG"), (f"ESN00", esn), (f"TNumber00", tnum)])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="Data", index=False)
        desc.to_excel(w, sheet_name="Description", index=False, header=False)


def _jg_n03_master(path: str, day: str, esn: str, tnum: str, vth: float) -> None:
    df = pd.DataFrame({"ESN": [esn], "TNumber": [tnum], "condition": ["A"],
                       "Vth_bg0": [vth], "day": [day]})
    write_master(path, {"JG_n03": df})


def _timepoint(root: str, day: str, esn: str, tnum: str, vth_shift: float) -> str:
    folder = os.path.join(root, f"tp_{day}")
    os.makedirs(folder, exist_ok=True)
    _raw_jg_n03(os.path.join(folder, f"E0000_2026_01_0{day}_N01_n03_JG_.xlsx"), esn, tnum, vth_shift)
    mdir = os.path.join(folder, "analysis", "A", "JG")
    os.makedirs(mdir, exist_ok=True)
    _jg_n03_master(os.path.join(mdir, "master_JG.xlsx"), day, esn, tnum, -1.0 + vth_shift)
    return folder


def test_sweep_evolution_writes_one_file_per_device_and_back_gate(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", vth_shift=0.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", vth_shift=0.3)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_jg_sweep_evolution
    compare_mod.plot_jg_sweep_evolution = lambda dc, p, t, **kw: (calls.append(dc), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)
    finally:
        compare_mod.plot_jg_sweep_evolution = orig

    assert len(calls) == 2   # one device x two back-gates (Vbg=0, Vbg=4), separate figures
    for curves in calls:
        assert {c["day_num"] for c in curves} == {0.0, 2.0}
        assert len({c["bg"] for c in curves}) == 1     # each figure is a single back-gate
        assert len(curves) == 2                        # 2 days x 1 back-gate
    assert {next(iter({c["bg"] for c in curves})) for curves in calls} == {0, 4}

    swdir = os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "sweeps")
    assert os.path.isfile(os.path.join(swdir, "ESN111_D1T1_Vbg0.png"))
    assert os.path.isfile(os.path.join(swdir, "ESN111_D1T1_Vbg4.png"))
    collapse_dir = os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "collapsed")
    assert os.path.isfile(os.path.join(collapse_dir, "ESN111_D1T1_Vbg0.png"))
    assert os.path.isfile(os.path.join(collapse_dir, "ESN111_D1T1_Vbg4.png"))


def test_sweep_evolution_curve_shifts_between_timepoints(tmp_path):
    """The actual Id(Vj) data passed to the plot must differ between day 0
    and day 2 (the turn-on was shifted by vth_shift=0.3 when generating the
    raw file) -- this is the whole point: seeing the curve SHAPE move."""
    t0 = _timepoint(str(tmp_path), "0", "111", "11", vth_shift=0.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", vth_shift=0.3)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_jg_sweep_evolution
    compare_mod.plot_jg_sweep_evolution = lambda dc, p, t, **kw: (calls.append(dc), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)
    finally:
        compare_mod.plot_jg_sweep_evolution = orig

    curves = calls[0]
    day0_bg0 = next(c for c in curves if c["day_num"] == 0.0 and c["bg"] == 0)
    day2_bg0 = next(c for c in curves if c["day_num"] == 2.0 and c["bg"] == 0)
    assert not np.allclose(day0_bg0["id"], day2_bg0["id"])


def test_sweep_evolution_uses_the_fixed_protocol_vj_range(tmp_path):
    """The x-axis must be pinned to the sweep's own Vj range, not auto-scaled
    per day (despiking otherwise clips the off-region differently per day and
    the frame wobbles between curves)."""
    t0 = _timepoint(str(tmp_path), "0", "111", "11", vth_shift=0.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", vth_shift=0.3)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_jg_sweep_evolution
    compare_mod.plot_jg_sweep_evolution = lambda dc, p, t, **kw: (calls.append(kw), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)
    finally:
        compare_mod.plot_jg_sweep_evolution = orig

    assert calls[0]["xlim"] == (DEFAULT.jg.sweep_start_vj, DEFAULT.jg.sweep_close_vj)


def test_sweep_evolution_honours_custom_degradation_vj_range(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", vth_shift=0.0)
    t2 = _timepoint(str(tmp_path), "2", "111", "11", vth_shift=0.3)
    out = str(tmp_path / "out")

    calls = []
    orig = compare_mod.plot_jg_sweep_evolution
    compare_mod.plot_jg_sweep_evolution = lambda dc, p, t, **kw: (calls.append(kw), orig(dc, p, t, **kw))[1]
    try:
        run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0",
                        make_plots=True, jg_sweep_xlim=(-2.0, 0.1),
                        raw_plot_mode="scatter")
    finally:
        compare_mod.plot_jg_sweep_evolution = orig

    assert calls
    assert all(call["xlim"] == (-2.0, 0.1) for call in calls)
    assert all(call["plot_mode"] == "scatter" for call in calls)


def test_sweep_evolution_skips_missing_timepoint_with_warning(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", "111", "11", vth_shift=0.0)
    empty = tmp_path / "tp_empty"
    empty.mkdir()
    out = str(tmp_path / "out")

    logs = []
    run_degradation([("0", t0), ("5", str(empty))], out_dir=out, baseline_day="0",
                    make_plots=True, log=logs.append)
    assert any("no raw n03 file found" in m for m in logs)


def test_all_lines_solid_and_only_day_legend(tmp_path, monkeypatch):
    """Every plotted curve is solid; the only legend is the day/colour one
    (both back-gates share a day's colour, told apart by position only)."""
    import matplotlib.legend
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    curves = [
        {"day_num": 0.0, "day_label": "0", "bg": 0, "vj": VJ, "id": np.full(VJ.size, 100.0)},
        {"day_num": 0.0, "day_label": "0", "bg": 4, "vj": VJ, "id": np.full(VJ.size, 200.0)},
        {"day_num": 2.0, "day_label": "2", "bg": 0, "vj": VJ, "id": np.full(VJ.size, 300.0)},
        {"day_num": 2.0, "day_label": "2", "bg": 4, "vj": VJ, "id": np.full(VJ.size, 400.0)},
    ]
    plot_jg_sweep_evolution(curves, str(tmp_path / "p.png"), "title")
    ax = plt.gcf().axes[0]

    styles = {ln.get_linestyle() for ln in ax.get_lines()}
    assert styles == {"-"}, styles                       # all solid, no dashes/dots
    assert {ln.get_linewidth() for ln in ax.get_lines()} == {2.5}

    legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert len(legends) == 1                             # only the Day legend
    assert legends[0].get_title().get_text() == "Day"
    labels = {t.get_text() for t in legends[0].get_texts()}
    assert labels == {"Baseline", "2 d"}                          # one entry per day, no back-gate entries
    plt.close("all")
    plt.close("all")


def test_jg_raw_overlay_can_be_scatter_without_connecting_lines(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    curves = [
        {"day_num": 0.0, "day_label": "0", "bg": 0, "vj": VJ,
         "id": np.linspace(100.0, 1000.0, VJ.size)},
        {"day_num": 2.0, "day_label": "2", "bg": 0, "vj": VJ,
         "id": np.linspace(120.0, 1200.0, VJ.size)},
    ]

    plot_jg_sweep_evolution(curves, str(tmp_path / "scatter.png"), "title",
                            plot_mode="scatter")
    ax = plt.gcf().axes[0]

    assert not ax.lines
    points = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(points) == 2
    assert all(np.asarray(c.get_sizes()) == pytest.approx([4.5 ** 2]) for c in points)
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"Baseline", "2 d"}
    plt.close("all")
