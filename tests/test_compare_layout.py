"""Unit tests for run_degradation's sheet filtering and trends folder layout."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import sensorlab.compare as compare_mod
from sensorlab.compare import run_degradation
from sensorlab.io.master_writer import write_master


def _jg_master(path: str, day: str, vth_bg0: float, vth_bg4: float) -> None:
    sheets = {}
    for n in ("JG_n03", "JG_n05", "JG_n06"):
        df = pd.DataFrame({"ESN": ["1"], "TNumber": ["11"], "condition": ["A"],
                           "Vth_bg0": [vth_bg0], "Vth_bg4": [vth_bg4], "day": [day]})
        sheets[n] = df
    write_master(path, sheets)


def _timepoint(root: str, day: str, vth: float, vth_bg4: float | None = None) -> str:
    folder = os.path.join(root, f"tp_{day}")
    mdir = os.path.join(folder, "analysis", "A", "JG")
    os.makedirs(mdir, exist_ok=True)
    _jg_master(os.path.join(mdir, "master_JG.xlsx"), day, vth, vth_bg4 if vth_bg4 is not None else vth)
    return folder


def test_jg_n05_n06_are_skipped_but_n03_kept(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", 1.0)
    t2 = _timepoint(str(tmp_path), "2", 1.5)
    out = str(tmp_path / "out")

    run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)

    root = os.path.join(compare_mod.active_degradation_root(out), "A")
    assert os.path.isdir(os.path.join(root, "JG_n03"))
    assert not os.path.isdir(os.path.join(root, "JG_n05"))
    assert not os.path.isdir(os.path.join(root, "JG_n06"))


def test_exclude_cond_day_drops_it_from_calc_and_output(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", 1.0)
    t2 = _timepoint(str(tmp_path), "2", 1.5)
    t4 = _timepoint(str(tmp_path), "4", 2.0)
    out = str(tmp_path / "out")

    run_degradation([("0", t0), ("2", t2), ("4", t4)], out_dir=out, baseline_day="0",
                    make_plots=False, exclude_cond_days={("A", "2")})

    agg = pd.read_excel(os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "aggregate.xlsx"))
    days = set(agg["day"].astype(str))
    assert "2" not in days              # the excluded condition-day is gone
    assert {"0", "4"} <= days           # the rest stay


def test_cycle_axis_exports_estimated_cycles_without_replacing_days(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", 1.0)
    t2 = _timepoint(str(tmp_path), "2", 1.5)
    out = str(tmp_path / "out")

    run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0",
                    make_plots=True, time_axis="cycles", hours_per_cycle=2.0)

    agg = pd.read_excel(os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "aggregate.xlsx"))
    assert set(agg["day_num"]) == {0, 2}       # original elapsed time is preserved
    assert set(agg["cycle_num"]) == {0, 24}    # 2 days * 24 h/day / 2 h/cycle
    row = agg[(agg["metric"] == "Vth_bg0") & (agg["day_num"] == 2)].iloc[0]
    assert row["delta_mean"] == pytest.approx(-0.5)  # calculation is unchanged
    assert os.path.isfile(os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03",
                                       "trends", "average", "Vth.png"))


@pytest.mark.parametrize("hours", [0, -2, float("nan"), float("inf"), "bad"])
def test_cycle_axis_rejects_invalid_cycle_duration(tmp_path, hours):
    with pytest.raises(ValueError, match="Hours per cycle"):
        run_degradation([], out_dir=str(tmp_path), make_plots=False,
                        time_axis="cycles", hours_per_cycle=hours)


def test_trends_are_sorted_into_family_subfolders(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", 1.0)
    t2 = _timepoint(str(tmp_path), "2", 1.5)
    out = str(tmp_path / "out")

    run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)

    tdir = os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "trends")
    for sub in ("devices", "average", "degradation"):
        files = os.listdir(os.path.join(tdir, sub))
        # Vth_bg0/Vth_bg4 are grouped -> one "Vth.png", not "Vth_bg0.png"
        assert "Vth.png" in files, (sub, files)
        assert not any(f.startswith("Vth_bg") for f in files), (sub, files)
    # no flat files directly under trends/ itself
    assert not [f for f in os.listdir(tdir) if f.endswith(".png")]
    # no separate 4th family -- just two extra plots dropped into the
    # existing devices/average folders
    assert not os.path.isdir(os.path.join(tdir, "delta"))


def test_jg_n03_gets_delta_vth_in_mv_alongside_normal_plots(tmp_path):
    t0 = _timepoint(str(tmp_path), "0", 1.000)
    t2 = _timepoint(str(tmp_path), "2", 0.985)   # -15 mV shift
    out = str(tmp_path / "out")

    run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)

    tdir = os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "trends")
    for sub in ("devices", "average"):
        assert os.path.isfile(os.path.join(tdir, sub, "delta_Vth.png"))
    # degradation/ (the % family) gets no delta_Vth counterpart
    assert not os.path.isfile(os.path.join(tdir, "degradation", "delta_Vth.png"))

    agg = pd.read_excel(os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "aggregate.xlsx"))
    row = agg[(agg["metric"] == "Vth_bg0") & (agg["day_num"] == 2)].iloc[0]
    assert row["delta_mean"] == pytest.approx(0.015)   # raw V in the table, unscaled


def test_delta_vth_is_wired_with_negative_scale(tmp_path, monkeypatch):
    """Regression for the double-negative bug: the pipeline's generic `delta`
    column is baseline-value (positive = 'degraded'), which flips sign for a
    negative-domain Vth (confirmed below: raw delta is +0.2646 V even though
    Vth itself moved from -0.619 V to -0.884 V, i.e. MORE negative). The fix
    is to negate that column back before scaling to mV, so the plot shows the
    natural value-baseline shift instead. (The y-axis is only *labelled*
    |ΔVth|; the data keeps its sign.)

    Spied on the compare.py -> trend_plot.py call boundary rather than
    rendering and re-parsing a live matplotlib figure: this checks the one
    thing that can regress here (the sign compare.py passes for `scale`)
    without depending on axis/label internals that plot_grouped_average's own
    tests in test_trend_plot.py already cover directly."""
    calls = []
    real = compare_mod.plot_grouped_average
    monkeypatch.setattr(compare_mod, "plot_grouped_average",
                        lambda *a, **kw: (calls.append((a, kw)), real(*a, **kw))[1])

    t0 = _timepoint(str(tmp_path), "0", -0.619265)
    t2 = _timepoint(str(tmp_path), "2", -0.883864)
    out = str(tmp_path / "out")
    run_degradation([("0", t0), ("2", t2)], out_dir=out, baseline_day="0", make_plots=True)

    agg = pd.read_excel(os.path.join(compare_mod.active_degradation_root(out), "A", "JG_n03", "aggregate.xlsx"))
    row = agg[(agg["metric"] == "Vth_bg0") & (agg["day_num"] == 2)].iloc[0]
    assert row["delta_mean"] == pytest.approx(0.264599, abs=1e-5)   # raw pipeline delta: positive

    delta_calls = [kw for _a, kw in calls if kw.get("value") == "delta"]
    assert len(delta_calls) == 1
    assert delta_calls[0]["scale"] < 0             # negated (signed data), not just +1000
    assert not delta_calls[0].get("abs_value")     # data keeps its sign; only the label is |ΔVth|
