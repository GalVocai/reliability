"""End-to-end pipeline test on a synthetic results folder.

Verifies the full chain scan -> classify -> states -> analyze -> master/plot
layout, the condition split, and the master values themselves — plus the two
folder-level regressions: Excel ``~$`` lock files must be skipped, and a stray
noisy Vj sample must not misclassify an SE run as ZOZO.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from sensorlab.io.device_map import ConditionCache
from sensorlab.io.result_reader import _classify, scan_result_files
from sensorlab.model import MajorKind
from sensorlab.pipeline import analyze_folder, load_folder
from sensorlab.plotting.style import StyleConfig

# constant per-part levels -> trailing-window means are exact
LEVELS = {0: 10000.0, 1: 13000.0, 2: 12000.0, 3: 11000.0}


def _write_se_result(path: str) -> None:
    rows = []
    for part, level in LEVELS.items():
        for i in range(60):
            t = part * 60 + i
            rows.append({"Time": float(t), "Part": part, "Vb": 4.0, "Vj": -0.5,
                         "Id00": level, "Id01": level / 2})
    data = pd.DataFrame(rows)
    desc = pd.DataFrame([
        ("TestType", "SE_STATE"),
        ("ESN00", 111), ("TNumber00", 11),
        ("ESN01", 222), ("TNumber01", 12),
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="Data", index=False)
        desc.to_excel(w, sheet_name="Description", index=False, header=False)


def _write_script(path: str) -> None:
    states = pd.DataFrame([
        {"Number": 2, "StateName": "State1", "PortJG1": -0.5, "PortBG": 4.0},
        {"Number": 2, "StateName": "State2", "PortJG1": -0.7, "PortBG": 4.0},
        {"Number": 3, "StateName": "State1", "PortJG1": -0.3, "PortBG": 4.0},
        {"Number": 3, "StateName": "State2", "PortJG1": -0.5, "PortBG": 4.0},
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        states.to_excel(w, sheet_name="States", index=False)


@pytest.fixture
def result_folder(tmp_path):
    folder = tmp_path / "E0001_2026_01_01_N01_run"
    folder.mkdir()
    _write_se_result(str(folder / "E0001_2026_01_01_N01_n04_SE_STATE_.xlsx"))
    _write_script(str(folder / "SE_Script.xlsx"))
    # Excel lock file: carries the major token + a valid extension but is
    # unreadable garbage — scan must skip it or the whole analysis crashes.
    (folder / "~$E0001_2026_01_01_N01_n04_SE_STATE_.xlsx").write_bytes(b"\x00" * 64)
    return str(folder)


def test_scan_skips_excel_lock_files(result_folder):
    names = [os.path.basename(p) for p in scan_result_files(result_folder)]
    assert names == ["E0001_2026_01_01_N01_n04_SE_STATE_.xlsx"]


def test_load_folder_classifies_and_attaches_states(result_folder):
    runs = load_folder(result_folder)
    assert [r.kind for r in runs] == [MajorKind.SE]
    run = runs[0]
    assert [(c.esn, c.tnumber) for c in run.channels] == [("111", "11"), ("222", "12")]
    # ports (2, 3) map positionally onto channel indices (0, 1)
    assert run.states[("111", "11")]["State1"] == (-0.5, 4.0)
    assert run.states[("222", "12")]["State1"] == (-0.3, 4.0)


def test_noisy_vj_sample_does_not_misclassify_se_as_zozo():
    data = pd.DataFrame({
        "Time": np.arange(40, dtype=float),
        "Part": np.repeat([0, 1, 2, 3], 10),
        "Vj": -0.5,
    })
    data.loc[15, "Vj"] = -0.9        # single noisy sample inside part 1
    assert _classify("SE_STATE", data) == MajorKind.SE


def test_analyze_folder_layout_and_master_values(result_folder, tmp_path, monkeypatch):
    # hermetic style: don't let the developer's ~/.sensorlab config leak in
    monkeypatch.setattr("sensorlab.pipeline.load_style", lambda: StyleConfig())
    out = str(tmp_path / "analysis")
    conditions = ConditionCache(tmp_path / "cond.json", {"111": "condA"})  # 222 -> unassigned

    written = analyze_folder(result_folder, out_dir=out, conditions=conditions, day="7d")

    assert set(written) == {"SE", "WP"}
    assert sorted(os.path.basename(p) for p in written["SE"]) == ["master_SE.xlsx"] * 2

    # working points land in their own per-condition master (for degradation)
    wp = pd.read_excel(os.path.join(out, "condA", "WP", "master_WP.xlsx"))
    assert wp[["ESN", "Vj_State1", "Vb_State1", "Vj_State2"]].iloc[0].tolist() == [111, -0.5, 4.0, -0.7]
    assert wp["day"].tolist() == ["7d"]

    # layout: <condition>/SE/{master_SE.xlsx, by_device/<label>.png}
    for cond, esn, label in (("condA", "111", "ESN111_D1T1"), ("unassigned", "222", "ESN222_D1T2")):
        master = os.path.join(out, cond, "SE", "master_SE.xlsx")
        assert os.path.isfile(master)
        assert os.path.isfile(os.path.join(out, cond, "SE", "by_device", f"{label}.png"))
        df = pd.read_excel(master)
        assert df["ESN"].astype(str).tolist() == [esn]   # condition split is exact
        assert df["day"].tolist() == ["7d"]

    master = pd.read_excel(os.path.join(out, "condA", "SE", "master_SE.xlsx"))
    row = master.iloc[0]
    # levels are dropped from the master -- only deltas/percentages remain
    assert "baseline" not in master.columns and "response" not in master.columns
    assert row["response_delta"] == pytest.approx(LEVELS[1] - LEVELS[0])
    assert row["response_pct"] == pytest.approx(30.0)
    # corrected recovery residual = recovery + (response - settle) - baseline
    assert row["recovery_delta"] == pytest.approx(LEVELS[3] + (LEVELS[1] - LEVELS[2]) - LEVELS[0])
    # recovery% = (settle - recovery) / (response - baseline)
    assert row["recovery_pct"] == pytest.approx(100.0 * (LEVELS[2] - LEVELS[3]) / (LEVELS[1] - LEVELS[0]))
    assert row["flags"] == "ok"      # clean device; 'ok' survives the Excel round-trip
    assert row["group"] == "normal"
    # the groups sheet keeps baseline for diagnostics
    grp = pd.read_excel(os.path.join(out, "condA", "SE", "master_SE.xlsx"), sheet_name="SE_n04_groups")
    assert "baseline" in grp.columns
