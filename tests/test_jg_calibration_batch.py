"""Optional JG correction and one-click raw-timepoint discovery."""

from pathlib import Path
import json

import pandas as pd
import pytest
import numpy as np

from sensorlab.batch import analyze_all_timepoints, discover_raw_timepoints
from sensorlab.compare import _calibration_from_masters
from sensorlab.degradation import active_analysis_root, find_masters
from sensorlab.io.device_map import ConditionCache
from sensorlab.jg_calibration import (CalibrationOffset, DeviceAssignment,
                                      JgCalibration, apply_jg_calibration,
                                      import_blank_sweeps)
from sensorlab.measurements.jg import _fixed_current_ss
from sensorlab.model import Channel, MajorKind, Run


def _calibration() -> JgCalibration:
    return JgCalibration(
        [CalibrationOffset("card-A", "Id02", 0, -30),
         CalibrationOffset("card-A", "Id02", 4, -40)],
        [DeviceAssignment("1216", "11", "card-A", "Id02")],
    )


def test_jg_correction_uses_physical_card_mapping_and_preserves_raw() -> None:
    raw = pd.DataFrame({"Part": [0, 2, 2], "Vb": [0, 0, 4],
                        "Vj": [-2.8, -2.8, -2.8], "Id00": [-31., 120., -41.]})
    run = Run("n03.xlsx", 3, MajorKind.JG, "JG", raw,
              [Channel("Id00", 0, "1216", "11")], {})
    corrected = apply_jg_calibration(run, _calibration())
    assert corrected.data["Id00"].tolist() == [-1., 150., -1.]
    assert run.data["Id00"].tolist() == [-31., 120., -41.]
    assert corrected.parts()[2]["Id00"].tolist() == [150., -1.]


def test_calibration_roundtrip_and_missing_assignment(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _calibration().save_json(path)
    loaded = JgCalibration.from_json(path)
    assert loaded.offset_for("1216", "11", 0) == -30
    with pytest.raises(ValueError, match="No JG calibration"):
        loaded.offset_for("1232", "11", 0)


def test_import_blank_sweep_means_by_card_channel_and_gate(
        monkeypatch: pytest.MonkeyPatch) -> None:
    description = pd.DataFrame({0: ["ESN00", "TNumber00"], 1: ["card-A", 11]})
    data = pd.DataFrame({"Part": [2, 2, 2, 2], "Vb": [0, 0, 4, 4],
                         "Vj": [-2.8, -2.77, -2.8, -2.77],
                         "Id00": [-31., -29., -41., -39.]})
    mapping = pd.DataFrame({"esn": ["1216"], "tnumber": ["11"],
                            "card_id": ["card-A"], "calibration_channel": [None]})
    monkeypatch.setattr("sensorlab.jg_calibration._mapping_table", lambda _path: mapping)
    monkeypatch.setattr("sensorlab.jg_calibration.pd.read_excel",
                        lambda _path, sheet_name, **_kwargs:
                        description if sheet_name == "Description" else data)
    imported = import_blank_sweeps(["blank.xlsx"], "mapping.csv")
    assert imported.offset_for("1216", "11", 0) == -30
    assert imported.offset_for("1216", "11", 4) == -40
    assert imported.as_json_dict()["offsets"][0]["n"] == 2


def test_import_fixed_state_blank_uses_same_dc_offset_at_both_back_gates(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    description = pd.DataFrame({0: ["ESN00", "TNumber00"], 1: [119, 11]})
    data = pd.DataFrame({"Part": [0, 0, 0], "Vb": [4, 4, 4],
                         "Vj": [-0.37, -0.37, -0.37],
                         "Id00": [-41., -39., -40.]})
    mapping = pd.DataFrame({"esn": ["1074"], "tnumber": ["11"],
                            "card_id": ["119"], "calibration_channel": [None]})
    monkeypatch.setattr("sensorlab.jg_calibration._mapping_table", lambda _path: mapping)
    monkeypatch.setattr("sensorlab.jg_calibration.pd.read_excel",
                        lambda _path, sheet_name, **_kwargs:
                        description if sheet_name == "Description" else data)
    imported = import_blank_sweeps(["card_state.xlsx"], "mapping.xlsx")
    assert imported.offset_for("1074", "11", 0) == -40
    assert imported.offset_for("1074", "11", 4) == -40
    assert imported.reference_conditions == {
        "type": "fixed_state", "part": 0, "measured_Vbg_V": 4.0,
        "measured_Vj_V": -0.37, "applies_to_all_Vbg": True,
    }
    path = tmp_path / "calibration.json"
    imported.save_json(path)
    assert JgCalibration.from_json(path).offset_for("1074", "11", 0) == -40


def test_import_direct_device_blank_sweep_without_card_mapping(
        monkeypatch: pytest.MonkeyPatch) -> None:
    description = pd.DataFrame({0: ["ESN00", "TNumber00"], 1: [1074, 11]})
    data = pd.DataFrame({"Part": [0, 0, 0, 0], "Vb": [0, 0, 4, 4],
                         "Vj": [-2.8, 0.3, -2.8, 0.3],
                         "Id00": [-31., -29., -41., -39.]})
    monkeypatch.setattr("sensorlab.jg_calibration.pd.read_excel",
                        lambda _path, sheet_name, **_kwargs:
                        description if sheet_name == "Description" else data)
    imported = import_blank_sweeps(["direct_blank.xlsx"])
    assert len(imported.assignments) == 1
    assert imported.offset_for("1074", "11", 0) == -30
    assert imported.offset_for("1074", "11", 4) == -40
    assert imported.reference_conditions == {
        "type": "JG_blank_sweep", "part": 0,
        "Vj_min_V": -2.8, "Vj_max_V": 0.3,
    }


def test_fixed_current_ss_requires_measured_points_and_linearity() -> None:
    current = np.array([20., 30., 45., 68., 100.])
    vj = 0.05 * np.log10(current)
    ss, n, r2, span, status = _fixed_current_ss(vj, current, (20, 100))
    assert ss == pytest.approx(50)
    assert (n, r2, status) == (5, pytest.approx(1), "ok")
    assert span > 0.6
    assert _fixed_current_ss(vj, current, (20, 35))[-1] == "too_few_points"


def test_degradation_detects_mixed_correction_settings(tmp_path: Path) -> None:
    day0 = tmp_path / "baseline"
    day2 = tmp_path / "2D"
    (day0 / "analysis").mkdir(parents=True)
    (day2 / "analysis").mkdir(parents=True)
    _calibration().save_json(day0 / "analysis" / "jg_calibration_used.json")
    with pytest.raises(ValueError, match="Some timepoint masters"):
        _calibration_from_masters([("baseline", str(day0)), ("2D", str(day2))])


def _raw_timepoint(root: Path, day: str) -> Path:
    run = root / day / "E123_example"
    run.mkdir(parents=True)
    (run / "E123_n03_JG_.xlsx").write_bytes(b"raw-result-fixture")
    return run


def test_discovers_unanalysed_timepoints_and_activates_new_versions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _raw_timepoint(tmp_path, "baseline")
    day2 = _raw_timepoint(tmp_path, "2D")
    (baseline / "analysis").mkdir()
    (baseline / "analysis" / "stale.txt").write_text("old")
    assert discover_raw_timepoints(tmp_path) == [("baseline", baseline), ("2D", day2)]

    def fake_analyze(folder: str, out_dir: str, **_kwargs) -> dict:
        output = Path(out_dir)
        output.mkdir(parents=True)
        (output / "fresh.txt").write_text(Path(folder).parent.name)
        return {"JG": [str(output / "fresh.txt")]}

    monkeypatch.setattr("sensorlab.batch.analyze_folder", fake_analyze)
    completed = analyze_all_timepoints(
        tmp_path, conditions=ConditionCache(tmp_path / "conditions.json"))
    assert [day for day, _ in completed] == ["baseline", "2D"]
    assert (baseline / "analysis" / "stale.txt").read_text() == "old"
    assert (active_analysis_root(baseline) / "fresh.txt").read_text() == "baseline"
    assert (active_analysis_root(day2) / "fresh.txt").read_text() == "2D"
    assert len(list(tmp_path.glob(".sensorlab-analysis-batch-*.json"))) == 1
    assert (baseline / "E123_n03_JG_.xlsx").read_bytes() == b"raw-result-fixture"


def test_failed_batch_keeps_previous_analysis_active(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _raw_timepoint(tmp_path, "baseline")
    _raw_timepoint(tmp_path, "2D")
    (baseline / "analysis").mkdir()
    (baseline / "analysis" / "old.txt").write_text("preserve")

    def fake_analyze(folder: str, out_dir: str, **_kwargs) -> dict:
        if Path(folder).parent.name == "2D":
            raise RuntimeError("simulated analysis error")
        output = Path(out_dir)
        output.mkdir(parents=True)
        (output / "new.txt").write_text("new")
        return {"JG": [str(output / "new.txt")]}

    monkeypatch.setattr("sensorlab.batch.analyze_folder", fake_analyze)
    with pytest.raises(RuntimeError, match="simulated analysis error"):
        analyze_all_timepoints(
            tmp_path, conditions=ConditionCache(tmp_path / "conditions.json"))
    assert (baseline / "analysis" / "old.txt").read_text() == "preserve"
    assert active_analysis_root(baseline) == baseline / "analysis"
    assert not list(tmp_path.glob(".sensorlab-analysis-batch-*.json"))


def test_active_batch_drives_masters_and_calibration_snapshots(tmp_path: Path) -> None:
    folders = [_raw_timepoint(tmp_path, day) for day in ("baseline", "2D")]
    entries = {}
    for folder in folders:
        (folder / "analysis").mkdir()
        active = folder / "analysis_batch_test123"
        master = active / "Hot storage" / "JG" / "master_JG.xlsx"
        master.parent.mkdir(parents=True)
        master.write_bytes(b"new master")
        _calibration().save_json(active / "jg_calibration_used.json")
        entries[folder.relative_to(tmp_path).as_posix()] = active.name
    manifest = tmp_path / ".sensorlab-analysis-batch-20260926_test123.json"
    manifest.write_text(json.dumps({"version": 1, "timepoints": entries}))
    assert active_analysis_root(folders[0]) == folders[0] / "analysis_batch_test123"
    assert list(find_masters(str(folders[0])))[0][2].endswith("master_JG.xlsx")
    assert _calibration_from_masters([("baseline", str(folders[0])),
                                      ("2D", str(folders[1]))]) is not None
