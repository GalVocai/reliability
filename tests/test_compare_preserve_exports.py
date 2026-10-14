"""Regeneration must not erase hand-exported degradation figures."""

import json
from pathlib import Path

import pytest

from sensorlab import compare


def test_degradation_rebuild_preserves_custom_figures_and_previous_tree(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "degradation"
    average = current / "Hot storage" / "JG_n03" / "trends" / "average"
    average.mkdir(parents=True)
    manual = average / "delta_Vth_powerlaw_Vbg0.pdf"
    manual.write_bytes(b"hand exported")
    automatic = average / "Vth.png"
    automatic.write_bytes(b"old generated")

    def fake_build(_timepoints: list, stage_parent: str, **kwargs: object) -> str:
        stage = Path(stage_parent) / str(kwargs["output_name"])
        new_average = stage / "Hot storage" / "JG_n03" / "trends" / "average"
        new_average.mkdir(parents=True)
        (new_average / "Vth.png").write_bytes(b"new generated")
        return str(stage)

    monkeypatch.setattr(compare, "_build_degradation_output", fake_build)
    result = Path(compare.run_degradation([], str(tmp_path)))
    assert result != current
    assert result.name.startswith("degradation_batch_")
    assert compare.active_degradation_root(tmp_path) == result
    assert manual.read_bytes() == b"hand exported"
    assert automatic.read_bytes() == b"old generated"
    assert (result / manual.relative_to(current)).read_bytes() == b"hand exported"
    assert (result / automatic.relative_to(current)).read_bytes() == b"new generated"
    generated = json.loads((result / ".sensorlab_generated_files.json").read_text())
    assert automatic.relative_to(current).as_posix() in generated["files"]
    assert manual.relative_to(current).as_posix() not in generated["files"]


def test_failed_build_leaves_existing_degradation_untouched(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "degradation"
    current.mkdir()
    (current / "manual.pdf").write_bytes(b"original")

    def failing_build(_timepoints: list, stage_parent: str, **kwargs: object) -> str:
        stage = Path(stage_parent) / str(kwargs["output_name"])
        stage.mkdir()
        (stage / "partial.png").write_bytes(b"partial")
        raise RuntimeError("simulated plot failure")

    monkeypatch.setattr(compare, "_build_degradation_output", failing_build)
    with pytest.raises(RuntimeError, match="simulated plot failure"):
        compare.run_degradation([], str(tmp_path))
    assert (current / "manual.pdf").read_bytes() == b"original"
    assert compare.active_degradation_root(tmp_path) == current
    assert not list(tmp_path.glob(".sensorlab-degradation-batch-*.json"))
