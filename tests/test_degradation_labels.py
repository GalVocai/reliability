"""Unit tests for the user-editable degradation-labels config."""

from __future__ import annotations

import json

from sensorlab.degradation_labels import DegradationLabels, load_degradation_labels


def test_defaults_and_underscore_fallback():
    cfg = DegradationLabels()
    assert cfg.ylabel("Vth") == r"$\mathbf{V_{Th}}$ (V)"
    assert cfg.ylabel("delta_Vth_H2") == r"$\mathbf{|\Delta V_{Th}|}$ Sensing (V)"    # JG_H2 in volts, |·| label
    assert cfg.deg_ylabel() == "%"
    # not configured -> '_' becomes ' '
    assert cfg.ylabel("some_new_metric") == "Some New Metric"


def test_titles_have_no_avdev_and_fill_placeholders():
    cfg = DegradationLabels()
    assert "AVDEV" not in cfg.title("Vth", "average", 0)
    assert cfg.title("Vth", "average", 0) == "Vth — average"
    assert cfg.title("Response", "degradation", 7) == "Response — % of day 7 (100 = baseline)"


def test_user_overrides_merge_over_defaults():
    cfg = DegradationLabels(
        ylabels={"Vth": "Threshold [mV]"},               # override one
        titles={"average": "{label}"},                    # override one family
    )
    assert cfg.ylabel("Vth") == "Threshold [mV]"          # overridden
    assert cfg.ylabel("Ion") == "Ion (nA)"                # default kept
    assert cfg.title("Ion (nA)", "average", 0) == "Ion (nA)"
    assert cfg.title("Ion", "devices", 0) == "Ion — per device"   # default family kept


def test_hidden_metrics():
    cfg = DegradationLabels(hidden=["Vth_d2", "SNR_off"])
    assert cfg.is_hidden("Vth_d2")
    assert not cfg.is_hidden("Vth")


def test_load_creates_file_when_missing_then_round_trips(tmp_path):
    p = tmp_path / "labels.json"
    assert not p.exists()

    cfg = load_degradation_labels(p)          # first call writes defaults
    assert p.is_file()
    assert cfg.ylabel("Vth") == r"$\mathbf{V_{Th}}$ (V)"

    # a hand-edit is picked up on the next load, defaults still fill the gaps
    data = json.loads(p.read_text(encoding="utf-8"))
    data["ylabels"]["Vth"] = "MY LABEL"
    data["hidden"] = ["R2"]
    p.write_text(json.dumps(data), encoding="utf-8")

    cfg2 = load_degradation_labels(p)
    assert cfg2.ylabel("Vth") == "MY LABEL"
    assert cfg2.ylabel("Ion") == "Ion (nA)"   # untouched default
    assert cfg2.is_hidden("R2")


def test_load_falls_back_on_corrupt_file(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    cfg = load_degradation_labels(p)          # must not raise
    assert cfg.ylabel("Vth") == r"$\mathbf{V_{Th}}$ (V)"     # defaults
