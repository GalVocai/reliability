"""Unit tests for the device population grouping (measurements/population.py)."""

import pandas as pd

from sensorlab.measurements.population import classify_se, classify_steps


def _se_master() -> pd.DataFrame:
    """9 devices, one condition: 5 normal + one of each anomaly."""
    rows = [
        # ESN, T, baseline, resp_pct, resp_delta, recovery_pct
        ("1", "11", 10000.0, 10.0, 1000.0, 70.0),   # normal
        ("1", "12", 12000.0, 10.5, 1260.0, 72.0),   # normal
        ("2", "11", 11000.0,  9.5, 1045.0, 68.0),   # normal
        ("2", "12", 13000.0, 10.2, 1326.0, 71.0),   # normal
        ("3", "11", 10500.0,  9.8, 1029.0, 69.0),   # normal
        ("4", "11",   200.0, 90.0,  180.0, 70.0),   # dead baseline (tiny nA)
        ("4", "12", 11500.0, -9.9, -1138.0, 65.0),  # responds DOWN vs group
        ("5", "11", 12500.0, 10.1, 1262.0, -6.0),   # rises after gas off
        ("5", "12", 11800.0, 95.0, 11210.0, 70.0),  # extreme magnitude
    ]
    return pd.DataFrame([{
        "ESN": e, "TNumber": t, "condition": "A", "baseline": b,
        "response_pct": rp, "response_delta": rd, "recovery_pct": rec,
    } for e, t, b, rp, rd, rec in rows])


def test_classify_se_groups():
    df, summary = classify_se(_se_master())
    got = dict(zip(zip(df["ESN"], df["TNumber"]), df["group"]))
    assert got[("1", "11")] == "normal"
    assert got[("4", "11")] == "dead_baseline"
    assert got[("4", "12")] == "opposite_direction"
    assert got[("5", "11")] == "no_recovery"
    assert got[("5", "12")] == "outlier_high"
    # summary explains the call: z of the extreme device is far past threshold
    s = summary.set_index(["ESN", "TNumber"])
    assert s.loc[("5", "12"), "z_ref"] > s.loc[("5", "12"), "z_thresh"]
    assert s.loc[("4", "12"), "direction"] == "down"


def test_classify_se_missing_columns_is_noop():
    df = pd.DataFrame({"ESN": ["1"], "TNumber": ["11"], "P0_level": [5.0]})
    out, summary = classify_se(df)
    assert "group" not in out.columns
    assert summary.empty


def test_classify_steps_no_recovery_rules():
    base = {
        "condition": "A", "baseline": 10000.0,
        "1000ppm_resp_pct": 50.0, "1000ppm_delta": 5000.0,
    }
    df = pd.DataFrame([
        {"ESN": "1", "TNumber": "11", **base, "recovery_pct": 5.0},    # normal
        {"ESN": "1", "TNumber": "12", **base, "recovery_pct": -3.0},   # overshoot
        {"ESN": "2", "TNumber": "11", **base, "recovery_pct": 60.0},   # residual > response
        {"ESN": "2", "TNumber": "12", **base, "recovery_pct": 6.0},    # normal
    ])
    out, _ = classify_steps(df)
    assert list(out["group"]) == ["normal", "no_recovery", "no_recovery", "normal"]
