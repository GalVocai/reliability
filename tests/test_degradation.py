"""Unit test for the degradation math (delta, retention, mean/sample SD)."""

import math

import pandas as pd

from sensorlab.degradation import aggregate_tidy, compute_tidy


def _long():
    rows = [
        # ESN, TNumber, condition, day, day_num, metric, value
        (1, 11, "A", "baseline", 0.0, "Vth", -1.0),
        (1, 11, "A", "7d", 7.0, "Vth", -0.9),
        (1, 12, "A", "baseline", 0.0, "Vth", -2.0),
        (1, 12, "A", "7d", 7.0, "Vth", -1.8),
    ]
    return pd.DataFrame(rows, columns=["ESN", "TNumber", "condition", "day", "day_num", "metric", "value"])


def test_delta_and_degradation():
    # delta = baseline - value (positive = the metric dropped from baseline)
    tidy = compute_tidy(_long(), ["ESN", "TNumber"], baseline_num=0.0)
    d7 = tidy[tidy["day_num"] == 7.0].set_index("TNumber")
    assert math.isclose(d7.loc[11, "delta"], -0.1, abs_tol=1e-9)             # -1.0 - (-0.9)
    assert math.isclose(d7.loc[12, "delta"], -0.2, abs_tol=1e-9)             # -2.0 - (-1.8)
    assert math.isclose(d7.loc[11, "degradation_pct"], 10.0, abs_tol=1e-9)   # -0.1 / -1.0
    assert math.isclose(d7.loc[12, "degradation_pct"], 10.0, abs_tol=1e-9)   # -0.2 / -2.0


def test_aggregate_mean_avdev():
    tidy = compute_tidy(_long(), ["ESN", "TNumber"], baseline_num=0.0)
    agg = aggregate_tidy(tidy, ["ESN", "TNumber"])
    row = agg[(agg["day_num"] == 7.0) & (agg["metric"] == "Vth")].iloc[0]
    assert row["n"] == 2
    assert math.isclose(row["deg_mean"], 10.0, abs_tol=1e-9)
    assert math.isclose(row["deg_avdev"], 0.0, abs_tol=1e-9)
    assert math.isclose(row["value_mean"], -1.35, abs_tol=1e-9)              # (-0.9 + -1.8)/2
    assert math.isclose(row["value_avdev"], 0.45, abs_tol=1e-9)
    assert math.isclose(row["value_sd"], 0.45 * math.sqrt(2), abs_tol=1e-9)
    assert math.isclose(row["delta_sd"], 0.05 * math.sqrt(2), abs_tol=1e-9)
    assert math.isclose(row["deg_sd"], 0.0, abs_tol=1e-9)


def test_aggregate_include_devices():
    tidy = compute_tidy(_long(), ["ESN", "TNumber"], baseline_num=0.0)
    agg = aggregate_tidy(tidy, ["ESN", "TNumber"], include_devices={("1", "11")})
    row = agg[(agg["day_num"] == 7.0) & (agg["metric"] == "Vth")].iloc[0]
    assert row["n"] == 1                                          # only the included device
    assert math.isclose(row["value_mean"], -0.9, abs_tol=1e-9)   # T11's day-7 value only
    assert math.isnan(row["value_sd"])
