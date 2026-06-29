"""Unit test for the degradation math (delta, degradation%, mean/std)."""

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
    tidy = compute_tidy(_long(), ["ESN", "TNumber"], baseline_num=0.0)
    d7 = tidy[tidy["day_num"] == 7.0].set_index("TNumber")
    assert math.isclose(d7.loc[11, "delta"], 0.1, abs_tol=1e-9)
    assert math.isclose(d7.loc[12, "delta"], 0.2, abs_tol=1e-9)
    assert math.isclose(d7.loc[11, "degradation_pct"], -10.0, abs_tol=1e-9)  # 0.1 / -1.0
    assert math.isclose(d7.loc[12, "degradation_pct"], -10.0, abs_tol=1e-9)  # 0.2 / -2.0


def test_aggregate_mean_std():
    tidy = compute_tidy(_long(), ["ESN", "TNumber"], baseline_num=0.0)
    agg = aggregate_tidy(tidy, ["ESN", "TNumber"])
    row = agg[(agg["day_num"] == 7.0) & (agg["metric"] == "Vth")].iloc[0]
    assert row["n"] == 2
    assert math.isclose(row["deg_mean"], -10.0, abs_tol=1e-9)
    assert math.isclose(row["deg_std"], 0.0, abs_tol=1e-9)
    assert math.isclose(row["value_mean"], -1.35, abs_tol=1e-9)              # (-0.9 + -1.8)/2
