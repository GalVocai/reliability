"""Unit test for the ZOZO single-exponential settle fit."""

import math

import numpy as np

from sensorlab.measurements.zozo import fit_segment_single


def test_single_exp_recovers_parameters():
    t = np.linspace(0, 100, 200)
    y = 100.0 + 50.0 * np.exp(-t / 10.0)        # DC=100, A=50, tau=10
    f = fit_segment_single({"time_local": t, "current": y})
    assert math.isclose(f["level"], 100.0, abs_tol=0.5)
    assert math.isclose(f["A"], 50.0, abs_tol=0.5)
    assert math.isclose(f["tau"], 10.0, abs_tol=0.2)
    assert f["R2"] > 0.999
