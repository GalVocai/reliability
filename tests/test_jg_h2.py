"""Unit test for JG_H2 air-vs-gas comparison metrics."""

import math

import numpy as np

from sensorlab.measurements.jg_h2 import compare_condition


def test_air_vs_h2_metrics():
    vj = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    air = (vj, np.array([1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]))     # Vth(100nA) = -1.0
    h2 = (vj, np.array([2.0, 20.0, 120.0, 1200.0, 6000.0, 11000.0]))      # Vth(100nA) = -1.1
    m = compare_condition(air, h2)
    assert math.isclose(m["Vth_air"], -1.0, abs_tol=1e-3)
    assert math.isclose(m["Vth_gas"], -1.1, abs_tol=1e-3)
    assert math.isclose(m["delta_Vth"], 0.1, abs_tol=1e-3)
    assert math.isclose(m["resp_at_Vth"], 20.0, abs_tol=1.0)              # (120-100)/100
    assert m["max_resp"] > 0 and not math.isnan(m["Vj_max_resp"])
