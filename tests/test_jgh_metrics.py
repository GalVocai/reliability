"""Unit test for JGH hysteresis metrics on a synthetic forward+reverse loop."""

import math

import numpy as np

from sensorlab.measurements.jgh import jgh_curve_metrics


def test_jgh_hysteresis():
    # forward sweep -2 -> 0.3, then reverse 0.3 -> -2 (shifted -> hysteresis)
    vj_f = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    id_f = np.array([1.0, 5.0, 10.0, 1000.0, 10000.0, 50000.0])
    vj_r = np.array([0.3, 0.0, -0.5, -1.0, -1.5, -2.0])
    id_r = np.array([50000.0, 40000.0, 20000.0, 5000.0, 10.0, 5.0])
    vj = np.concatenate([vj_f, vj_r])
    idv = np.concatenate([id_f, id_r])

    m = jgh_curve_metrics(vj, idv)
    assert math.isclose(m["Vth_fwd"], -1.0, abs_tol=1e-3)
    assert math.isclose(m["Vth_rev"], -1.5, abs_tol=1e-3)
    assert math.isclose(m["dVth"], -0.5, abs_tol=1e-3)
    assert m["Ion"] == 50000.0
    assert m["loop_area"] > 0          # reverse sits above forward
