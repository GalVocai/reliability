"""Unit test for JG transfer-curve metrics on a synthetic monotonic sweep."""

import math

import numpy as np

from sensorlab.measurements.jg import jg_curve_metrics
from sensorlab.protocol import JGProtocol, Protocol

PROTO = Protocol(jg=JGProtocol(vth_target_nA=10.0))   # explicit target, default-independent


def test_jg_curve_metrics_basic():
    vj = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    idv = np.array([0.5, 2.0, 20.0, 200.0, 5000.0, 50000.0])
    m = jg_curve_metrics(vj, idv, PROTO)
    assert m["Ion"] == 50000.0                 # on end
    assert m["Ioff"] == 0.5                     # off end
    assert m["Ion_Ioff"] == 50000.0            # Ioff floored at noise (1.0)
    # Vth: interpolate Vj where Id crosses 10 nA, between (-1.5, 2) and (-1.0, 20)
    assert math.isclose(m["Vth"], -1.2778, abs_tol=1e-3)
    # SS over band (1, 0.05*Ion=2500): pts Id=2,20,200 -> 500 mV/dec
    assert math.isclose(m["SS"], 500.0, abs_tol=1.0)
