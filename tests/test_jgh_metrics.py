"""Unit test for JGH hysteresis metrics on a synthetic forward+reverse loop."""

import math

import numpy as np
import pandas as pd

from sensorlab.measurements.jgh import analyze_jgh_run, jgh_curve_metrics
from sensorlab.model import Channel, MajorKind, Run
from sensorlab.protocol import JGProtocol, Protocol

PROTO = Protocol(jg=JGProtocol(vth_target_nA=10.0))   # explicit target, default-independent


def test_jgh_hysteresis():
    # forward sweep -2 -> 0.3, then reverse 0.3 -> -2 (shifted -> hysteresis)
    vj_f = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3])
    id_f = np.array([1.0, 5.0, 10.0, 1000.0, 10000.0, 50000.0])
    vj_r = np.array([0.3, 0.0, -0.5, -1.0, -1.5, -2.0])
    id_r = np.array([50000.0, 40000.0, 20000.0, 5000.0, 10.0, 5.0])
    vj = np.concatenate([vj_f, vj_r])
    idv = np.concatenate([id_f, id_r])

    m = jgh_curve_metrics(vj, idv, PROTO)
    assert math.isclose(m["Vth_FWD"], -1.0, abs_tol=1e-3)
    assert math.isclose(m["Vth_BWD"], -1.5, abs_tol=1e-3)
    assert math.isclose(m["Delta_Vth"], 0.5, abs_tol=1e-3)   # fwd - rev
    assert m["Ion"] == 50000.0
    assert m["loop_area"] > 0          # area-based hysteresis
    assert 0.0 < m["HI"] <= 1.0        # normalized loop area


def _loop_rows(vb: float) -> list[dict]:
    vj_f = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.3]
    id_f = [1.0, 5.0, 10.0, 1000.0, 10000.0, 50000.0]
    vj_r = [0.3, 0.0, -0.5, -1.0, -1.5, -2.0]
    id_r = [50000.0, 40000.0, 20000.0, 5000.0, 10.0, 5.0]
    return [{"Part": 0, "Vb": vb, "Vj": v, "Id00": i}
            for v, i in zip(vj_f + vj_r, id_f + id_r)]


def test_analyze_jgh_run_skips_and_trims_reported_columns():
    """JGH master: bg -4/2 excluded (protocol.jg.jgh_skip_bg), and only
    Delta_Vth/loop_area/HI reported per remaining back-gate -- Vth_FWD/BWD/Ion
    belong to the JG mode's master, not duplicated here."""
    rows = []
    for vb in (-4.0, 0.0, 2.0, 4.0):
        rows += _loop_rows(vb)
    ch = Channel(column="Id00", index=0, esn="1", tnumber="11")
    run = Run(path="syn", major=2, kind=MajorKind.JGH, test_type="JGH",
             data=pd.DataFrame(rows), channels=[ch], description={})

    df = analyze_jgh_run(run)
    cols = set(df.columns)
    assert not any(c.endswith("_bg-4") or c.endswith("_bg2") for c in cols)
    for vb in ("bg0", "bg4"):
        assert {f"Delta_Vth_{vb}", f"loop_area_{vb}", f"HI_{vb}", f"flags_{vb}"} <= cols
        assert f"Vth_FWD_{vb}" not in cols and f"Ion_{vb}" not in cols
