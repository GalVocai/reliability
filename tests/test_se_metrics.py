"""Unit test for the SE settle-correction math on a synthetic 4-part run."""

import pandas as pd

from sensorlab.measurements.se import analyze_se_run
from sensorlab.model import Channel, MajorKind, Run


def _make_run() -> Run:
    # 4 parts; each part's samples all fall inside the 20 s trailing window.
    rows = []
    for part, (t0, level) in enumerate([(0, 100.0), (30, 200.0), (60, 220.0), (90, 150.0)]):
        for k in range(5):
            rows.append({"Time": t0 + k, "Part": part, "Id00": level})
    data = pd.DataFrame(rows)
    ch = Channel(column="Id00", index=0, esn="1074", tnumber="11")
    return Run(path="synthetic", major=4, kind=MajorKind.SE, test_type="SE_STATE",
               data=data, channels=[ch], description={})


def test_se_correction_levels_and_percentages():
    df = analyze_se_run(_make_run())
    r = df.iloc[0]
    assert r["baseline"] == 100.0
    assert r["response"] == 200.0
    assert r["settle_raw"] == 220.0
    assert r["shift"] == -20.0                 # response - settle
    assert r["settle_corr"] == 200.0           # corrected onto response
    assert r["recovery_corr"] == 130.0         # 150 + (-20)
    assert r["response_pct"] == 100.0          # (200-100)/100
    assert r["recovery_pct"] == 70.0           # (200-130)/(200-100)
