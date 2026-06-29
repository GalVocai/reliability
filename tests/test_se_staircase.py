"""Unit test for the standalone ppm-staircase (STEPS) mode."""

import pandas as pd

from sensorlab.measurements.steps import analyze_steps_run
from sensorlab.model import Channel, MajorKind, Run


def _staircase_run() -> Run:
    # 7 parts; levels chosen for known responses vs baseline=100
    levels = [100.0, 110.0, 125.0, 150.0, 200.0, 300.0, 120.0]   # ppm 0,10,25,50,100,1000,0
    rows = []
    for part, lvl in enumerate(levels):
        for k in range(5):
            rows.append({"Time": part * 100 + k, "Part": part, "Vj": -0.58, "Id00": lvl})
    ch = Channel(column="Id00", index=0, esn="1", tnumber="11")
    return Run(path="syn", major=7, kind=MajorKind.STEPS, test_type="SE_STATE",
               data=pd.DataFrame(rows), channels=[ch], description={})


def test_staircase_response_per_step():
    r = analyze_steps_run(_staircase_run()).iloc[0]
    assert r["baseline"] == 100.0
    assert r["10ppm_resp_pct"] == 10.0
    assert r["25ppm_resp_pct"] == 25.0
    assert r["50ppm_resp_pct"] == 50.0
    assert r["100ppm_resp_pct"] == 100.0
    assert r["1000ppm_resp_pct"] == 200.0
    assert r["recovery_pct"] == 20.0
