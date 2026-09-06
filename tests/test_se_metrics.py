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
    assert r["response_delta"] == 100.0        # nA vs baseline
    assert r["settle_raw"] == 220.0
    assert r["shift"] == -20.0                 # response - settle
    assert r["settle_corr"] == 200.0           # corrected onto response
    assert r["recovery_corr"] == 130.0         # 150 + (-20)
    assert r["recovery_delta"] == 30.0         # 130 - 100 (corrected residual)
    assert r["response_pct"] == 100.0          # (200-100)/100
    assert r["recovery_pct"] == 70.0           # (200-130)/(200-100)


def _make_midrise_run() -> Run:
    """Response part starts mid-rise: the first samples already read 50 while
    the baseline part sat at 0. The 90% anchor must be the baseline level (0),
    not the first samples of the response part."""
    rows = []
    for k in range(60):                                   # P0 baseline: flat 0
        rows.append({"Time": float(k), "Part": 0, "Id00": 0.0})
    for k in range(60):                                   # P1: ramp 50->100 in 10 s, then plateau
        y = 50.0 + 5.0 * k if k < 10 else 100.0
        rows.append({"Time": 60.0 + k, "Part": 1, "Id00": y})
    for k in range(60):                                   # P2 settle: flat 100
        rows.append({"Time": 120.0 + k, "Part": 2, "Id00": 100.0})
    for k in range(60):                                   # P3 recovery: instant drop to 10
        rows.append({"Time": 180.0 + k, "Part": 3, "Id00": 10.0})
    ch = Channel(column="Id00", index=0, esn="1074", tnumber="11")
    return Run(path="synthetic", major=4, kind=MajorKind.SE, test_type="SE_STATE",
               data=pd.DataFrame(rows), channels=[ch], description={})


def test_t90_is_anchored_on_previous_part_level():
    r = analyze_se_run(_make_midrise_run()).iloc[0]
    # baseline anchor: target = 0 + 0.9*(100-0) = 90 nA -> reached at k=8 (t=8 s).
    # A first-samples anchor (y0=60) would give target 96 -> t=10 s instead.
    assert r["t90_response_s"] == 8.0
    # recovery drops instantly: target = 100 + 0.9*(10-100) = 19 -> first sample.
    assert r["t90_recovery_s"] == 0.0
    assert "t90_not_reached" not in r["flags"]
