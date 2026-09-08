from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sensorlab.plotting.overview_plot import (
    _collect_overview_data,
    generate_overview_plots,
)


def _write_excel(path: Path, frame: pd.DataFrame, sheet: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, sheet_name=sheet, index=False)


def _make_condition(path: Path, days: list[int]) -> None:
    # The aggregate rows already represent the population chosen in the
    # degradation window. Values are deliberately simple but non-zero so both
    # the regular and normalized figures can be generated.
    jg_rows = []
    h2_rows = []
    se_rows = []
    for day in days:
        jg_rows.append({
            "metric": "Vth_bg4", "day_num": day,
            "delta_mean": -0.01 * day, "delta_avdev": 0.001, "n": 2,
        })
        for metric, baseline, rate in (
            ("delta_Vth_H2", 0.12, -0.001),
            ("max_resp_H2_pct", 800.0, -5.0),
            ("delta_Vth_recovery", 0.04, 0.001),
        ):
            h2_rows.append({
                "Vbg": 4, "metric": metric, "day_num": day,
                "value_mean": baseline + rate * day,
                "value_avdev": abs(baseline) * 0.05, "n": 2,
            })
        se_rows.append({
            "metric": "t90_response_s", "day_num": day,
            "value_mean": 1000.0 + 10.0 * day,
            "value_avdev": 50.0, "n": 2,
        })
    _write_excel(path / "JG_n03" / "aggregate.xlsx", pd.DataFrame(jg_rows), "aggregate")
    _write_excel(path / "JG_H2" / "aggregate.xlsx", pd.DataFrame(h2_rows), "aggregate")
    _write_excel(path / "SE_n04" / "aggregate.xlsx", pd.DataFrame(se_rows), "aggregate")

    # Two selected devices reproduce the aggregate. A third, very steep
    # device is present in the tidy table but was unticked in Degradation.
    # The report must recover and preserve that selection.
    selected = (("1", "11"), ("2", "11"))
    all_devices = (*selected, ("9", "11"))
    step_rows = []
    aggregate_rows = []
    for day in days:
        slopes = {
            ("1", "11"): 2.0 - 0.02 * day,
            ("2", "11"): 4.0 - 0.02 * day,
            ("9", "11"): 100.0,
        }
        for ppm in (10, 25, 50, 100):
            metric = f"{ppm}ppm_delta"
            selected_values = []
            for esn, tnumber in all_devices:
                value = slopes[(esn, tnumber)] * ppm + float(int(esn))
                step_rows.append({
                    "ESN": esn, "TNumber": tnumber, "day_num": day,
                    "metric": metric, "value": value,
                })
                if (esn, tnumber) in selected:
                    selected_values.append(value)
            values = pd.Series(selected_values, dtype=float)
            aggregate_rows.append({
                "metric": metric, "day_num": day,
                "value_mean": values.mean(),
                "value_avdev": (values - values.mean()).abs().mean(),
                "n": len(selected_values),
            })
    _write_excel(
        path / "STEPS_n07" / "degradation.xlsx",
        pd.DataFrame(step_rows), "degradation",
    )
    _write_excel(
        path / "STEPS_n07" / "aggregate.xlsx",
        pd.DataFrame(aggregate_rows), "aggregate",
    )


def test_overview_is_regenerated_from_all_current_checkpoints(tmp_path: Path) -> None:
    conditions = {
        "hot_storage": tmp_path / "Hot storage",
        "85c_85rh": tmp_path / "85C_85RH",
        "cycles": tmp_path / "cycles",
    }
    _make_condition(conditions["hot_storage"], [0, 2, 7])
    _make_condition(conditions["85c_85rh"], [0, 2, 7])
    _make_condition(conditions["cycles"], [0, 13, 21])

    data = _collect_overview_data(conditions)

    assert data["stress_delta_vth"]["cycles"]["day_num"].tolist() == [0, 13, 21]
    sensitivity = data["sensitivity"]["cycles"].set_index("day_num")
    assert sensitivity.loc[0, "value_mean"] == pytest.approx(3.0)
    assert sensitivity.loc[21, "value_mean"] == pytest.approx(2.58)
    assert sensitivity.loc[21, "n"] == 2

    generated = generate_overview_plots(conditions, tmp_path / "output")

    assert len(generated) == 11
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
    assert all(path.with_suffix(".pdf").is_file() for path in generated)
    required = tmp_path / "output" / "publication_plots" / "regular" / "stress_induced_deltaVTh.png"
    assert required in generated
