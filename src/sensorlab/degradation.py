"""Cross-timepoint degradation: compare per-device metrics against a baseline.

Operates on the per-timepoint masters (not raw data). Each timepoint's masters
are tagged with a `day`; metrics are melted to a tidy long form, joined to the
baseline day per device, and delta / degradation% are computed. An aggregate
table gives mean +/- std across devices (per condition) for the average trend.
"""

from __future__ import annotations

import os
import re

import pandas as pd

_KEY_CANDIDATES = ["ESN", "TNumber", "Vbg", "segment", "state_Vj"]
_NON_METRIC = {"n_points", "major", "day_num"}
_DAY_RE = re.compile(r"(\d+(?:\.\d+)?)")


def day_to_num(label: object) -> float:
    """'baseline' -> 0, '7d' -> 7, '12D' -> 12, '14' -> 14, else NaN."""
    s = str(label).strip().lower()
    if s in ("", "baseline", "base", "t0", "day0", "0d", "0"):
        return 0.0
    m = _DAY_RE.search(s)
    return float(m.group(1)) if m else float("nan")


def find_masters(folder: str):
    """Yield (condition, mode, path) for every master under folder[/analysis]."""
    root = os.path.join(folder, "analysis")
    if not os.path.isdir(root):
        root = folder
    if not os.path.isdir(root):
        return
    for cond in sorted(os.listdir(root)):
        cdir = os.path.join(root, cond)
        if not os.path.isdir(cdir):
            continue
        for mode in sorted(os.listdir(cdir)):
            path = os.path.join(cdir, mode, f"master_{mode}.xlsx")
            if os.path.isfile(path):
                yield cond, mode, path


def load_timepoint(folder: str, day: object | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """{(condition, sheet): df} for one timepoint, tagged with day / day_num."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for cond, _mode, path in find_masters(folder):
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            d = day if day is not None else (df["day"].iloc[0] if "day" in df.columns and len(df) else "")
            df["day"] = d
            df["day_num"] = day_to_num(d)
            out[(cond, sheet)] = df
    return out


def to_long(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Melt metric columns -> (id..., condition, day, day_num, metric, value)."""
    id_cols = [c for c in _KEY_CANDIDATES if c in df.columns]
    meta = [c for c in ("condition", "day", "day_num") if c in df.columns]
    metric_cols = [
        c for c in df.columns
        if c not in id_cols + meta + list(_NON_METRIC) and pd.api.types.is_numeric_dtype(df[c])
    ]
    long = df.melt(id_vars=id_cols + meta, value_vars=metric_cols, var_name="metric", value_name="value")
    return long, id_cols


def compute_tidy(all_long: pd.DataFrame, id_cols: list[str], baseline_num: float) -> pd.DataFrame:
    """Add baseline_value, delta and degradation_pct (vs the baseline day)."""
    gkey = id_cols + ["metric"]
    base = (all_long[all_long["day_num"] == baseline_num][gkey + ["value"]]
            .rename(columns={"value": "baseline_value"})
            .drop_duplicates(gkey))
    tidy = all_long.merge(base, on=gkey, how="left")
    tidy["delta"] = tidy["value"] - tidy["baseline_value"]
    tidy["degradation_pct"] = tidy["delta"] / tidy["baseline_value"].where(tidy["baseline_value"] != 0) * 100
    return tidy


def aggregate_tidy(tidy: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Mean +/- std across devices, per condition (+ sub-device structure) per day."""
    struct = [c for c in id_cols if c not in ("ESN", "TNumber")]   # keep Vbg/segment/state
    keys = ([c for c in ["condition"] if c in tidy.columns] + struct + ["metric", "day", "day_num"])
    return (tidy.groupby(keys, dropna=False)
            .agg(value_mean=("value", "mean"), value_std=("value", "std"),
                 delta_mean=("delta", "mean"), delta_std=("delta", "std"),
                 deg_mean=("degradation_pct", "mean"), deg_std=("degradation_pct", "std"),
                 n=("value", "count"))
            .reset_index()
            .sort_values(keys).reset_index(drop=True))


def build_degradation(timepoints: list[tuple[object, str]],
                      baseline_day: object | None = None) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]]:
    """timepoints: list of (day_label, folder). Returns {(condition, sheet): (tidy, agg)}."""
    per_key: dict[tuple[str, str], list[pd.DataFrame]] = {}
    id_cols_by_key: dict[tuple[str, str], list[str]] = {}
    for day_label, folder in timepoints:
        for (cond, sheet), df in load_timepoint(folder, day=day_label).items():
            long, idc = to_long(df)
            per_key.setdefault((cond, sheet), []).append(long)
            id_cols_by_key[(cond, sheet)] = idc

    base_num = day_to_num(baseline_day) if baseline_day is not None else None
    results: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key, longs in per_key.items():
        idc = id_cols_by_key[key]
        all_long = pd.concat(longs, ignore_index=True)
        bnum = base_num if base_num is not None else float(all_long["day_num"].min())
        tidy = compute_tidy(all_long, idc, bnum)
        results[key] = (tidy.sort_values(idc + ["metric", "day_num"]).reset_index(drop=True),
                        aggregate_tidy(tidy, idc))
    return results
