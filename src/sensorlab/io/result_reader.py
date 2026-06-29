"""Read one instrument result .xlsx into a typed :class:`Run`."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..model import Channel, MajorKind, Run, device_columns_of
from ..protocol import (
    ALLOWED_EXCEL_EXTENSIONS,
    DATA_SHEET,
    DEFAULT,
    DESCRIPTION_SHEET,
    ESN_KEY_RE,
    MAJOR_RE,
    TNUMBER_KEY_RE,
)
from ..util import normalize_identifier, normalize_tnumber_value


def _read_description(path: str) -> dict[str, object]:
    """Description sheet is a 2-column key/value block."""
    raw = pd.read_excel(path, sheet_name=DESCRIPTION_SHEET, header=None)
    out: dict[str, object] = {}
    for key, value in zip(raw.iloc[:, 0], raw.iloc[:, 1]):
        if pd.isna(key):
            continue
        out[str(key).strip()] = value
    return out


def _build_channels(data: pd.DataFrame, description: dict[str, object]) -> list[Channel]:
    channels: list[Channel] = []
    for col in device_columns_of(data):
        idx = int(col[2:])                 # 'Id07' -> 7
        esn = normalize_identifier(description.get(f"ESN{idx:02d}"))
        tnum = normalize_tnumber_value(description.get(f"TNumber{idx:02d}"))
        channels.append(Channel(column=col, index=idx, esn=esn, tnumber=tnum))
    return channels


def _classify(test_type: str, data: pd.DataFrame) -> MajorKind:
    t = test_type.strip().upper()
    if t == "JGH":
        return MajorKind.JGH
    if t == "JG":
        return MajorKind.JG
    if t.startswith("SE"):
        # ZOZO is an SE_STATE run that toggles Vj between distinct levels.
        if "Vj" in data.columns and data["Vj"].round(3).nunique() > 1:
            return MajorKind.ZOZO
        # constant-Vj SE: 4 parts = main sensing; ppm-staircase length = STEPS.
        nparts = data["Part"].nunique() if "Part" in data.columns else 1
        if nparts == len(DEFAULT.se.ppm_steps):
            return MajorKind.STEPS
        return MajorKind.SE
    return MajorKind.UNKNOWN


def _major_from_name(path: str) -> int:
    m = MAJOR_RE.search(Path(path).name)
    return int(m.group(1)) if m else -1


def read_run(path: str) -> Run:
    data = pd.read_excel(path, sheet_name=DATA_SHEET, header=0)
    data.columns = [str(c).strip() for c in data.columns]
    description = _read_description(path)
    test_type = str(description.get("TestType", "")).strip()
    if not test_type:
        # fall back to filename token, e.g. ..._n02_JGH_
        name = Path(path).stem.upper()
        test_type = "JGH" if "JGH" in name else "JG" if "_JG_" in name else "SE_STATE"
    channels = _build_channels(data, description)
    return Run(
        path=path,
        major=_major_from_name(path),
        kind=_classify(test_type, data),
        test_type=test_type,
        data=data,
        channels=channels,
        description=description,
    )


def scan_result_files(folder: str) -> list[str]:
    """Result files in a run folder (those carrying an ``_n0X_`` major token)."""
    found: list[str] = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXCEL_EXTENSIONS and MAJOR_RE.search(name):
            found.append(os.path.join(folder, name))
    return sorted(found, key=lambda p: _major_from_name(p))
