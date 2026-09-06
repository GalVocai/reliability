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


def _toggles_vj(data: pd.DataFrame) -> bool:
    """True when Vj is held at distinct levels across Parts (a ZOZO run).

    Compares per-Part medians rather than raw unique values, so a stray noisy
    Vj sample in a constant-Vj SE run cannot misclassify it as ZOZO.
    """
    if "Vj" not in data.columns:
        return False
    if "Part" in data.columns:
        return data.groupby("Part")["Vj"].median().round(3).nunique() > 1
    return data["Vj"].round(3).nunique() > 1


def _classify(test_type: str, data: pd.DataFrame) -> MajorKind:
    t = test_type.strip().upper()
    if t == "JGH":
        return MajorKind.JGH
    if t == "JG":
        return MajorKind.JG
    if t.startswith("SE"):
        # ZOZO is an SE_STATE run that toggles Vj between distinct levels.
        if _toggles_vj(data):
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


def _is_junk_file(name: str) -> bool:
    """Excel ``~$`` lock files (created while a workbook is open) and hidden
    files match the extension + major-token filters but are not readable."""
    return name.startswith("~$") or name.startswith(".")


def scan_result_files(folder: str) -> list[str]:
    """Result files in a run folder (those carrying an ``_n0X_`` major token)."""
    found: list[str] = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if _is_junk_file(name):
            continue
        if ext in ALLOWED_EXCEL_EXTENSIONS and MAJOR_RE.search(name):
            found.append(os.path.join(folder, name))
    return sorted(found, key=lambda p: _major_from_name(p))


STATES_SHEET = "States"


def load_states(folder: str) -> dict[int, dict[str, tuple[float, float]]]:
    """Per-channel working points from the script file's ``States`` sheet.

    The script .xlsx sits in the results folder (no ``_n0X_`` token in its
    name) and defines, per channel Number and per StateName (State1/State2),
    the setpoints PortJG1 (Vj) and PortBG (Vb).

    Returns {channel index: {"State1": (Vj, Vb), "State2": (Vj, Vb)}} — empty
    dict when no script file is found.
    """
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if _is_junk_file(name) or ext not in ALLOWED_EXCEL_EXTENSIONS or MAJOR_RE.search(name):
            continue
        path = os.path.join(folder, name)
        try:
            xl = pd.ExcelFile(path)
            if STATES_SHEET not in xl.sheet_names:
                continue
            df = xl.parse(STATES_SHEET)
        except Exception:
            continue
        out: dict[int, dict[str, tuple[float, float]]] = {}
        for _, r in df.iterrows():
            try:
                idx = int(r["Number"])
                out.setdefault(idx, {})[str(r["StateName"]).strip()] = (
                    float(r["PortJG1"]), float(r["PortBG"]))
            except (KeyError, TypeError, ValueError):
                continue
        if out:
            return out
    return {}
