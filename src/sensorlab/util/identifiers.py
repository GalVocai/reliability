"""Identifier / name normalization helpers.

Ported verbatim from the legacy apps, where `normalize_identifier`,
`sanitize_filename`, `sanitize_sheet_name`, etc. were duplicated across
``excel_device_splitter_gui.py`` and ``se_timeseries_gui.py``. Here they live
exactly once, so the two implementations can never drift apart again.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from ..protocol import INVALID_FILENAME_CHARS, INVALID_SHEET_CHARS


def normalize_identifier(value: object) -> str:
    """Normalize an ESN/identifier value to a clean string (e.g. 727.0 -> '727')."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not math.isfinite(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def normalize_tnumber_value(value: object) -> str:
    """Normalize a TNumber value (same rules as `normalize_identifier`)."""
    text = normalize_identifier(value)
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not math.isfinite(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def normalize_compact_token(value: str) -> str:
    """Lowercase and strip everything except [a-z0-9] -- for fuzzy column matching."""
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def sanitize_filename(name: str) -> str:
    safe = INVALID_FILENAME_CHARS.sub("_", name).strip().strip(".")
    return safe or "UNKNOWN_ESN"


def sanitize_sheet_name(name: str) -> str:
    safe = INVALID_SHEET_CHARS.sub("_", name).strip()
    if not safe:
        safe = "Sheet"
    return safe[:31]


def unique_sheet_name(base_name: str, existing_names: set[str]) -> str:
    base = sanitize_sheet_name(base_name)
    candidate = base
    index = 1
    while candidate in existing_names:
        suffix = f"_{index}"
        max_base_len = 31 - len(suffix)
        candidate = f"{base[:max_base_len]}{suffix}"
        index += 1
    existing_names.add(candidate)
    return candidate


def find_column_case_insensitive(columns: list[object], target: str) -> Optional[object]:
    target_lower = target.strip().lower()
    for column in columns:
        if str(column).strip().lower() == target_lower:
            return column
    return None


def find_column_by_normalized_prefix(columns: list[object], prefix: str) -> Optional[object]:
    prefix_norm = normalize_compact_token(prefix)
    for column in columns:
        column_norm = normalize_compact_token(str(column))
        if column_norm.startswith(prefix_norm):
            return column
    return None
