"""Smoke tests for the shared identifier helpers -- proves the harness runs."""

from sensorlab.util import (
    normalize_identifier,
    normalize_tnumber_value,
    sanitize_sheet_name,
    unique_sheet_name,
)


def test_normalize_identifier_strips_float_zeros():
    assert normalize_identifier(727.0) == "727"
    assert normalize_identifier("727.0") == "727"
    assert normalize_identifier(" 3 ") == "3"


def test_normalize_identifier_handles_empty_and_nan():
    assert normalize_identifier(None) == ""
    assert normalize_identifier(float("nan")) == ""
    assert normalize_identifier("") == ""


def test_normalize_tnumber_keeps_nonnumeric_text():
    assert normalize_tnumber_value("T-12A") == "T-12A"
    assert normalize_tnumber_value(3.0) == "3"


def test_sheet_name_sanitized_and_truncated():
    assert sanitize_sheet_name("ESN727/Tnum:3") == "ESN727_Tnum_3"
    assert len(sanitize_sheet_name("x" * 50)) == 31


def test_unique_sheet_name_dedupes():
    used: set[str] = set()
    a = unique_sheet_name("Data", used)
    b = unique_sheet_name("Data", used)
    assert a == "Data"
    assert b != a and b.startswith("Data")
