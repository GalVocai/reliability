"""Single source of truth for *true* constants and conventions.

The experiment timeline is NOT here -- it is read from the data (the ``Part``
column + ``Time``, and the ``Description`` sheet). What lives here is only:
  * structural patterns (how columns/keys are named),
  * genuine analysis parameters (windows, thresholds),
  * the Part->phase role mapping for the fixed SE protocol (named in one place,
    validated against the data at runtime so a protocol change fails loud).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# --- structural patterns -----------------------------------------------------

DEVICE_RE = re.compile(r"^Id\d+$")          # Id00 .. Id39 (dynamic count!)
ESN_KEY_RE = re.compile(r"^ESN(\d+)$")       # ESN00 .. in Description
TNUMBER_KEY_RE = re.compile(r"^TNumber(\d+)$")
MAJOR_RE = re.compile(r"_n(\d+)_")           # ..._n04_... in result filenames

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
INVALID_SHEET_CHARS = re.compile(r"[\[\]\:\*\?\/\\]")

ALLOWED_EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")

DATA_SHEET = "Data"
DESCRIPTION_SHEET = "Description"


# --- analysis parameters -----------------------------------------------------

@dataclass(frozen=True)
class SEProtocol:
    """Sensing (SE_STATE) analysis parameters."""

    # Trailing seconds averaged to define a stabilized level within a part.
    window_s: float = 20.0

    # Part->phase roles for the 4-part main sensing run (baseline/exposure/
    # settle/recovery). Named here, ONE place; cross-checked against the gas
    # log at runtime so "the order changed" fails loud instead of silent.
    baseline_part: int = 0    # air baseline
    response_part: int = 1    # H2 exposure (clean, before the JG sweep)
    settle_part: int = 2      # post-sweep settle (capacitance offset lives here)
    recovery_part: int = 3    # recovery in air

    # Reject samples > this many MADs from the window median before averaging,
    # so a single measurement spike can never distort a level.
    outlier_mad_k: float = 6.0


@dataclass(frozen=True)
class JGProtocol:
    """JG / JGH sweep parameters."""

    sweep_close_vj: float = 0.3          # a sweep block closes when Vj returns to 0.3
    sweep_close_tol: float = 1e-6
    noise_floor_nA: float = 1.0
    vth_target_nA: float = 10.0
    relevant_rep_is_last: bool = True    # of the 3 JG reps, the last Part is used


@dataclass(frozen=True)
class Protocol:
    se: SEProtocol = SEProtocol()
    jg: JGProtocol = JGProtocol()


DEFAULT = Protocol()
