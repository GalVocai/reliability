"""Single source of truth for the (fixed) experimental protocol.

Every "magic number" that used to be scattered across
``excel_device_splitter_gui.py`` and ``se_timeseries_gui.py`` lives here, named
and documented. The lab protocol is fixed, so these values rarely change -- but
when they do, this is the *one* place to edit, instead of hunting through
thousands of lines in two files.

Design note (the lesson from "the order changed once"):
    Some legacy values address samples by *absolute index* (e.g. recovery
    reference at index 100..120). That silently breaks if the sampling cadence
    changes or a row is dropped during cleaning -- even when the protocol is
    nominally identical. Such fields are marked `LEGACY INDEX` below; the port
    replaces them with physical time windows (see `RECOVERY_REFERENCE_WINDOW_S`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --- Structural patterns (data shape -- fixed) -------------------------------

DEVICE_COLUMN_PATTERN = re.compile(r"^Id\d{2}$")   # Id00, Id01, ...
ESN_KEY_PATTERN = re.compile(r"^ESN(\d{2})$")       # ESN00, ESN01, ... in Description
TIMEPOINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[dD]\s*$")  # "1D", "7D", "12d"

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
INVALID_SHEET_CHARS = re.compile(r"[\[\]\:\*\?\/\\]")

ALLOWED_EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")
DEVICE_COLUMNS = ("Id00", "Id01", "Id02", "Id03")


@dataclass(frozen=True)
class SensingProtocol:
    """SE (hydrogen sensing) timeline parameters."""

    # Physical anchor: the gas valve opens this many seconds after the H2
    # recording starts. Baseline is measured just before this moment.
    h2_exposure_start_s: float = 900.0

    # Window aggregations (counts of trailing samples) -- cadence-independent
    # because they are "last N points", not "index N".
    baseline_points: int = 20
    response_points: int = 20
    recovery_points: int = 20

    # Response/recovery threshold for T90 (90% of the total step).
    response_threshold_pct: float = 90.0

    # --- recovery reference: PORT TARGET ------------------------------------
    # LEGACY INDEX (fragile): recovery reference was iloc[100:120].
    # Preferred physical equivalent: a time window measured from recovery start.
    # Fill RECOVERY_REFERENCE_WINDOW_S once you confirm the real-time span that
    # indices 100..120 corresponded to for the fixed cadence, then the metric
    # code uses the window instead of the index.
    recovery_reference_start_index: int = 100   # LEGACY INDEX
    recovery_reference_points: int = 20         # LEGACY INDEX
    recovery_air_start_index: int = 120         # LEGACY INDEX
    recovery_reference_window_s: tuple[float, float] | None = None  # e.g. (0.0, 20.0)

    # Sanity bounds -- raise (don't silently miscompute) if data disagrees with
    # the fixed protocol. Tune once against a known-good dataset.
    h2_exposure_start_bounds_s: tuple[float, float] = (850.0, 950.0)


@dataclass(frozen=True)
class JGProtocol:
    """JG sweep parameters."""

    sweep_close_vj: float = 0.3          # a sweep block closes when Vj returns to 0.3
    sweep_close_tol: float = 1e-6
    noise_floor_nA: float = 1.0
    y_axis_min_nA: float = 1e-1
    vth_target_nA: float = 10.0          # constant-current Vth extraction target


@dataclass(frozen=True)
class Protocol:
    """Top-level fixed protocol. Import `DEFAULT` for the current lab setup."""

    sensing: SensingProtocol = field(default_factory=SensingProtocol)
    jg: JGProtocol = field(default_factory=JGProtocol)
    device_columns: tuple[str, ...] = DEVICE_COLUMNS


# The one instance the whole pipeline reads from.
DEFAULT = Protocol()
