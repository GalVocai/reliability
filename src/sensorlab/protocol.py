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

    # Quality-flag thresholds (surface dubious cases, don't silently report).
    settle_drift_frac: float = 0.02   # settle still moving > 2% of response span -> not stabilized
    recovery_pct_min: float = 0.0     # flag recovery outside this range
    recovery_pct_max: float = 100.0
    low_baseline_nA: float = 1000.0   # flag very low baseline current (noisy %)

    # ppm-staircase run (n07): concentration per Part. NOT recorded in the data
    # (the gas log only flags the 1000 ppm step), so the fixed protocol lives
    # here. P0 = air baseline, last 0 = recovery. Edit if the staircase changes.
    ppm_steps: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0, 1000.0, 0.0)


@dataclass(frozen=True)
class JGProtocol:
    """JG / JGH sweep parameters."""

    sweep_close_vj: float = 0.3          # a sweep block closes when Vj returns to 0.3
    sweep_close_tol: float = 1e-6
    noise_floor_nA: float = 1.0
    # Constant-current Vth target (all modes). 100 nA sits clearly above the
    # off-noise (which swings up to ~60-120 nA) yet still catches the abrupt
    # turn-on. Paired with an onset guard so noise spikes can't trigger it.
    vth_target_nA: float = 100.0
    vth_onset_guard: int = 3             # crossing must stay above target for N points
    relevant_rep_is_last: bool = True    # of the 3 JG reps, the last Part is used

    # SS is fit over the log-linear turn-on: Id in (noise_floor, ss_upper_frac*Ion).
    # (The textbook 1-10 nA band is too narrow for these steep devices.)
    ss_upper_frac: float = 0.05
    # Flag a device as leaky / degraded when on/off ratio is below this.
    min_on_off: float = 100.0


@dataclass(frozen=True)
class Protocol:
    se: SEProtocol = SEProtocol()
    jg: JGProtocol = JGProtocol()


DEFAULT = Protocol()
