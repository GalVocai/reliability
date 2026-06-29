"""Typed objects parsed from the self-describing result files."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .protocol import DEVICE_RE


class MajorKind(str, Enum):
    SE = "SE"        # sensing (Id vs time at fixed bias)
    ZOZO = "ZOZO"    # SE_STATE alternating between two states
    JG = "JG"        # unidirectional transfer sweep, repeated
    JGH = "JGH"      # bidirectional hysteresis sweep
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Channel:
    """One device channel (e.g. column 'Id07')."""

    column: str          # 'Id07'
    index: int           # 7
    esn: str             # '1232'  (the card)
    tnumber: str         # '32'    (the transistor)

    @property
    def label(self) -> str:
        return f"ESN{self.esn}_T{self.tnumber}"


@dataclass
class Run:
    """One result file = one major (n0X)."""

    path: str
    major: int                       # 4  (from filename _n04_)
    kind: MajorKind
    test_type: str                   # raw 'SE_STATE' / 'JG' / 'JGH'
    data: pd.DataFrame               # Time, Part, Vb, Vj, ... Id00..IdNN
    channels: list[Channel]
    description: dict[str, object]   # raw Description key/value

    @property
    def device_columns(self) -> list[str]:
        return [c.column for c in self.channels]

    def parts(self) -> dict[int, pd.DataFrame]:
        """Split Data by the explicit ``Part`` column (the 'minor' index)."""
        return {int(p): g.reset_index(drop=True) for p, g in self.data.groupby("Part")}


def device_columns_of(df: pd.DataFrame) -> list[str]:
    """All Id columns, in order -- dynamic (40 here, not a hardcoded 4)."""
    return [c for c in df.columns if DEVICE_RE.match(str(c))]
