"""Optional signed DC correction for JG transfer sweeps.

The calibration JSON is an intermediate generated from blank-card sweeps.  Its
device map refers to physical calibration cards/channels, which need not match
the Id column number in a results workbook.  No correction is applied unless
the caller explicitly supplies a calibration to the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .model import MajorKind, Run
from .util import normalize_identifier, normalize_tnumber_value


@dataclass(frozen=True)
class CalibrationOffset:
    card_id: str
    channel: str
    vbg: float | None
    mean_nA: float
    sd_nA: float | None = None
    n: int | None = None


@dataclass(frozen=True)
class DeviceAssignment:
    esn: str
    tnumber: str
    card_id: str
    channel: str


class JgCalibration:
    """Validated per-card/channel/back-gate blank-current means."""

    def __init__(self, offsets: list[CalibrationOffset],
                 assignments: list[DeviceAssignment], source: str = "",
                 reference_conditions: dict | None = None) -> None:
        if not offsets or not assignments:
            raise ValueError("Calibration must contain offsets and device assignments")
        self.source = source
        self.reference_conditions = reference_conditions or {}
        self.offset_records = list(offsets)
        self.offsets: dict[tuple[str, str, float | None], float] = {}
        self.assignments: dict[tuple[str, str], DeviceAssignment] = {}
        for row in offsets:
            key = (row.card_id, row.channel,
                   round(row.vbg, 2) if row.vbg is not None else None)
            if key in self.offsets or not np.isfinite(row.mean_nA):
                raise ValueError(f"Duplicate or non-finite calibration offset: {key}")
            self.offsets[key] = float(row.mean_nA)
        for row in assignments:
            key = (row.esn, row.tnumber)
            if key in self.assignments:
                raise ValueError(f"Duplicate device assignment: {key}")
            self.assignments[key] = row

    @classmethod
    def from_json(cls, path: str | Path) -> JgCalibration:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"Unsupported calibration JSON version in {source}")
        try:
            offsets = [CalibrationOffset(
                card_id=str(row["card_id"]).strip(),
                channel=str(row["channel"]).strip(),
                vbg=float(row["vbg"]) if row.get("vbg") is not None else None,
                mean_nA=float(row["mean_nA"]),
                sd_nA=float(row["sd_nA"]) if row.get("sd_nA") is not None else None,
                n=int(row["n"]) if row.get("n") is not None else None,
            ) for row in data["offsets"]]
            assignments = [DeviceAssignment(
                esn=str(row["esn"]).strip(),
                tnumber=str(row["tnumber"]).strip(),
                card_id=str(row["card_id"]).strip(),
                channel=str(row["channel"]).strip(),
            ) for row in data["assignments"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid calibration JSON: {source}") from exc
        if any(not x.card_id or not x.channel for x in offsets):
            raise ValueError("Calibration offsets need a card_id and channel")
        if any(not x.esn or not x.tnumber or not x.card_id or not x.channel
               for x in assignments):
            raise ValueError("Every device assignment needs ESN, TNumber, card_id and channel")
        return cls(offsets, assignments, source=str(data.get("source") or source),
                   reference_conditions=data.get("reference_conditions"))

    def offset_for(self, esn: str, tnumber: str, vbg: float) -> float:
        device = (str(esn), str(tnumber))
        try:
            assigned = self.assignments[device]
        except KeyError as exc:
            raise ValueError(f"No JG calibration card/channel assignment for {device}") from exc
        key = (assigned.card_id, assigned.channel, round(float(vbg), 2))
        if key in self.offsets:
            return self.offsets[key]
        gate_independent_key = (assigned.card_id, assigned.channel, None)
        if gate_independent_key in self.offsets:
            return self.offsets[gate_independent_key]
        raise ValueError(f"No JG calibration offset for {key} (device {device})")

    def as_json_dict(self) -> dict:
        """Portable snapshot used to reproduce a corrected analysis later."""
        return {
            "version": 1,
            "source": self.source,
            "reference_conditions": self.reference_conditions,
            "offsets": [
                {"card_id": row.card_id, "channel": row.channel, "vbg": row.vbg,
                 "mean_nA": row.mean_nA, "sd_nA": row.sd_nA, "n": row.n}
                for row in sorted(self.offset_records,
                                  key=lambda item: (item.card_id, item.channel,
                                                    float("-inf") if item.vbg is None
                                                    else item.vbg))
            ],
            "assignments": [
                {"esn": row.esn, "tnumber": row.tnumber,
                 "card_id": row.card_id, "channel": row.channel}
                for _, row in sorted(self.assignments.items())
            ],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_json_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")


def apply_jg_calibration(run: Run, calibration: JgCalibration) -> Run:
    """Return a corrected JG run; never mutate the raw workbook or input Run.

    For every row, I_corrected = I_measured - mean_blank_current.  Offsets are
    validated for *all* channels and back gates before any data is changed.
    All JG parts are corrected so later plots and Part-2 metrics use the same
    current convention.
    """
    if run.kind != MajorKind.JG:
        return run
    gates = run.data["Vb"].round(2)
    unique_gates = sorted(float(v) for v in gates.dropna().unique())
    lookups = {(ch.column, vbg): calibration.offset_for(ch.esn, ch.tnumber, vbg)
               for ch in run.channels for vbg in unique_gates}
    corrected = run.data.copy()
    for (column, vbg), offset in lookups.items():
        rows = gates == vbg
        corrected.loc[rows, column] = pd.to_numeric(
            corrected.loc[rows, column], errors="coerce") - offset
    return replace(run, data=corrected, _parts_cache=None)


def _mapping_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    table = (pd.read_csv(source) if source.suffix.lower() == ".csv"
             else pd.read_excel(source))
    columns = {str(column).strip().lower(): column for column in table.columns}
    required = ("esn", "tnumber", "card_id")
    if any(name not in columns for name in required):
        raise ValueError("Mapping needs ESN, TNumber and card_id columns")
    chosen = table.rename(columns={original: name for name, original in columns.items()})
    if "calibration_channel" not in chosen:
        chosen["calibration_channel"] = None
    chosen["esn"] = chosen["esn"].map(normalize_identifier)
    chosen["tnumber"] = chosen["tnumber"].map(normalize_tnumber_value)
    chosen["card_id"] = chosen["card_id"].map(normalize_identifier)
    return chosen[list(required) + ["calibration_channel"]]


def import_blank_sweeps(
    workbooks: list[str | Path], mapping_file: str | Path | None = None,
    *, part: int = 2,
) -> JgCalibration:
    """Build signed DC offsets from blank sweeps or fixed-state records.

    When the workbook Description names the actual device ESN and TNumber,
    omit mapping_file. An optional mapping file supports earlier records that
    identify physical calibration cards instead. It has ESN, TNumber, card_id
    and optionally calibration_channel (Id00, Id01, ...).
    A fixed-state record yields a gate-independent DC offset. Its measured
    Vb and Vj are recorded in the JSON for traceability.
    """
    if not workbooks:
        raise ValueError("Select at least one blank-card sweep workbook")
    mapping = _mapping_table(mapping_file) if mapping_file is not None else None
    expected_cards = set(mapping["card_id"]) if mapping is not None else None
    # Pool repeated blank measurements by physical card/channel/back gate.
    values: dict[tuple[str, str, float | None], list[np.ndarray]] = {}
    channel_tnumbers: dict[tuple[str, str], str] = {}
    direct_assignments: dict[tuple[str, str], DeviceAssignment] = {}
    reference_conditions: dict = {}
    import_mode: str | None = None
    for workbook in map(Path, workbooks):
        description_raw = pd.read_excel(workbook, sheet_name="Description", header=None)
        description = {str(k).strip(): v for k, v in
                       zip(description_raw.iloc[:, 0], description_raw.iloc[:, 1])
                       if pd.notna(k)}
        data = pd.read_excel(workbook, sheet_name="Data")
        if "Part" not in data or "Vb" not in data:
            raise ValueError(f"Blank workbook lacks Part or Vb: {workbook}")
        parts = pd.to_numeric(data["Part"], errors="coerce")
        if (parts == part).any():
            mode = "JG_sweep"
            data = data.loc[parts == part]
            conditions = {"type": mode, "part": part}
        elif set(parts.dropna().unique()) == {0} and "Vj" in data:
            gates = pd.to_numeric(data["Vb"], errors="coerce").dropna().unique()
            vjs = pd.to_numeric(data["Vj"], errors="coerce").dropna().unique()
            if len(vjs) > 1:
                mode = "JG_blank_sweep"
                conditions = {"type": mode, "part": 0,
                              "Vj_min_V": float(min(vjs)),
                              "Vj_max_V": float(max(vjs))}
            elif len(gates) == 1 and len(vjs) == 1:
                mode = "fixed_state"
                conditions = {"type": mode, "part": 0,
                              "measured_Vbg_V": float(gates[0]),
                              "measured_Vj_V": float(vjs[0]),
                              "applies_to_all_Vbg": True}
            else:
                raise ValueError(f"Unrecognized blank measurement grid: {workbook}")
        else:
            raise ValueError(f"No JG Part {part} or fixed-state blank data: {workbook}")
        if import_mode is not None and mode != import_mode:
            raise ValueError("Do not mix JG sweeps and fixed-state blank records")
        if reference_conditions and conditions != reference_conditions:
            raise ValueError("Blank workbooks have incompatible reference conditions")
        import_mode = mode
        reference_conditions = conditions
        if expected_cards is not None:
            matching_values = {normalize_identifier(value) for value in description.values()
                               if pd.notna(value)} & expected_cards
            global_card = next(iter(matching_values)) if len(matching_values) == 1 else None
        else:
            global_card = None
        for channel in (str(col) for col in data if re.fullmatch(r"Id\d+", str(col))):
            index = int(channel[2:])
            tnumber_value = description.get(f"TNumber{index:02d}")
            tnumber = normalize_tnumber_value(tnumber_value)
            if mapping is None:
                esn = normalize_identifier(description.get(f"ESN{index:02d}"))
                if not esn or not tnumber:
                    raise ValueError(f"Missing ESN/TNumber for {channel} in {workbook}")
                card = esn
                key = (esn, tnumber)
                assigned = DeviceAssignment(esn, tnumber, card, channel)
                if key in direct_assignments and direct_assignments[key] != assigned:
                    raise ValueError(f"Conflicting channel assignment for ESN{esn} T{tnumber}")
                direct_assignments[key] = assigned
            else:
                card = normalize_identifier(description.get(f"ESN{index:02d}"))
                if card not in expected_cards:
                    card = normalize_identifier(description.get(f"Card{index:02d}"))
                if card not in expected_cards:
                    card = global_card
                if card is None or card not in expected_cards:
                    continue  # unrelated card/channel in a shared blank workbook
            if pd.notna(tnumber_value):
                key = (card, channel)
                if key in channel_tnumbers and channel_tnumbers[key] != tnumber:
                    raise ValueError(f"Conflicting TNumber for calibration {key}")
                channel_tnumbers[key] = tnumber
            gate_values = pd.to_numeric(data["Vb"], errors="coerce").round(2)
            groups = [None] if mode == "fixed_state" else sorted(gate_values.dropna().unique())
            for vbg in groups:
                selected = data if vbg is None else data.loc[gate_values == vbg]
                samples = pd.to_numeric(selected[channel],
                                        errors="coerce").dropna().to_numpy(float)
                if samples.size:
                    values.setdefault((card, channel,
                                       float(vbg) if vbg is not None else None), []).append(samples)
    if not values:
        raise ValueError("No blank-sweep channels matched the mapping card IDs")
    offsets = []
    for (card, channel, vbg), chunks in sorted(
            values.items(), key=lambda item: (item[0][0], item[0][1],
                                              float("-inf") if item[0][2] is None
                                              else item[0][2])):
        samples = np.concatenate(chunks)
        offsets.append(CalibrationOffset(
            card, channel, vbg, float(np.mean(samples)),
            sd_nA=float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
            n=len(samples)))
    available = {(row.card_id, row.channel) for row in offsets}
    if mapping is None:
        assignments = list(direct_assignments.values())
        for row in assignments:
            if (row.card_id, row.channel) not in available:
                raise ValueError(f"No blank samples for ESN{row.esn} T{row.tnumber}")
    else:
        assignments = []
        for row in mapping.itertuples(index=False):
            channel = row.calibration_channel
            if pd.isna(channel) or not str(channel).strip():
                matches = [candidate for candidate, tnum in channel_tnumbers.items()
                           if candidate[0] == row.card_id and tnum == row.tnumber]
                if len(matches) != 1:
                    raise ValueError(f"Cannot uniquely infer calibration channel for "
                                     f"ESN{row.esn} T{row.tnumber} on card {row.card_id}; "
                                     "supply calibration_channel in the mapping")
                channel = matches[0][1]
            else:
                channel = str(channel).strip()
                if channel.isdigit():
                    channel = f"Id{int(channel):02d}"
            if (row.card_id, channel) not in available:
                raise ValueError(f"No blank samples for card {row.card_id}, channel {channel}")
            assignments.append(DeviceAssignment(row.esn, row.tnumber, row.card_id, channel))
    return JgCalibration(offsets, assignments,
                         source=", ".join(str(Path(p).resolve()) for p in workbooks),
                         reference_conditions=reference_conditions)
