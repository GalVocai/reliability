"""Shared plot labels: axis titles and the device title built from GUI fields."""

from __future__ import annotations

from ..model import Channel

# axis labels (shared so every plot is consistent)
X_TIME = "Time (seconds)"
X_VJ = "Vjg [V]"
Y_ID = "Ids (nA)"


def device_title(experiment_type: str, day: str, ch: Channel) -> str:
    """'{experiment type} {Day#} ESN{esn} {Tnum}'.

    `experiment_type` and `day` are supplied by the GUI (not in the data);
    ESN and Tnum come from the file. Empty fields are dropped.
    """
    fields = [str(experiment_type).strip(), str(day).strip(), f"ESN{ch.esn}", ch.tnum_label]
    return " ".join(f for f in fields if f)
