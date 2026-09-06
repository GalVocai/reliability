"""Shared plot labels: axis titles and the device title built from GUI fields."""

from __future__ import annotations

from ..model import Channel

# axis labels (shared so every plot is consistent). Written in the academic-
# paper convention: italic variable with subscript via mathtext, unit in
# parentheses. The publication style bolds them (see StyleConfig.apply_publication).
X_TIME = "Time (s)"
X_VJ = r"$\mathbf{V_{JG}}$ (V)"
Y_ID = r"$\mathbf{I_{DS}}$ (nA)"
Y_RESP = "Response (%)"
Y_RESP_MAX = "Maximum Sensor Response (%)"


def channel_states(states: dict | None, ch: Channel) -> dict:
    """This channel's own working points, keyed by device identity (ESN, TNumber)."""
    return (states or {}).get((ch.esn, ch.tnumber), {})


def wp_label(states: dict | None, ch: Channel, both_states: bool = False) -> str:
    """Working-point string for one channel from the States sheet.

    Single state: 'Vj=-0.58V Vb=4V' (State1). With both_states (ZOZO):
    'Vj=-0.58<->-0.78V Vb=4V'."""
    st = channel_states(states, ch)
    s1, s2 = st.get("State1"), st.get("State2")
    if not s1:
        return ""
    if both_states and s2:
        if s1[1] == s2[1]:
            return f"Vj={s1[0]:g}↔{s2[0]:g}V Vb={s1[1]:g}V"
        return f"Vj={s1[0]:g}↔{s2[0]:g}V Vb={s1[1]:g}↔{s2[1]:g}V"
    return f"Vj={s1[0]:g}V Vb={s1[1]:g}V"


def wp_label_stack(states: dict | None, channels: list[Channel]) -> str:
    """Working points of a stacked card: per-T Vj list + the (shared) Vb."""
    pts = [channel_states(states, c).get("State1") for c in channels]
    pts = [p for p in pts if p]
    if not pts:
        return ""
    vjs = ", ".join(f"{p[0]:g}" for p in pts)
    vbs = {p[1] for p in pts}
    vb = f"{pts[0][1]:g}" if len(vbs) == 1 else "/".join(f"{v:g}" for v in sorted(vbs))
    return f"Vj=[{vjs}]V Vb={vb}V"


def device_title(experiment_type: str, day: str, ch: Channel) -> str:
    """'{experiment type} {Day#} ESN{esn} {Tnum}'.

    `experiment_type` and `day` are supplied by the GUI (not in the data);
    ESN and Tnum come from the file. Empty fields are dropped.
    """
    fields = [str(experiment_type).strip(), str(day).strip(), f"ESN{ch.esn}", ch.tnum_label]
    return " ".join(f for f in fields if f)
