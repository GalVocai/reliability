"""User-editable labels for the degradation trend plots.

One JSON file (``~/.sensorlab/degradation_labels.json``) is the single source
of truth for every y-axis label and title in the degradation study. It is
written out pre-filled with the current defaults on first use, so it can be
edited in any text editor; the study reads it and falls back to the defaults
for anything missing (new metrics added later keep working untouched).

Three sections:
  * ``ylabels`` -- metric name -> y-axis label (with units). Keys include the
    grouped-plot pseudo-metrics ``ppm_resp_pct`` / ``ppm_delta`` / ``ppm_t90_s``
    and the special ``@degradation`` (the retention "%") / ``@delta_Vth`` axes.
  * ``titles``  -- plot family -> title template, placeholders ``{label}`` and
    ``{base}`` (the baseline day).
  * ``hidden``  -- metric names to skip entirely (no plot produced).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_LABELS_PATH = Path.home() / ".sensorlab" / "degradation_labels.json"

# Split off mathtext ($...$) and bracketed units/qualifiers ((...) / [...]) so
# they are never touched by title-casing.
_PROTECTED = re.compile(r"(\$[^$]*\$|\([^)]*\)|\[[^\]]*\])")


def titlecase_label(s: str) -> str:
    """Capitalise the first letter of every word in a label, leaving mathtext
    and bracketed units untouched (nA/ppm, mV/dec, (V) stay as-is)."""
    out = []
    for part in _PROTECTED.split(s):
        if part[:1] in "$([":
            out.append(part)                       # math / units: verbatim
        else:
            out.append(re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), part))
    return "".join(out)

_DEFAULT_YLABELS: dict[str, str] = {
    # JG_n03 transfer sweep
    "Vth": r"$\mathbf{V_{Th}}$ (V)", "Vth_d2": r"$\mathbf{V_{Th}}$ (d2) (V)", "SS": "SS (mV/dec)",
    "gm": "gm (nA/V)", "Ion": "Ion (nA)", "Ioff": "Ioff (nA)",
    "Ion_Ioff": "Ion/Ioff", "SNR_off": "SNR (off)",
    # JGH hysteresis
    "Delta_Vth": r"$\mathbf{\Delta V_{Th}}$ hysteresis (V)", "loop_area": "Loop area (nA·V)", "HI": "HI",
    # SE / STEPS sensing (shared names)
    "response_delta": "Response (nA)", "response_pct": "Response (%)",
    "recovery_delta": "Recovery (nA)", "recovery_pct": "Recovery (%)",
    "t90_response_s": "t90 response (s)", "t90_recovery_s": "t90 recovery (s)",
    "recovery_t90_s": "Recovery t90 (s)", "LOD_ppm": "LOD (ppm)",
    "sensitivity_pct_per_dec": "Sensitivity (%/decade)",
    # STEPS extra graphs
    "sensitivity_nA_per_ppm": "Sensitivity (nA/ppm)",
    "@resp_vs_ppm": "ΔResponse (nA)",          # y-axis of response-vs-concentration
    # JG_H2 gas comparison (ΔVth in V, by request)
    "Vth_air": r"$\mathbf{V_{Th}}$ air (V)", "Vth_H2": r"$\mathbf{V_{Th}}$ H2 (V)",
    "Vth_recovery": r"$\mathbf{V_{Th}}$ recovery (V)",
    "delta_Vth_H2": r"$\mathbf{|\Delta V_{Th}|}$ sensing (V)",
    "delta_Vth_recovery": r"$\mathbf{|\Delta V_{Th}|}$ recovery (V)",
    "resp_at_Vth_H2_pct": r"Response @$\mathbf{V_{Th}}$ H2 (%)",
    "resp_at_Vth_recovery_pct": r"Response @$\mathbf{V_{Th}}$ recovery (%)",
    "max_resp_H2_pct": "Maximum Response (%)", "max_resp_recovery_pct": "Max response recovery (%)",
    "Vj_max_resp_H2": "Vj @max response H2 (V)", "Vj_max_resp_recovery": "Vj @max response recovery (V)",
    # working points
    "Vj_State1": "Vj State1 (V)", "Vj_State2": "Vj State2 (V)",
    "Vb_State1": "Vb State1 (V)", "Vb_State2": "Vb State2 (V)",
    # ZOZO transient
    "tau": "τ (s)", "A": "Amplitude (nA)", "level": "Level (nA)",
    "R2": "R²", "lin_slope": "Slope (nA/s)",
    # grouped-plot pseudo-metrics (STEPS ppm concentrations on one axis)
    "ppm_resp_pct": "H2 response (%)", "ppm_delta": "H2 response delta (nA)",
    "ppm_t90_s": "H2 t90 (s)",
    # special axes
    "@degradation": "%",                       # retention view (100 = baseline)
    "@delta_Vth": r"$\mathbf{\Delta V_{Th}}$ (mV)",   # JG_n03 day-vs-baseline shift
}

_DEFAULT_TITLES: dict[str, str] = {
    "devices": "{label} — per device",
    "average": "{label} — average",
    "degradation": "{label} — % of day {base} (100 = baseline)",
    "delta": "{label} — Δ vs day {base} (mV)",
}


class DegradationLabels:
    """Resolved labels/titles, user overrides merged over the defaults."""

    def __init__(self, ylabels: dict | None = None, titles: dict | None = None,
                 hidden: list | None = None, path: Path = DEFAULT_LABELS_PATH):
        self.path = Path(path)
        self.ylabels = {**_DEFAULT_YLABELS, **(ylabels or {})}
        self.titles = {**_DEFAULT_TITLES, **(titles or {})}
        self.hidden = [str(m) for m in (hidden or [])]

    # -- lookups --------------------------------------------------------------
    def ylabel(self, metric: str) -> str:
        """Y-axis label for a metric ('_' -> ' ' fallback), Title-Cased (every
        word capitalised; mathtext and bracketed units left untouched)."""
        raw = self.ylabels.get(str(metric), str(metric).replace("_", " "))
        return titlecase_label(raw)

    def deg_ylabel(self) -> str:
        return self.ylabels.get("@degradation", "%")

    def title(self, label: str, family: str, base: object) -> str:
        tmpl = self.titles.get(family, "{label}")
        return tmpl.replace("{label}", str(label)).replace("{base}", str(base))

    def is_hidden(self, metric: str) -> bool:
        return str(metric) in self.hidden

    # -- persistence ----------------------------------------------------------
    def as_dict(self) -> dict:
        return {"ylabels": self.ylabels, "titles": self.titles, "hidden": self.hidden}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
                             encoding="utf-8")


def load_degradation_labels(path: Path = DEFAULT_LABELS_PATH) -> DegradationLabels:
    """Load the label config; create it (pre-filled with defaults) if missing,
    so there is always a file on disk to edit. Never raises on a bad/locked
    file -- it just falls back to the built-in defaults."""
    p = Path(path)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return DegradationLabels(data.get("ylabels"), data.get("titles"),
                                     data.get("hidden"), path=p)
        except Exception:
            return DegradationLabels(path=p)
    cfg = DegradationLabels(path=p)
    try:
        cfg.save()
    except Exception:
        pass
    return cfg
