"""User-editable plot style, set in the Plotter window and used by every plot.

Persisted to ~/.sensorlab/plot_style.json so the interactive previewer and the
batch pipeline share one source of truth for colors / line styles / gradients.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_STYLE_PATH = Path.home() / ".sensorlab" / "plot_style.json"

_BG_LINESTYLE = {-4: ":", 0: "-", 4: "--"}


@dataclass
class StyleConfig:
    # --- back-gate (Vbg) lines, used by JG / JGH -------------------------
    bg_colors: dict = field(default_factory=lambda: {"-4": "#2ca02c", "0": "#1f77b4", "4": "#d62728"})
    show_vbg: list = field(default_factory=lambda: [0, 4])     # which back-gates to draw
    same_vbg_color: bool = False                               # color both/all Vbg identically
    single_vbg_color: str = "#1f77b4"

    # --- gas-condition colors, used by JG_H2 -----------------------------
    air_color: str = "#1f77b4"
    h2_color: str = "#d62728"
    recovery_color: str = "#2ca02c"
    show_recovery: bool = False

    # --- lines -----------------------------------------------------------
    linewidth: float = 1.0

    # --- SE / time-series ------------------------------------------------
    se_color: str = "#1f77b4"
    se_stack_per_esn: bool = False        # overlay a card's transistors on one figure

    # --- STEPS ppm gradient ---------------------------------------------
    steps_gradient: bool = True
    steps_cmap: str = "Blues"
    steps_intensity: float = 0.9          # 0..1 max shade depth

    # -- helpers ----------------------------------------------------------
    def bg_color(self, vb: int) -> str:
        if self.same_vbg_color:
            return self.single_vbg_color
        return self.bg_colors.get(str(int(vb)), "#333333")

    def bg_linestyle(self, vb: int) -> str:
        return _BG_LINESTYLE.get(int(vb), "-")

    def shows(self, vb: int) -> bool:
        return int(vb) in [int(v) for v in self.show_vbg]

    def save(self, path: Path = DEFAULT_STYLE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_style(path: Path = DEFAULT_STYLE_PATH) -> StyleConfig:
    p = Path(path)
    s = StyleConfig()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except Exception:
            pass
    return s


DEFAULT_STYLE = StyleConfig()
