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

# Academic-paper (Origin-style) categorical palette.  The marker cycle is kept
# for backwards compatibility with explicitly requested marker plots, while the
# standard sensorlab output is line-only.
PUB_PALETTE = ["#000000", "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
               "#ff7f0e", "#e377c2", "#17becf", "#8c564b", "#bcbd22"]
PUB_MARKERS = ["o", "s", "^", "v", "D", "<", ">", "p", "*", "h"]


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
    show_recovery: bool = False           # recovery (n06) is excluded from JG_H2 plots
    jg_h2_inset: bool = True              # zoom inset on the Vth region with ΔVth arrows
    # inset position/size [x0, y0, w, h] in axes fraction (draggable in the studio)
    jg_h2_inset_bbox: list = field(default_factory=lambda: [0.58, 0.07, 0.40, 0.44])
    # zoom region shown in the inset (the grey rectangle): [Vj_lo, Vj_hi, Ids_lo,
    # Ids_hi]. Empty = auto (fit around the Vth values).
    jg_h2_inset_zoom: list = field(default_factory=list)
    # ΔVth arrows in the inset: line width, arrow-head size, and vertical
    # position (as a multiple of the Vth target current) for the H2 / recovery families
    jg_h2_arrow_lw: float = 1.6
    jg_h2_arrow_scale: float = 12.0
    jg_h2_arrow_h2_y: float = 2.0
    jg_h2_arrow_rec_y: float = 0.6

    # --- lines -----------------------------------------------------------
    linewidth: float = 2.5                  # standard sweep/trace line width (pt)
    linestyle: str = "-"                   # "-", "--", ":", "-." (SE/STEPS/JG main line)

    # --- SE / time-series ------------------------------------------------
    se_color: str = "#1f77b4"
    se_stack_per_esn: bool = False        # overlay a card's transistors on one figure
    se_gradient: bool = True              # shaded band over the H2 exposure parts
    se_cmap: str = "Blues"
    se_intensity: float = 0.3             # 0..1 shade depth of the exposure band
    se_uniform_y: bool = True             # same Y range for every device of the run
    se_stack_dc_shift: bool = False       # stack: shift each trace so its minimum sits at 0
    se_normalize: bool = False            # plot (I-I0)/I0*100 instead of nA
    steps_uniform_y: bool = True
    steps_stack_dc_shift: bool = False
    steps_normalize: bool = False
    # per-card series hiding: esn -> [tnum labels] hidden in THAT card's stack only
    stack_hidden: dict = field(default_factory=dict)
    uniform_y_margin_nA: float = 1000.0   # padding below/above the run's min/max

    # --- STEPS ppm gradient ---------------------------------------------
    steps_gradient: bool = True
    steps_cmap: str = "Blues"
    steps_intensity: float = 0.9          # 0..1 max shade depth
    steps_stack_per_esn: bool = False     # overlay a card's transistors on one staircase

    # --- JG_H2 Vbg linestyle override ------------------------------------
    same_vbg_linestyle: bool = False   # when True: use style.linestyle for all BG in JG_H2

    # --- Y-axis scale override -------------------------------------------
    yscale_override: str = "auto"      # "auto", "log", or "linear"

    # --- axis limits ([] = auto; [min, max] = fixed) ---------------------
    xlim: list = field(default_factory=list)
    ylim: list = field(default_factory=list)

    # --- legend ----------------------------------------------------------
    show_legend: bool = True
    legend_metrics: bool = True         # include Vth/SS/dVth/… numbers in legend labels
    h2_label: str = "100ppm H2"         # display label for the H2 exposure condition

    # --- markers (JG / JG_H2) -------------------------------------------
    show_vth_line: bool = False          # horizontal line at Vth target current
    show_delta_vth: bool = False         # ΔVth double-arrow annotation (JG_H2 only)
    show_vth_marker: bool = False        # red dot at the Vth point (JG)

    # --- publication style (match academic-paper figure format) ----------
    publication: bool = True             # master toggle for the paper look
    pub_palette: list = field(default_factory=lambda: list(PUB_PALETTE))
    pub_markers_on: bool = False         # standard figures use coloured lines only
    pub_marker_size: float = 5.0
    pub_markevery: int = 0               # 0 = auto-thin markers on dense curves
    pub_bold_labels: bool = True         # bold axis titles (methane-paper look)
    pub_show_title: bool = False         # in-figure plot title (papers use captions)
    pub_minor_ticks: bool = True         # minor ticks on linear axes
    pub_ticks_in: bool = True            # ticks point inward
    pub_ticks_all_sides: bool = False    # ticks on top/right too (only for a real
                                         # secondary axis; off = bottom/left only)
    pub_frame_lw: float = 1.3            # spine / tick line width
    pub_tick_size: float = 14.0          # tick-label font size
    pub_label_size: float = 16.0         # axis-title font size
    pub_title_size: float = 15.0         # plot-title font size
    pub_legend_size: float = 11.0        # legend entries and legend-title font size

    # --- output resolution -----------------------------------------------
    dpi: float = 300.0                   # saved-image resolution (was 110)

    # --- grid & major ticks ----------------------------------------------
    show_major_grid: bool = False
    x_major_step: float = 0.0           # 0 = auto; >0 sets MultipleLocator interval
    y_major_step: float = 0.0           # 0 = auto; ignored on log Y scale

    # --- ZOZO overlays ----------------------------------------------------
    zozo_show_exp_fit: bool = True      # dashed single-exp fit per segment
    zozo_show_lin_slope: bool = False   # straight line first-4 -> last-4 points + slope
    zozo_stack_per_esn: bool = False    # overlay a card's transistors on one figure

    # --- per-mode title templates ------------------------------------------
    # mode -> template with {ESN} {T} {Mode} {Day} placeholders ("" = default)
    title_templates: dict = field(default_factory=dict)

    # --- per-mode axis settings ----------------------------------------------
    # mode -> {"yscale","xlim","ylim","grid","x_step","y_step"}; the flat fields
    # above (yscale_override/xlim/...) are only the legacy fallback.
    mode_axes: dict = field(default_factory=lambda: {
        # Voltage-sweep figures use the same readable 0.5 V major-tick spacing.
        "JG": {"x_step": 0.5},
        "JG_H2": {"x_step": 0.5},
        "JGH": {"x_step": 0.5},
    })

    # -- helpers ----------------------------------------------------------
    def bg_color(self, vb: int) -> str:
        if self.same_vbg_color:
            return self.single_vbg_color
        return self.bg_colors.get(str(int(vb)), "#333333")

    def bg_linestyle(self, vb: int) -> str:
        if self.same_vbg_linestyle:
            return self.linestyle
        return _BG_LINESTYLE.get(int(vb), "-")

    def shows(self, vb: int) -> bool:
        return int(vb) in [int(v) for v in self.show_vbg]

    def title_for(self, mode: str, default: str, esn: str = "", t: str = "",
                  day: str = "", wp: str = "") -> str:
        """Per-mode custom title. Placeholders: {ESN} {T} {Mode} {Day} {WP}; ''=default."""
        tmpl = str((self.title_templates or {}).get(mode, "")).strip()
        if not tmpl:
            return default
        return (tmpl.replace("{ESN}", str(esn)).replace("{T}", str(t))
                    .replace("{Mode}", mode).replace("{Day}", str(day))
                    .replace("{WP}", str(wp)))

    def apply_axes(self, ax, mode: str, default_log: bool = False) -> None:
        """Apply this mode's own scale / limits / grid / ticks (falling back to
        the legacy flat fields when the mode has no entry). Keeps every mode's
        axis settings independent — in the studio AND in the batch pipeline."""
        from matplotlib.ticker import MultipleLocator
        # deliberately NO fallback to the flat yscale_override/xlim/... fields:
        # those held one global value and leaked the last-edited tab's settings
        # into every mode's plots
        cfg = (self.mode_axes or {}).get(mode, {})
        yscale = cfg.get("yscale", "auto")
        if yscale == "log":
            ax.set_yscale("log")
        elif yscale == "linear":
            ax.set_yscale("linear")
        elif default_log:
            ax.set_yscale("log")
        for lim, setter in ((cfg.get("xlim", []), ax.set_xlim),
                            (cfg.get("ylim", []), ax.set_ylim)):
            if lim and len(lim) == 2:
                try:
                    setter(float(lim[0]), float(lim[1]))
                except (TypeError, ValueError):
                    pass
        try:
            x_step = float(cfg.get("x_step", 0) or 0)
            y_step = float(cfg.get("y_step", 0) or 0)
        except (TypeError, ValueError):
            x_step = y_step = 0.0
        if x_step > 0:
            ax.xaxis.set_major_locator(MultipleLocator(x_step))
        if y_step > 0 and ax.get_yscale() != "log":
            ax.yaxis.set_major_locator(MultipleLocator(y_step))
        if cfg.get("grid", False):
            ax.grid(True, which="major", ls=":", alpha=0.35, color="#888888")

    def apply_scale(self, ax, default_log: bool = False) -> None:
        """Apply Y-axis scale: override if set, else fall back to mode default."""
        if self.yscale_override == "log":
            ax.set_yscale("log")
        elif self.yscale_override == "linear":
            ax.set_yscale("linear")
        elif default_log:
            ax.set_yscale("log")

    def apply_limits(self, ax) -> None:
        """Apply user-specified axis limits (skip if list is empty or None)."""
        if self.xlim and len(self.xlim) == 2:
            try:
                ax.set_xlim(float(self.xlim[0]), float(self.xlim[1]))
            except (TypeError, ValueError):
                pass
        if self.ylim and len(self.ylim) == 2:
            try:
                ax.set_ylim(float(self.ylim[0]), float(self.ylim[1]))
            except (TypeError, ValueError):
                pass

    def apply_ticks(self, ax) -> None:
        """Configure major tick interval and optional grid."""
        from matplotlib.ticker import MultipleLocator
        if self.x_major_step > 0:
            ax.xaxis.set_major_locator(MultipleLocator(self.x_major_step))
        if self.y_major_step > 0 and ax.get_yscale() != "log":
            ax.yaxis.set_major_locator(MultipleLocator(self.y_major_step))
        if self.show_major_grid:
            ax.grid(True, which="major", ls=":", alpha=0.35, color="#888888")

    # -- publication (academic-paper) styling -----------------------------
    def pub_color(self, i: int) -> str:
        """Categorical colour for the i-th series (paper palette)."""
        pal = self.pub_palette or PUB_PALETTE
        return pal[i % len(pal)]

    def pub_marker(self, i: int):
        """Distinct marker shape for the i-th series ('' when markers off)."""
        if not self.pub_markers_on:
            return ""
        return PUB_MARKERS[i % len(PUB_MARKERS)]

    def series_kwargs(self, i: int, n: int = 0, fallback_color: str | None = None,
                      with_marker: bool = False) -> dict:
        """plot() kwargs for the i-th series.

        Standard output is line-only.  A caller can still opt into markers
        explicitly with ``with_marker=True`` for a specialised figure.
        """
        if self.publication:
            kw = {"color": self.pub_color(i)}
            mk = self.pub_marker(i) if with_marker else ""
            if mk:
                kw.update(marker=mk, markersize=self.pub_marker_size,
                          markevery=self.markevery_for(n))
            return kw
        return {"color": fallback_color} if fallback_color else {}

    def markevery_for(self, n: int):
        """Thin markers so a dense curve shows ~20 of them, like the papers."""
        if self.pub_markevery and self.pub_markevery > 0:
            return self.pub_markevery
        return max(1, n // 20) if n and n > 40 else 1

    def apply_publication(self, ax) -> None:
        """Give an axes the academic-paper frame: full box, inward ticks on
        all four sides, minor ticks, bold sized axis titles. No-op when the
        publication toggle is off. Call after labels + scale are set."""
        if not self.publication:
            return
        from matplotlib.ticker import AutoMinorLocator
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(self.pub_frame_lw)
            sp.set_color("black")
        d = "in" if self.pub_ticks_in else "out"
        sides = self.pub_ticks_all_sides   # no phantom secondary scale unless wanted
        ax.tick_params(which="major", direction=d, top=sides, right=sides,
                       length=5.5, width=self.pub_frame_lw, labelsize=self.pub_tick_size)
        ax.tick_params(which="minor", direction=d, top=sides, right=sides,
                       length=3.0, width=self.pub_frame_lw * 0.8)
        if self.pub_minor_ticks:
            if ax.get_xscale() != "log":
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            if ax.get_yscale() != "log":
                ax.yaxis.set_minor_locator(AutoMinorLocator())
        weight = "bold" if self.pub_bold_labels else "normal"
        for lab in (ax.xaxis.label, ax.yaxis.label):
            lab.set_fontsize(self.pub_label_size)
            lab.set_fontweight(weight)
        # ``tick_params(labelsize=...)`` does not expose font weight.  Apply it
        # directly so both the numbers and any scientific-notation offset are
        # visually consistent with the bold axis labels.
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight(weight)
        for offset_text in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            offset_text.set_fontsize(self.pub_tick_size)
            offset_text.set_fontweight(weight)
        if self.pub_show_title:
            ax.title.set_fontsize(self.pub_title_size)
            ax.title.set_fontweight(weight)
        else:
            ax.set_title("")             # papers caption the figure instead

    @staticmethod
    def simple_legend_handles(handles):
        """Replace ErrorbarContainer handles with plain series proxies.

        Matplotlib's default error-bar legend handler repeats the vertical bar
        and caps inside the legend. The plots keep their error bars; the legend
        shows only the series line in line mode or its marker in scatter mode.
        """
        from matplotlib.container import ErrorbarContainer
        from matplotlib.lines import Line2D

        clean = []
        for handle in handles:
            if isinstance(handle, ErrorbarContainer):
                data_line = handle.lines[0]
                if data_line is not None:
                    handle = Line2D(
                        [], [], color=data_line.get_color(),
                        linestyle=data_line.get_linestyle(),
                        linewidth=data_line.get_linewidth(),
                        marker=data_line.get_marker(),
                        markersize=data_line.get_markersize(),
                        markerfacecolor=data_line.get_markerfacecolor(),
                        markeredgecolor=data_line.get_markeredgecolor(),
                        alpha=data_line.get_alpha(),
                    )
            clean.append(handle)
        return clean

    def legend_fontsize(self, fallback: float = 8.0) -> float:
        """Use the larger shared publication size, else the caller's fallback."""
        return float(self.pub_legend_size) if self.publication else float(fallback)

    def apply_legend(self, ax, fontsize: float = 8, loc: str = "best",
                     handles=None, labels=None, **kwargs) -> None:
        """Show a consistently sized legend, honouring show_legend."""
        if self.show_legend:
            if handles is None:
                handles, auto_labels = ax.get_legend_handles_labels()
                if labels is None:
                    labels = auto_labels
            if handles:
                handles = self.simple_legend_handles(list(handles))
                kwargs.setdefault("frameon", not self.publication)
                size = self.legend_fontsize(fontsize)
                kwargs.setdefault("title_fontsize", size)
                legend_items = {"handles": handles}
                if labels is not None:
                    legend_items["labels"] = labels
                ax.legend(fontsize=size, loc=loc, **legend_items, **kwargs)
        else:
            lg = ax.get_legend()
            if lg:
                lg.remove()

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
