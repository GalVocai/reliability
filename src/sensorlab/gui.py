"""Thin Tkinter front-end for the sensorlab pipeline.

Pick a results folder, enter the Day#, fill the condition per card (cached and
remembered), then run. All analysis lives in the headless core; this is only a
shell around `analyze_folder` (+ a degradation window around `run_degradation`).
"""

from __future__ import annotations

import os
import math
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import colorchooser

from .compare import discover_timepoints, run_degradation
from .degradation import (conditions_in_folder, day_to_num, devices_in_folder,
                          find_masters)
from .io.device_map import load_condition_cache
from .model import MajorKind
from .pipeline import all_esns, analyze_folder, group_by_kind, load_folder
from .plotting import (
    load_style, render_jg, render_jg_h2, render_jgh, render_se, render_se_stack,
    render_steps, render_steps_stack, render_zozo, render_zozo_stack,
)
from .protocol import DEFAULT as PROTO
from .report_builder import (ConditionSource, ReportBuildResult,
                             available_report_devices, build_results_report,
                             discover_degradation_sources)

# --- palette / fonts ---------------------------------------------------------
BG = "#eef0f3"
CARD = "#f7f8fa"
ACCENT = "#2563eb"
ACCENT_DK = "#1d4ed8"
TEXT = "#1f2937"
MUTED = "#6b7280"
FONT = ("Segoe UI", 10)
MONO = ("Consolas", 9)


def _setup_style(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=TEXT, font=FONT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED)
    style.configure("TCheckbutton", background=BG)
    style.configure("TRadiobutton", background=BG)
    style.configure("TLabelframe", background=BG, borderwidth=1, relief="solid", bordercolor="#d1d5db")
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
    style.configure("TEntry", fieldbackground="white", padding=3)
    style.configure("TButton", padding=6, background="#e5e7eb")
    style.map("TButton", background=[("active", "#d1d5db")])
    style.configure("Accent.TButton", padding=8, foreground="white", background=ACCENT,
                    font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton",
              background=[("active", ACCENT_DK), ("disabled", "#9ca3af")],
              foreground=[("disabled", "#e5e7eb")])


def _header(parent: tk.Misc, title: str, subtitle: str) -> None:
    bar = tk.Frame(parent, bg=ACCENT)
    bar.pack(fill="x")
    tk.Label(bar, text=title, bg=ACCENT, fg="white", font=("Segoe UI", 16, "bold")).pack(
        anchor="w", padx=14, pady=(10, 0))
    tk.Label(bar, text=subtitle, bg=ACCENT, fg="#dbeafe", font=("Segoe UI", 9)).pack(
        anchor="w", padx=14, pady=(0, 10))


def _make_log(parent: tk.Misc) -> ScrolledText:
    log = ScrolledText(parent, height=12, state="disabled", font=MONO,
                       background="#0f172a", foreground="#e2e8f0", insertbackground="#e2e8f0",
                       borderwidth=0, relief="flat")
    return log


def _tnum_label(t: str) -> str:
    return f"D{t[0]}T{t[1]}" if len(t) == 2 and t.isdigit() else f"T{t}"


class SensorLabApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("sensorlab")
        self.geometry("820x680")
        self.configure(bg=BG)
        _setup_style(self)
        self.folder: str | None = None
        self.cache = load_condition_cache()
        self.cond_vars: dict[str, tk.StringVar] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._build()
        self.after(100, self._drain_log)

    # ---- layout ------------------------------------------------------------
    def _build(self) -> None:
        _header(self, "sensorlab", "Transistor / sensor analysis — load a results folder and run")
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        # 1. folder
        f1 = ttk.LabelFrame(body, text="  1 · Results folder  ", padding=10)
        f1.pack(fill="x", pady=(0, 10))
        ttk.Button(f1, text="Select folder…", command=self._pick_folder).pack(side="left")
        self.folder_lbl = ttk.Label(f1, text="no folder selected", style="Muted.TLabel")
        self.folder_lbl.pack(side="left", padx=12)

        # 2. options
        f2 = ttk.LabelFrame(body, text="  2 · Options  ", padding=10)
        f2.pack(fill="x", pady=(0, 10))
        ttk.Label(f2, text="Day # :").pack(side="left")
        self.day_var = tk.StringVar(value="baseline")
        ttk.Entry(f2, textvariable=self.day_var, width=16).pack(side="left", padx=(4, 20))
        self.plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f2, text="Generate plots", variable=self.plots_var).pack(side="left")

        # 3. conditions
        f3 = ttk.LabelFrame(body, text="  3 · Condition per card  (filled once, remembered)  ", padding=10)
        f3.pack(fill="x", pady=(0, 10))
        self.cond_frame = ttk.Frame(f3)
        self.cond_frame.pack(fill="x")
        ttk.Label(self.cond_frame, text="Select a folder to list the cards…",
                  style="Muted.TLabel").pack(anchor="w")

        # actions
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(0, 10))
        self.run_btn = ttk.Button(actions, text="▶  Run analysis", style="Accent.TButton",
                                  command=self._run, state="disabled")
        self.run_btn.pack(side="left")
        ttk.Button(actions, text="Compare timepoints (degradation)…",
                   command=lambda: DegradationWindow(self)).pack(side="left", padx=10)
        ttk.Button(actions, text="Plot studio…",
                   command=lambda: PlotStudio(self, self.folder)).pack(side="left")
        ttk.Button(actions, text="Results report…",
                   command=lambda: ResultsReportWindow(self)).pack(side="left", padx=(10, 0))

        # log
        f4 = ttk.LabelFrame(body, text="  Log  ", padding=6)
        f4.pack(fill="both", expand=True)
        self.log = _make_log(f4)
        self.log.pack(fill="both", expand=True)

    # ---- actions -----------------------------------------------------------
    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select results folder")
        if not folder:
            return
        try:
            runs = load_folder(folder)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        if not runs:
            messagebox.showwarning("No data", "No result files (…_n0X_…) found in this folder.")
            return
        self.folder = folder
        self.folder_lbl.config(text=folder, style="TLabel")
        esns = all_esns(runs)
        self._log(f"Loaded {len(runs)} majors; {len(esns)} cards (ESN).")
        self._build_condition_rows(esns)
        self.run_btn.config(state="normal")

    def _build_condition_rows(self, esns: list[str]) -> None:
        for w in self.cond_frame.winfo_children():
            w.destroy()
        self.cond_vars.clear()
        for i, esn in enumerate(sorted(esns, key=lambda x: (len(x), x))):
            r, c = divmod(i, 2)
            cell = ttk.Frame(self.cond_frame)
            cell.grid(row=r, column=c, sticky="w", padx=6, pady=3)
            ttk.Label(cell, text=f"ESN{esn}", width=10).pack(side="left")
            var = tk.StringVar(value=self.cache.get(esn))
            ttk.Entry(cell, textvariable=var, width=24).pack(side="left")
            self.cond_vars[esn] = var

    def _run(self) -> None:
        if not self.folder:
            return
        self.cache.update({esn: v.get().strip() for esn, v in self.cond_vars.items()})
        self.cache.save()
        self._log(f"Saved conditions -> {self.cache.path}")
        self.run_btn.config(state="disabled")
        # Tk variables must only be read on the UI thread.  Snapshot all input
        # before handing the CPU-heavy analysis to the worker.
        folder = self.folder
        make_plots = self.plots_var.get()
        day = self.day_var.get().strip()
        threading.Thread(target=self._worker, args=(folder, make_plots, day), daemon=True).start()

    def _worker(self, folder: str, make_plots: bool, day: str) -> None:
        try:
            written = analyze_folder(folder, conditions=self.cache,
                                     make_plots=make_plots, day=day,
                                     log=self._queue.put)
            self._queue.put("__DONE__:" + ", ".join(written))
        except Exception as exc:
            self._queue.put("__ERROR__:" + str(exc))

    def _drain_log(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.startswith("__DONE__:"):
                    self._log("[done] " + msg[len("__DONE__:"):])
                    self.run_btn.config(state="normal")
                    messagebox.showinfo("Done", "Analysis complete.")
                elif msg.startswith("__ERROR__:"):
                    self._log("[error] " + msg[len("__ERROR__:"):])
                    self.run_btn.config(state="normal")
                    messagebox.showerror("Error", msg[len("__ERROR__:"):])
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


def _peek_day(folder: str) -> str:
    for _cond, _mode, path in find_masters(folder):
        try:
            df = pd.read_excel(path, nrows=1)
            if "day" in df.columns and len(df):
                return str(df["day"].iloc[0])
        except Exception:
            pass
    return ""


class DegradationWindow(tk.Toplevel):
    """Pick timepoint folders, mark the baseline, choose devices, build the study."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.title("sensorlab — degradation")
        self.geometry("860x780")
        self.configure(bg=BG)
        self.rows: list[dict] = []
        self.dev_vars: dict[tuple[str, str], tk.BooleanVar] = {}
        self.excl_cond_days: set[tuple[str, str]] = set()   # (condition, day) dropped
        self.baseline = tk.IntVar(value=0)
        self.out_dir: str | None = None
        self._queue: queue.Queue[str] = queue.Queue()

        _header(self, "Compare timepoints", "Degradation across days — delta, % degradation, averages ± AVDEV")
        # The controls can exceed a laptop display. Keep the entire form in a
        # vertically scrollable canvas so the run button and log stay reachable.
        scroll_host = ttk.Frame(self)
        scroll_host.pack(fill="both", expand=True)
        scroll_canvas = tk.Canvas(scroll_host, bg=BG, highlightthickness=0)
        scroll_bar = ttk.Scrollbar(scroll_host, orient="vertical",
                                   command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(scroll_canvas, padding=12)
        body_window = scroll_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _e: scroll_canvas.configure(
            scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>", lambda e: scroll_canvas.itemconfigure(
            body_window, width=e.width))

        def _wheel(event) -> None:
            scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

        scroll_host.bind("<Enter>", lambda _e: self.bind_all("<MouseWheel>", _wheel))
        scroll_host.bind("<Leave>", lambda _e: self.unbind_all("<MouseWheel>"))

        f1 = ttk.LabelFrame(body, text="  Timepoints  (pick the baseline)  ", padding=10)
        f1.pack(fill="x", pady=(0, 10))
        bt = ttk.Frame(f1)
        bt.pack(fill="x")
        ttk.Button(bt, text="Auto-detect from parent…", command=self._auto_detect).pack(side="left")
        ttk.Button(bt, text="Add timepoint…", command=self._add).pack(side="left", padx=6)
        ttk.Button(bt, text="Output folder…", command=self._pick_out).pack(side="left", padx=8)
        self.out_lbl = ttk.Label(bt, text="output: parent of first folder", style="Muted.TLabel")
        self.out_lbl.pack(side="left")
        self.rows_frame = ttk.Frame(f1)
        self.rows_frame.pack(fill="x", pady=(8, 0))

        f2 = ttk.LabelFrame(body, text="  Devices / transistors in the average  ", padding=10)
        f2.pack(fill="x", pady=(0, 10))
        tb = ttk.Frame(f2)
        tb.pack(fill="x")
        ttk.Button(tb, text="All", width=6, command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(tb, text="None", width=6, command=lambda: self._set_all(False)).pack(side="left", padx=4)
        self.dev_frame = ttk.Frame(f2)
        self.dev_frame.pack(fill="x", pady=(8, 0))

        f_axis = ttk.LabelFrame(body, text="  Degradation time axis  ", padding=8)
        f_axis.pack(fill="x", pady=(0, 10))
        self.time_axis_var = tk.StringVar(value="days")
        self.hours_per_cycle_var = tk.StringVar(value="2.0")
        ttk.Radiobutton(f_axis, text="Days", value="days", variable=self.time_axis_var,
                        command=self._sync_time_axis_controls).pack(side="left")
        ttk.Radiobutton(f_axis, text="Cycle #", value="cycles", variable=self.time_axis_var,
                        command=self._sync_time_axis_controls).pack(side="left", padx=(10, 4))
        ttk.Label(f_axis, text="Hours per cycle:").pack(side="left", padx=(12, 3))
        self.hours_per_cycle_entry = ttk.Entry(
            f_axis, textvariable=self.hours_per_cycle_var, width=7)
        self.hours_per_cycle_entry.pack(side="left")
        ttk.Label(f_axis, text="Timepoint fields remain elapsed days; cycle count is estimated.",
                  style="Muted.TLabel").pack(side="left", padx=(12, 0))
        self._sync_time_axis_controls()

        # STEPS concentration selection filters the data independently of the
        # optional Time/Ids viewport limits.
        f_steps = ttk.LabelFrame(body, text="  STEPS overlay  ", padding=8)
        f_steps.pack(fill="x", pady=(0, 10))
        ppm_row = ttk.Frame(f_steps)
        ppm_row.pack(fill="x")
        ttk.Label(ppm_row, text="Show ppm:").pack(side="left")
        self.ppm_vars: dict[int, tk.BooleanVar] = {}
        for ppm in (0, 10, 25, 50, 100, 1000):
            v = tk.BooleanVar(value=True)
            self.ppm_vars[ppm] = v
            ttk.Checkbutton(ppm_row, text=str(ppm), variable=v).pack(side="left", padx=2)

        bounds_row = ttk.Frame(f_steps)
        bounds_row.pack(fill="x", pady=(6, 0))
        ttk.Label(bounds_row, text="Time (s):", style="Muted.TLabel").pack(side="left")
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        ttk.Entry(bounds_row, textvariable=self.xmin_var, width=7).pack(side="left", padx=(2, 0))
        ttk.Label(bounds_row, text="to", style="Muted.TLabel").pack(side="left", padx=2)
        ttk.Entry(bounds_row, textvariable=self.xmax_var, width=7).pack(side="left")
        ttk.Label(bounds_row, text="     Ids (nA):", style="Muted.TLabel").pack(side="left")
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()
        ttk.Entry(bounds_row, textvariable=self.ymin_var, width=7).pack(side="left", padx=(2, 0))
        ttk.Label(bounds_row, text="to", style="Muted.TLabel").pack(side="left", padx=2)
        ttk.Entry(bounds_row, textvariable=self.ymax_var, width=7).pack(side="left")

        f_jg = ttk.LabelFrame(body, text="  JG sweep overlay  ", padding=8)
        f_jg.pack(fill="x", pady=(0, 10))
        ttk.Label(f_jg, text="VJG range (V):").pack(side="left")
        self.jg_xmin_var = tk.StringVar()
        self.jg_xmax_var = tk.StringVar()
        ttk.Entry(f_jg, textvariable=self.jg_xmin_var, width=8).pack(side="left", padx=(4, 0))
        ttk.Label(f_jg, text="to", style="Muted.TLabel").pack(side="left", padx=3)
        ttk.Entry(f_jg, textvariable=self.jg_xmax_var, width=8).pack(side="left")
        ttk.Label(f_jg, text="  (blank = protocol range)", style="Muted.TLabel").pack(side="left")

        f_styles = ttk.LabelFrame(body, text="  Plot styles  ", padding=8)
        f_styles.pack(fill="x", pady=(0, 10))

        trend_row = ttk.Frame(f_styles)
        trend_row.pack(fill="x")
        self.plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(trend_row, text="Generate plots",
                        variable=self.plots_var).pack(side="left")
        ttk.Label(trend_row, text="Trend style:").pack(side="left", padx=(12, 2))
        self.trend_style_var = tk.StringVar(value="line")
        ttk.Radiobutton(trend_row, text="Lines", variable=self.trend_style_var,
                        value="line").pack(side="left")
        ttk.Radiobutton(trend_row, text="Scatter", variable=self.trend_style_var,
                        value="scatter").pack(side="left", padx=(2, 6))

        raw_row = ttk.Frame(f_styles)
        raw_row.pack(fill="x", pady=(5, 0))
        ttk.Label(raw_row, text="Raw overlay style:").pack(side="left")
        self.raw_style_var = tk.StringVar(value="line")
        ttk.Radiobutton(raw_row, text="Lines", variable=self.raw_style_var,
                        value="line").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(raw_row, text="Scatter", variable=self.raw_style_var,
                        value="scatter").pack(side="left", padx=(2, 6))
        ttk.Label(raw_row, text="(JG sweeps, STEPS and SE_n04)",
                  style="Muted.TLabel").pack(side="left", padx=(8, 0))

        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(0, 10))
        self.excl_btn = ttk.Button(bar, text="Exclude days…", command=self._open_excl_dialog)
        self.excl_btn.pack(side="left", padx=6)
        self.run_btn = ttk.Button(bar, text="▶  Run degradation", style="Accent.TButton",
                                  command=self._run, state="disabled")
        self.run_btn.pack(side="left", padx=10)

        f3 = ttk.LabelFrame(body, text="  Log  ", padding=6)
        f3.pack(fill="both", expand=True)
        self.log = _make_log(f3)
        self.log.pack(fill="both", expand=True)
        self.after(100, self._drain)

    def _auto_detect(self) -> None:
        """Point at the parent folder (holds baseline / 2D / 7D / …) and load
        every analysed timepoint in one shot."""
        parent = filedialog.askdirectory(
            title="Select the parent folder (contains baseline / 2D / 7D …)")
        if not parent:
            return
        found = discover_timepoints(parent)
        if not found:
            messagebox.showwarning(
                "Nothing found",
                f"No analysed timepoints found under:\n{parent}\n\n"
                "Expected subfolders named baseline / 2D / 7D / … each holding "
                "an 'E…' run folder with an 'analysis' subfolder.")
            return
        self._clear_rows()
        for day_label, folder in found:
            self._add_row(folder, day_label=day_label)
        # degradation/ should land next to the timepoints -> default out = parent
        self.out_dir = parent
        self.out_lbl.config(text=f"output: {parent}", style="TLabel")
        self._log(f"Auto-detected {len(found)} timepoints: "
                  + ", ".join(d for d, _ in found))

    def _add(self) -> None:
        folder = filedialog.askdirectory(title="Select a timepoint folder")
        if not folder:
            return
        if not list(find_masters(folder)):
            messagebox.showwarning(
                "No masters",
                f"No analysis masters found under:\n{folder}\n\n"
                "Expected <condition>/<MODE>/master_<MODE>.xlsx inside an "
                "'analysis' subfolder (or directly under the selected folder).\n"
                "Pick the experiment folder itself (not its parent), run Analyze "
                "on it first if needed, and if it is on OneDrive make sure the "
                "'analysis' subfolder is fully synced.")
            return
        self._add_row(folder)

    def _add_row(self, folder: str, day_label: str | None = None) -> None:
        i = len(self.rows)
        row = ttk.Frame(self.rows_frame)
        row.pack(fill="x", pady=1)
        ttk.Radiobutton(row, variable=self.baseline, value=i).pack(side="left")
        ttk.Label(row, text=os.path.basename(folder), width=44).pack(side="left")
        day = day_label if day_label is not None else (_peek_day(folder) or "")
        day_var = tk.StringVar(value=day)
        ttk.Label(row, text="day:").pack(side="left")
        ttk.Entry(row, textvariable=day_var, width=10).pack(side="left")
        self.rows.append({"folder": folder, "day": day_var})
        self._auto_baseline()
        self.run_btn.config(state="normal")
        if not self.dev_vars:
            self._populate_devices(folder)

    def _clear_rows(self) -> None:
        for w in self.rows_frame.winfo_children():
            w.destroy()
        self.rows.clear()
        for w in self.dev_frame.winfo_children():
            w.destroy()
        self.dev_vars.clear()
        self.excl_cond_days.clear()
        self._update_excl_btn()
        self.baseline.set(0)
        self.run_btn.config(state="disabled")

    def _auto_baseline(self) -> None:
        """Point the baseline radio at the EARLIEST day — not at whichever
        folder happened to be added first (a '2D' folder sorts before
        'baseline' in the picker, silently inverting the study)."""
        nums = [day_to_num(r["day"].get()) for r in self.rows]
        finite = [(n, i) for i, n in enumerate(nums) if n == n]
        if finite:
            self.baseline.set(min(finite)[1])

    def _populate_devices(self, folder: str) -> None:
        for w in self.dev_frame.winfo_children():
            w.destroy()
        self.dev_vars.clear()
        by_esn: dict[str, list[str]] = {}
        for esn, tnum in devices_in_folder(folder):
            by_esn.setdefault(esn, []).append(tnum)
        for r, (esn, tnums) in enumerate(by_esn.items()):
            ttk.Button(self.dev_frame, text=f"ESN{esn}", width=10,
                       command=lambda e=esn: self._toggle_card(e)).grid(row=r, column=0, sticky="w", pady=1)
            for c, t in enumerate(sorted(tnums)):
                v = tk.BooleanVar(value=True)
                self.dev_vars[(esn, t)] = v
                ttk.Checkbutton(self.dev_frame, text=_tnum_label(t), variable=v).grid(
                    row=r, column=c + 1, sticky="w", padx=3)

    def _toggle_card(self, esn: str) -> None:
        vars_ = [v for (e, _t), v in self.dev_vars.items() if e == esn]
        new = not all(v.get() for v in vars_)
        for v in vars_:
            v.set(new)

    def _sync_time_axis_controls(self) -> None:
        """Hours/cycle is meaningful only when the cycle presentation is selected."""
        state = "normal" if self.time_axis_var.get() == "cycles" else "disabled"
        self.hours_per_cycle_entry.config(state=state)

    def _set_all(self, state: bool) -> None:
        for v in self.dev_vars.values():
            v.set(state)

    def _update_excl_btn(self) -> None:
        n = len(self.excl_cond_days)
        self.excl_btn.config(text=f"Exclude days… ({n})" if n else "Exclude days…")

    def _open_excl_dialog(self) -> None:
        """A (condition x day) grid: untick a cell to drop that condition on
        that day from the whole study (calc + every plot). Baseline is locked."""
        if not self.rows:
            messagebox.showinfo("No timepoints", "Add timepoints first.")
            return
        days = [r["day"].get().strip() for r in self.rows]
        base_idx = self.baseline.get()
        base_day = days[base_idx] if 0 <= base_idx < len(days) else None
        conds = sorted({c for r in self.rows for c in conditions_in_folder(r["folder"])})
        if not conds:
            messagebox.showwarning("No conditions",
                                   "No conditions found in the masters of these timepoints.")
            return
        win = tk.Toplevel(self)
        win.title("Exclude condition-days")
        win.configure(bg=BG)
        win.transient(self)
        ttk.Label(win, text="Untick a cell to drop that condition on that day (from the "
                            "calculation and all plots). Baseline is locked.",
                  padding=(10, 8)).grid(row=0, column=0, columnspan=len(days) + 1, sticky="w")
        for c, day in enumerate(days):
            ttk.Label(win, text=(day or f"col{c + 1}"), width=9, anchor="center").grid(
                row=1, column=c + 1, padx=4, pady=2)
        cell_vars: dict[tuple[str, str], tuple[tk.BooleanVar, bool]] = {}
        for r, cond in enumerate(conds):
            ttk.Label(win, text=cond, width=18, anchor="w").grid(
                row=r + 2, column=0, sticky="w", padx=(10, 4))
            for c, day in enumerate(days):
                is_base = base_day is not None and day == base_day
                included = is_base or (cond, day) not in self.excl_cond_days
                v = tk.BooleanVar(value=included)
                cell_vars[(cond, day)] = (v, is_base)
                cb = ttk.Checkbutton(win, variable=v)
                cb.grid(row=r + 2, column=c + 1, padx=4)
                if is_base:
                    cb.state(["disabled", "selected"])

        def _ok() -> None:
            self.excl_cond_days = {(cond, day)
                                   for (cond, day), (v, is_base) in cell_vars.items()
                                   if not is_base and not v.get()}
            self._update_excl_btn()
            win.destroy()

        btns = ttk.Frame(win)
        btns.grid(row=len(conds) + 2, column=0, columnspan=len(days) + 1, pady=10)
        ttk.Button(btns, text="OK", command=_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=6)

    def _pick_out(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_dir = d
            self.out_lbl.config(text=f"output: {d}", style="TLabel")

    def _run(self) -> None:
        timepoints = [(r["day"].get().strip(), r["folder"]) for r in self.rows]
        baseline_day = timepoints[self.baseline.get()][0]
        out = self.out_dir or os.path.dirname(self.rows[0]["folder"])
        selected = {k for k, v in self.dev_vars.items() if v.get()}
        include = selected if self.dev_vars else None
        time_axis = self.time_axis_var.get()
        hours_per_cycle = 2.0
        if time_axis == "cycles":
            try:
                hours_per_cycle = float(self.hours_per_cycle_var.get())
                if not math.isfinite(hours_per_cycle) or hours_per_cycle <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid cycle duration",
                                     "Hours per cycle must be a positive number (for example, 2.0).")
                return

        def _num(s: str):
            s = s.strip()
            try:
                return float(s)
            except ValueError:
                return None
        sel_ppm = {p for p, v in self.ppm_vars.items() if v.get()}
        # all ticked (or none ticked) = full staircase -> no cut
        steps_ppm = None if (not sel_ppm or sel_ppm == set(self.ppm_vars)) else sel_ppm
        xmin, xmax = _num(self.xmin_var.get()), _num(self.xmax_var.get())
        ymin, ymax = _num(self.ymin_var.get()), _num(self.ymax_var.get())
        jg_xmin, jg_xmax = _num(self.jg_xmin_var.get()), _num(self.jg_xmax_var.get())
        jg_sweep_xlim = ((jg_xmin, jg_xmax)
                         if (jg_xmin is not None or jg_xmax is not None) else None)
        sensing_xlim = (xmin, xmax) if (xmin is not None or xmax is not None) else None
        sensing_ylim = (ymin, ymax) if (ymin is not None or ymax is not None) else None
        make_plots = self.plots_var.get()
        trend_plot_mode = self.trend_style_var.get()
        raw_plot_mode = self.raw_style_var.get()
        excluded = set(self.excl_cond_days)

        self.run_btn.config(state="disabled")
        self._log(f"Baseline='{baseline_day}'  devices={len(selected)}/{len(self.dev_vars)}  out={out}"
                  + (f"  STEPS ppm={sorted(steps_ppm)}" if steps_ppm else "")
                  + (f"  STEPS Time={sensing_xlim}" if sensing_xlim else "")
                  + (f"  STEPS Ids={sensing_ylim}" if sensing_ylim else "")
                  + (f"  JG VJG={jg_sweep_xlim}" if jg_sweep_xlim else "")
                  + f"  trends={trend_plot_mode}  raw={raw_plot_mode}"
                  + (f"  axis=cycles ({hours_per_cycle:g} h/cycle)"
                     if time_axis == "cycles" else "  axis=days"))

        def worker() -> None:
            try:
                root = run_degradation(timepoints, out_dir=out, baseline_day=baseline_day,
                                       make_plots=make_plots, include_devices=include,
                                       steps_ppm=steps_ppm, jg_sweep_xlim=jg_sweep_xlim,
                                       sensing_xlim=sensing_xlim, sensing_ylim=sensing_ylim,
                                       trend_plot_mode=trend_plot_mode,
                                       raw_plot_mode=raw_plot_mode,
                                       exclude_cond_days=excluded, time_axis=time_axis,
                                       hours_per_cycle=hours_per_cycle, log=self._queue.put)
                self._queue.put("__DONE__:" + root)
            except Exception as exc:
                self._queue.put("__ERROR__:" + str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.startswith("__DONE__:"):
                    self._log("[done] degradation -> " + msg[len("__DONE__:"):])
                    self.run_btn.config(state="normal")
                    messagebox.showinfo("Done", "Degradation study complete.")
                elif msg.startswith("__ERROR__:"):
                    self._log("[error] " + msg[len("__ERROR__:"):])
                    self.run_btn.config(state="normal")
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


def _sensorlab_project_root() -> Path:
    """Project root for both a source checkout and the one-file executable."""

    if getattr(sys, "frozen", False):
        # dist/SensorLab.exe -> sensorlab project root
        return Path(sys.executable).resolve().parent.parent
    # src/sensorlab/gui.py -> sensorlab project root
    return Path(__file__).resolve().parents[2]


class ResultsReportWindow(tk.Toplevel):
    """Build the thesis Results Outline from one or more degradation roots."""

    _ORDER = ("hot_storage", "85c_85rh", "cycles")

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.title("sensorlab — results report")
        self.geometry("980x800")
        self.minsize(820, 650)
        self.configure(bg=BG)
        self.degradation_roots: list[Path] = []
        self.sources: dict[str, ConditionSource] = {}
        self.device_vars: dict[str, tk.StringVar] = {}
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        _header(self, "Results report", "Build the Results Outline directly from SensorLab outputs")
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        files = ttk.LabelFrame(body, text="  Documents and supplemental plots  ", padding=10)
        files.pack(fill="x", pady=(0, 10))
        files.columnconfigure(1, weight=1)
        self.outline_var = tk.StringVar()
        self.supplemental_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self._file_row(files, 0, "Clean outline:", self.outline_var,
                       "Select…", self._pick_outline)
        self._file_row(files, 1, "Overview output folder:", self.supplemental_var,
                       "Select…", self._pick_supplemental)
        self._file_row(files, 2, "Output document:", self.output_var,
                       "Select…", self._pick_output)

        roots_box = ttk.LabelFrame(body, text="  Degradation sources  ", padding=10)
        roots_box.pack(fill="x", pady=(0, 10))
        root_buttons = ttk.Frame(roots_box)
        root_buttons.pack(fill="x", pady=(0, 6))
        ttk.Button(root_buttons, text="Add degradation folder…",
                   command=self._add_root).pack(side="left")
        ttk.Button(root_buttons, text="Remove selected",
                   command=self._remove_root).pack(side="left", padx=6)
        ttk.Label(root_buttons,
                  text="Add both the Hot Storage/85-85 root and the Cycles root.",
                  style="Muted.TLabel").pack(side="left", padx=8)
        self.root_list = tk.Listbox(roots_box, height=3, exportselection=False,
                                    background="white", relief="solid", borderwidth=1)
        self.root_list.pack(fill="x")

        conditions = ttk.LabelFrame(body, text="  Detected conditions and representative devices  ", padding=10)
        conditions.pack(fill="x", pady=(0, 10))
        self.conditions_frame = ttk.Frame(conditions)
        self.conditions_frame.pack(fill="x")

        note = ttk.LabelFrame(body, text="  Build policy  ", padding=8)
        note.pack(fill="x", pady=(0, 10))
        ttk.Label(
            note,
            text=("Cross-test overview plots are regenerated from the selected degradation tables. "
                  "Missing specialized plots are shown as explicit placeholders and listed in the manifest. "
                  "The clean outline is never overwritten."),
            style="Muted.TLabel",
            wraplength=900,
        ).pack(anchor="w")

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(0, 10))
        self.build_btn = ttk.Button(actions, text="▶  Build results report",
                                    style="Accent.TButton", command=self._run,
                                    state="disabled")
        self.build_btn.pack(side="left")
        ttk.Button(actions, text="Open output folder",
                   command=self._open_output_folder).pack(side="left", padx=8)

        log_box = ttk.LabelFrame(body, text="  Log  ", padding=6)
        log_box.pack(fill="both", expand=True)
        self.log = _make_log(log_box)
        self.log.pack(fill="both", expand=True)

        self._load_defaults()
        self.after(100, self._drain_report_queue)

    def _file_row(self, parent: tk.Misc, row: int, label: str,
                  variable: tk.StringVar, button: str, command) -> None:
        ttk.Label(parent, text=label, width=20).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=(4, 8), pady=3)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, pady=3)

    def _load_defaults(self) -> None:
        project = _sensorlab_project_root()
        self.supplemental_var.set(str(project / "output"))
        reliability_data = project.parent / "reliability"
        outline = reliability_data / "Thesis" / "Results outline.docx"
        if outline.is_file():
            self.outline_var.set(str(outline))
            self.output_var.set(str(outline.with_name("Results outline - SensorLab.docx")))
        for candidate in (
            reliability_data / "degradation",
            reliability_data / "cycles" / "degradation",
        ):
            if candidate.is_dir():
                self.degradation_roots.append(candidate.resolve())
        self._sync_root_list()
        self._refresh_sources(show_error=False)

    def _pick_outline(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the clean Results outline",
            filetypes=[("Word document", "*.docx"), ("All files", "*.*")],
        )
        if not path:
            return
        self.outline_var.set(path)
        current_output = self.output_var.get().strip()
        if not current_output or Path(current_output).name == "Results outline - SensorLab.docx":
            source = Path(path)
            self.output_var.set(str(source.with_name(source.stem + " - SensorLab.docx")))
        self._sync_build_state()

    def _pick_supplemental(self) -> None:
        path = filedialog.askdirectory(title="Select the SensorLab overview output folder")
        if path:
            self.supplemental_var.set(path)
            self._sync_build_state()

    def _pick_output(self) -> None:
        initial = self.output_var.get().strip()
        path = filedialog.asksaveasfilename(
            title="Save Results Outline as",
            initialdir=str(Path(initial).parent) if initial else None,
            initialfile=Path(initial).name if initial else "Results outline - SensorLab.docx",
            defaultextension=".docx",
            filetypes=[("Word document", "*.docx")],
        )
        if path:
            self.output_var.set(path)
            self._sync_build_state()

    def _add_root(self) -> None:
        path = filedialog.askdirectory(title="Select a SensorLab degradation folder")
        if not path:
            return
        candidate = Path(path).resolve()
        if candidate in self.degradation_roots:
            return
        self.degradation_roots.append(candidate)
        try:
            self._refresh_sources(show_error=True)
        except ValueError:
            self.degradation_roots.pop()
            self._refresh_sources(show_error=False)

    def _remove_root(self) -> None:
        selected = list(self.root_list.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.degradation_roots[index]
        self._sync_root_list()
        self._refresh_sources(show_error=False)

    def _sync_root_list(self) -> None:
        self.root_list.delete(0, "end")
        for root in self.degradation_roots:
            self.root_list.insert("end", str(root))

    def _refresh_sources(self, *, show_error: bool) -> None:
        old_devices = {key: var.get() for key, var in self.device_vars.items()}
        try:
            self.sources = discover_degradation_sources(self.degradation_roots) \
                if self.degradation_roots else {}
        except (OSError, ValueError) as exc:
            self.sources = {}
            if show_error:
                messagebox.showerror("Invalid degradation source", str(exc), parent=self)
            self._render_conditions(old_devices)
            self._sync_build_state()
            if show_error:
                raise ValueError(str(exc)) from exc
            return
        self._sync_root_list()
        self._render_conditions(old_devices)
        self._sync_build_state()

    def _render_conditions(self, old_devices: dict[str, str]) -> None:
        for widget in self.conditions_frame.winfo_children():
            widget.destroy()
        self.device_vars.clear()
        if not self.sources:
            ttk.Label(self.conditions_frame, text="No conditions detected.",
                      style="Muted.TLabel").pack(anchor="w")
            return
        order = [key for key in self._ORDER if key in self.sources]
        order.extend(sorted(key for key in self.sources if key not in order))
        for row, key in enumerate(order):
            source = self.sources[key]
            devices = available_report_devices(source)
            selected = old_devices.get(key, "")
            if selected not in devices:
                selected = devices[0] if devices else ""
            variable = tk.StringVar(value=selected)
            self.device_vars[key] = variable
            line = ttk.Frame(self.conditions_frame)
            line.pack(fill="x", pady=2)
            ttk.Label(line, text=source.label, width=22).pack(side="left")
            ttk.Label(line, text=str(source.directory), style="Muted.TLabel",
                      width=65).pack(side="left", fill="x", expand=True)
            ttk.Label(line, text="Device:").pack(side="left", padx=(8, 3))
            combo = ttk.Combobox(line, textvariable=variable, values=devices,
                                 state="readonly", width=21)
            combo.pack(side="left")

    def _sync_build_state(self) -> None:
        ready = (
            all(key in self.sources for key in self._ORDER)
            and all(self.device_vars.get(key) and self.device_vars[key].get()
                    for key in self._ORDER)
            and bool(self.outline_var.get().strip())
            and bool(self.output_var.get().strip())
            and bool(self.supplemental_var.get().strip())
        )
        self.build_btn.config(state="normal" if ready else "disabled")

    def _run(self) -> None:
        outline = self.outline_var.get().strip()
        supplement = self.supplemental_var.get().strip()
        output = self.output_var.get().strip()
        devices = {key: var.get() for key, var in self.device_vars.items()}
        roots = [str(root) for root in self.degradation_roots]
        if Path(output).exists():
            overwrite = messagebox.askyesno(
                "Replace existing report?",
                f"The output already exists:\n{output}\n\nReplace it?",
                parent=self,
            )
            if not overwrite:
                return
        else:
            overwrite = False
        self.build_btn.config(state="disabled")
        self._log_report(f"Building from {len(roots)} degradation roots…")

        def worker() -> None:
            try:
                result = build_results_report(
                    outline, roots, supplement, output, devices, overwrite=overwrite)
                self._queue.put(("done", result))
            except Exception as exc:
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_report_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "done":
                    result = payload
                    assert isinstance(result, ReportBuildResult)
                    self._log_report(f"Report written: {result.output_path}")
                    self._log_report(f"Manifest written: {result.manifest_path}")
                    self._log_report(
                        f"Overview plots regenerated: {len(result.overview_generated)}")
                    self._log_report(
                        f"Images added: {result.images_added}; missing: {len(result.missing_images)}")
                    self.build_btn.config(state="normal")
                    messagebox.showinfo(
                        "Results report complete",
                        f"Report created with {result.images_added} plots.\n"
                        f"Missing placeholders: {len(result.missing_images)}\n\n"
                        f"{result.output_path}",
                        parent=self,
                    )
                else:
                    self._log_report(str(payload))
                    self.build_btn.config(state="normal")
                    messagebox.showerror("Results report error", str(payload), parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain_report_queue)

    def _open_output_folder(self) -> None:
        output = self.output_var.get().strip()
        folder = Path(output).expanduser().parent if output else Path.cwd()
        if folder.is_dir():
            os.startfile(str(folder))

    def _log_report(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


class PlotStudio(tk.Toplevel):
    """Interactive plot previewer — one tab per mode, only relevant controls shown."""

    TYPES = ["SE", "STEPS", "JG", "JG_H2", "JGH", "ZOZO"]
    CMAPS = ["Blues", "Greens", "Oranges", "Purples", "Reds", "Greys", "viridis"]

    def __init__(self, master: tk.Tk, folder: str | None = None) -> None:
        super().__init__(master)
        self.title("sensorlab — plot studio")
        self.geometry("1180x720")
        self.configure(bg=BG)
        self.style_cfg = load_style()
        self.channels: list = []
        self.grouped: dict = {}
        self.se_run = self.steps_run = self.jg_run = self.air = self.h2 = self.jgh_run = None
        self.rec = None                 # JG_H2 recovery run (n06)
        self._inset_ax = None           # the draggable JG_H2 Vth inset (when present)
        self._drag = None               # active inset drag/resize state
        self._suppress = False          # guard: skip _apply_simple during programmatic var sync
        self.zozo_run = None
        self._color_buttons: dict[str, list] = {}

        _header(self, "Plot studio", "Style the plots live, then Save as default for every generated plot")
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=330)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.fig = Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # drag / resize the JG_H2 Vth inset directly on the canvas
        self.canvas.mpl_connect("button_press_event", self._inset_press)
        self.canvas.mpl_connect("motion_notify_event", self._inset_motion)
        self.canvas.mpl_connect("button_release_event", self._inset_release)

        self._init_vars()
        self._build_left(left)
        if folder:
            self._load(folder)

    # ---- shared variables (all tabs bind to these) ----------------------
    def _init_vars(self) -> None:
        s = self.style_cfg
        self.type_var = tk.StringVar(value="SE")
        self.dev_var  = tk.StringVar()
        self.ls_var       = tk.StringVar(value=s.linestyle)
        self.lw_var       = tk.DoubleVar(value=s.linewidth)
        self.vbg_vars    = {v: tk.BooleanVar(value=v in s.show_vbg) for v in (-4, 0, 4)}
        self.same_var    = tk.BooleanVar(value=s.same_vbg_color)
        self.same_ls_var = tk.BooleanVar(value=s.same_vbg_linestyle)
        self.stack_se_var    = tk.BooleanVar(value=s.se_stack_per_esn)
        self.se_grad_var     = tk.BooleanVar(value=s.se_gradient)
        self.se_cmap_var     = tk.StringVar(value=s.se_cmap)
        self.se_intensity_var = tk.DoubleVar(value=s.se_intensity)
        self.se_uniform_var  = tk.BooleanVar(value=s.se_uniform_y)
        self.se_dcshift_var  = tk.BooleanVar(value=s.se_stack_dc_shift)
        self.se_norm_var     = tk.BooleanVar(value=s.se_normalize)
        self.steps_stack_var = tk.BooleanVar(value=s.steps_stack_per_esn)
        self.steps_uniform_var = tk.BooleanVar(value=s.steps_uniform_y)
        self.steps_dcshift_var = tk.BooleanVar(value=s.steps_stack_dc_shift)
        self.steps_norm_var    = tk.BooleanVar(value=s.steps_normalize)
        self.day_var = tk.StringVar(value="")
        self.title_vars = {m: tk.StringVar(value=str((s.title_templates or {}).get(m, "")))
                           for m in self.TYPES}
        self.grad_var        = tk.BooleanVar(value=s.steps_gradient)
        self.cmap_var        = tk.StringVar(value=s.steps_cmap)
        self.intensity_var   = tk.DoubleVar(value=s.steps_intensity)
        self.show_legend_var    = tk.BooleanVar(value=s.show_legend)
        self.legend_metrics_var = tk.BooleanVar(value=s.legend_metrics)
        self.h2_label_var       = tk.StringVar(value=s.h2_label)
        self.show_recovery_var  = tk.BooleanVar(value=s.show_recovery)
        self.jg_h2_inset_var    = tk.BooleanVar(value=s.jg_h2_inset)
        self.arrow_lw_var       = tk.StringVar(value=f"{s.jg_h2_arrow_lw:g}")
        self.arrow_scale_var    = tk.StringVar(value=f"{s.jg_h2_arrow_scale:g}")
        self.arrow_h2y_var      = tk.StringVar(value=f"{s.jg_h2_arrow_h2_y:g}")
        self.arrow_recy_var     = tk.StringVar(value=f"{s.jg_h2_arrow_rec_y:g}")
        _bb = list(s.jg_h2_inset_bbox) + [0.58, 0.07, 0.40, 0.44]
        self.inset_x0_var = tk.StringVar(value=f"{_bb[0]:.3f}")
        self.inset_y0_var = tk.StringVar(value=f"{_bb[1]:.3f}")
        self.inset_w_var  = tk.StringVar(value=f"{_bb[2]:.3f}")
        self.inset_h_var  = tk.StringVar(value=f"{_bb[3]:.3f}")
        _z = list(s.jg_h2_inset_zoom) + ["", "", "", ""]
        self.zoom_vjlo_var  = tk.StringVar(value=(f"{_z[0]:g}" if _z[0] != "" else ""))
        self.zoom_vjhi_var  = tk.StringVar(value=(f"{_z[1]:g}" if _z[1] != "" else ""))
        self.zoom_idslo_var = tk.StringVar(value=(f"{_z[2]:g}" if _z[2] != "" else ""))
        self.zoom_idshi_var = tk.StringVar(value=(f"{_z[3]:g}" if _z[3] != "" else ""))
        self.show_vth_line_var  = tk.BooleanVar(value=s.show_vth_line)
        self.show_delta_vth_var = tk.BooleanVar(value=s.show_delta_vth)
        self.show_vth_marker_var = tk.BooleanVar(value=s.show_vth_marker)
        self.zozo_exp_fit_var   = tk.BooleanVar(value=s.zozo_show_exp_fit)
        self.zozo_lin_slope_var = tk.BooleanVar(value=s.zozo_show_lin_slope)
        self.zozo_stack_var     = tk.BooleanVar(value=s.zozo_stack_per_esn)
        # per-mode axis/grid settings — each tab has its own independent state,
        # seeded from (and saved back into) style.mode_axes
        axes = s.mode_axes if isinstance(s.mode_axes, dict) else {}
        def _lim_str(m, key, i):
            lim = axes.get(m, {}).get(key, [])
            return str(lim[i]) if isinstance(lim, list) and len(lim) == 2 else ""
        def _step_str(m, key):
            v = axes.get(m, {}).get(key, 0)
            try:
                return "" if float(v) == 0 else str(v)
            except (TypeError, ValueError):
                return ""
        self.yscale_vars    = {m: tk.StringVar(value=axes.get(m, {}).get("yscale", "auto")) for m in self.TYPES}
        self.xmin_vars      = {m: tk.StringVar(value=_lim_str(m, "xlim", 0)) for m in self.TYPES}
        self.xmax_vars      = {m: tk.StringVar(value=_lim_str(m, "xlim", 1)) for m in self.TYPES}
        self.ymin_vars      = {m: tk.StringVar(value=_lim_str(m, "ylim", 0)) for m in self.TYPES}
        self.ymax_vars      = {m: tk.StringVar(value=_lim_str(m, "ylim", 1)) for m in self.TYPES}
        self.show_grid_vars = {m: tk.BooleanVar(value=bool(axes.get(m, {}).get("grid", False))) for m in self.TYPES}
        self.x_step_vars    = {m: tk.StringVar(value=_step_str(m, "x_step")) for m in self.TYPES}
        self.y_step_vars    = {m: tk.StringVar(value=_step_str(m, "y_step")) for m in self.TYPES}
        # stack series show/hide: tnum label → BooleanVar (built after load)
        self.series_vars:    dict = {}
        self._series_frames: dict = {}   # mode → LabelFrame holding the checkboxes
        # legend label overrides: auto_label → StringVar with custom text
        self._legend_label_map:    dict = {}
        self._legend_label_traced: set  = set()
        self._legend_visible_map:  dict = {}  # auto_label → BooleanVar (show in legend)
        self._legend_vis_traced:   set  = set()
        self._legend_frames:       dict = {}  # mode → LabelFrame
        self._legend_rows:         dict = {}  # mode → list[widget]
        self._legend_order:        dict = {}  # mode → list[auto_label] (custom legend order)
        # traces only registered once here (not per-tab)
        self.ls_var.trace_add("write", lambda *a: self._apply_simple())
        self.cmap_var.trace_add("write", lambda *a: self._apply_simple())
        self.se_cmap_var.trace_add("write", lambda *a: self._apply_simple())
        self.h2_label_var.trace_add("write", lambda *a: self._apply_simple())
        for _v in (self.arrow_lw_var, self.arrow_scale_var, self.arrow_h2y_var, self.arrow_recy_var,
                   self.inset_x0_var, self.inset_y0_var, self.inset_w_var, self.inset_h_var,
                   self.zoom_vjlo_var, self.zoom_vjhi_var, self.zoom_idslo_var, self.zoom_idshi_var):
            _v.trace_add("write", lambda *a: self._apply_simple())
        self.day_var.trace_add("write", lambda *a: self._redraw())
        for tv in self.title_vars.values():
            tv.trace_add("write", lambda *a: self._apply_simple())

    # ---- left panel layout ----------------------------------------------
    def _build_left(self, p: tk.Misc) -> None:
        ttk.Button(p, text="Load results folder…", command=self._pick).pack(fill="x", pady=(0, 4))
        dr = ttk.Frame(p)
        dr.pack(fill="x", pady=(0, 4))
        ttk.Label(dr, text="Device:").pack(side="left")
        self.dev_cb = ttk.Combobox(dr, textvariable=self.dev_var, state="readonly")
        self.dev_cb.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.dev_cb.bind("<<ComboboxSelected>>",
                         lambda e: (self._rebuild_series_boxes(), self._redraw()))
        ttk.Label(dr, text="Day#:").pack(side="left", padx=(6, 0))
        ttk.Entry(dr, textvariable=self.day_var, width=8).pack(side="left", padx=(4, 0))

        self.nb = ttk.Notebook(p)
        self.nb.pack(fill="both", expand=True, pady=(2, 4))
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        for mode in self.TYPES:
            self._add_tab(mode)

        ttk.Button(p, text="💾  Save as default style", style="Accent.TButton",
                   command=self._save).pack(fill="x")

    def _add_tab(self, mode: str) -> None:
        outer = ttk.Frame(self.nb)
        self.nb.add(outer, text=f"  {mode}  ")
        cvs  = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb   = ttk.Scrollbar(outer, orient="vertical", command=cvs.yview)
        innr = ttk.Frame(cvs)
        sb.pack(side="right", fill="y")
        cvs.pack(side="left", fill="both", expand=True)
        wid = cvs.create_window((0, 0), window=innr, anchor="nw")
        innr.bind("<Configure>", lambda e, c=cvs:          c.configure(scrollregion=c.bbox("all")))
        cvs.bind( "<Configure>", lambda e, c=cvs, w=wid:  c.itemconfig(w, width=e.width))
        _s = lambda ev, c=cvs: c.yview_scroll(int(-1 * (ev.delta / 120)), "units")
        outer.bind("<Enter>", lambda e, f=_s: self.bind_all("<MouseWheel>", f))
        outer.bind("<Leave>", lambda e:       self.unbind_all("<MouseWheel>"))
        getattr(self, f"_tab_{mode.lower()}")(innr)
        self._tab_common(innr, mode)

    def _on_tab_change(self, _event: object) -> None:
        try:
            self.type_var.set(self.TYPES[self.nb.index(self.nb.select())])
        except Exception:
            pass
        # re-read this tab's own axis/grid vars into the style before drawing,
        # so limits set on another tab never leak into this one
        self._apply_simple()

    # ---- colour helpers (synced across all tabs) -------------------------
    def _lf(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=f"  {title}  ", padding=8)
        f.pack(fill="x", padx=4, pady=3)
        return f

    def _color_btn(self, parent: tk.Misc, label: str, attr: str,
                   bg_gate: str | None = None) -> None:
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=1)
        ttk.Label(row, text=label, width=14).pack(side="left")
        key = f"bg_{bg_gate}" if bg_gate is not None else attr
        cur = (self.style_cfg.bg_colors.get(bg_gate, "#333") if bg_gate
               else getattr(self.style_cfg, attr, "#333"))
        b = tk.Button(row, width=3, bg=cur, relief="ridge",
                      command=lambda: self._pick_color(attr, bg_gate))
        b.pack(side="left")
        self._color_buttons.setdefault(key, []).append(b)

    def _pick_color(self, attr: str, bg_gate: str | None = None) -> None:
        key = f"bg_{bg_gate}" if bg_gate is not None else attr
        cur = (self.style_cfg.bg_colors.get(bg_gate) if bg_gate
               else getattr(self.style_cfg, attr))
        c = colorchooser.askcolor(color=cur, parent=self)[1]
        if c:
            if bg_gate is not None:
                self.style_cfg.bg_colors[bg_gate] = c
            else:
                setattr(self.style_cfg, attr, c)
            for btn in self._color_buttons.get(key, []):
                btn.config(bg=c)
            self._redraw()

    def _vbg_row(self, p: tk.Misc, *, same_color: bool = False, same_ls: bool = False) -> None:
        f = self._lf(p, "Back-gate (Vbg)")
        for v in (-4, 0, 4):
            ttk.Checkbutton(f, text=f"BG {v}", variable=self.vbg_vars[v],
                            command=self._apply_vbg).pack(side="left", padx=2)
        if same_color:
            ttk.Checkbutton(f, text="same color", variable=self.same_var,
                            command=self._apply_simple).pack(side="left", padx=6)
        if same_ls:
            ttk.Checkbutton(f, text="same LS", variable=self.same_ls_var,
                            command=self._apply_simple).pack(side="left", padx=6)

    def _bg_colors(self, p: tk.Misc) -> None:
        f = self._lf(p, "Colors")
        self._color_btn(f, "BG 0", "bg_colors", "0")
        self._color_btn(f, "BG 4", "bg_colors", "4")
        self._color_btn(f, "Same / single", "single_vbg_color")

    # ---- per-mode tab knobs --------------------------------------------
    def _tab_se(self, p: tk.Misc) -> None:
        cf = self._lf(p, "Colors")
        self._color_btn(cf, "SE / line", "se_color")
        sf = self._lf(p, "Stack per ESN")
        ttk.Checkbutton(sf, text="Stack transistors per card",
                        variable=self.stack_se_var, command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(sf, text="DC-shift traces (each min to 0, stack)",
                        variable=self.se_dcshift_var, command=self._apply_simple).pack(anchor="w")
        self._series_frames["SE"] = self._lf(p, "Stack series (show/hide)")
        ttk.Label(self._series_frames["SE"], text="(load data to list transistors)",
                  style="Muted.TLabel").pack(anchor="w")
        yf = self._lf(p, "Scaling")
        ttk.Checkbutton(yf, text="auto-center Y (data min/max ±1000 nA)",
                        variable=self.se_uniform_var, command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(yf, text="normalized view (response % vs baseline)",
                        variable=self.se_norm_var, command=self._apply_simple).pack(anchor="w")
        gf = self._lf(p, "Exposure band")
        ttk.Checkbutton(gf, text="gradient on", variable=self.se_grad_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Combobox(gf, textvariable=self.se_cmap_var, values=self.CMAPS,
                     state="readonly").pack(fill="x", pady=2)
        ttk.Label(gf, text="intensity").pack(anchor="w")
        ttk.Scale(gf, from_=0.05, to=1.0, variable=self.se_intensity_var,
                  command=lambda v: self._apply_simple()).pack(fill="x")

    def _tab_steps(self, p: tk.Misc) -> None:
        cf = self._lf(p, "Colors")
        self._color_btn(cf, "Line", "se_color")
        sf = self._lf(p, "Stack per ESN")
        ttk.Checkbutton(sf, text="Stack transistors per card",
                        variable=self.steps_stack_var, command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(sf, text="DC-shift traces (each min to 0, stack)",
                        variable=self.steps_dcshift_var, command=self._apply_simple).pack(anchor="w")
        self._series_frames["STEPS"] = self._lf(p, "Stack series (show/hide)")
        ttk.Label(self._series_frames["STEPS"], text="(load data to list transistors)",
                  style="Muted.TLabel").pack(anchor="w")
        yf = self._lf(p, "Scaling")
        ttk.Checkbutton(yf, text="auto-center Y (data min/max ±1000 nA)",
                        variable=self.steps_uniform_var, command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(yf, text="normalized view (response % vs baseline)",
                        variable=self.steps_norm_var, command=self._apply_simple).pack(anchor="w")
        gf = self._lf(p, "Gradient")
        ttk.Checkbutton(gf, text="gradient on", variable=self.grad_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Combobox(gf, textvariable=self.cmap_var, values=self.CMAPS,
                     state="readonly").pack(fill="x", pady=2)
        ttk.Label(gf, text="intensity").pack(anchor="w")
        ttk.Scale(gf, from_=0.2, to=1.0, variable=self.intensity_var,
                  command=lambda v: self._apply_simple()).pack(fill="x")

    def _tab_jg(self, p: tk.Misc) -> None:
        self._vbg_row(p, same_color=True)
        self._bg_colors(p)
        mf = self._lf(p, "Markers")
        ttk.Checkbutton(mf, text="Vth marker (red dot)",
                        variable=self.show_vth_marker_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(mf, text=f"Vth current line ({PROTO.jg.vth_target_nA:g} nA)",
                        variable=self.show_vth_line_var,
                        command=self._apply_simple).pack(anchor="w")

    def _tab_jg_h2(self, p: tk.Misc) -> None:
        self._vbg_row(p, same_ls=True)
        cf = self._lf(p, "Colors")
        self._color_btn(cf, "Air", "air_color")
        self._color_btn(cf, "H2", "h2_color")
        h2r = ttk.Frame(cf)
        h2r.pack(anchor="w", pady=(4, 0))
        ttk.Label(h2r, text="H2 label:", width=10).pack(side="left")
        ttk.Entry(h2r, textvariable=self.h2_label_var, width=14).pack(side="left", padx=4)

        rf = self._lf(p, "Recovery + Vth inset")
        ttk.Checkbutton(rf, text="Recovery curve (n06)", variable=self.show_recovery_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(rf, text="Zoom inset  (drag to move on the plot)",
                        variable=self.jg_h2_inset_var,
                        command=self._apply_simple).pack(anchor="w")

        def _fields(title, pairs):
            ttk.Label(rf, text=title).pack(anchor="w", pady=(4, 0))
            grid = ttk.Frame(rf); grid.pack(anchor="w")
            for i, (lbl, var) in enumerate(pairs):
                ttk.Label(grid, text=lbl, width=7).grid(row=i // 2, column=(i % 2) * 2, sticky="w")
                ttk.Entry(grid, textvariable=var, width=7).grid(
                    row=i // 2, column=(i % 2) * 2 + 1, padx=(0, 8), pady=1)

        _fields("Inset position / size (0–1):",
                [("x0", self.inset_x0_var), ("y0", self.inset_y0_var),
                 ("width", self.inset_w_var), ("height", self.inset_h_var)])
        _fields("Zoom region  (blank = auto):",
                [("Vj min", self.zoom_vjlo_var), ("Vj max", self.zoom_vjhi_var),
                 ("Ids min", self.zoom_idslo_var), ("Ids max", self.zoom_idshi_var)])

        mf = self._lf(p, "Markers")
        ttk.Checkbutton(mf, text=f"Vth current line ({PROTO.jg.vth_target_nA:g} nA)",
                        variable=self.show_vth_line_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(mf, text="ΔVth arrow",
                        variable=self.show_delta_vth_var,
                        command=self._apply_simple).pack(anchor="w")

    def _tab_jgh(self, p: tk.Misc) -> None:
        self._vbg_row(p, same_color=True)
        self._bg_colors(p)

    def _tab_zozo(self, p: tk.Misc) -> None:
        cf = self._lf(p, "Colors")
        self._color_btn(cf, "Line", "se_color")
        sf = self._lf(p, "Stack per ESN")
        ttk.Checkbutton(sf, text="Stack transistors per card",
                        variable=self.zozo_stack_var, command=self._apply_simple).pack(anchor="w")
        self._series_frames["ZOZO"] = self._lf(p, "Stack series (show/hide)")
        ttk.Label(self._series_frames["ZOZO"], text="(load data to list transistors)",
                  style="Muted.TLabel").pack(anchor="w")
        of = self._lf(p, "Overlays")
        ttk.Checkbutton(of, text="exponential fit (dashed)",
                        variable=self.zozo_exp_fit_var,
                        command=self._apply_simple).pack(anchor="w")
        ttk.Checkbutton(of, text="linear slope line (first 4 → last 4 pts)",
                        variable=self.zozo_lin_slope_var,
                        command=self._apply_simple).pack(anchor="w")

    # ---- common bottom controls (Title / Line / Legend / Scale / Limits / Grid) -
    def _tab_common(self, p: tk.Misc, mode: str) -> None:
        tf = self._lf(p, "Title  (blank = default)")
        ttk.Entry(tf, textvariable=self.title_vars[mode]).pack(fill="x")
        ttk.Label(tf, text="{ESN} = card (1074)   {T} = transistor (D1T2)",
                  style="Muted.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(tf, text="{Mode} = this tab      {Day} = Day# field",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Label(tf, text="{WP} = working point Vj/Vb (States sheet;",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Label(tf, text="   SE/STEPS: State1, ZOZO: State1↔State2)",
                  style="Muted.TLabel").pack(anchor="w")

        lf = self._lf(p, "Line")
        lr = ttk.Frame(lf)
        lr.pack(fill="x")
        ttk.Label(lr, text="style:").pack(side="left")
        ttk.Combobox(lr, textvariable=self.ls_var, width=6, state="readonly",
                     values=["-", "--", ":", "-."]).pack(side="left", padx=4)
        ttk.Label(lr, text="width").pack(side="left", padx=(10, 2))
        ttk.Scale(lr, from_=0.5, to=3.0, variable=self.lw_var,
                  command=lambda v: self._apply_simple()).pack(side="left", fill="x", expand=True)

        lgf = self._lf(p, "Legend")
        ttk.Checkbutton(lgf, text="show legend", variable=self.show_legend_var,
                        command=self._apply_simple).pack(side="left")
        ttk.Checkbutton(lgf, text="metrics in labels", variable=self.legend_metrics_var,
                        command=self._apply_simple).pack(side="left", padx=8)

        ysf = self._lf(p, "Y-axis scale")
        for lbl in ("auto", "log", "linear"):
            ttk.Radiobutton(ysf, text=lbl, variable=self.yscale_vars[mode], value=lbl,
                            command=self._apply_simple).pack(side="left", padx=4)

        limf = self._lf(p, "Axis limits  (blank = auto)")
        xr = ttk.Frame(limf)
        xr.pack(fill="x")
        ttk.Label(xr, text="X:", width=3).pack(side="left")
        ttk.Entry(xr, textvariable=self.xmin_vars[mode], width=7).pack(side="left", padx=2)
        ttk.Label(xr, text="—").pack(side="left")
        ttk.Entry(xr, textvariable=self.xmax_vars[mode], width=7).pack(side="left", padx=2)
        yr = ttk.Frame(limf)
        yr.pack(fill="x", pady=(3, 0))
        ttk.Label(yr, text="Y:", width=3).pack(side="left")
        ttk.Entry(yr, textvariable=self.ymin_vars[mode], width=7).pack(side="left", padx=2)
        ttk.Label(yr, text="—").pack(side="left")
        ttk.Entry(yr, textvariable=self.ymax_vars[mode], width=7).pack(side="left", padx=2)
        ttk.Button(limf, text="Apply", command=self._apply_simple).pack(anchor="e", pady=(4, 0))

        gf = self._lf(p, "Grid & ticks")
        ttk.Checkbutton(gf, text="major grid", variable=self.show_grid_vars[mode],
                        command=self._apply_simple).pack(anchor="w")
        xsr = ttk.Frame(gf)
        xsr.pack(fill="x", pady=(3, 0))
        ttk.Label(xsr, text="X step:", width=7).pack(side="left")
        ttk.Entry(xsr, textvariable=self.x_step_vars[mode], width=8).pack(side="left", padx=2)
        ttk.Label(xsr, text="(blank=auto)", style="Muted.TLabel").pack(side="left", padx=4)
        ysr = ttk.Frame(gf)
        ysr.pack(fill="x", pady=(2, 0))
        ttk.Label(ysr, text="Y step:", width=7).pack(side="left")
        ttk.Entry(ysr, textvariable=self.y_step_vars[mode], width=8).pack(side="left", padx=2)
        ttk.Label(ysr, text="(linear only)", style="Muted.TLabel").pack(side="left", padx=4)
        ttk.Button(gf, text="Apply", command=self._apply_simple).pack(anchor="e", pady=(4, 0))

        lglf = self._lf(p, "Legend labels")
        self._legend_frames[mode] = lglf
        self._legend_rows[mode]   = []

    # ---- legend label editor -------------------------------------------
    def _rebuild_legend_editor(self, mode: str) -> None:
        frame = self._legend_frames.get(mode)
        if frame is None:
            return
        for w in self._legend_rows.get(mode, []):
            w.destroy()
        self._legend_rows[mode] = []
        handles, auto_labels = self.ax.get_legend_handles_labels()
        if not handles:
            lbl = ttk.Label(frame, text="(no legend entries)", style="Muted.TLabel")
            lbl.pack(anchor="w")
            self._legend_rows[mode].append(lbl)
            return
        # display in the stored custom order; new entries append in natural order
        order = [l for l in self._legend_order.get(mode, []) if l in auto_labels]
        order += [l for l in auto_labels if l not in order]
        self._legend_order[mode] = order
        for pos, auto in enumerate(order):
            if auto not in self._legend_label_map:
                self._legend_label_map[auto] = tk.StringVar(value=auto)
            if auto not in self._legend_label_traced:
                self._legend_label_map[auto].trace_add("write", lambda *a: self._apply_legend_labels())
                self._legend_label_traced.add(auto)
            if auto not in self._legend_visible_map:
                self._legend_visible_map[auto] = tk.BooleanVar(value=True)
            if auto not in self._legend_vis_traced:
                self._legend_visible_map[auto].trace_add("write", lambda *a: self._apply_legend_labels())
                self._legend_vis_traced.add(auto)
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=1)
            ttk.Checkbutton(row, variable=self._legend_visible_map[auto]).pack(side="left")
            up = ttk.Button(row, text="▲", width=2,
                            command=lambda a=auto: self._legend_move(mode, a, -1))
            up.pack(side="left")
            up.state(["disabled"] if pos == 0 else ["!disabled"])
            dn = ttk.Button(row, text="▼", width=2,
                            command=lambda a=auto: self._legend_move(mode, a, +1))
            dn.pack(side="left")
            dn.state(["disabled"] if pos == len(order) - 1 else ["!disabled"])
            ttk.Entry(row, textvariable=self._legend_label_map[auto]).pack(side="left", fill="x", expand=True, padx=2)
            self._legend_rows[mode].append(row)

    def _legend_move(self, mode: str, auto: str, delta: int) -> None:
        """Move a legend entry up/down in the custom order, then re-render."""
        order = self._legend_order.get(mode, [])
        if auto not in order:
            return
        i = order.index(auto)
        j = i + delta
        if 0 <= j < len(order):
            order[i], order[j] = order[j], order[i]
            self._legend_order[mode] = order
            self._rebuild_legend_editor(mode)
            self._apply_legend_labels()

    _MODE_LEGEND = {
        "SE":    ("best",        8),
        "STEPS": ("best",        8),
        "JG":    ("upper left",  8),
        "JG_H2": ("upper left",  7),
        "JGH":   ("upper left",  8),
        "ZOZO":  ("best",        8),
    }

    def _apply_legend_labels(self) -> None:
        handles, auto_labels = self.ax.get_legend_handles_labels()
        if not handles:
            self.canvas.draw()
            return
        mode = self.type_var.get()
        loc, fs = self._MODE_LEGEND.get(mode, ("best", 8))
        # reorder handles/labels by the stored custom order (new entries kept last)
        order = self._legend_order.get(mode)
        if order:
            hmap = dict(zip(auto_labels, handles))
            pairs = [(hmap[l], l) for l in order if l in hmap]
            pairs += [(h, l) for h, l in zip(handles, auto_labels) if l not in order]
            handles, auto_labels = [p[0] for p in pairs], [p[1] for p in pairs]
        visible = [
            (h, self._legend_label_map.get(l, tk.StringVar(value=l)).get() or l)
            for h, l in zip(handles, auto_labels)
            if self._legend_visible_map.get(l, tk.BooleanVar(value=True)).get()
        ]
        if self.style_cfg.show_legend:
            if visible:
                fh, fl = zip(*visible)
                self.ax.legend(handles=list(fh), labels=list(fl), fontsize=fs, loc=loc)
            else:
                lg = self.ax.get_legend()
                if lg:
                    lg.remove()
        self.canvas.draw()

    # ---- stack series show/hide (per card / ESN) --------------------------
    def _rebuild_series_boxes(self) -> None:
        ch = self._channel()
        esn = ch.esn if ch else None
        labels = sorted({c.tnum_label for c in self.channels if esn is not None and c.esn == esn})
        hidden = set((self.style_cfg.stack_hidden or {}).get(str(esn), []))
        self.series_vars = {t: tk.BooleanVar(value=t not in hidden) for t in labels}
        for frame in self._series_frames.values():
            frame.config(text=f"  Stack series ESN{esn} (show/hide)  " if esn else "  Stack series (show/hide)  ")
            for w in frame.winfo_children():
                w.destroy()
            if not labels:
                ttk.Label(frame, text="(load data to list transistors)",
                          style="Muted.TLabel").pack(anchor="w")
                continue
            for t in labels:
                ttk.Checkbutton(frame, text=t, variable=self.series_vars[t],
                                command=self._apply_series).pack(side="left", padx=3)

    def _apply_series(self) -> None:
        ch = self._channel()
        if ch is None:
            return
        if not isinstance(self.style_cfg.stack_hidden, dict):
            self.style_cfg.stack_hidden = {}
        self.style_cfg.stack_hidden[str(ch.esn)] = [
            t for t, v in self.series_vars.items() if not v.get()]
        self._redraw()

    # ---- apply / draw --------------------------------------------------
    def _apply_vbg(self) -> None:
        self.style_cfg.show_vbg = [v for v, bv in self.vbg_vars.items() if bv.get()]
        self._redraw()

    @staticmethod
    def _parse_lim(mn: tk.StringVar, mx: tk.StringVar) -> list:
        try:
            return [float(mn.get()), float(mx.get())]
        except (ValueError, TypeError):
            return []

    def _apply_simple(self) -> None:
        if getattr(self, "_suppress", False):     # programmatic var updates (drag sync)
            return
        s = self.style_cfg
        s.same_vbg_color     = self.same_var.get()
        s.same_vbg_linestyle = self.same_ls_var.get()
        s.se_stack_per_esn   = self.stack_se_var.get()
        s.se_gradient        = self.se_grad_var.get()
        s.se_cmap            = self.se_cmap_var.get()
        s.se_intensity       = float(self.se_intensity_var.get())
        s.se_uniform_y       = self.se_uniform_var.get()
        s.se_stack_dc_shift  = self.se_dcshift_var.get()
        s.se_normalize       = self.se_norm_var.get()
        s.steps_stack_per_esn = self.steps_stack_var.get()
        s.steps_uniform_y     = self.steps_uniform_var.get()
        s.steps_stack_dc_shift = self.steps_dcshift_var.get()
        s.steps_normalize      = self.steps_norm_var.get()
        s.steps_gradient     = self.grad_var.get()
        s.steps_cmap         = self.cmap_var.get()
        s.steps_intensity    = float(self.intensity_var.get())
        s.linewidth          = float(self.lw_var.get())
        s.linestyle          = self.ls_var.get()
        s.show_legend        = self.show_legend_var.get()
        s.legend_metrics     = self.legend_metrics_var.get()
        s.h2_label           = self.h2_label_var.get() or "100ppm H2"
        s.show_recovery      = self.show_recovery_var.get()
        s.jg_h2_inset        = self.jg_h2_inset_var.get()

        def _pf(var, default):
            try:    return float(var.get())
            except (ValueError, TypeError): return default
        s.jg_h2_arrow_lw     = _pf(self.arrow_lw_var, s.jg_h2_arrow_lw)
        s.jg_h2_arrow_scale  = _pf(self.arrow_scale_var, s.jg_h2_arrow_scale)
        s.jg_h2_arrow_h2_y   = _pf(self.arrow_h2y_var, s.jg_h2_arrow_h2_y)
        s.jg_h2_arrow_rec_y  = _pf(self.arrow_recy_var, s.jg_h2_arrow_rec_y)
        s.jg_h2_inset_bbox   = [_pf(self.inset_x0_var, 0.58), _pf(self.inset_y0_var, 0.07),
                                _pf(self.inset_w_var, 0.40), _pf(self.inset_h_var, 0.44)]
        _zv = [self.zoom_vjlo_var.get(), self.zoom_vjhi_var.get(),
               self.zoom_idslo_var.get(), self.zoom_idshi_var.get()]
        if all(v.strip() for v in _zv):
            try:    s.jg_h2_inset_zoom = [float(v) for v in _zv]
            except ValueError: s.jg_h2_inset_zoom = []
        else:
            s.jg_h2_inset_zoom = []
        s.show_vth_line      = self.show_vth_line_var.get()
        s.show_delta_vth     = self.show_delta_vth_var.get()
        s.show_vth_marker    = self.show_vth_marker_var.get()
        s.zozo_show_exp_fit   = self.zozo_exp_fit_var.get()
        s.zozo_show_lin_slope = self.zozo_lin_slope_var.get()
        s.zozo_stack_per_esn  = self.zozo_stack_var.get()
        mode = self.type_var.get()
        if not isinstance(s.title_templates, dict):
            s.title_templates = {}
        s.title_templates[mode] = self.title_vars[mode].get()
        # write EVERY mode's axis settings so Save persists all tabs, not just
        # the active one (fixes limits leaking between modes in the pipeline)
        if not isinstance(s.mode_axes, dict):
            s.mode_axes = {}
        for m in self.TYPES:
            def _f(var):
                try:    return float(var.get())
                except (ValueError, TypeError): return 0.0
            s.mode_axes[m] = {
                "yscale": self.yscale_vars[m].get(),
                "xlim":   self._parse_lim(self.xmin_vars[m], self.xmax_vars[m]),
                "ylim":   self._parse_lim(self.ymin_vars[m], self.ymax_vars[m]),
                "grid":   self.show_grid_vars[m].get(),
                "x_step": _f(self.x_step_vars[m]),
                "y_step": _f(self.y_step_vars[m]),
            }
        self._redraw()

    def _pick(self) -> None:
        d = filedialog.askdirectory(title="Select results folder")
        if d:
            self._load(d)

    def _load(self, folder: str) -> None:
        try:
            runs = load_folder(folder)
            self.grouped = group_by_kind(runs)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc), parent=self)
            return
        g = self.grouped
        self.se_run    = next(iter(g.get(MajorKind.SE, [])), None)
        self.steps_run = next(iter(g.get(MajorKind.STEPS, [])), None)
        jg = g.get(MajorKind.JG, [])
        by = {r.major: r for r in jg}
        self.jg_run = by.get(PROTO.jg.h2_air_major) or (jg[0] if jg else None)
        self.air, self.h2 = by.get(PROTO.jg.h2_air_major), by.get(PROTO.jg.h2_exposure_major)
        self.rec = by.get(PROTO.jg.h2_recovery_major)
        self.jgh_run = next(iter(g.get(MajorKind.JGH, [])), None)
        self.zozo_run = next(iter(g.get(MajorKind.ZOZO, [])), None)
        avail = {"SE": bool(self.se_run), "STEPS": bool(self.steps_run),
                 "JG": bool(jg), "JG_H2": bool(self.air and self.h2),
                 "JGH": bool(self.jgh_run), "ZOZO": bool(self.zozo_run)}
        for i, mode in enumerate(self.TYPES):
            self.nb.tab(i, state="normal" if avail[mode] else "disabled")
        for i, mode in enumerate(self.TYPES):
            if avail[mode]:
                self.nb.select(i)
                self.type_var.set(mode)
                break
        rep = self.se_run or self.jg_run or self.jgh_run or self.steps_run or self.zozo_run
        self.channels = rep.channels if rep else []
        self.dev_cb.config(values=[c.label for c in self.channels])
        if self.channels:
            self.dev_var.set(self.channels[0].label)
        self._rebuild_series_boxes()
        self._redraw()

    def _channel(self) -> object | None:
        for c in self.channels:
            if c.label == self.dev_var.get():
                return c
        return self.channels[0] if self.channels else None

    # -- draggable JG_H2 Vth inset ---------------------------------------
    def _inset_frac(self, event):
        """Mouse position in figure fraction, or None if off-canvas."""
        if event.x is None or event.y is None:
            return None
        return event.x / self.fig.bbox.width, event.y / self.fig.bbox.height

    def _inset_press(self, event) -> None:
        if self.type_var.get() != "JG_H2" or self._inset_ax is None:
            return
        pt = self._inset_frac(event)
        if pt is None:
            return
        fx, fy = pt
        pos = self._inset_ax.get_position()
        if not (pos.x0 <= fx <= pos.x1 and pos.y0 <= fy <= pos.y1):
            return
        corner = fx > pos.x1 - 0.04 and fy > pos.y1 - 0.04   # top-right -> resize
        self._drag = {"mode": "resize" if corner else "move", "fx": fx, "fy": fy,
                      "bbox0": list(self.style_cfg.jg_h2_inset_bbox)}

    def _inset_apply_bbox(self, bbox) -> None:
        self.style_cfg.jg_h2_inset_bbox = bbox
        ap = self.ax.get_position()
        self._inset_ax.set_position([ap.x0 + bbox[0] * ap.width, ap.y0 + bbox[1] * ap.height,
                                     bbox[2] * ap.width, bbox[3] * ap.height])
        self.canvas.draw_idle()

    def _inset_motion(self, event) -> None:
        if not self._drag or self._inset_ax is None:
            return
        pt = self._inset_frac(event)
        if pt is None:
            return
        ap = self.ax.get_position()
        dfx = (pt[0] - self._drag["fx"]) / ap.width      # delta in axes fraction
        dfy = (pt[1] - self._drag["fy"]) / ap.height
        x0, y0, w, h = self._drag["bbox0"]
        if self._drag["mode"] == "move":
            x0 = min(max(x0 + dfx, 0.0), 1.0 - w)
            y0 = min(max(y0 + dfy, 0.0), 1.0 - h)
        else:                                            # resize from the top-right corner
            w = min(max(w + dfx, 0.15), 1.0 - x0)
            h = min(max(h + dfy, 0.15), 1.0 - y0)
        self._inset_apply_bbox([x0, y0, w, h])

    def _inset_release(self, event) -> None:
        if self._drag:
            self._drag = None
            self._sync_inset_fields()   # reflect the dragged bbox in the entry fields
            self._redraw()              # re-anchor the connector lines at the new position
            self.canvas.draw()

    def _sync_inset_fields(self) -> None:
        b = list(self.style_cfg.jg_h2_inset_bbox)
        self._suppress = True
        self.inset_x0_var.set(f"{b[0]:.3f}"); self.inset_y0_var.set(f"{b[1]:.3f}")
        self.inset_w_var.set(f"{b[2]:.3f}");  self.inset_h_var.set(f"{b[3]:.3f}")
        self._suppress = False

    def _redraw(self) -> None:
        self.ax.clear()
        self._inset_ax = None
        ch = self._channel()
        t  = self.type_var.get()
        s  = self.style_cfg
        day = self.day_var.get().strip()
        try:
            if ch is None:
                self.ax.text(0.5, 0.5, "Load a results folder", ha="center", va="center")
            elif t == "SE" and self.se_run:
                chs = [c for c in self.se_run.channels if c.esn == ch.esn] if s.se_stack_per_esn else None
                (render_se_stack if chs else render_se)(self.ax, self.se_run, chs or ch, s, day=day)
            elif t == "STEPS" and self.steps_run:
                chs = [c for c in self.steps_run.channels if c.esn == ch.esn] if s.steps_stack_per_esn else None
                (render_steps_stack if chs else render_steps)(self.ax, self.steps_run, chs or ch, s, day=day)
            elif t == "JG" and self.jg_run:
                render_jg(self.ax, self.jg_run, ch, s, day=day)
            elif t == "JG_H2" and self.air and self.h2:
                self._inset_ax = render_jg_h2(self.ax, self.air, self.h2, ch, s, day=day, rec=self.rec)
            elif t == "JGH" and self.jgh_run:
                render_jgh(self.ax, self.jgh_run, ch, s, day=day)
            elif t == "ZOZO" and self.zozo_run:
                chs = [c for c in self.zozo_run.channels if c.esn == ch.esn] if s.zozo_stack_per_esn else None
                (render_zozo_stack if chs else render_zozo)(self.ax, self.zozo_run, chs or ch, s, day=day)
            else:
                self.ax.text(0.5, 0.5, f"No {t} data in this folder", ha="center", va="center")
        except Exception as exc:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}",
                         ha="center", va="center", wrap=True)
        self.fig.tight_layout()
        self._rebuild_legend_editor(self.type_var.get())
        self._apply_legend_labels()

    def _save(self) -> None:
        self.style_cfg.save()
        messagebox.showinfo("Saved", "Style saved as default — used by every generated plot.", parent=self)


def main() -> None:
    SensorLabApp().mainloop()


if __name__ == "__main__":
    main()
