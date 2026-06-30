"""Thin Tkinter front-end for the sensorlab pipeline.

Pick a results folder, enter the Day#, fill the condition per card (cached and
remembered), then run. All analysis lives in the headless core; this is only a
shell around `analyze_folder` (+ a degradation window around `run_degradation`).
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import colorchooser

from .compare import run_degradation
from .degradation import devices_in_folder, find_masters
from .io.device_map import load_condition_cache
from .model import MajorKind
from .pipeline import all_esns, analyze_folder, group_by_kind, load_folder
from .plotting import (
    load_style, render_jg, render_jg_h2, render_jgh, render_se, render_se_stack,
    render_steps, render_steps_stack,
)
from .protocol import DEFAULT as PROTO

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
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            written = analyze_folder(self.folder, conditions=self.cache,
                                     make_plots=self.plots_var.get(), day=self.day_var.get().strip(),
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
        self.geometry("820x780")
        self.configure(bg=BG)
        self.rows: list[dict] = []
        self.dev_vars: dict[tuple[str, str], tk.BooleanVar] = {}
        self.baseline = tk.IntVar(value=0)
        self.out_dir: str | None = None
        self._queue: queue.Queue[str] = queue.Queue()

        _header(self, "Compare timepoints", "Degradation across days — delta, % degradation, averages ± std")
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        f1 = ttk.LabelFrame(body, text="  Timepoints  (pick the baseline)  ", padding=10)
        f1.pack(fill="x", pady=(0, 10))
        bt = ttk.Frame(f1)
        bt.pack(fill="x")
        ttk.Button(bt, text="Add timepoint…", command=self._add).pack(side="left")
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

        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(0, 10))
        self.plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Trend plots", variable=self.plots_var).pack(side="left")
        self.run_btn = ttk.Button(bar, text="▶  Run degradation", style="Accent.TButton",
                                  command=self._run, state="disabled")
        self.run_btn.pack(side="left", padx=10)

        f3 = ttk.LabelFrame(body, text="  Log  ", padding=6)
        f3.pack(fill="both", expand=True)
        self.log = _make_log(f3)
        self.log.pack(fill="both", expand=True)
        self.after(100, self._drain)

    def _add(self) -> None:
        folder = filedialog.askdirectory(title="Select a timepoint folder")
        if not folder:
            return
        if not list(find_masters(folder)):
            messagebox.showwarning("No masters", "No analysis masters found in this folder.")
            return
        i = len(self.rows)
        row = ttk.Frame(self.rows_frame)
        row.pack(fill="x", pady=1)
        ttk.Radiobutton(row, variable=self.baseline, value=i).pack(side="left")
        ttk.Label(row, text=os.path.basename(folder), width=44).pack(side="left")
        day_var = tk.StringVar(value=_peek_day(folder) or "")
        ttk.Label(row, text="day:").pack(side="left")
        ttk.Entry(row, textvariable=day_var, width=10).pack(side="left")
        self.rows.append({"folder": folder, "day": day_var})
        self.run_btn.config(state="normal")
        if not self.dev_vars:
            self._populate_devices(folder)

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

    def _set_all(self, state: bool) -> None:
        for v in self.dev_vars.values():
            v.set(state)

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
        self.run_btn.config(state="disabled")
        self._log(f"Baseline='{baseline_day}'  devices={len(selected)}/{len(self.dev_vars)}  out={out}")

        def worker() -> None:
            try:
                root = run_degradation(timepoints, out_dir=out, baseline_day=baseline_day,
                                       make_plots=self.plots_var.get(), include_devices=include,
                                       log=self._queue.put)
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


class PlotStudio(tk.Toplevel):
    """Interactive plot previewer: tweak the style live, save it as the default."""

    TYPES = ["SE", "STEPS", "JG", "JG_H2", "JGH"]
    CMAPS = ["Blues", "Greens", "Oranges", "Purples", "Reds", "Greys", "viridis"]

    def __init__(self, master: tk.Tk, folder: str | None = None) -> None:
        super().__init__(master)
        self.title("sensorlab — plot studio")
        self.geometry("1120x680")
        self.configure(bg=BG)
        self.style_cfg = load_style()
        self.channels: list = []
        self.grouped: dict = {}
        self.se_run = self.steps_run = self.jg_run = self.air = self.h2 = self.jgh_run = None

        _header(self, "Plot studio", "Style the plots live, then Save as default for every generated plot")
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root, width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # preview canvas
        self.fig = Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._build_controls(left)
        if folder:
            self._load(folder)

    # ---- controls ----------------------------------------------------------
    def _build_controls(self, p: tk.Misc) -> None:
        ttk.Button(p, text="Load results folder…", command=self._pick).pack(fill="x", pady=(0, 6))

        sel = ttk.LabelFrame(p, text="  Plot  ", padding=8)
        sel.pack(fill="x", pady=4)
        self.type_var = tk.StringVar(value="SE")
        self.type_cb = ttk.Combobox(sel, textvariable=self.type_var, values=self.TYPES, state="readonly")
        self.type_cb.pack(fill="x")
        self.type_cb.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        self.dev_var = tk.StringVar()
        self.dev_cb = ttk.Combobox(sel, textvariable=self.dev_var, state="readonly")
        self.dev_cb.pack(fill="x", pady=(4, 0))
        self.dev_cb.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        vbg = ttk.LabelFrame(p, text="  Back-gate (Vbg)  ", padding=8)
        vbg.pack(fill="x", pady=4)
        self.vbg_vars = {}
        for v in (-4, 0, 4):
            bv = tk.BooleanVar(value=v in self.style_cfg.show_vbg)
            self.vbg_vars[v] = bv
            ttk.Checkbutton(vbg, text=f"BG {v}", variable=bv, command=self._apply_vbg).pack(side="left", padx=3)
        self.same_var = tk.BooleanVar(value=self.style_cfg.same_vbg_color)
        ttk.Checkbutton(vbg, text="same color", variable=self.same_var,
                        command=self._apply_simple).pack(side="left", padx=8)

        colors = ttk.LabelFrame(p, text="  Colors  ", padding=8)
        colors.pack(fill="x", pady=4)
        self._bg_btn0 = self._bg_color_btn(colors, "BG 0", "0")
        self._bg_btn4 = self._bg_color_btn(colors, "BG 4", "4")
        self._single_btn = self._attr_color_btn(colors, "Same / single", "single_vbg_color")
        self._air_btn = self._attr_color_btn(colors, "Air", "air_color")
        self._h2_btn = self._attr_color_btn(colors, "H2", "h2_color")
        self._se_btn = self._attr_color_btn(colors, "SE / line", "se_color")

        lines = ttk.LabelFrame(p, text="  Line  ", padding=8)
        lines.pack(fill="x", pady=4)
        ttk.Label(lines, text="style:").pack(side="left")
        self.ls_var = tk.StringVar(value=self.style_cfg.linestyle)
        ttk.Combobox(lines, textvariable=self.ls_var, width=8, state="readonly",
                     values=["-", "--", ":", "-."]).pack(side="left", padx=4)
        self.ls_var.trace_add("write", lambda *a: self._apply_simple())
        ttk.Label(lines, text="width").pack(side="left", padx=(10, 2))
        self.lw = ttk.Scale(lines, from_=0.5, to=3.0, value=self.style_cfg.linewidth,
                            command=lambda v: self._apply_simple())
        self.lw.pack(side="left", fill="x", expand=True)

        se = ttk.LabelFrame(p, text="  Stack per ESN  ", padding=8)
        se.pack(fill="x", pady=4)
        self.stack_var = tk.BooleanVar(value=self.style_cfg.se_stack_per_esn)
        ttk.Checkbutton(se, text="SE", variable=self.stack_var,
                        command=self._apply_simple).pack(side="left")
        self.steps_stack_var = tk.BooleanVar(value=self.style_cfg.steps_stack_per_esn)
        ttk.Checkbutton(se, text="STEPS", variable=self.steps_stack_var,
                        command=self._apply_simple).pack(side="left", padx=10)

        steps = ttk.LabelFrame(p, text="  STEPS gradient  ", padding=8)
        steps.pack(fill="x", pady=4)
        self.grad_var = tk.BooleanVar(value=self.style_cfg.steps_gradient)
        ttk.Checkbutton(steps, text="gradient on", variable=self.grad_var,
                        command=self._apply_simple).pack(anchor="w")
        self.cmap_var = tk.StringVar(value=self.style_cfg.steps_cmap)
        ttk.Combobox(steps, textvariable=self.cmap_var, values=self.CMAPS, state="readonly").pack(fill="x", pady=2)
        self.cmap_var.trace_add("write", lambda *a: self._apply_simple())
        ttk.Label(steps, text="intensity").pack(anchor="w")
        self.intensity = ttk.Scale(steps, from_=0.2, to=1.0, value=self.style_cfg.steps_intensity,
                                   command=lambda v: self._apply_simple())
        self.intensity.pack(fill="x")

        ttk.Button(p, text="💾  Save as default style", style="Accent.TButton",
                   command=self._save).pack(fill="x", pady=(8, 0))

    def _attr_color_btn(self, parent, label, attr):
        f = ttk.Frame(parent)
        f.pack(anchor="w", pady=1)
        ttk.Label(f, text=label, width=14).pack(side="left")
        b = tk.Button(f, width=3, bg=getattr(self.style_cfg, attr), relief="ridge",
                      command=lambda: self._pick_attr(attr, b))
        b.pack(side="left")
        return b

    def _bg_color_btn(self, parent, label, key):
        f = ttk.Frame(parent)
        f.pack(anchor="w", pady=1)
        ttk.Label(f, text=label, width=14).pack(side="left")
        b = tk.Button(f, width=3, bg=self.style_cfg.bg_colors.get(key, "#333"), relief="ridge",
                      command=lambda: self._pick_bg(key, b))
        b.pack(side="left")
        return b

    def _pick_attr(self, attr, btn):
        c = colorchooser.askcolor(color=getattr(self.style_cfg, attr), parent=self)[1]
        if c:
            setattr(self.style_cfg, attr, c)
            btn.config(bg=c)
            self._redraw()

    def _pick_bg(self, key, btn):
        c = colorchooser.askcolor(color=self.style_cfg.bg_colors.get(key), parent=self)[1]
        if c:
            self.style_cfg.bg_colors[key] = c
            btn.config(bg=c)
            self._redraw()

    def _apply_vbg(self):
        self.style_cfg.show_vbg = [v for v, bv in self.vbg_vars.items() if bv.get()]
        self._redraw()

    def _apply_simple(self):
        s = self.style_cfg
        s.same_vbg_color = self.same_var.get()
        s.se_stack_per_esn = self.stack_var.get()
        s.steps_stack_per_esn = self.steps_stack_var.get()
        s.steps_gradient = self.grad_var.get()
        s.steps_cmap = self.cmap_var.get()
        s.steps_intensity = float(self.intensity.get())
        s.linewidth = float(self.lw.get())
        s.linestyle = self.ls_var.get()
        self._redraw()

    # ---- data + draw -------------------------------------------------------
    def _pick(self):
        d = filedialog.askdirectory(title="Select results folder")
        if d:
            self._load(d)

    def _load(self, folder: str):
        try:
            runs = load_folder(folder)
            self.grouped = group_by_kind(runs)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc), parent=self)
            return
        g = self.grouped
        self.se_run = next(iter(g.get(MajorKind.SE, [])), None)
        self.steps_run = next(iter(g.get(MajorKind.STEPS, [])), None)
        jg = g.get(MajorKind.JG, [])
        by = {r.major: r for r in jg}
        self.jg_run = by.get(PROTO.jg.h2_air_major) or (jg[0] if jg else None)
        self.air, self.h2 = by.get(PROTO.jg.h2_air_major), by.get(PROTO.jg.h2_exposure_major)
        self.jgh_run = next(iter(g.get(MajorKind.JGH, [])), None)
        avail = ([t for t, ok in (("SE", self.se_run), ("STEPS", self.steps_run), ("JG", bool(jg)),
                  ("JG_H2", self.air and self.h2), ("JGH", self.jgh_run)) if ok])
        self.type_cb.config(values=avail or self.TYPES)
        if avail:
            self.type_var.set(avail[0])
        rep = self.se_run or self.jg_run or self.jgh_run or self.steps_run
        self.channels = rep.channels if rep else []
        self.dev_cb.config(values=[c.label for c in self.channels])
        if self.channels:
            self.dev_var.set(self.channels[0].label)
        self._redraw()

    def _channel(self):
        for c in self.channels:
            if c.label == self.dev_var.get():
                return c
        return self.channels[0] if self.channels else None

    def _redraw(self):
        self.ax.clear()
        ch = self._channel()
        t = self.type_var.get()
        s = self.style_cfg
        try:
            if ch is None:
                self.ax.text(0.5, 0.5, "Load a results folder", ha="center", va="center")
            elif t == "SE" and self.se_run:
                if s.se_stack_per_esn:
                    chs = [c for c in self.se_run.channels if c.esn == ch.esn]
                    render_se_stack(self.ax, self.se_run, chs, s, day="preview")
                else:
                    render_se(self.ax, self.se_run, ch, s, day="preview")
            elif t == "STEPS" and self.steps_run:
                if s.steps_stack_per_esn:
                    chs = [c for c in self.steps_run.channels if c.esn == ch.esn]
                    render_steps_stack(self.ax, self.steps_run, chs, s, day="preview")
                else:
                    render_steps(self.ax, self.steps_run, ch, s, day="preview")
            elif t == "JG" and self.jg_run:
                render_jg(self.ax, self.jg_run, ch, s, day="preview")
            elif t == "JG_H2" and self.air and self.h2:
                render_jg_h2(self.ax, self.air, self.h2, ch, s, day="preview")
            elif t == "JGH" and self.jgh_run:
                render_jgh(self.ax, self.jgh_run, ch, s, day="preview")
            else:
                self.ax.text(0.5, 0.5, f"No {t} data in this folder", ha="center", va="center")
        except Exception as exc:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}", ha="center", va="center", wrap=True)
        self.fig.tight_layout()
        self.canvas.draw()

    def _save(self):
        self.style_cfg.save()
        messagebox.showinfo("Saved", "Style saved as default — used by every generated plot.", parent=self)


def main() -> None:
    SensorLabApp().mainloop()


if __name__ == "__main__":
    main()
