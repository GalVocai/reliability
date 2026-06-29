"""Thin Tkinter front-end for the sensorlab pipeline.

Pick a results folder, enter the Day#, fill the condition per card (cached and
remembered), then run. All analysis lives in the headless core; this is only a
shell around `analyze_folder`.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from .compare import run_degradation
from .degradation import devices_in_folder, find_masters
from .io.device_map import load_condition_cache
from .pipeline import all_esns, analyze_folder, load_folder


class SensorLabApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("sensorlab")
        self.geometry("760x620")
        self.folder: str | None = None
        self.cache = load_condition_cache()
        self.cond_vars: dict[str, tk.StringVar] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._build()
        self.after(100, self._drain_log)

    # ---- layout ------------------------------------------------------------
    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Select results folder…", command=self._pick_folder).pack(side="left")
        self.folder_lbl = ttk.Label(top, text="(no folder selected)", foreground="#666")
        self.folder_lbl.pack(side="left", padx=10)

        opts = ttk.Frame(self)
        opts.pack(fill="x", **pad)
        ttk.Label(opts, text="Day #:").pack(side="left")
        self.day_var = tk.StringVar(value="baseline")
        ttk.Entry(opts, textvariable=self.day_var, width=16).pack(side="left", padx=(4, 16))
        self.plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Generate plots", variable=self.plots_var).pack(side="left")

        ttk.Label(self, text="Condition per card (filled once, remembered):").pack(anchor="w", **pad)
        self.cond_frame = ttk.Frame(self)
        self.cond_frame.pack(fill="x", **pad)

        run = ttk.Frame(self)
        run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run, text="Run analysis", command=self._run, state="disabled")
        self.run_btn.pack(side="left")
        ttk.Button(run, text="Compare timepoints (degradation)…",
                   command=lambda: DegradationWindow(self)).pack(side="left", padx=12)

        ttk.Label(self, text="Log:").pack(anchor="w", **pad)
        self.log = ScrolledText(self, height=14, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, **pad)

    # ---- actions -----------------------------------------------------------
    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select results folder")
        if not folder:
            return
        self.folder = folder
        self.folder_lbl.config(text=folder)
        try:
            runs = load_folder(folder)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        if not runs:
            messagebox.showwarning("No data", "No result files (…_n0X_…) found in this folder.")
            return
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
            cell.grid(row=r, column=c, sticky="w", padx=6, pady=2)
            ttk.Label(cell, text=f"ESN{esn}", width=10).pack(side="left")
            var = tk.StringVar(value=self.cache.get(esn))
            ttk.Entry(cell, textvariable=var, width=22).pack(side="left")
            self.cond_vars[esn] = var

    def _run(self) -> None:
        if not self.folder:
            return
        # persist conditions
        self.cache.update({esn: v.get().strip() for esn, v in self.cond_vars.items()})
        self.cache.save()
        self._log(f"Saved conditions -> {self.cache.path}")
        self.run_btn.config(state="disabled")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            written = analyze_folder(
                self.folder,
                conditions=self.cache,
                make_plots=self.plots_var.get(),
                day=self.day_var.get().strip(),
                log=self._queue.put,
            )
            self._queue.put("__DONE__:" + ", ".join(written))
        except Exception as exc:  # surface errors in the log
            self._queue.put("__ERROR__:" + str(exc))

    # ---- log plumbing ------------------------------------------------------
    def _drain_log(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.startswith("__DONE__:"):
                    self._log("✔ " + msg[len("__DONE__:"):])
                    self.run_btn.config(state="normal")
                    messagebox.showinfo("Done", "Analysis complete.")
                elif msg.startswith("__ERROR__:"):
                    self._log("ERROR: " + msg[len("__ERROR__:"):])
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
    """Read the 'day' tag from any master in a timepoint folder."""
    for _cond, _mode, path in find_masters(folder):
        try:
            df = pd.read_excel(path, nrows=1)
            if "day" in df.columns and len(df):
                return str(df["day"].iloc[0])
        except Exception:
            pass
    return ""


class DegradationWindow(tk.Toplevel):
    """Pick timepoint folders, mark the baseline, and build the degradation study."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.title("Compare timepoints — degradation")
        self.geometry("760x720")
        self.rows: list[dict] = []
        self.dev_vars: dict[tuple[str, str], tk.BooleanVar] = {}
        self.baseline = tk.IntVar(value=0)
        self.out_dir: str | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Add timepoint folder…", command=self._add).pack(side="left")
        ttk.Button(top, text="Output folder…", command=self._pick_out).pack(side="left", padx=8)
        self.out_lbl = ttk.Label(top, text="(output: parent of first folder)", foreground="#666")
        self.out_lbl.pack(side="left")

        ttk.Label(self, text="Timepoints (select the baseline):").pack(anchor="w", **pad)
        self.rows_frame = ttk.Frame(self)
        self.rows_frame.pack(fill="x", **pad)

        devhdr = ttk.Frame(self)
        devhdr.pack(fill="x", **pad)
        ttk.Label(devhdr, text="Devices / transistors in the average:").pack(side="left")
        ttk.Button(devhdr, text="All", width=5, command=lambda: self._set_all(True)).pack(side="left", padx=4)
        ttk.Button(devhdr, text="None", width=5, command=lambda: self._set_all(False)).pack(side="left")
        self.dev_frame = ttk.Frame(self)
        self.dev_frame.pack(fill="x", **pad)

        bar = ttk.Frame(self)
        bar.pack(fill="x", **pad)
        self.plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Trend plots", variable=self.plots_var).pack(side="left")
        self.run_btn = ttk.Button(bar, text="Run degradation", command=self._run, state="disabled")
        self.run_btn.pack(side="left", padx=10)

        self.log = ScrolledText(self, height=14, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, **pad)
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
        ttk.Label(row, text=os.path.basename(folder), width=42).pack(side="left")
        day_var = tk.StringVar(value=_peek_day(folder) or "")
        ttk.Label(row, text="day:").pack(side="left")
        ttk.Entry(row, textvariable=day_var, width=10).pack(side="left")
        self.rows.append({"folder": folder, "day": day_var})
        self.run_btn.config(state="normal")
        if not self.dev_vars:                     # populate device grid from the first folder
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
                       command=lambda e=esn: self._toggle_card(e)).grid(row=r, column=0, sticky="w")
            for c, t in enumerate(sorted(tnums)):
                v = tk.BooleanVar(value=True)
                self.dev_vars[(esn, t)] = v
                lbl = f"D{t[0]}T{t[1]}" if len(t) == 2 and t.isdigit() else f"T{t}"
                ttk.Checkbutton(self.dev_frame, text=lbl, variable=v).grid(row=r, column=c + 1, sticky="w", padx=2)

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
            self.out_lbl.config(text=f"output: {d}")

    def _run(self) -> None:
        timepoints = [(r["day"].get().strip(), r["folder"]) for r in self.rows]
        baseline_day = timepoints[self.baseline.get()][0]
        out = self.out_dir or os.path.dirname(self.rows[0]["folder"])
        selected = {k for k, v in self.dev_vars.items() if v.get()}
        include = selected if self.dev_vars else None
        self.run_btn.config(state="disabled")
        self._log(f"Baseline = '{baseline_day}'  |  devices in average: {len(selected)}/{len(self.dev_vars)}  |  output -> {out}")

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
                    self._log("✔ degradation -> " + msg[len("__DONE__:"):])
                    self.run_btn.config(state="normal")
                    messagebox.showinfo("Done", "Degradation study complete.")
                elif msg.startswith("__ERROR__:"):
                    self._log("ERROR: " + msg[len("__ERROR__:"):])
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


def main() -> None:
    SensorLabApp().mainloop()


if __name__ == "__main__":
    main()
