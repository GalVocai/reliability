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


def main() -> None:
    SensorLabApp().mainloop()


if __name__ == "__main__":
    main()
