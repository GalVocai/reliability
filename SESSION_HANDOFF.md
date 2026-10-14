# sensorlab — Session Handoff

A working context document so a new conversation can continue without
re-deriving everything. Upload this to the new project.

**Project root:** `C:\Users\GalFridman\Desktop\sensorlab`
**Package:** `src/sensorlab` (editable install; run from the project root)
**Tests:** `python -m pytest -q` → **53 passing** as of this handoff.
**GUI launcher:** `Desktop\sensorlab.bat` (sets `PYTHONPATH=…\src`, runs `python -m sensorlab.gui`).
**Model note:** the GUI loads the code once at startup — after any code change you must **close and relaunch** the GUI. The label JSON (below) is re-read every run, so editing it does NOT need a relaunch.

---

## 1. What the project is

Unified analysis pipeline for transistor / sensor electrical characterization.
Replaces two legacy GUI apps ("Excel Device Splitter" + "SETimeSeriesProcessor").
Headless, tested core + thin CLI/GUI frontends. All fixed parameters live in
one place: [`protocol.py`](src/sensorlab/protocol.py).

Each experiment run folder contains instrument result files named
`E<id>_<date>_N01_n0X_<TYPE>_.xlsx` (the `_n0X_` token = the "major"), plus a
script `.xlsx` (no `_n0X_` token) whose `States` sheet holds per-channel working
points. Majors: **n01 ZOZO, n02 JGH, n03 JG (air), n04 SE, n05 JG (H2),
n06 JG (recovery), n07 STEPS**.

**Real data (OneDrive):**
`C:\Users\GalFridman\OneDrive - VOCAI Ltd\V2MOS - V2MOS Documents\R&D Sensors development\Projects\reliability\`
with timepoint subfolders `baseline` (day 0), `2D` (day 2), `7D` (day 7),
`14D` (day 14). Each has the run folder + an `analysis/` output subfolder.
Two conditions in the data: **`85C_85RH`** and **`Hot storage`** (assigned per
ESN via the condition cache `~/.sensorlab/conditions.json`).

---

## 2. Architecture — two independent stages

**Stage 1 — `analyze_folder`** ([pipeline.py](src/sensorlab/pipeline.py)), runs per timepoint folder. **This is the only place metrics are computed.**
- Reads raw `_n0X_` files → classifies (`io/result_reader.py`) → runs analyzers (`measurements/`) → writes **master workbooks**:
  `‹folder›/analysis/‹condition›/‹MODE›/master_‹MODE›.xlsx`, one row per device, tagged with `day` + `condition`.
- Also writes `master_WP.xlsx` (working points, transcribed from the script `States` sheet — NOT measured).
- Population grouping ([measurements/population.py](src/sensorlab/measurements/population.py)) adds a `group` column + a `_groups` diagnostic sheet flagging `dead_baseline` / `opposite_direction` / `no_recovery` / `outlier_high|low` (never drops rows — flag only).

**Stage 2 — `run_degradation`** ([compare.py](src/sensorlab/compare.py)), runs across timepoints. **Does NOT recompute** — it READS the masters.
- `build_degradation` ([degradation.py](src/sensorlab/degradation.py)): reads masters → melts to tidy long form → joins to baseline day → computes `delta` and `degradation_pct` → aggregates mean ± **AVDEV** across devices.
- Writes `degradation.xlsx` (tidy), `aggregate.xlsx`, and the plots.
- **Wipes the whole `degradation/` output root** at the start of every call (full regeneration; a dropped sheet must not leave a stale folder).

**Consequence:** changing a metric formula (in `measurements/`) requires re-running `analyze_folder`; `run_degradation` alone won't see it.

---

## 3. Degradation output structure

```
‹out›/degradation/‹condition›/‹sheet›/
    degradation.xlsx   aggregate.xlsx
    trends/
        devices/‹metric›.png        # one line per device (raw value)
        average/‹metric›.png        # mean ± AVDEV (physical quantity)
        degradation/‹metric›.png    # RETENTION %  (see §4)
    sweeps/‹device›.png             # JG_n03 only: raw Id-vs-Vj across days
    sensing/‹device›.png            # SE_n04 / STEPS_n07: raw Id-vs-time across days
```

**Trends** come from the master tables (scalars). **sweeps/ + sensing/**
("evolution" plots) re-read the RAW result files (they need full curve shape).

**Metric grouping:** metrics ending `_bg0`/`_bg4` (JG/JGH) are drawn on ONE
graph (`Vth.png`, both back-gates overlaid); STEPS `<ppm>_resp_pct` etc. are
grouped by concentration onto one graph (`ppm_resp_pct.png`). JG_n05/JG_n06 are
**skipped** in the degradation study (they only feed JG_H2).

---

## 4. Conventions / decisions made this session (preserve these)

- **Vth threshold current = 150 nA** (`protocol.jg.vth_target_nA`, was 100). Feeds all Vth calcs (JG/JGH/JG_H2).
- **Retention view for the `degradation/` family:** plot `100 − degradation_pct` (= value/baseline×100), so the curve starts at **100 % on the baseline day** and drops as it degrades. Implemented as `scale=-1, offset=100`; error-bar magnitude uses `abs(scale)`. Y-axis for this family is just **`%`**. Title: `"‹label› — % of day ‹base› (100 = baseline)"`.
- **AVDEV, not std**, for the aggregate spread (Excel `AVEDEV` = mean absolute deviation). Grows linearly with an outlier's distance, so it's less inflated than std. `(mean +/- AVDEV)` text was **removed from every y-axis and title** by request.
- **SE `t90` / STEPS `t90`** are anchored on the PREVIOUS phase's stabilized level (seconds from the gas switch to 90 % of the step), not on the part's first samples.
- **`response_delta` / `<ppm>_delta`** columns (nA, comparable across devices) replaced the absolute `_level` columns in the SE/STEPS masters. Level columns are dropped from the master (kept in the `_groups` sheet for diagnostics).
- **delta_Vth (JG_n03):** an extra `delta_Vth.png` (devices+average) shows the day-vs-baseline Vth shift in **mV**, with the sign NEGATED so it reads naturally: negative = Vth got more negative. (The generic pipeline `delta = baseline − value` double-negates for a signed quantity, hence `scale = -1000`.) The `%` degradation plot for Vth is mathematically unstable (Vth crosses zero) — use `delta_Vth` (mV) for threshold shift.
- **Sensing evolution (SE/STEPS) DC-shift = subtract the BASELINE LEVEL** (Part-0 stabilized level), NOT the curve minimum. Minimum-anchoring breaks for downward responders (their min is mid-response); baseline-anchoring aligns baselines correctly. `_sensing_curve()` in compare.py computes the anchor (SE from `_continuous`'s 4th return; STEPS via `_trailing_level` of the first part).
- **Sweep plots plot `|Id|`** (helper renamed `positive_despiked` → **`abs_despiked`** in `_clean.py`). Off-region current swings through zero; a log axis can't show negatives, so we used to drop them (gaps). Now they're kept as magnitude — the off-region noise floor stays visible. Applied to ALL sweep plots: sweep-evolution + JG/JGH/JG_H2 (Plot Studio side too). Exact zeros still dropped (log10(0) undefined).
- **Colours = shades of green everywhere in degradation plots.** Evolution plots: day = green gradient (light = early, dark = late). Grouped plots (ppm/back-gate) + per-device + average: green gradient via `_GROUP_CMAP="Greens"` in `trend_plot.py`. (The regular Plot Studio analysis plots are UNCHANGED — they still use `StyleConfig` colours the user sets in the GUI.)
- **Sweep-evolution styling:** all solid lines, only colour (= day) distinguishes; x-axis fixed to `[protocol.jg.sweep_start_vj, sweep_close_vj]` = `[-2.8, 0.3]` so every day is framed identically. Two back-gates share a day's colour (told apart by position; no linestyle).
- **Back-gate legend labels** carry units: `BG=0 [V]`, `BG=4 [V]`. JGH master trimmed to `Delta_Vth`/`loop_area`/`HI` only, and skips back-gates `-4` and `2` (`protocol.jg.jgh_skip_bg`).
- **Robustness fixes:** Excel `~$` lock files skipped in `scan_result_files`; `back_gate_groups` matches Vb rounded to 10 mV; a cloud-sync-safe `_savefig` retries on `FileNotFoundError` (OneDrive race); degradation baseline day must be the earliest (warns otherwise).

---

## 5. The label config file (the "wizard")

**`~/.sensorlab/degradation_labels.json`** — single source of truth for every
degradation-plot y-axis label and title. Auto-created (pre-filled with defaults)
on first `run_degradation`; edit in any text editor; **re-read every run** (no
GUI relaunch needed). Managed by [degradation_labels.py](src/sensorlab/degradation_labels.py).

Three sections:
- `ylabels` — metric → y-axis label (with units). Includes grouped pseudo-metrics `ppm_resp_pct` / `ppm_delta` / `ppm_t90_s` and specials `@degradation` (the `%`) / `@delta_Vth` (the mV axis). Unlisted → `_`→space fallback.
- `titles` — family (`devices`/`average`/`degradation`/`delta`) → template with `{label}` and `{base}` placeholders.
- `hidden` — list of metric names to skip entirely (e.g. `["Vth_d2","SNR_off","R2"]`).

Corrupt/locked file → silently falls back to defaults. Delete the file to reset.

---

## 6. File map

| File | Role |
|---|---|
| `protocol.py` | ALL fixed params: Vth target 150 nA, sweep range, ppm steps, jgh_skip_bg, windows/thresholds |
| `model.py` | `Run`/`Channel`; `run.parts()` cached |
| `pipeline.py` | Stage 1: `analyze_folder`, `load_folder`, WP master |
| `compare.py` | Stage 2: `run_degradation` + trend/evolution orchestration, grouping, retention wiring |
| `degradation.py` | tidy/aggregate math (delta, degradation_pct, AVDEV) |
| `degradation_labels.py` | the editable-label config |
| `measurements/{se,steps,jg,jgh,jg_h2,zozo}.py` | per-mode metric calcs |
| `measurements/population.py` | device grouping / outlier flags |
| `plotting/trend_plot.py` | degradation trend plots (green, AVDEV-free, ylabel_override) |
| `plotting/evolution_plot.py` | sweep + sensing evolution (green day gradient, baseline DC-shift, `|Id|`) |
| `plotting/{jg,jgh,jg_h2,se,steps,zozo}_plot.py` | Plot Studio per-device analysis plots (StyleConfig-driven) |
| `plotting/_clean.py` | `abs_despiked` (|Id|), `despiked_linear`, `stable_min` |
| `plotting/style.py` | `StyleConfig` for Plot Studio (`~/.sensorlab/plot_style.json`) |
| `gui.py` | tkinter: analyze folder, Plot Studio, Degradation window |

---

## 7. Current state & next steps

- **Code is fully updated** and 53 tests pass. Nothing committed (all changes uncommitted in git; new untracked files: `degradation_labels.py`, `population.py`, `evolution_plot.py`, several `tests/test_*.py`).
- **The OneDrive images are STALE** — the last full regeneration predates the most recent changes (label config, AVDEV-in-title removal, `|Id|`, baseline-anchor). **To see everything current, run a full refresh:**
  ```python
  from sensorlab.pipeline import analyze_folder
  from sensorlab.compare import run_degradation
  rel = r"C:\Users\GalFridman\OneDrive - VOCAI Ltd\V2MOS - V2MOS Documents\R&D Sensors development\Projects\reliability"
  tps = {
    "0":  rel + r"\baseline\E1074_2026_06_26_N01_SE_State1_10cards_Hot_storage_8585_baseline",
    "2":  rel + r"\2D\E1074_2026_07_02_N01_SE_State1_10cards_Hot_storage_8585_2D",
    "7":  rel + r"\7D\E1074_2026_07_08_N01_SE_State1_10cards_Hot_storage_8585_7D",
    "14": rel + r"\14D\...",   # confirm exact 14D folder name
  }
  for day, f in tps.items():
      analyze_folder(f, day=day, log=print)          # regenerate masters (needed for |Id|, delta cols, groups)
  run_degradation([(d, f) for d, f in tps.items()], out_dir=rel, baseline_day="0", log=print)
  ```
  (`analyze_folder` per timepoint is needed because the master content changed; then one `run_degradation` over all days.)
- **Device exclusion:** in the GUI Degradation window, untick devices to drop them from the average (per-device plots grey them out; tidy keeps all). Programmatically: `run_degradation(..., include_devices={("1100","11"), ...})`.

---

## 8. Open items the user may pick up

- Optionally **hide** noisy/meaningless metrics via the `hidden` list (candidates discussed: `Vth_d2`, `SNR_off`, `R2`, and the `%`-degradation plot for Vth which is unstable).
- Optionally add a small GUI editor on top of the label JSON (user chose the JSON-file approach for now; GUI was deferred).
- Nothing is committed to git — consider committing once the refreshed output is confirmed good.

---

## How to run tests
```bash
cd C:\Users\GalFridman\Desktop\sensorlab
python -m pytest -q
```
