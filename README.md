# sensorlab

Unified analysis pipeline for transistor / sensor electrical characterization.

Replaces the two legacy desktop apps with **one headless, testable core** plus
thin frontends (CLI + GUI):

- **Excel Device Splitter** -> `split` / `merge` / `analyze` stages (per device)
- **SETimeSeriesProcessor** -> `degradation` / `plot` stages (across timepoints)

## Why this rewrite

The legacy apps work but are two ~3k-5k line GUI monoliths with duplicated
helpers, GUI-coupled logic, no tests, and metric code that addresses samples by
*absolute index* (which silently miscomputes if the cadence drifts). This port
fixes three things that matter even though the experiment is **fixed**:

1. **Single source of truth** -- every magic number lives in
   [`protocol.py`](src/sensorlab/protocol.py), documented, in one place.
2. **Semantic anchoring** -- index-based windows (`iloc[100:120]`) become
   physical time windows, so they survive dropped rows / cadence changes.
3. **Fail loud** -- `assert`s validate that data matches the fixed protocol,
   turning silent wrong numbers into visible errors.

## Approach: staged port with golden tests

Each stage is ported into the clean package and verified to produce the **same
numbers** as the legacy app on a known dataset (`tests/fixtures/`). Nothing is
"believed" correct -- it is proven against ground truth before the old app is
retired.

## Layout

```
src/sensorlab/
  protocol.py     # all fixed parameters, one place
  model.py        # typed Run / Channel objects parsed from result files
  pipeline.py     # folder orchestration: load -> classify -> analyze -> masters
  util/           # shared helpers (deduplicated from both legacy files)
  io/             # result reader, condition cache, master writer
  measurements/   # SE / STEPS / ZOZO / JG / JGH / JG_H2 metrics
  plotting/       # per-device plots + shared style config
  degradation.py  # cross-timepoint tidy tables (delta / degradation%)
  compare.py      # degradation study orchestration + trend plots
  cli.py gui.py   # thin frontends
tests/
  fixtures/       # legacy raw + *_analyzed.xlsx = regression ground truth
```

## Develop

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
pytest
```

## Build the Windows GUI executable

Install the project and PyInstaller once, then run the build script:

```powershell
py -3 -m pip install -e ".[dev]" pyinstaller
.\build_exe.ps1
```

The windowed, single-file executable is written to `dist\SensorLab.exe`.

## Degradation time axis

The degradation window can present checkpoints as either **Days** or
**Cycle #**. Keep the timepoint fields as their original elapsed-day labels
(`baseline`, `2D`, `7D`, ...). In Cycle # mode, set the cycle duration in
hours; SensorLab uses:

```
cycle_num = day_num * 24 / hours_per_cycle
```

The day columns are preserved in the exported workbooks and an estimated
`cycle_num` column is added. With the default 2-hour cycle, 2D, 7D, and 13D
are displayed as cycles 24, 84, and 156 respectively.

## Status

- [x] Phase 0 -- skeleton, `protocol.py`, shared utils, test harness, fixtures
- [x] Phase 1 -- SE sensing metrics (time-window anchoring) + golden test
- [x] Phase 2 -- JG / JGH hysteresis / JG_H2 / STEPS / ZOZO / degradation
- [x] Phase 3 -- CLI + thin GUI (load folder, Day#, condition cache, plot studio)
- [x] Phase 4 -- package to .exe
