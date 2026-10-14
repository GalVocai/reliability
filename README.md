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
  report_builder.py # direct degradation-output -> Results Outline DOCX
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

## JMP data export

Open **Compare timepoints (degradation)**, auto-detect or add the analyzed
timepoints, choose the baseline and devices, then click **Export data for JMP**.
Search the available metrics and tick those to include. **Select shown** and
**Clear shown** act on the current search result. Choose **By day** to place
all devices at 0D before the next checkpoint, or **By device** to place all
checkpoints for each transistor together. Both choices export the same values.

The output workbook has `All_Data`, `PostBaseline`, `CheckpointSummary`, and
`ColumnInfo` sheets. Two matching CSV files and a QC text report are saved
beside it. `CheckpointSummary` is in long form by condition, day, metric and
measure; when exactly two metrics are selected, both summaries use the same
complete device pairs. Each metric gets its original value and a retention column. The
retention is `100 * value(t) / value(0)` for the same transistor; the
`JG_H2/delta_Vth_H2` retention uses absolute magnitudes, matching the earlier
H2 sensing JMP export. Missing or zero baseline values yield blank retention.
Vbg and other measurement states are written into column names so one row
still represents one transistor at one day and condition. If every selected
metric uses the same Vbg, the workbook also includes a numeric `Vbg` column.
For an n03/JG_H2-only selection, it includes the documented 100 ppm H2
concentration. `Day` is stored as
a number; use the `ColumnInfo` sheet to set it to Nominal in JMP when fitting
checkpoint effects. The export reads the active analysis masters and respects
the selected devices and excluded condition-days. It does not replace the
existing degradation outputs or plots.

## Results Outline report

Each exported report now includes a final-checkpoint numerical summary for
the six overview metrics and all three reliability conditions. Values come
from the same population means used to draw the graphs. Each series uses its
own last available day and the same reference day as its normalized graph.
For example, a final normalized value of 0.60 means a 40% decrease. Longer
response times are labeled as slower response; signed voltage shifts are
reported without assigning them a performance-loss score.

An `<report name>.endpoints.xlsx` workbook is saved beside the Word report,
with baseline/final values, AVDEV, sample counts, normalized final values,
percentage changes, and all plotted source values. These numbers are also
included in the report's JSON manifest. There is no combined performance
percentage across different metrics.

The GUI's **Results report** window builds a graph-populated Word document
directly from SensorLab outputs; PowerPoint is not involved. Select the clean
`Results outline.docx`, add one or more degradation roots, choose a
representative device for each detected condition, and select the SensorLab
`output` folder where the cross-test publication plots should be written.

Multiple roots are intentional: the normal reliability degradation root may
contain `Hot storage` and `85C_85RH`, while a second root contains `cycles`.
The builder merges these condition folders logically and regenerates the
cross-test regular/normalized figures from their saved population aggregates
on every build. This keeps newly added checkpoints (for example Cycles 21D /
252 cycles) synchronized with the report. It records every chosen image in a
JSON manifest beside the generated DOCX and never overwrites the clean outline
without an explicit confirmation.

New degradation runs also generate report-ready JG curves aligned to the
baseline threshold and JG-H2 overlays for the baseline and final checkpoint.
Older degradation folders that predate these plots are still accepted; the
DOCX receives visible placeholders and the manifest lists every missing file.

## Optional JG blank-current correction and batch analysis

On the main screen, **Apply JG blank-current correction** is unchecked by
default. An unchecked run uses the existing JG calculations and raw currents.
When checked, SensorLab subtracts the signed blank-current mean for each
physical calibration card, channel, and back gate from every JG sweep (n03
air, n05 hydrogen, n06 recovery). The original workbooks are never edited.
Corrected runs use a configurable fixed-current SS window (default 20–100 nA)
instead of the legacy Ion-relative SS window. The master records the measured
point count, R², current span, and fit status. SS is missing if the window has
fewer than four measured points, less than 0.3 decades of current span, or
R² below 0.95; no points are invented by interpolation.

Use **Import blank-card data…** to select a JG calibration workbook whose
Description identifies the measured `ESN` and `TNumber` for each `Id` column.
SensorLab averages the blank current separately for each device and back gate.
It accepts blank JG sweeps recorded as Part 0 or Part 2. The generated JSON
contains the signed mean, SD, sample count, device identity, and source file.
Select this JSON with **Calibration JSON…** before enabling the correction.
A copy is saved in each corrected `analysis` directory so later degradation
plots use the same correction. The Python importer also supports an optional
card-mapping table for older blank-card workbooks that do not identify devices.

**Select all timepoints…** detects raw `baseline`, `2D`, `7D`, etc. folders
from one parent directory, even when they have never been analyzed. **Run
analysis** processes them in day order. Each batch writes new
`analysis_batch_<id>` directories and leaves existing `analysis` folders
untouched. After every day succeeds, a study-level
`.sensorlab-analysis-batch-<run id>.json` activates the complete set. The
degradation reader automatically uses this active set; a failed batch leaves
the previously active masters in use. Raw result workbooks remain unchanged.
Run **Compare timepoints (degradation)…** after the batch to refresh the
cross-day trends.

Each degradation run writes a new `degradation_batch_<id>` directory beside
the existing `degradation` directory. After the build succeeds, a study-level
`.sensorlab-degradation-batch-<id>.json` marks the new directory as active.
The Results report automatically loads that active version. Existing files
that the new run did not generate, including hand-exported figures, are copied
into the new version at the same paths. The original directory stays intact.
The new version records generated and preserved files in
`.sensorlab_generated_files.json`. If the build fails, the active version
remains unchanged. The first run may retain obsolete plots from the legacy
directory; later runs use the manifest to omit obsolete generated files.

## Status

- [x] Phase 0 -- skeleton, `protocol.py`, shared utils, test harness, fixtures
- [x] Phase 1 -- SE sensing metrics (time-window anchoring) + golden test
- [x] Phase 2 -- JG / JGH hysteresis / JG_H2 / STEPS / ZOZO / degradation
- [x] Phase 3 -- CLI + thin GUI (load folder, Day#, condition cache, plot studio)
- [x] Phase 4 -- package to .exe
