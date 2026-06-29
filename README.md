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
  util/           # shared helpers (deduplicated from both legacy files)
  io/             # excel read/write            (next)
  sensing.py      # SE metrics                  (first port target)
  jg.py idvt.py hysteresis.py degradation.py    (later)
  cli.py gui.py   # thin frontends             (later)
tests/
  fixtures/       # legacy raw + *_analyzed.xlsx = regression ground truth
```

## Develop

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
pytest
```

## Status

- [x] Phase 0 -- skeleton, `protocol.py`, shared utils, test harness, fixtures
- [ ] Phase 1 -- port SE sensing metrics (time-window anchoring) + golden test
- [ ] Phase 2 -- JG / IdVt / hysteresis / ET / degradation
- [x] Phase 3 -- CLI + thin GUI (load folder, Day#, condition cache)
- [ ] Phase 4 -- package to .exe
