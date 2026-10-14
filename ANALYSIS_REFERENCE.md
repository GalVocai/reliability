# sensorlab — Experiment, Analysis & Data Reference

A self-contained reference to the experiment, every metric formula, and the
data that comes out. Upload to any conversation for context.

---

## 1. The experiment

Reliability study of transistor-based gas sensors. Devices are organised as
**cards (ESN)** × **transistors (TNumber)**: TNumber `11/12/31/32` →
`D1T1/D1T2/D3T1/D3T2`. Typical run: **10 cards × 4 transistors = 40 devices**,
split across two stress conditions: **85C/85RH** and **Hot storage**.

Measured at **timepoints** — `baseline` (day 0), `2D`, `7D`, `14D`, `20D`, … —
to track how each metric drifts (degrades) over time.

Each timepoint runs a fixed sequence of **7 majors** (`_n0X_` result files):

| Major | Mode | What it measures |
|---|---|---|
| n01 | **ZOZO** | Id vs time, Vj alternating between two states (transient relaxation) |
| n02 | **JGH** | bidirectional transfer sweep (hysteresis) |
| n03 | **JG** (air) | transfer sweep Id vs Vj in air |
| n04 | **SE** | sensing: Id vs time, baseline → H2 exposure → recovery |
| n05 | **JG** (H2) | transfer sweep under H2 |
| n06 | **JG** (recovery) | transfer sweep after H2 |
| n07 | **STEPS** | H2 ppm staircase (10/25/50/100/1000 ppm) |

Working points (Vj, Vb per state) come from the run's script `States` sheet.
Sweeps span **Vj = −2.8 → 0.3 V**; back-gates **Vb = 0, 4 V** (JGH also −4, 2).

---

## 2. Pipeline (two stages)

- **Stage 1 — `analyze_folder`** (per timepoint): raw `_n0X_` files → per-device
  **master workbooks** `analysis/‹condition›/‹MODE›/master_‹MODE›.xlsx`. This is
  where all metrics are computed.
- **Stage 2 — `run_degradation`** (across timepoints): reads the masters →
  computes delta / degradation% vs the baseline day → trend + evolution plots.

---

## 3. Common building blocks

**Stabilized level** (robust trailing mean of a phase) — mean of `Id` over the
last `window_s` (=20 s) of a part, after dropping samples far from the median:

```
MAD    = median( |x − median(x)| )
keep x  where  |x − median(x)| ≤ mad_k · MAD          (mad_k = 6)
level  = mean(kept x)
```

**Safe percent:** `pct(num, den) = num / den × 100`  (NaN if den = 0).

---

## 4. Metric formulas by mode

### 4.1 SE sensing (n04)
Four parts: **baseline(0) / response(1) / settle(2) / recovery(3)**. The
in-exposure JG sweep adds a capacitive offset to settle+recovery, corrected by
shifting them onto the response level:

```
baseline   = level(part0)
response   = level(part1)
settle     = level(part2)
recovery   = level(part3)
shift      = response − settle                 (settle_corr = settle + shift = response)

response_delta = response − baseline                                   [nA]
response_pct   = (response − baseline) / baseline × 100                 [%]
recovery_delta = (recovery + shift) − baseline                         [nA]
recovery_pct   = [(settle+shift) − (recovery+shift)]
                 / [(settle+shift) − baseline] × 100                    [%]
```

**Response time t90** — seconds from a part's start (= the gas switch) to the
first *sustained* crossing of 90 % of the step:

```
target = y0 + 0.9 · (yf − y0)
   y0 = the PREVIOUS phase's stabilized level  (baseline for response, settle for recovery)
   yf = this part's stabilized (trailing) level
t90 = time of the first sample past `target` that stays past it for `guard` samples
```

### 4.2 STEPS ppm staircase (n07)
Seven parts, concentrations `ppm = (0, 10, 25, 50, 100, 1000, 0)`;
`baseline = level(part0)`.

```
<ppm>_delta    = level(part_i) − baseline                              [nA]
<ppm>_resp_pct = (level(part_i) − baseline) / baseline × 100           [%]
<ppm>_t90_s    = t90 (anchored on the PREVIOUS step's level)           [s]

baseline_noise = std of the trailing baseline window (MAD-filtered)    [nA]
noise_pct      = baseline_noise / |baseline| × 100                     [%]
```

**Sensitivity & LOD** — least-squares fit over the exposure steps (ppm > 0):

```
resp% = a · log10(ppm) + b
sensitivity = a                                                        [%/decade]
LOD  = 10 ^ [ ( sign(a)·3·noise_pct − b ) / a ]                        [ppm]
```
(LOD = the concentration where the fitted line reaches 3σ of the baseline noise.)

### 4.3 JG transfer sweep (n03 / n05 / n06)
Per back-gate (Vb = 0, 4), last of the 3 reps. Curve cleaned (dup-Vj averaged,
sorted). `Ion = Id at Vj=0.3` (on end), `Ioff = Id at Vj=−2.8` (off end).

```
Vth  (constant-current, target = 150 nA) : first sustained upward crossing of the
     target, linearly interpolated:
        Vth = x0 + (target − y0)·(x1 − x0)/(y1 − y0)
Vth_d2  : Vj at the zero-crossing of d²Id/dVj² near the max-gm point (inflection)
SS  (subthreshold swing) : fit  Vj = m·log10(Id) + c  over the turn-on
        (noise_floor < Id < ss_upper_frac·Ion),  SS = |m| × 1000        [mV/decade]
gm  = max( dId/dVj )                                                    [nA/V]
Ion_Ioff = Ion / max(Ioff, noise_floor)
SNR_off  = |mean(off-region Id)| / std(off-region Id)
```

### 4.4 JGH hysteresis (n02)
Forward sweep (Vj −2.8→0.3) + reverse (0.3→−2.8) per back-gate. Only
back-gates 0 and 4 reported (`jgh_skip_bg = -4, 2`).

```
Delta_Vth = Vth_FWD − Vth_BWD                     (constant-current Vth on each half) [V]
loop_area = ∫ |Id_fwd − Id_rev| dVj               (over the overlapping Vj range)     [nA·V]
HI        = loop_area / ∫ max(Id_fwd, Id_rev) dVj  (normalized hysteresis index)
```

### 4.5 JG_H2 gas comparison (air n03 vs H2 n05 / recovery n06)
Part-2 transfer sweep, per back-gate.

```
delta_Vth   = Vth(air) − Vth(gas)                                       [V]
resp_at_Vth = (I_gas − I_air) / I_air × 100   at the air Vth            [%]
max_resp    = max over Vj of (I_gas − I_air)/I_air × 100                [%]
              (searched from the air turn-on onset rightward)
Vj_max_resp = the Vj where max_resp occurs                             [V]
```

### 4.6 ZOZO transient (n01)
Each held-state segment relaxes; a **shared slow + per-segment fast**
bi-exponential is fit across a state's segments (scipy `curve_fit`):

```
Id(t) = DC_i + A_slow·exp(−t_abs/τ_slow) + A_fast_i·exp(−t_rel/τ_fast_i)
```
Per-segment single-exponential settle (reported as level/A/τ):
```
Id(t) = DC + A·exp(−t/τ)      →   level = DC, A, tau = τ
R²        = 1 − RSS/TSS
lin_slope = ( mean(last 4 pts) − mean(first 4 pts) ) / (t_last4 − t_first4)  [nA/s]
```

### 4.7 Working points (WP)
Transcribed from the script `States` sheet (setpoints, not measured):
`Vj_State1, Vb_State1, Vj_State2, Vb_State2` [V].

### 4.8 Population grouping (per timepoint, per condition)
A robust z-score flags devices whose behaviour departs from the group
(diagnostic only — no rows are dropped):

```
z = 0.6745 · (x − median(x)) / MAD          (per condition, excluding dead devices)

group priority:
  dead_baseline       |baseline| < 1000 nA
  opposite_direction  response-delta sign opposes the group majority
  no_recovery         recovery outside its physical range
  outlier_high/low    |z| > 3.5
  normal              otherwise
```

---

## 5. Degradation computation (across days)

Per device, vs its **own** baseline-day value:

```
delta           = baseline_value − value                               [same unit as metric]
degradation_pct = delta / baseline_value × 100          (NaN if baseline = 0)
```

Aggregate across devices (per condition, per day):

```
deg_mean = mean( degradation_pct )                       (mean of per-device ratios)
deg_sd   = sqrt( Σ(degradation_pct − deg_mean)² / (n−1) ) (sample SD, n ≥ 2)
```

**What the `degradation/` plots draw — retention view:**

```
plotted = 100 − degradation_pct = (value / baseline) × 100
```
→ starts at **100 % on the baseline day**, drops as the metric degrades. Y-axis
names the metric and Retention (%). Error bars show sample SD across devices;
the shift does not change the spread.

**Worked example** (response 4000 nA on day 0 → 3000 nA on day 7):
`delta = 1000 nA`; `degradation_pct = 25 %`; plotted `= 100 − 25 = 75 %`.

> **Note on Vth:** the % degradation is unstable for Vth (it crosses zero, so
> dividing by a near-zero baseline explodes). Use the dedicated **`delta_Vth`
> plot in mV** for threshold shift; its sign is negated so *negative = Vth
> became more negative*.

---

## 6. Data that comes out

**Per-timepoint masters** (`analysis/‹condition›/‹MODE›/master_‹MODE›.xlsx`):
one row per device, columns = the metrics above, tagged with `day` +
`condition`. A `_groups` sheet carries the population flags. `master_WP.xlsx`
holds working points.

**Degradation study** (`degradation/‹condition›/‹sheet›/`):
- `degradation.xlsx` — tidy long form, **per device per day**: `value`,
  `baseline_value`, `delta`, `degradation_pct`.
- `aggregate.xlsx` — mean and sample SD across devices: `value_mean/sd`,
  `delta_mean/sd`, `deg_mean/sd`, `n`. The older `*_avdev` columns remain for
  compatibility with existing reports.
- `trends/devices/‹metric›.png` — one line per device (raw value).
- `trends/average/‹metric›.png` — mean ± sample SD (physical quantity).
- `trends/degradation/‹metric›.png` — retention `%` (100 = baseline).
- `sweeps/‹device›.png` — JG_n03 raw Id-vs-Vj overlaid across days (log |Id|).
- `sensing/‹device›.png` — SE/STEPS raw Id-vs-time overlaid across days
  (DC-shifted to each day's baseline).

Every number in a plot is reproducible from these `.xlsx` files.

Conventions: JG/JGH `_bg0`/`_bg4` metrics share a combined graph, and retention
also has separate `_Vbg0`/`_Vbg4` files. STEPS ppm steps share one graph;
JG_n05/n06 are excluded from degradation (they only feed JG_H2). Plot
labels/titles are editable in `~/.sensorlab/degradation_labels.json`.
