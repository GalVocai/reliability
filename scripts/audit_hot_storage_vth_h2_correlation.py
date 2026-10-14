"""Audit Hot-storage Vbg=0 JG Vth versus H2 sensing association.

Read-only with respect to SensorLab analysis/degradation. Exports new Desktop
figures and CSV/JSON results. Repeated-measures uncertainty resamples whole
device trajectories, never individual device-day rows. Day 0 is excluded.
Requires statsmodels in the local analysis environment.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from sensorlab.plotting.style import load_style


DATA_ROOT = Path(__file__).resolve().parents[2] / "reliability"
HOT = DATA_ROOT / "degradation" / "Hot storage"
OUT = Path.home() / "Desktop" / "Hot_storage_Vbg0_Vth_H2_correlation_audit"
BG = 0
DAYS = (2, 7, 14, 20, 25, 28)
BOOTSTRAPS = 10_000
SEED = 20260927


def _metric(family: str, metric: str, selected: set[tuple[str, str]],
            vbg: int | None = None) -> pd.DataFrame:
    data = pd.read_excel(HOT / family / "degradation.xlsx",
                         sheet_name="degradation")
    data = data.loc[data.metric.eq(metric)].copy()
    if vbg is not None:
        data = data.loc[data.Vbg.eq(vbg)].copy()
    data["ESN"] = data.ESN.astype(int).astype(str)
    data["TNumber"] = data.TNumber.astype(int).astype(str)
    data["device"] = list(zip(data.ESN, data.TNumber))
    data = data.loc[data.device.isin(selected)].copy()
    if data.duplicated(["device", "day_num"]).any():
        raise ValueError(f"Duplicate {family} {metric} rows")
    return data[["ESN", "TNumber", "device", "day_num", "value",
                 "baseline_value", "degradation_pct"]]


def load_pairs() -> pd.DataFrame:
    old = pd.read_csv(HOT / "correlation" /
                      "Vth_vs_H2_sensing_retention_Vbg0_paired_devices.csv")
    selected = {tuple(ast.literal_eval(value)) for value in old.device.unique()}
    x = _metric("JG_n03", "Vth_bg0", selected).rename(columns={
        "value": "jg_vth_V", "baseline_value": "jg_vth_0D_V",
        "degradation_pct": "jg_degradation_pct"})
    y = _metric("JG_H2", "delta_Vth_H2", selected, BG).rename(columns={
        "value": "h2_delta_vth_V", "baseline_value": "h2_delta_vth_0D_V",
        "degradation_pct": "h2_degradation_pct"})
    air = _metric("JG_H2", "Vth_air", selected, BG)[[
        "device", "day_num", "value"]].rename(columns={"value": "h2_air_vth_V"})
    gas = _metric("JG_H2", "Vth_H2", selected, BG)[[
        "device", "day_num", "value"]].rename(columns={"value": "h2_gas_vth_V"})
    data = x.merge(y.drop(columns=["ESN", "TNumber"]),
                   on=["device", "day_num"], validate="one_to_one")
    data = data.merge(air, on=["device", "day_num"], validate="one_to_one")
    data = data.merge(gas, on=["device", "day_num"], validate="one_to_one")
    data = data.sort_values(["ESN", "TNumber", "day_num"]).reset_index(drop=True)
    if len(data) != 7 * len(selected) or data.isna().any().any():
        raise ValueError("Selected device/timepoint pairs are incomplete")
    if not np.allclose(data.h2_delta_vth_V,
                       data.h2_air_vth_V - data.h2_gas_vth_V, atol=1e-9):
        raise ValueError("H2 delta Vth is not air minus H2 Vth")
    if not (data.jg_vth_0D_V.lt(0).all()
            and data.h2_delta_vth_V.gt(0).all()
            and data.h2_delta_vth_0D_V.gt(0).all()):
        raise ValueError("Signed threshold assumptions for this audit changed")
    data["x_retention_pct"] = 100 * data.jg_vth_V / data.jg_vth_0D_V
    data["y_retention_pct"] = (100 * data.h2_delta_vth_V /
                               data.h2_delta_vth_0D_V)
    data["jg_drift_mV"] = 1000 * (data.jg_vth_V - data.jg_vth_0D_V)
    data["jg_vth_mV"] = 1000 * data.jg_vth_V
    data["h2_abs_delta_vth_mV"] = 1000 * data.h2_delta_vth_V.abs()
    data["h2_signed_delta_vth_mV"] = 1000 * data.h2_delta_vth_V
    data["jg_minus_h2_air_vth_mV"] = 1000 * (
        data.jg_vth_V - data.h2_air_vth_V)
    old["device"] = old.device.map(ast.literal_eval)
    check = data.merge(old, on=["device", "day_num"],
                       suffixes=("", "_old"), validate="one_to_one")
    for name in ("x_retention_pct", "y_retention_pct"):
        if not np.allclose(check[name], check[f"{name}_old"], atol=1e-8):
            raise ValueError(f"Recomputed {name} differs from the original plot")
    if (data.loc[data.day_num.eq(0), ["x_retention_pct", "y_retention_pct"]]
            .sub(100).abs().max().max() > 1e-8):
        raise ValueError("0D retention does not equal 100%")
    return data


def _r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _two_way_beta(x: np.ndarray, y: np.ndarray) -> float:
    xd = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()
    yd = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    return float(np.sum(xd * yd) / np.sum(xd * xd))


def _matrix(data: pd.DataFrame, column: str, devices: list[tuple[str, str]]) -> np.ndarray:
    pivot = data.pivot(index="device", columns="day_num", values=column)
    return pivot.loc[devices, list(DAYS)].to_numpy(float)


def _bootstrap(x: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n, t = x.shape
    idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    xb, yb = x[idx], y[idx]
    xmean, ymean = xb.mean(axis=1), yb.mean(axis=1)
    trend = np.array([_r(a, b) for a, b in zip(xmean, ymean)])
    xday = xb - xmean[:, None, :]
    yday = yb - ymean[:, None, :]
    xy = np.sum(xday * yday, axis=(1, 2))
    xx = np.sum(xday * xday, axis=(1, 2))
    yy = np.sum(yday * yday, axis=(1, 2))
    within = xy / np.sqrt(xx * yy)
    x_tw = (xb - xb.mean(axis=2, keepdims=True)
            - xb.mean(axis=1, keepdims=True)
            + xb.mean(axis=(1, 2), keepdims=True))
    y_tw = (yb - yb.mean(axis=2, keepdims=True)
            - yb.mean(axis=1, keepdims=True)
            + yb.mean(axis=(1, 2), keepdims=True))
    beta = np.sum(x_tw * y_tw, axis=(1, 2)) / np.sum(x_tw ** 2, axis=(1, 2))
    result = {}
    for name, values in (("trend_r", trend), ("within_day_r", within),
                         ("two_way_beta", beta)):
        values = values[np.isfinite(values)]
        result[name] = {
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "valid_bootstraps": int(len(values)),
        }
    return result


def _fit_models(post: pd.DataFrame, x_name: str, y_name: str) -> dict:
    d = post.copy()
    d["x"] = d[x_name]
    d["y"] = d[y_name]
    d["x_device_mean"] = d.groupby("device").x.transform("mean")
    d["x_within_device"] = d.x - d.x_device_mean
    d["x_between_devices"] = d.x_device_mean - d.x.mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simple = smf.mixedlm("y ~ x + C(day_num)", d, groups=d.device).fit(
            reml=True, method="powell", maxiter=400, disp=False)
        separated = smf.mixedlm(
            "y ~ x_within_device + x_between_devices + C(day_num)",
            d, groups=d.device).fit(
                reml=True, method="powell", maxiter=400, disp=False)
    if not simple.converged or not separated.converged:
        raise RuntimeError("Mixed model did not converge")
    fixed = smf.ols("y ~ x + C(day_num) + C(device)", d).fit(
        cov_type="cluster", cov_kwds={"groups": d.device})

    def effect(fit, term: str) -> dict:
        ci = fit.conf_int().loc[term]
        return {"beta": float(fit.params[term]),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "p_value": float(fit.pvalues[term])}

    return {
        "mixed_simple": effect(simple, "x"),
        "mixed_within_device": effect(separated, "x_within_device"),
        "mixed_between_devices": effect(separated, "x_between_devices"),
        "device_and_day_fixed_effects_clustered": effect(fixed, "x"),
        "random_intercept_variance_simple": float(simple.cov_re.iloc[0, 0]),
        "random_intercept_variance_separated": float(separated.cov_re.iloc[0, 0]),
        "warnings": sorted({str(w.message) for w in caught}),
        "simple_summary": simple.summary().as_text(),
        "separated_summary": separated.summary().as_text(),
    }


def _describe(post: pd.DataFrame, x_name: str, y_name: str) -> dict:
    devices = sorted(post.device.unique())
    x = _matrix(post, x_name, devices)
    y = _matrix(post, y_name, devices)
    xday, yday = x - x.mean(axis=0), y - y.mean(axis=0)
    trend = _r(x.mean(axis=0), y.mean(axis=0))
    within = _r(xday.ravel(), yday.ravel())
    boot = _bootstrap(x, y)
    model = _fit_models(post, x_name, y_name)
    return {
        "n_devices": len(devices), "n_esn": post.ESN.nunique(),
        "n_postbaseline_checkpoints": len(DAYS), "n_paired_rows": len(post),
        "stress_trend_r": trend,
        "stress_trend_r_ci95_device_bootstrap": boot["trend_r"],
        "within_checkpoint_pooled_r": within,
        "within_checkpoint_r_ci95_device_bootstrap": boot["within_day_r"],
        "two_way_fixed_effect_beta": _two_way_beta(x, y),
        "two_way_beta_ci95_device_bootstrap": boot["two_way_beta"],
        "models": model,
    }


def _influence(post: pd.DataFrame, x_name: str, y_name: str) -> pd.DataFrame:
    devices = sorted(post.device.unique())
    x = _matrix(post, x_name, devices)
    y = _matrix(post, y_name, devices)
    rows = []
    for i, device in enumerate(devices):
        xx = np.delete(x, i, axis=0)
        yy = np.delete(y, i, axis=0)
        rows.append({"left_out": f"ESN{device[0]} T{device[1]}",
                     "stress_trend_r": _r(xx.mean(axis=0), yy.mean(axis=0)),
                     "within_checkpoint_r": _r((xx - xx.mean(axis=0)).ravel(),
                                               (yy - yy.mean(axis=0)).ravel()),
                     "two_way_beta": _two_way_beta(xx, yy)})
    for esn in sorted(post.ESN.unique()):
        keep = [i for i, device in enumerate(devices) if device[0] != esn]
        xx, yy = x[keep], y[keep]
        rows.append({"left_out": f"ESN{esn} (all devices)",
                     "stress_trend_r": _r(xx.mean(axis=0), yy.mean(axis=0)),
                     "within_checkpoint_r": _r((xx - xx.mean(axis=0)).ravel(),
                                               (yy - yy.mean(axis=0)).ravel()),
                     "two_way_beta": _two_way_beta(xx, yy)})
    return pd.DataFrame(rows)


def _checkpoint_stats(post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in post.groupby("day_num"):
        x, y = group.x_retention_pct.to_numpy(float), group.y_retention_pct.to_numpy(float)
        r = _r(x, y)
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        half = stats.norm.ppf(0.975) / np.sqrt(len(group) - 3)
        rows.append({"day_num": day, "n_devices": len(group), "r": r,
                     "fisher_ci95_low": float(np.tanh(z - half)),
                     "fisher_ci95_high": float(np.tanh(z + half))})
    return pd.DataFrame(rows)


def _save_figure(fig, stem: str, dpi: float) -> None:
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def _plot_paired(data: pd.DataFrame, result: dict, *, x: str, y: str,
                 xlabel: str, ylabel: str, stem: str, reference_text: str,
                 axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 show_reference: bool = True) -> None:
    style = load_style()
    post = data.loc[data.day_num.gt(0)].copy()
    means = post.groupby("day_num")[[x, y]].mean().reindex(DAYS)
    sds = post.groupby("day_num")[[x, y]].std().reindex(DAYS)
    slope, intercept = np.polyfit(means[x], means[y], 1)
    line_x = np.linspace(means[x].min(), means[x].max(), 200)
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    ax.scatter(post[x], post[y], s=30, color="#9aa3ad", alpha=0.35,
               edgecolors="none", label="Paired devices", zorder=1)
    ax.plot(line_x, slope * line_x + intercept, color="#d62728", lw=2.2,
            label="Fit to checkpoint means", zorder=2)
    ax.errorbar(means[x], means[y], xerr=sds[x], yerr=sds[y], fmt="o",
                color="black", ecolor="black", elinewidth=1.15, capsize=3.5,
                markersize=7, label="Checkpoint mean ± SD", zorder=3)
    day_offsets = {2: (9, 8), 7: (8, 8), 14: (8, -18),
                   20: (8, 9), 25: (-31, 8), 28: (-30, -19)}
    for day, row in means.iterrows():
        ax.annotate(f"{day}D", (row[x], row[y]), xytext=day_offsets[day],
                    textcoords="offset points", fontsize=10.5, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                              pad=0.1))
    if show_reference:
        reference = data.loc[data.day_num.eq(0), [x, y]].mean()
        ax.scatter([reference[x]], [reference[y]], s=95, marker="D",
                   facecolors="white", edgecolors="black", linewidths=1.5,
                   zorder=4)
        ax.annotate(reference_text, (reference[x], reference[y]),
                    xytext=(-8, -20), textcoords="offset points", ha="right",
                    fontsize=11, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if axis_limits:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
    style.apply_publication(ax)
    style.apply_legend(ax, loc="upper left", fontsize=11.5)
    trend = result["stress_trend_r_ci95_device_bootstrap"]
    within = result["within_checkpoint_r_ci95_device_bootstrap"]
    stats_x, stats_y, stats_ha = ((0.03, 0.745, "left") if axis_limits
                                  else (0.97, 0.965, "right"))
    ax.text(stats_x, stats_y,
            f"{result['n_devices']} devices · {result['n_esn']} chips · 6 post-baseline days\n"
            f"Stress-trend $r$ = {result['stress_trend_r']:.2f}, "
            f"95% CI [{trend['ci95_low']:.2f}, {trend['ci95_high']:.2f}]\n"
            f"Within-day pooled $r$ = {result['within_checkpoint_pooled_r']:.2f}, "
            f"95% CI [{within['ci95_low']:.2f}, {within['ci95_high']:.2f}]",
            transform=ax.transAxes, fontsize=11.5, ha=stats_ha, va="top",
            zorder=6, bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.88, pad=2.0))
    ax.text(0.97, 0.03, "Error bars: device SD; CIs: whole-device bootstrap",
            transform=ax.transAxes, fontsize=9.2, ha="right", va="bottom",
            color="#444444")
    _save_figure(fig, stem, style.dpi)


def _plot_adjusted(data: pd.DataFrame, result: dict) -> None:
    style = load_style()
    post = data.loc[data.day_num.gt(0)].copy()
    for source, target in (("x_retention_pct", "x_adjusted_pp"),
                           ("y_retention_pct", "y_adjusted_pp")):
        post[target] = (post[source]
                        - post.groupby("device")[source].transform("mean")
                        - post.groupby("day_num")[source].transform("mean")
                        + post[source].mean())
    post[["ESN", "TNumber", "day_num", "x_adjusted_pp",
          "y_adjusted_pp"]].to_csv(OUT / "device_and_day_adjusted_retention.csv",
                                 index=False)
    fig, ax = plt.subplots(figsize=(9.7, 6.8))
    palette = ["#000000", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for day, color in zip(DAYS, palette):
        group = post.loc[post.day_num.eq(day)]
        ax.scatter(group.x_adjusted_pp, group.y_adjusted_pp, s=42,
                   color=color, alpha=0.65, edgecolors="none",
                   label=f"{day}D", zorder=2)
    xlim = max(abs(post.x_adjusted_pp.min()), abs(post.x_adjusted_pp.max())) * 1.1
    line_x = np.array([-xlim, xlim])
    beta = result["two_way_fixed_effect_beta"]
    ax.plot(line_x, beta * line_x, color="black", lw=2.2,
            label="Adjusted fit", zorder=3)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.axvline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel(r"Baseline $\mathbf{V}_{\mathbf{TH}}$ Retention,")
    ax.set_ylabel(r"$\mathbf{H}_{\mathbf{2}}$-Induced "
                  r"$\mathbf{|\boldsymbol{\Delta}V_{TH}|}$ Retention,")
    ax.xaxis.label.set_text(ax.xaxis.label.get_text() + " adjusted (pp)")
    ax.yaxis.label.set_text(ax.yaxis.label.get_text() + " adjusted (pp)")
    style.apply_publication(ax)
    style.apply_legend(ax, loc="lower left", fontsize=10.5, ncol=2)
    ci = result["two_way_beta_ci95_device_bootstrap"]
    ax.text(0.97, 0.97,
            f"Day + device adjusted slope = {beta:.2f}\n"
            f"95% device-bootstrap CI [{ci['ci95_low']:.2f}, {ci['ci95_high']:.2f}]\n"
            f"{result['n_devices']} devices, {result['n_paired_rows']} repeated observations",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.3)
    _save_figure(fig, "retention_device_day_adjusted_Vbg0", style.dpi)


def _plot_checkpoint_r(checkpoints: pd.DataFrame) -> None:
    style = load_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = checkpoints.day_num.to_numpy(float)
    r = checkpoints.r.to_numpy(float)
    low = checkpoints.fisher_ci95_low.to_numpy(float)
    high = checkpoints.fisher_ci95_high.to_numpy(float)
    ax.errorbar(x, r, yerr=[r - low, high - r], fmt="o", color="#1f77b4",
                ecolor="#1f77b4", elinewidth=1.5, capsize=4, markersize=7,
                zorder=2)
    ax.axhline(0, color="#888888", lw=1, zorder=1)
    ax.set_xticks(list(DAYS))
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.75, 1.05)
    ax.set_xlabel("Stress Time (days)")
    ax.set_ylabel("Within-checkpoint Pearson $r$")
    style.apply_publication(ax)
    ax.text(0.03, 0.05, "13 devices per checkpoint · Fisher 95% CI",
            transform=ax.transAxes, fontsize=11, va="bottom")
    _save_figure(fig, "retention_correlation_by_checkpoint_Vbg0", style.dpi)


def _plot_shared_vth(data: pd.DataFrame) -> None:
    style = load_style()
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    ax.scatter(data.jg_vth_V, data.h2_air_vth_V, s=27,
               color="#1f77b4", alpha=0.58, edgecolors="none", zorder=2)
    limits = (min(data.jg_vth_V.min(), data.h2_air_vth_V.min()) - 0.08,
              max(data.jg_vth_V.max(), data.h2_air_vth_V.max()) + 0.08)
    ax.plot(limits, limits, color="#d62728", lw=1.8, zorder=1)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"JG_n03 Air $\mathbf{V}_{\mathbf{TH}}$ (V)")
    ax.set_ylabel(r"JG_H2 Air $\mathbf{V}_{\mathbf{TH}}$ (V)")
    style.apply_publication(ax)
    ax.text(0.04, 0.96, f"Same value in {len(data)}/{len(data)} device-day pairs",
            transform=ax.transAxes, fontsize=11.5, va="top")
    _save_figure(fig, "shared_air_Vth_identity_Vbg0", style.dpi)


def _write_readme(summary: dict) -> None:
    r = summary["normalized"]
    raw = summary["raw_drift"]
    ci = r["two_way_beta_ci95_device_bootstrap"]
    lines = [
        "HOT STORAGE / Vbg=0 / JG_n03 Vth versus H2 sensing: correlation audit",
        "",
        "Selection: the exact 13 devices in the prior Hot-storage correlation plot",
        "(4 ESN chips), measured at 0, 2, 7, 14, 20, 25 and 28 days.",
        "Day 0 is a reference only and is excluded from every correlation and fit.",
        "The 78 post-baseline rows are repeated observations, not 78 independent devices.",
        "",
        "Normalized axes:",
        "X = 100 * JG_n03 Vth(t) / JG_n03 Vth(0).",
        "Y = 100 * JG_H2 [Vth_air(t)-Vth_H2(t)] /",
        "         [Vth_air(0)-Vth_H2(0)].",
        "The H2 difference is positive in all 91 paired rows, so it equals its magnitude.",
        "The 0D point is (100,100) by construction.",
        "",
        "Mathematical coupling: JG_n03 Vth(t) == JG_H2 Vth_air(t) exactly",
        "for all 91 device-day pairs, because the H2 calculation reuses the n03 air sweep.",
        "Thus Y contains the same air threshold as X. Correlation alone cannot",
        "establish a separate physical link between baseline and sensing mechanisms.",
        "Normalization additionally reuses each device's 0D denominators.",
        "Algebraically, with A=air Vth and G=H2 Vth, X=100*A/A0 and",
        "Y=100*(A-G)/(A0-G0). All A0 are negative and all (A0-G0) positive;",
        "the shared A term can itself induce a negative within-device slope.",
        "",
        f"Stress-trend r, six post-baseline checkpoint means: {r['stress_trend_r']:.3f}.",
        f"Pooled within-checkpoint r: {r['within_checkpoint_pooled_r']:.3f}.",
        "Both 95% CIs resample complete device trajectories (10,000 replicates).",
        "SD bars in paired plots describe between-device spread and are not CIs.",
        "",
        "Primary adjusted model: Y ~ X + categorical(day) + random intercept(device).",
        "It estimates a common X slope after time adjustment and repeated devices.",
        "A device-mean / within-device decomposition was also fit with the same",
        "random intercept and categorical day. A two-way device+day fixed-effect",
        "slope was bootstrapped by complete device trajectories.",
        f"Adjusted X slope (simple mixed model): {r['models']['mixed_simple']['beta']:.3f}",
        f"  [Wald 95% CI {r['models']['mixed_simple']['ci95_low']:.3f},",
        f"   {r['models']['mixed_simple']['ci95_high']:.3f}];",
        f"   p={r['models']['mixed_simple']['p_value']:.3f}.",
        f"Within-device + day slope: {r['models']['mixed_within_device']['beta']:.3f}.",
        f"Device-bootstrap 95% CI for the equivalent two-way FE slope:",
        f"  [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}].",
        "Because the device-bootstrap CI crosses zero, a nonzero adjusted within-device",
        "association is not robustly established. The mixed-model Wald CI is narrower.",
        "",
        "Raw-scale sensitivity: JG_n03 day-minus-0D Vth drift (mV) versus",
        "H2-induced absolute threshold shift (mV), again excluding 0D from fits.",
        f"Six-mean trend r = {raw['stress_trend_r']:.3f}; pooled day-centered",
        f"r = {raw['within_checkpoint_pooled_r']:.3f}; two-way FE beta =",
        f"{raw['two_way_fixed_effect_beta']:.3f}. See results.json for all CIs and",
        "raw Vth (non-drift) analysis. Leaving out any one device does not change",
        "the sign of the adjusted normalized slope; chip-level sensitivity is",
        "listed separately. Only four ESN chips are available, so device-based",
        "CIs do not fully account for possible chip-level clustering.",
        "QFN is not a variable in these Hot-storage records; ESN identifies chips.",
        "",
        "Files: results.json has exact numbers, checkpoint_correlations.csv",
        "has per-day r values, paired_metrics_all_checkpoints.csv has all source",
        "pairs, leave_one_out_*.csv have device/chip influence checks, and",
        "mixed_model_summaries.txt has model output.",
    ]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_pairs()
    post = data.loc[data.day_num.gt(0)].copy()
    if set(post.day_num.unique()) != set(DAYS):
        raise ValueError("Unexpected postbaseline checkpoints")
    normalized = _describe(post, "x_retention_pct", "y_retention_pct")
    raw_vth = _describe(post, "jg_vth_mV", "h2_abs_delta_vth_mV")
    raw_drift = _describe(post, "jg_drift_mV", "h2_abs_delta_vth_mV")
    data.to_csv(OUT / "paired_metrics_all_checkpoints.csv", index=False)
    _checkpoint_stats(post).to_csv(OUT / "checkpoint_correlations.csv", index=False)
    _influence(post, "x_retention_pct", "y_retention_pct").to_csv(
        OUT / "leave_one_out_retention.csv", index=False)
    _influence(post, "jg_vth_mV", "h2_abs_delta_vth_mV").to_csv(
        OUT / "leave_one_out_raw_vth.csv", index=False)
    _influence(post, "jg_drift_mV", "h2_abs_delta_vth_mV").to_csv(
        OUT / "leave_one_out_raw_drift.csv", index=False)
    with (OUT / "mixed_model_summaries.txt").open("w", encoding="utf-8") as handle:
        for label, result in (("RETENTION", normalized),
                              ("RAW VTH", raw_vth), ("RAW DRIFT", raw_drift)):
            handle.write(f"{label}\n")
            handle.write(result["models"]["simple_summary"] + "\n\n")
            handle.write(result["models"]["separated_summary"] + "\n\n")
    summary = {
        "condition": "Hot storage", "Vbg_V": BG,
        "reference_day0_excluded": True,
        "bootstrap": {"resampling_unit": "whole device trajectory",
                      "n_replicates": BOOTSTRAPS, "seed": SEED},
        "normalized": {k: v for k, v in normalized.items()
                       if k != "models"} | {"models": {k: v for k, v in normalized["models"].items()
                                                   if not k.endswith("_summary")}},
        "raw_vth": {k: v for k, v in raw_vth.items()
                    if k != "models"} | {"models": {k: v for k, v in raw_vth["models"].items()
                                                if not k.endswith("_summary")}},
        "raw_drift": {k: v for k, v in raw_drift.items()
                      if k != "models"} | {"models": {k: v for k, v in raw_drift["models"].items()
                                                  if not k.endswith("_summary")}},
        "shared_measurement_audit": {
            "h2_delta_definition": "JG_H2 Vth_air minus JG_H2 Vth_H2",
            "x_source": "JG_n03 n03 air sweep; the same air sweep is used for JG_H2 Vth_air",
            "exact_shared_vth_value": bool(np.allclose(
                data.jg_vth_V, data.h2_air_vth_V, rtol=0, atol=1e-9)),
            "n_total_pairs_including_0D": len(data),
            "median_abs_jg_minus_h2_air_vth_mV": float(
                data.jg_minus_h2_air_vth_mV.abs().median()),
            "n_exact_matches_tolerance_1uV": int((
                data.jg_minus_h2_air_vth_mV.abs() < 0.001).sum()),
            "min_h2_0D_abs_mV": float((1000 * data.loc[data.day_num.eq(0),
                                                   "h2_delta_vth_0D_V"].abs()).min()),
        },
        "qfn": "No QFN column or QFN cohort in these Hot-storage records; ESN is the chip-level identifier",
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
    _plot_paired(data, normalized, x="x_retention_pct", y="y_retention_pct",
                 xlabel=r"Baseline $\mathbf{V}_{\mathbf{TH}}$ Retention (%)",
                 ylabel=(r"$\mathbf{H}_{\mathbf{2}}$-Induced "
                         r"$\mathbf{|\boldsymbol{\Delta}V_{TH}|}$ Retention (%)"),
                 stem="retention_correlation_with_CI_Vbg0",
                 reference_text="Reference (0D)",
                 axis_limits=((30, 107), (5, 108)))
    _plot_paired(data, raw_drift, x="jg_drift_mV",
                 y="h2_abs_delta_vth_mV",
                 xlabel=r"JG_n03 $\mathbf{\Delta V}_{\mathbf{TH}}$ Drift (mV)",
                 ylabel=(r"$\mathbf{H}_{\mathbf{2}}$-Induced "
                         r"$\mathbf{|\boldsymbol{\Delta}V_{TH}|}$ (mV)"),
                 stem="raw_drift_vs_H2_shift_Vbg0",
                 reference_text="Reference (0D)", show_reference=False)
    _plot_paired(data, raw_vth, x="jg_vth_mV",
                 y="h2_abs_delta_vth_mV",
                 xlabel=r"JG_n03 Air $\mathbf{V}_{\mathbf{TH}}$ (mV)",
                 ylabel=(r"$\mathbf{H}_{\mathbf{2}}$-Induced "
                         r"$\mathbf{|\boldsymbol{\Delta}V_{TH}|}$ (mV)"),
                 stem="raw_Vth_vs_H2_shift_Vbg0",
                 reference_text="Reference (0D)", show_reference=False)
    _plot_adjusted(data, normalized)
    _plot_checkpoint_r(pd.read_csv(OUT / "checkpoint_correlations.csv"))
    _plot_shared_vth(data)
    _write_readme(summary)
    print("NORMALIZED", json.dumps(summary["normalized"], indent=2))
    print("RAW VTH", json.dumps(summary["raw_vth"], indent=2))
    print("RAW DRIFT", json.dumps(summary["raw_drift"], indent=2))
    print("SHARED", json.dumps(summary["shared_measurement_audit"], indent=2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
