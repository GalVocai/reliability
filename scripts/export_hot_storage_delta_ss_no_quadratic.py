"""Re-export the Hot Storage Delta SS figure without the quadratic check.

Reads the already saved numerical checkpoint summary. The mean, sample SD,
selection, and full-curve model are unchanged.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT = Path.home() / "Desktop" / "Hot_storage_SS_full_sweep_Vbg0"
SOURCE = OUTPUT / "Hot_storage_SS_full_sweep_Vbg0_checkpoint_summary.csv"
STEM = "Hot_storage_Delta_SS_full_sweep_Vbg0_mean_SD_no_quadratic"
EXPECTED_DAYS = np.array([0, 2, 7, 14, 20, 25, 28], dtype=float)


def main() -> None:
    summary = pd.read_csv(SOURCE).sort_values("day_num")
    if (not np.array_equal(summary.day_num.to_numpy(float), EXPECTED_DAYS)
            or not summary.n_selected.eq(13).all()
            or not summary.n_model_QC_pass.eq(13).all()
            or summary[["delta_model_mean", "delta_model_SD"]].isna().any().any()):
        raise ValueError("Saved Hot Storage checkpoint summary is incomplete")

    fig, ax = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    x = summary.day_num.to_numpy(float)
    ax.errorbar(x, summary.delta_model_mean, yerr=summary.delta_model_SD,
                color="#1f77b4", ecolor="#555555", marker="o", markersize=6,
                linewidth=2.2, elinewidth=1.5, capsize=4,
                label="Full-curve model: mean ± SD")
    ax.axhline(0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Stress Time (days)", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"Model-derived $\Delta SS_{100\,\mathrm{nA}}$ (mV/dec)",
                  fontsize=15, fontweight="bold")
    ax.tick_params(direction="in", labelsize=12)
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.margins(x=0.05, y=0.18)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{STEM}.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
