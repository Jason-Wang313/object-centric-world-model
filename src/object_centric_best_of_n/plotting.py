"""Figure generation for the object-centric Best-of-N paper scaffold."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _line(df: pd.DataFrame, x: str, y: str, label_col: str, ax: plt.Axes) -> None:
    for label, group in df.groupby(label_col, sort=True):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", linewidth=2, label=str(label))


def figure1_selected_tail_binding_failure(main: pd.DataFrame, out: Path) -> None:
    df = main[(main["scenario"] == "raw") & (main["selector"] == "raw")].sort_values("N")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(df["N"], df["selected_object_score_mean"], marker="o", color="#2f6f9f", label="selected object score")
    ax1.plot(df["N"], df["selected_real_utility_mean"], marker="s", color="#b23b3b", label="selected real utility")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Best-of-N samples")
    ax1.set_ylabel("mean selected value")
    ax1.set_title("Selected-tail object binding failure")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    _save(fig, out / "figure1_selected_tail_binding_failure.png")


def figure2_repair_comparison(main: pd.DataFrame, out: Path) -> None:
    selectors = ["raw", "identity_consistent", "property_calibrated", "targeted_probe", "combined_repair", "random", "oracle"]
    df = main[(main["scenario"] == "raw") & (main["selector"].isin(selectors))]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _line(df, "N", "selected_real_utility_mean", "selector", ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Best-of-N samples")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("Object-specific repair comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, out / "figure2_repair_comparison.png")


def figure3_tail_diagnostics(main: pd.DataFrame, out: Path) -> None:
    df = main[(main["scenario"] == "raw") & (main["selector"] == "raw")].sort_values("N")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["N"], df["identity_error_mean"], marker="o", label="identity error")
    ax.plot(df["N"], df["swap_rate_mean"], marker="s", label="slot swap")
    ax.plot(df["N"], df["object_real_gap_mean"], marker="^", label="object-real gap")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Best-of-N samples")
    ax.set_ylabel("mean diagnostic")
    ax.set_title("Selected-tail diagnostics")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "figure3_tail_diagnostics.png")


def figure4_targeted_probe(seed_df: pd.DataFrame, out: Path) -> None:
    df = seed_df[(seed_df["scenario"] == "hidden_property") & (seed_df["selector"].isin(["raw", "targeted_probe"]))]
    agg = df.groupby(["selector", "N"], as_index=False)["selected_real_utility"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _line(agg, "N", "selected_real_utility", "selector", ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Best-of-N samples")
    ax.set_ylabel("selected real utility")
    ax.set_title("Targeted hidden-property probing")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "figure4_targeted_probe_before_after.png")


def figure5_exact_law_validation(law_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(law_df["predicted_selected_utility"], law_df["empirical_selected_utility"], s=38, alpha=0.80)
    low = min(law_df["predicted_selected_utility"].min(), law_df["empirical_selected_utility"].min())
    high = max(law_df["predicted_selected_utility"].max(), law_df["empirical_selected_utility"].max())
    ax.plot([low, high], [low, high], color="#333333", linewidth=1)
    ax.set_xlabel("exact law prediction")
    ax.set_ylabel("Monte Carlo empirical")
    ax.set_title("Finite tie-aware law validation")
    ax.grid(alpha=0.25)
    _save(fig, out / "figure5_exact_law_validation.png")


def write_all_figures(main: pd.DataFrame, seed_df: pd.DataFrame, law_df: pd.DataFrame, figure_dir: str | Path) -> None:
    out = Path(figure_dir)
    figure1_selected_tail_binding_failure(main, out)
    figure2_repair_comparison(main, out)
    figure3_tail_diagnostics(main, out)
    figure4_targeted_probe(seed_df, out)
    figure5_exact_law_validation(law_df, out)
