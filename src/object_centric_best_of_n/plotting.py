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


def figure6_stress_robustness(stress: pd.DataFrame, out: Path) -> None:
    if stress.empty:
        return
    df = stress[(stress["N"] == stress["N"].max()) & (stress["selector"].isin(["raw", "combined_repair", "oracle"]))]
    pivot = df.pivot_table(
        index="scenario",
        columns="selector",
        values="selected_real_utility_mean",
        aggfunc="mean",
    ).reindex(columns=["raw", "combined_repair", "oracle"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    pivot.plot(kind="bar", ax=ax, color=["#b23b3b", "#3c7c5a", "#2f6f9f"])
    ax.set_xlabel("stress scenario")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("High-N stress robustness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "figure6_stress_robustness.png")


def figure7_learned_learning_curve(learned_curve: pd.DataFrame, out: Path) -> None:
    if learned_curve.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    axes[0].plot(learned_curve["train_scenes"], learned_curve["property_accuracy"], marker="o", label="model")
    axes[0].plot(learned_curve["train_scenes"], learned_curve["random_property_accuracy"], linestyle="--", label="baseline")
    axes[0].set_title("hidden property")
    axes[0].set_ylabel("accuracy")
    axes[1].plot(learned_curve["train_scenes"], learned_curve["identity_alignment_accuracy"], marker="o")
    axes[1].plot(learned_curve["train_scenes"], learned_curve["random_identity_alignment_accuracy"], linestyle="--")
    axes[1].set_title("identity alignment")
    axes[2].plot(learned_curve["train_scenes"], learned_curve["transition_mse"], marker="o", label="model")
    axes[2].plot(learned_curve["train_scenes"], learned_curve["constant_transition_mse"], linestyle="--", label="baseline")
    axes[2].set_title("transition")
    axes[2].set_ylabel("MSE")
    for ax in axes:
        ax.set_xlabel("train scenes")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out / "figure7_learned_object_model.png")


def figure8_repair_ablation(ablation: pd.DataFrame, out: Path) -> None:
    if ablation.empty:
        return
    df = ablation.set_index("scenario")[
        ["raw_selected_real_utility", "best_single_repair_utility", "combined_repair_utility", "oracle_utility"]
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    df.plot(kind="bar", ax=ax, color=["#b23b3b", "#d1963a", "#3c7c5a", "#2f6f9f"])
    ax.set_xlabel("scenario")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("Repair ablation at high N")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["raw", "best single repair", "combined repair", "oracle"], frameon=False, fontsize=8)
    _save(fig, out / "figure8_repair_ablation.png")


def figure9_seed_block_robustness(robustness: pd.DataFrame, out: Path) -> None:
    if robustness.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    x = robustness["block_id"].astype(str)
    axes[0].bar(x, robustness["raw_tail_score_gain"], color="#2f6f9f")
    axes[0].set_title("raw score gain")
    axes[1].bar(x, robustness["raw_tail_utility_drop"], color="#b23b3b")
    axes[1].set_title("raw utility drop")
    axes[2].bar(x, robustness["combined_raw_nmax_gain"], color="#3c7c5a")
    axes[2].set_title("combined gain")
    for ax in axes:
        ax.set_xlabel("seed block")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("effect size")
    _save(fig, out / "figure9_seed_block_robustness.png")


def figure10_score_calibration(calibration: pd.DataFrame, out: Path) -> None:
    if calibration.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = calibration["score_bin"].astype(str)
    ax.plot(x, calibration["mean_raw_object_score"], marker="o", label="mean object score")
    ax.plot(x, calibration["mean_real_utility"], marker="s", label="mean real utility")
    ax.bar(x, calibration["object_real_gap"], alpha=0.25, label="object-real gap")
    ax.set_xlabel("raw-score quantile bin")
    ax.set_ylabel("mean value")
    ax.set_title("Raw object-score calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "figure10_score_calibration.png")


def figure11_sensitivity(sensitivity: pd.DataFrame, out: Path) -> None:
    if sensitivity.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for selector, group in sensitivity.groupby("selector", sort=True):
        group = group.sort_values("score_noise")
        ax.plot(group["score_noise"], group["selected_real_utility_mean"], marker="o", linewidth=2, label=str(selector))
    ax.set_xlabel("score noise std")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("Score-noise sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "figure11_score_noise_sensitivity.png")


def figure12_negative_control(negative: pd.DataFrame, out: Path) -> None:
    if negative.empty:
        return
    df = negative[negative["contrast"].isin(["good_control", "corrupted_mean"])].set_index("contrast")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    df[["selected_real_utility_mean", "identity_error_mean", "object_real_gap_mean"]].plot(
        kind="bar",
        ax=ax,
        color=["#3c7c5a", "#b23b3b", "#d1963a"],
    )
    ax.set_xlabel("raw high-N contrast")
    ax.set_ylabel("mean value")
    ax.set_title("Negative control: good scene vs corrupted scenes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["real utility", "identity error", "object-real gap"], frameon=False, fontsize=8)
    _save(fig, out / "figure12_negative_control.png")


def figure13_learned_ablation(learned_ablation: pd.DataFrame, out: Path) -> None:
    if learned_ablation.empty:
        return
    df = learned_ablation.set_index("ablation")[
        ["property_accuracy", "identity_alignment_accuracy", "reward_correlation"]
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    df.plot(kind="bar", ax=ax, color=["#2f6f9f", "#3c7c5a", "#d1963a"])
    ax.set_xlabel("learned feature set")
    ax.set_ylabel("metric")
    ax.set_title("Learned object-model ablations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["hidden property", "identity alignment", "reward"], frameon=False, fontsize=8)
    _save(fig, out / "figure13_learned_ablation.png")


def figure14_ood_stress(ood: pd.DataFrame, out: Path) -> None:
    if ood.empty:
        return
    df = ood[(ood["selector"].isin(["raw", "combined_repair", "oracle"])) & (ood["N"] == ood["N"].max())]
    pivot = df.pivot_table(
        index="scenario",
        columns="selector",
        values="selected_real_utility_mean",
        aggfunc="mean",
    ).reindex(columns=["raw", "combined_repair", "oracle"])
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    pivot.plot(kind="bar", ax=ax, color=["#b23b3b", "#3c7c5a", "#2f6f9f"])
    ax.set_xlabel("OOD synthetic variant")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("OOD object-count stress")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "figure14_ood_object_count_stress.png")


def figure15_model_family_proxies(family: pd.DataFrame, out: Path) -> None:
    if family.empty:
        return
    selectors = [
        "raw",
        "latent_global_proxy",
        "relational_slot_proxy",
        "diffusion_score_proxy",
        "combined_repair",
        "oracle",
    ]
    df = family[(family["selector"].isin(selectors)) & (family["N"] == family["N"].max())]
    pivot = df.pivot_table(
        index="scenario",
        columns="selector",
        values="selected_real_utility_mean",
        aggfunc="mean",
    ).reindex(columns=selectors)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    pivot.plot(kind="bar", ax=ax, color=["#b23b3b", "#8a6fba", "#d1963a", "#707070", "#3c7c5a", "#2f6f9f"])
    ax.set_xlabel("controlled synthetic scenario")
    ax.set_ylabel("mean selected real utility")
    ax.set_title("Toy model-family proxy selectors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, out / "figure15_model_family_proxies.png")


def write_all_figures(
    main: pd.DataFrame,
    seed_df: pd.DataFrame,
    law_df: pd.DataFrame,
    figure_dir: str | Path,
    stress_df: pd.DataFrame | None = None,
    learned_curve: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
    robustness_df: pd.DataFrame | None = None,
    calibration_df: pd.DataFrame | None = None,
    sensitivity_df: pd.DataFrame | None = None,
    negative_df: pd.DataFrame | None = None,
    learned_ablation_df: pd.DataFrame | None = None,
    ood_df: pd.DataFrame | None = None,
    family_df: pd.DataFrame | None = None,
) -> None:
    out = Path(figure_dir)
    figure1_selected_tail_binding_failure(main, out)
    figure2_repair_comparison(main, out)
    figure3_tail_diagnostics(main, out)
    figure4_targeted_probe(seed_df, out)
    figure5_exact_law_validation(law_df, out)
    if stress_df is not None:
        figure6_stress_robustness(stress_df, out)
    if learned_curve is not None:
        figure7_learned_learning_curve(learned_curve, out)
    if ablation_df is not None:
        figure8_repair_ablation(ablation_df, out)
    if robustness_df is not None:
        figure9_seed_block_robustness(robustness_df, out)
    if calibration_df is not None:
        figure10_score_calibration(calibration_df, out)
    if sensitivity_df is not None:
        figure11_sensitivity(sensitivity_df, out)
    if negative_df is not None:
        figure12_negative_control(negative_df, out)
    if learned_ablation_df is not None:
        figure13_learned_ablation(learned_ablation_df, out)
    if ood_df is not None:
        figure14_ood_stress(ood_df, out)
    if family_df is not None:
        figure15_model_family_proxies(family_df, out)
