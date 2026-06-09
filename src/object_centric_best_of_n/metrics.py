"""Metrics and aggregation helpers for object-centric Best-of-N experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .object_model import Candidate
from .repair import GATE_ACTIONS, conservative_selected_tail_stop_rule


MODEL_FAMILY_PROXY_SELECTORS = ("raw", "latent_global_proxy", "relational_slot_proxy", "diffusion_score_proxy")


def mean_ci(values: Iterable[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, mean - z * se, mean + z * se


def upper_tail_rank_correlation(candidates: list[Candidate], q: float = 0.75) -> float:
    if len(candidates) < 3:
        return 0.0
    scores = np.asarray([c.score for c in candidates], dtype=float)
    utilities = np.asarray([c.real_utility for c in candidates], dtype=float)
    threshold = np.quantile(scores, q)
    mask = scores >= threshold
    if np.sum(mask) < 3 or np.std(scores[mask]) < 1e-12 or np.std(utilities[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(scores[mask], utilities[mask])[0, 1])


def selection_record(
    experiment: str,
    scenario: str,
    selector: str,
    n: int,
    seed: int,
    selected: Candidate,
    candidates: list[Candidate],
) -> dict[str, float | int | str]:
    raw_scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
    raw_utilities = np.asarray([candidate.real_utility for candidate in candidates], dtype=float)
    return {
        "experiment": experiment,
        "scenario": scenario,
        "selector": selector,
        "N": int(n),
        "seed": int(seed),
        "selected_candidate_id": int(selected.candidate_id),
        "selected_object_score": float(selected.score),
        "selected_real_utility": float(selected.real_utility),
        "identity_error": float(selected.identity_error),
        "swap_rate": float(selected.swap),
        "merge_split_rate": float(selected.merge_split),
        "property_error": float(selected.property_error),
        "property_entropy": float(selected.property_entropy),
        "occlusion_error": float(selected.occlusion_error),
        "object_real_gap": float(selected.score - selected.real_utility),
        "candidate_mean_score": float(np.mean(raw_scores)),
        "candidate_mean_real_utility": float(np.mean(raw_utilities)),
        "candidate_best_real_utility": float(np.max(raw_utilities)),
        "regret": float(np.max(raw_utilities) - selected.real_utility),
        "oracle_gap": float(np.max(raw_utilities) - selected.real_utility),
        "upper_tail_rank_correlation": upper_tail_rank_correlation(candidates),
    }


def aggregate_seed_metrics(seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "scenario", "selector", "N"]
    value_cols = [
        "selected_object_score",
        "selected_real_utility",
        "identity_error",
        "swap_rate",
        "merge_split_rate",
        "property_error",
        "property_entropy",
        "occlusion_error",
        "object_real_gap",
        "regret",
        "oracle_gap",
        "upper_tail_rank_correlation",
    ]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in seed_df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for col in value_cols:
            mean, low, high = mean_ci(group[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_low"] = low
            row[f"{col}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _binomial_two_sided_p(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    tail = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail))


def _bootstrap_mean_ci(values: Iterable[float], reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(arr))
    if arr.size == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(reps), arr.size), replace=True)
    means = np.mean(draws, axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return estimate, float(low), float(high)


def _paired_gain_units(
    df: pd.DataFrame,
    scenario: str,
    treatment_selector: str,
    baseline_selector: str = "raw",
    n: int | None = None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if n is None:
        n = int(df["N"].max())
    key_cols = ["scenario", "N", "seed"]
    base = df[
        (df["scenario"] == scenario)
        & (df["selector"] == baseline_selector)
        & (df["N"] == n)
    ][key_cols + ["selected_real_utility"]].rename(columns={"selected_real_utility": "baseline"})
    treat = df[
        (df["scenario"] == scenario)
        & (df["selector"] == treatment_selector)
        & (df["N"] == n)
    ][key_cols + ["selected_real_utility"]].rename(columns={"selected_real_utility": "treatment"})
    merged = treat.merge(base, on=key_cols, how="inner")
    return merged["treatment"] - merged["baseline"]


def paired_selector_effects(seed_df: pd.DataFrame, baseline: str = "raw") -> pd.DataFrame:
    """Paired per-seed selector gains against a baseline selector."""

    key_cols = ["experiment", "scenario", "N", "seed"]
    baseline_df = seed_df[seed_df["selector"] == baseline][key_cols + ["selected_real_utility"]].rename(
        columns={"selected_real_utility": "baseline_selected_real_utility"}
    )
    rows: list[dict[str, float | int | str]] = []
    for selector, group in seed_df[seed_df["selector"] != baseline].groupby("selector", sort=True):
        merged = group.merge(baseline_df, on=key_cols, how="inner")
        for keys, pair_group in merged.groupby(["experiment", "scenario", "N"], sort=True):
            gains = pair_group["selected_real_utility"] - pair_group["baseline_selected_real_utility"]
            mean, low, high = mean_ci(gains)
            wins = int(np.sum(gains > 1e-12))
            losses = int(np.sum(gains < -1e-12))
            ties = int(gains.size - wins - losses)
            std = float(np.std(gains, ddof=1)) if gains.size > 1 else 0.0
            effect_size = float(mean / std) if std > 1e-12 else float("inf") if mean > 0 else 0.0
            row = dict(zip(["experiment", "scenario", "N"], keys))
            row.update(
                {
                    "selector": selector,
                    "baseline": baseline,
                    "n_pairs": int(gains.size),
                    "mean_gain": mean,
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate": float(wins / gains.size) if gains.size else 0.0,
                    "sign_test_p": _binomial_two_sided_p(wins, losses),
                    "paired_effect_size": effect_size,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def stress_summary(stress_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-N stress rows and add paired raw-to-repair gains."""

    if stress_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(stress_seed_df)
    paired = paired_selector_effects(stress_seed_df)
    paired = paired.rename(columns={"selector": "comparison_selector"})
    combined = paired[paired["comparison_selector"] == "combined_repair"][
        ["experiment", "scenario", "N", "mean_gain", "gain_ci_low", "gain_ci_high", "win_rate", "sign_test_p"]
    ].rename(
        columns={
            "mean_gain": "combined_vs_raw_gain_mean",
            "gain_ci_low": "combined_vs_raw_gain_ci_low",
            "gain_ci_high": "combined_vs_raw_gain_ci_high",
            "win_rate": "combined_vs_raw_win_rate",
            "sign_test_p": "combined_vs_raw_sign_test_p",
        }
    )
    return main.merge(combined, on=["experiment", "scenario", "N"], how="left")


def repair_ablation_summary(main_metrics: pd.DataFrame, paired_effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize high-N repair components against raw and oracle."""

    if main_metrics.empty:
        return pd.DataFrame()
    nmax = int(main_metrics["N"].max())
    selectors = [
        "raw",
        "identity_consistent",
        "property_calibrated",
        "targeted_probe",
        "combined_repair",
        "random",
        "oracle",
    ]
    scenarios = ["raw", "hidden_property", "occlusion", "swap", "merge_split"]
    df = main_metrics[
        (main_metrics["N"] == nmax)
        & (main_metrics["scenario"].isin(scenarios))
        & (main_metrics["selector"].isin(selectors))
    ].copy()
    if df.empty:
        return df
    pair_cols = ["scenario", "selector", "N", "mean_gain", "win_rate", "sign_test_p"]
    available_pair_cols = [col for col in pair_cols if col in paired_effects.columns]
    if not paired_effects.empty and set(["scenario", "selector", "N"]).issubset(paired_effects.columns):
        df = df.merge(
            paired_effects[available_pair_cols],
            on=["scenario", "selector", "N"],
            how="left",
        )
    rows: list[dict[str, float | int | str]] = []
    for scenario, group in df.groupby("scenario", sort=True):
        raw_value = float(group.loc[group["selector"] == "raw", "selected_real_utility_mean"].iloc[0])
        oracle_value = float(group.loc[group["selector"] == "oracle", "selected_real_utility_mean"].iloc[0])
        component_group = group[group["selector"].isin(["identity_consistent", "property_calibrated", "targeted_probe"])]
        best_component = float(component_group["selected_real_utility_mean"].max()) if not component_group.empty else float("nan")
        combined_value = float(group.loc[group["selector"] == "combined_repair", "selected_real_utility_mean"].iloc[0])
        combined_row = group[group["selector"] == "combined_repair"].iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "N": nmax,
                "raw_selected_real_utility": raw_value,
                "best_single_repair_utility": best_component,
                "combined_repair_utility": combined_value,
                "oracle_utility": oracle_value,
                "combined_vs_raw_gain": combined_value - raw_value,
                "combined_vs_best_single_gain": combined_value - best_component,
                "combined_oracle_gap": oracle_value - combined_value,
                "combined_win_rate_vs_raw": float(combined_row.get("win_rate", np.nan)),
                "combined_sign_test_p": float(combined_row.get("sign_test_p", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def observable_repair_summary(main_metrics: pd.DataFrame, paired_effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize observable-only repair against raw, controlled repair, and oracle."""

    if main_metrics.empty:
        return pd.DataFrame()
    nmax = int(main_metrics["N"].max())
    scenarios = ["raw", "hidden_property", "occlusion", "swap", "merge_split"]
    selectors = ["raw", "observable_repair", "combined_repair", "oracle"]
    df = main_metrics[
        (main_metrics["N"] == nmax)
        & (main_metrics["scenario"].isin(scenarios))
        & (main_metrics["selector"].isin(selectors))
    ].copy()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for scenario, group in df.groupby("scenario", sort=True):
        if not set(selectors).issubset(set(group["selector"])):
            continue
        raw_value = float(group.loc[group["selector"] == "raw", "selected_real_utility_mean"].iloc[0])
        observable_value = float(
            group.loc[group["selector"] == "observable_repair", "selected_real_utility_mean"].iloc[0]
        )
        combined_value = float(group.loc[group["selector"] == "combined_repair", "selected_real_utility_mean"].iloc[0])
        oracle_value = float(group.loc[group["selector"] == "oracle", "selected_real_utility_mean"].iloc[0])
        pair = paired_effects[
            (paired_effects["scenario"] == scenario)
            & (paired_effects["selector"] == "observable_repair")
            & (paired_effects["N"] == nmax)
        ]
        rows.append(
            {
                "scenario": scenario,
                "N": nmax,
                "raw_selected_real_utility": raw_value,
                "observable_repair_utility": observable_value,
                "combined_repair_utility": combined_value,
                "oracle_utility": oracle_value,
                "observable_vs_raw_gain": observable_value - raw_value,
                "combined_minus_observable_gap": combined_value - observable_value,
                "observable_oracle_gap": oracle_value - observable_value,
                "observable_win_rate_vs_raw": float(pair["win_rate"].iloc[0]) if not pair.empty else float("nan"),
                "observable_sign_test_p": float(pair["sign_test_p"].iloc[0]) if not pair.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def seed_block_robustness(seed_df: pd.DataFrame, block_size: int = 4) -> pd.DataFrame:
    """Check whether key effects survive contiguous seed blocks."""

    if seed_df.empty:
        return pd.DataFrame()
    seeds = sorted(int(seed) for seed in seed_df["seed"].unique())
    rows: list[dict[str, float | int | str]] = []
    for block_id, start in enumerate(range(0, len(seeds), block_size)):
        block_seeds = seeds[start : start + block_size]
        if not block_seeds:
            continue
        block = seed_df[seed_df["seed"].isin(block_seeds)]
        raw = block[(block["scenario"] == "raw") & (block["selector"] == "raw")]
        if raw.empty:
            continue
        raw_by_n = raw.groupby("N", sort=True)[["selected_object_score", "selected_real_utility", "identity_error"]].mean()
        raw_by_n = raw_by_n.sort_index()
        raw64 = block[(block["scenario"] == "raw") & (block["selector"] == "raw") & (block["N"] == raw_by_n.index.max())]
        combined64 = block[
            (block["scenario"] == "raw") & (block["selector"] == "combined_repair") & (block["N"] == raw_by_n.index.max())
        ]
        pair = raw64[["seed", "selected_real_utility"]].merge(
            combined64[["seed", "selected_real_utility"]],
            on="seed",
            suffixes=("_raw", "_combined"),
        )
        gains = pair["selected_real_utility_combined"] - pair["selected_real_utility_raw"]
        rows.append(
            {
                "block_id": int(block_id),
                "seed_min": int(min(block_seeds)),
                "seed_max": int(max(block_seeds)),
                "n_seeds": int(len(block_seeds)),
                "raw_tail_score_gain": float(raw_by_n["selected_object_score"].iloc[-1] - raw_by_n["selected_object_score"].iloc[0]),
                "raw_tail_utility_drop": float(raw_by_n["selected_real_utility"].iloc[0] - raw_by_n["selected_real_utility"].iloc[-1]),
                "raw_tail_identity_error": float(raw_by_n["identity_error"].iloc[-1]),
                "combined_raw_nmax_gain": float(np.mean(gains)) if len(gains) else float("nan"),
                "combined_raw_nmax_win_rate": float(np.mean(gains > 1e-12)) if len(gains) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def score_calibration_table(candidate_df: pd.DataFrame, bins: int = 6) -> pd.DataFrame:
    """Bin raw candidate object scores and compare them with real utility."""

    if candidate_df.empty:
        return pd.DataFrame()
    df = candidate_df.copy()
    df["score_bin"] = pd.qcut(df["raw_object_score"].rank(method="first"), q=bins, labels=False)
    rows: list[dict[str, float | int]] = []
    for score_bin, group in df.groupby("score_bin", sort=True):
        rows.append(
            {
                "score_bin": int(score_bin),
                "count": int(group.shape[0]),
                "mean_raw_object_score": float(group["raw_object_score"].mean()),
                "mean_real_utility": float(group["real_utility"].mean()),
                "object_real_gap": float(group["raw_object_score"].mean() - group["real_utility"].mean()),
                "identity_error_rate": float(group["identity_error"].mean()),
                "merge_split_rate": float(group["merge_split"].mean()),
                "property_error_rate": float(group["property_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_summary(sensitivity_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score-noise sensitivity rows."""

    if sensitivity_seed_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for keys, group in sensitivity_seed_df.groupby(["selector", "score_noise"], sort=True):
        selector, score_noise = keys
        utility_mean, utility_low, utility_high = mean_ci(group["selected_real_utility"])
        identity_mean, _, _ = mean_ci(group["identity_error"])
        rows.append(
            {
                "selector": str(selector),
                "score_noise": float(score_noise),
                "n_rows": int(group.shape[0]),
                "selected_real_utility_mean": utility_mean,
                "selected_real_utility_ci_low": utility_low,
                "selected_real_utility_ci_high": utility_high,
                "identity_error_mean": identity_mean,
            }
        )
    return pd.DataFrame(rows)


def negative_control_summary(main_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether high-N collapse is specific to corrupted object scenarios."""

    if main_metrics.empty:
        return pd.DataFrame()
    nmax = int(main_metrics["N"].max())
    raw = main_metrics[(main_metrics["selector"] == "raw") & (main_metrics["N"] == nmax)].copy()
    if raw.empty:
        return pd.DataFrame()
    corrupted = raw[raw["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"])]
    good = raw[raw["scenario"] == "good"]
    rows: list[dict[str, float | int | str]] = []
    if not good.empty:
        rows.append(
            {
                "contrast": "good_control",
                "N": nmax,
                "selected_real_utility_mean": float(good["selected_real_utility_mean"].iloc[0]),
                "identity_error_mean": float(good["identity_error_mean"].iloc[0]),
                "object_real_gap_mean": float(good["object_real_gap_mean"].iloc[0]),
            }
        )
    if not corrupted.empty:
        rows.append(
            {
                "contrast": "corrupted_mean",
                "N": nmax,
                "selected_real_utility_mean": float(corrupted["selected_real_utility_mean"].mean()),
                "identity_error_mean": float(corrupted["identity_error_mean"].mean()),
                "object_real_gap_mean": float(corrupted["object_real_gap_mean"].mean()),
            }
        )
    if not good.empty and not corrupted.empty:
        rows.append(
            {
                "contrast": "good_minus_corrupted",
                "N": nmax,
                "selected_real_utility_mean": float(good["selected_real_utility_mean"].iloc[0] - corrupted["selected_real_utility_mean"].mean()),
                "identity_error_mean": float(good["identity_error_mean"].iloc[0] - corrupted["identity_error_mean"].mean()),
                "object_real_gap_mean": float(good["object_real_gap_mean"].iloc[0] - corrupted["object_real_gap_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def ood_summary(ood_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OOD synthetic rows and attach combined-vs-raw paired gains."""

    if ood_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(ood_seed_df)
    paired = paired_selector_effects(ood_seed_df)
    combined = paired[paired["selector"] == "combined_repair"][
        ["experiment", "scenario", "N", "mean_gain", "gain_ci_low", "gain_ci_high", "win_rate", "sign_test_p"]
    ].rename(
        columns={
            "mean_gain": "combined_vs_raw_gain_mean",
            "gain_ci_low": "combined_vs_raw_gain_ci_low",
            "gain_ci_high": "combined_vs_raw_gain_ci_high",
            "win_rate": "combined_vs_raw_win_rate",
            "sign_test_p": "combined_vs_raw_sign_test_p",
        }
    )
    return main.merge(combined, on=["experiment", "scenario", "N"], how="left")


def domain_randomization_summary(domain_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out domain-randomized synthetic stress rows."""

    if domain_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(domain_seed_df)
    paired = paired_selector_effects(domain_seed_df)
    rows: list[pd.Series] = []
    for _, group in main.groupby(["experiment", "scenario", "N"], sort=True):
        raw = group[group["selector"] == "raw"]
        combined = group[group["selector"] == "combined_repair"]
        observable = group[group["selector"] == "observable_repair"]
        oracle = group[group["selector"] == "oracle"]
        raw_utility = float(raw["selected_real_utility_mean"].iloc[0]) if not raw.empty else float("nan")
        combined_utility = (
            float(combined["selected_real_utility_mean"].iloc[0]) if not combined.empty else float("nan")
        )
        observable_utility = (
            float(observable["selected_real_utility_mean"].iloc[0]) if not observable.empty else float("nan")
        )
        oracle_utility = float(oracle["selected_real_utility_mean"].iloc[0]) if not oracle.empty else float("nan")
        combined_pair = paired[
            (paired["scenario"] == group["scenario"].iloc[0])
            & (paired["N"] == group["N"].iloc[0])
            & (paired["selector"] == "combined_repair")
        ]
        observable_pair = paired[
            (paired["scenario"] == group["scenario"].iloc[0])
            & (paired["N"] == group["N"].iloc[0])
            & (paired["selector"] == "observable_repair")
        ]
        for _, row in group.iterrows():
            out = row.copy()
            out["domain_combined_vs_raw_gain_mean"] = combined_utility - raw_utility
            out["domain_observable_vs_raw_gain_mean"] = observable_utility - raw_utility
            out["domain_combined_oracle_gap_mean"] = oracle_utility - combined_utility
            out["domain_observable_oracle_gap_mean"] = oracle_utility - observable_utility
            out["domain_combined_win_rate"] = (
                float(combined_pair["win_rate"].iloc[0]) if not combined_pair.empty else float("nan")
            )
            out["domain_observable_win_rate"] = (
                float(observable_pair["win_rate"].iloc[0]) if not observable_pair.empty else float("nan")
            )
            rows.append(out)
    return pd.DataFrame(rows)


def counterfactual_target_summary(counter_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate target-swap rows where the true target is not object zero."""

    if counter_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(counter_seed_df)
    paired = paired_selector_effects(counter_seed_df)
    rows: list[pd.Series] = []
    for _, group in main.groupby(["experiment", "scenario", "N"], sort=True):
        raw = group[group["selector"] == "raw"]
        combined = group[group["selector"] == "combined_repair"]
        observable = group[group["selector"] == "observable_repair"]
        oracle = group[group["selector"] == "oracle"]
        raw_utility = float(raw["selected_real_utility_mean"].iloc[0]) if not raw.empty else float("nan")
        combined_utility = (
            float(combined["selected_real_utility_mean"].iloc[0]) if not combined.empty else float("nan")
        )
        observable_utility = (
            float(observable["selected_real_utility_mean"].iloc[0]) if not observable.empty else float("nan")
        )
        oracle_utility = float(oracle["selected_real_utility_mean"].iloc[0]) if not oracle.empty else float("nan")
        combined_pair = paired[
            (paired["scenario"] == group["scenario"].iloc[0])
            & (paired["N"] == group["N"].iloc[0])
            & (paired["selector"] == "combined_repair")
        ]
        observable_pair = paired[
            (paired["scenario"] == group["scenario"].iloc[0])
            & (paired["N"] == group["N"].iloc[0])
            & (paired["selector"] == "observable_repair")
        ]
        for _, row in group.iterrows():
            out = row.copy()
            out["counterfactual_combined_vs_raw_gain_mean"] = combined_utility - raw_utility
            out["counterfactual_observable_vs_raw_gain_mean"] = observable_utility - raw_utility
            out["counterfactual_combined_oracle_gap_mean"] = oracle_utility - combined_utility
            out["counterfactual_observable_oracle_gap_mean"] = oracle_utility - observable_utility
            out["counterfactual_combined_win_rate"] = (
                float(combined_pair["win_rate"].iloc[0]) if not combined_pair.empty else float("nan")
            )
            out["counterfactual_observable_win_rate"] = (
                float(observable_pair["win_rate"].iloc[0]) if not observable_pair.empty else float("nan")
            )
            rows.append(out)
    return pd.DataFrame(rows)


def pilot_calibration_summary(pilot_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out pilot-label calibration rows."""

    if pilot_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(pilot_seed_df)
    paired = paired_selector_effects(pilot_seed_df)
    rows: list[pd.Series] = []
    for _, group in main.groupby(["experiment", "scenario", "N"], sort=True):
        raw = group[group["selector"] == "raw"]
        pilot = group[group["selector"] == "pilot_calibrated"]
        observable = group[group["selector"] == "observable_repair"]
        combined = group[group["selector"] == "combined_repair"]
        oracle = group[group["selector"] == "oracle"]
        raw_utility = float(raw["selected_real_utility_mean"].iloc[0]) if not raw.empty else float("nan")
        pilot_utility = float(pilot["selected_real_utility_mean"].iloc[0]) if not pilot.empty else float("nan")
        observable_utility = (
            float(observable["selected_real_utility_mean"].iloc[0]) if not observable.empty else float("nan")
        )
        combined_utility = (
            float(combined["selected_real_utility_mean"].iloc[0]) if not combined.empty else float("nan")
        )
        oracle_utility = float(oracle["selected_real_utility_mean"].iloc[0]) if not oracle.empty else float("nan")
        pilot_pair = paired[
            (paired["scenario"] == group["scenario"].iloc[0])
            & (paired["N"] == group["N"].iloc[0])
            & (paired["selector"] == "pilot_calibrated")
        ]
        for _, row in group.iterrows():
            out = row.copy()
            out["pilot_vs_raw_gain_mean"] = pilot_utility - raw_utility
            out["pilot_vs_observable_gain_mean"] = pilot_utility - observable_utility
            out["pilot_vs_combined_gap_mean"] = combined_utility - pilot_utility
            out["pilot_oracle_gap_mean"] = oracle_utility - pilot_utility
            out["pilot_win_rate"] = float(pilot_pair["win_rate"].iloc[0]) if not pilot_pair.empty else float("nan")
            rows.append(out)
    return pd.DataFrame(rows)


def model_family_proxy_summary(family_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate toy model-family proxy selectors against combined repair.

    These rows are deliberately scoped to controlled synthetic proxy selectors.
    They are not evidence for broad benchmark superiority over model families.
    """

    if family_seed_df.empty:
        return pd.DataFrame()
    main = aggregate_seed_metrics(family_seed_df)
    rows: list[pd.Series] = []
    for _, group in main.groupby(["experiment", "scenario", "N"], sort=True):
        proxy_group = group[group["selector"].isin(MODEL_FAMILY_PROXY_SELECTORS)]
        combined = group[group["selector"] == "combined_repair"]
        oracle = group[group["selector"] == "oracle"]
        best_proxy = float(proxy_group["selected_real_utility_mean"].max()) if not proxy_group.empty else float("nan")
        best_proxy_selector = (
            str(proxy_group.sort_values("selected_real_utility_mean", ascending=False)["selector"].iloc[0])
            if not proxy_group.empty
            else ""
        )
        combined_utility = (
            float(combined["selected_real_utility_mean"].iloc[0]) if not combined.empty else float("nan")
        )
        oracle_utility = float(oracle["selected_real_utility_mean"].iloc[0]) if not oracle.empty else float("nan")
        for _, row in group.iterrows():
            out = row.copy()
            out["best_proxy_selector"] = best_proxy_selector
            out["best_proxy_utility_mean"] = best_proxy
            out["combined_vs_best_proxy_gain_mean"] = combined_utility - best_proxy
            out["combined_oracle_gap_mean"] = oracle_utility - combined_utility
            rows.append(out)
    return pd.DataFrame(rows)


def statistical_audit(
    seed_df: pd.DataFrame,
    ood_seed_df: pd.DataFrame | None = None,
    family_seed_df: pd.DataFrame | None = None,
    counterfactual_seed_df: pd.DataFrame | None = None,
    pilot_seed_df: pd.DataFrame | None = None,
    leave_one_failure_seed_df: pd.DataFrame | None = None,
    bootstrap_reps: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap confidence audit for the main controlled evidence claims."""

    rows: list[dict[str, float | int | str | bool]] = []

    def add_row(
        effect_id: str,
        description: str,
        units: Iterable[float],
        threshold: float,
        direction: str = "lower",
    ) -> None:
        values = list(units)
        estimate, low, high = _bootstrap_mean_ci(values, reps=bootstrap_reps, seed=seed + len(rows) * 997)
        ci_target = low if direction == "lower" else high
        passes = bool(ci_target >= threshold) if direction == "lower" else bool(ci_target <= threshold)
        rows.append(
            {
                "effect_id": effect_id,
                "description": description,
                "estimate": estimate,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "threshold": float(threshold),
                "direction": direction,
                "n_units": int(len(values)),
                "passes": passes,
            }
        )

    if not seed_df.empty:
        raw = seed_df[(seed_df["scenario"] == "raw") & (seed_df["selector"] == "raw")]
        if not raw.empty:
            n_min = int(raw["N"].min())
            n_max = int(raw["N"].max())
            low_n = raw[raw["N"] == n_min][["seed", "selected_object_score", "selected_real_utility"]].rename(
                columns={
                    "selected_object_score": "score_low_n",
                    "selected_real_utility": "utility_low_n",
                }
            )
            high_n = raw[raw["N"] == n_max][["seed", "selected_object_score", "selected_real_utility"]].rename(
                columns={
                    "selected_object_score": "score_high_n",
                    "selected_real_utility": "utility_high_n",
                }
            )
            merged = low_n.merge(high_n, on="seed", how="inner")
            add_row(
                "raw_tail_score_gain",
                "Raw selected object-score gain from lowest to highest N.",
                merged["score_high_n"] - merged["score_low_n"],
                threshold=0.25,
            )
            add_row(
                "raw_tail_utility_drop",
                "Raw selected real-utility drop from lowest to highest N.",
                merged["utility_low_n"] - merged["utility_high_n"],
                threshold=0.10,
            )
        add_row(
            "combined_repair_raw_gain",
            "Combined repair selected-utility gain over raw at high N.",
            _paired_gain_units(seed_df, "raw", "combined_repair"),
            threshold=0.50,
        )
        add_row(
            "observable_repair_raw_gain",
            "Observable-only repair selected-utility gain over raw at high N.",
            _paired_gain_units(seed_df, "raw", "observable_repair"),
            threshold=0.45,
        )
        add_row(
            "targeted_probe_hidden_gain",
            "Targeted probe selected-utility gain over raw for hidden-property scenes at high N.",
            _paired_gain_units(seed_df, "hidden_property", "targeted_probe"),
            threshold=0.10,
        )

    if ood_seed_df is not None and not ood_seed_df.empty:
        n_max = int(ood_seed_df["N"].max())
        corrupted = ood_seed_df[ood_seed_df["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"])]
        base = corrupted[(corrupted["selector"] == "raw") & (corrupted["N"] == n_max)][
            ["scenario", "seed", "selected_real_utility"]
        ].rename(columns={"selected_real_utility": "baseline"})
        treat = corrupted[(corrupted["selector"] == "combined_repair") & (corrupted["N"] == n_max)][
            ["scenario", "seed", "selected_real_utility"]
        ].rename(columns={"selected_real_utility": "treatment"})
        merged = treat.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "ood_combined_repair_gain",
            "Dense OOD corrupted-scene combined repair selected-utility gain over raw.",
            merged["treatment"] - merged["baseline"],
            threshold=0.60,
        )
        observable = corrupted[(corrupted["selector"] == "observable_repair") & (corrupted["N"] == n_max)][
            ["scenario", "seed", "selected_real_utility"]
        ].rename(columns={"selected_real_utility": "observable"})
        observable_merged = observable.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "ood_observable_repair_gain",
            "Dense OOD corrupted-scene observable-only repair selected-utility gain over raw.",
            observable_merged["observable"] - observable_merged["baseline"],
            threshold=0.50,
        )

    if family_seed_df is not None and not family_seed_df.empty:
        n_max = int(family_seed_df["N"].max())
        proxies = family_seed_df[
            (family_seed_df["selector"].isin(MODEL_FAMILY_PROXY_SELECTORS))
            & (family_seed_df["N"] == n_max)
        ]
        best_proxy = (
            proxies.groupby(["scenario", "seed"], as_index=False)["selected_real_utility"]
            .max()
            .rename(columns={"selected_real_utility": "best_proxy"})
        )
        combined = family_seed_df[(family_seed_df["selector"] == "combined_repair") & (family_seed_df["N"] == n_max)][
            ["scenario", "seed", "selected_real_utility"]
        ].rename(columns={"selected_real_utility": "combined"})
        merged = combined.merge(best_proxy, on=["scenario", "seed"], how="inner")
        add_row(
            "model_family_proxy_gain",
            "Combined repair selected-utility gain over the best toy model-family proxy.",
            merged["combined"] - merged["best_proxy"],
            threshold=0.20,
        )

    if counterfactual_seed_df is not None and not counterfactual_seed_df.empty:
        n_max = int(counterfactual_seed_df["N"].max())
        base = counterfactual_seed_df[
            (counterfactual_seed_df["selector"] == "raw") & (counterfactual_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "baseline"})
        combined = counterfactual_seed_df[
            (counterfactual_seed_df["selector"] == "combined_repair") & (counterfactual_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "combined"})
        combined_merged = combined.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "counterfactual_combined_repair_gain",
            "Retargeted true-object stress combined repair selected-utility gain over raw.",
            combined_merged["combined"] - combined_merged["baseline"],
            threshold=0.50,
        )
        observable = counterfactual_seed_df[
            (counterfactual_seed_df["selector"] == "observable_repair") & (counterfactual_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "observable"})
        observable_merged = observable.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "counterfactual_observable_repair_gain",
            "Retargeted true-object stress observable-only repair selected-utility gain over raw.",
            observable_merged["observable"] - observable_merged["baseline"],
            threshold=0.45,
        )

    if pilot_seed_df is not None and not pilot_seed_df.empty:
        n_max = int(pilot_seed_df["N"].max())
        base = pilot_seed_df[(pilot_seed_df["selector"] == "raw") & (pilot_seed_df["N"] == n_max)][
            ["scenario", "seed", "selected_real_utility"]
        ].rename(columns={"selected_real_utility": "baseline"})
        pilot = pilot_seed_df[
            (pilot_seed_df["selector"] == "pilot_calibrated") & (pilot_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "pilot"})
        merged = pilot.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "pilot_calibrated_repair_gain",
            "Held-out pilot-label calibrated selector selected-utility gain over raw.",
            merged["pilot"] - merged["baseline"],
            threshold=0.45,
        )

    if leave_one_failure_seed_df is not None and not leave_one_failure_seed_df.empty:
        n_max = int(leave_one_failure_seed_df["N"].max())
        base = leave_one_failure_seed_df[
            (leave_one_failure_seed_df["selector"] == "raw") & (leave_one_failure_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "baseline"})
        pilot = leave_one_failure_seed_df[
            (leave_one_failure_seed_df["selector"] == "pilot_calibrated") & (leave_one_failure_seed_df["N"] == n_max)
        ][["scenario", "seed", "selected_real_utility"]].rename(columns={"selected_real_utility": "pilot"})
        merged = pilot.merge(base, on=["scenario", "seed"], how="inner")
        add_row(
            "leave_one_failure_pilot_gain",
            "Leave-one-failure-out pilot-calibrated selected-utility gain over raw.",
            merged["pilot"] - merged["baseline"],
            threshold=0.40,
        )

    return pd.DataFrame(rows)


def exact_law_prediction_error(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(np.mean(np.abs(df["predicted_selected_utility"] - df["empirical_selected_utility"])))


def deployment_gate_from_metrics(main_metrics: pd.DataFrame) -> str:
    """Summarize high-N selected-tail risk with one allowed gate action."""

    if main_metrics.empty:
        return "collect_pilot_labels"
    high_n = main_metrics[main_metrics["N"] == main_metrics["N"].max()]
    raw = high_n[(high_n["selector"] == "raw") & (high_n["scenario"].isin(["raw", "occlusion", "hidden_property"]))]
    combined = high_n[(high_n["selector"] == "combined_repair") & (high_n["scenario"].isin(["raw", "occlusion", "hidden_property"]))]
    source = raw if not raw.empty else high_n
    repair_gain = 0.0
    if not raw.empty and not combined.empty:
        repair_gain = float(
            combined["selected_real_utility_mean"].mean() - raw["selected_real_utility_mean"].mean()
        )
    summary = {
        "N": int(high_n["N"].max()),
        "identity_error": float(source["identity_error_mean"].mean()),
        "object_real_gap": float(source["object_real_gap_mean"].mean()),
        "property_entropy": float(source["property_entropy_mean"].mean()),
        "repair_gain": repair_gain,
    }
    action = conservative_selected_tail_stop_rule(summary)
    if action not in GATE_ACTIONS:
        raise AssertionError(f"invalid gate action: {action}")
    return action
