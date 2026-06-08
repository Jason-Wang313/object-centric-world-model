"""Metrics and aggregation helpers for object-centric Best-of-N experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .object_model import Candidate
from .repair import GATE_ACTIONS, conservative_selected_tail_stop_rule


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
