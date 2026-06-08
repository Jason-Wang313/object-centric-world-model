"""Metrics and aggregation helpers for object-centric Best-of-N experiments."""

from __future__ import annotations

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
