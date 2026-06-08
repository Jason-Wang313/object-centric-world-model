"""Finite tie-aware Best-of-N selection laws.

The law here is intentionally generic: a finite candidate population is sampled
with replacement, the candidate with maximal score is selected, and exact ties
are broken uniformly among tied sample positions. Object-centric experiments
plug in object-level scores and real utilities, but the proof object is this
small finite selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """One score-equivalence class in descending score order."""

    score: float
    indices: tuple[int, ...]
    probability: float
    mean_utility: float


def _as_1d_float(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def score_tie_groups(scores: Iterable[float]) -> list[tuple[float, np.ndarray]]:
    """Return score tie groups sorted from highest score to lowest score."""

    score_arr = _as_1d_float(scores, "scores")
    unique_scores = np.unique(score_arr)[::-1]
    return [(float(score), np.flatnonzero(score_arr == score)) for score in unique_scores]


def selected_tie_groups(
    utilities: Iterable[float], scores: Iterable[float] | None = None, n: int = 1
) -> list[TieGroup]:
    """Exact selected-group probabilities for N i.i.d. finite draws.

    For score group g, the event that g is selected is "all samples have score
    at most score(g), and at least one sample has score exactly score(g)".
    This gives ((|g| + below) / m)^N - (below / m)^N. Conditional on that event,
    uniform tie-breaking over tied sample positions makes the expected utility
    the arithmetic mean of utilities in the group.
    """

    if n < 1:
        raise ValueError("n must be at least 1")
    utility_arr = _as_1d_float(utilities, "utilities")
    score_arr = utility_arr if scores is None else _as_1d_float(scores, "scores")
    if score_arr.shape != utility_arr.shape:
        raise ValueError("utilities and scores must have the same length")

    m = utility_arr.size
    groups: list[TieGroup] = []
    above = 0
    for score, idx in score_tie_groups(score_arr):
        group_size = int(idx.size)
        below = m - above - group_size
        probability = ((group_size + below) / m) ** n - (below / m) ** n
        groups.append(
            TieGroup(
                score=score,
                indices=tuple(int(i) for i in idx),
                probability=float(probability),
                mean_utility=float(np.mean(utility_arr[idx])),
            )
        )
        above += group_size
    return groups


def exact_best_of_n_expected_utility(
    utilities: Iterable[float], scores: Iterable[float] | None = None, n: int = 1
) -> float:
    """Return the exact expected real utility selected by Best-of-N."""

    groups = selected_tie_groups(utilities, scores=scores, n=n)
    return float(sum(group.probability * group.mean_utility for group in groups))


def exact_binary_expected_success(
    successes: Iterable[float], scores: Iterable[float] | None = None, n: int = 1
) -> float:
    """Exact selected success probability for binary utilities."""

    success_arr = _as_1d_float(successes, "successes")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("successes must be binary 0/1 values")
    return exact_best_of_n_expected_utility(success_arr, scores=scores, n=n)


def expected_curve(
    utilities: Iterable[float],
    scores: Iterable[float] | None,
    ns: Iterable[int],
) -> dict[int, float]:
    """Compute exact selected utility for several N values."""

    return {int(n): exact_best_of_n_expected_utility(utilities, scores, int(n)) for n in ns}


def monte_carlo_best_of_n(
    utilities: Iterable[float],
    scores: Iterable[float] | None = None,
    n: int = 1,
    trials: int = 10_000,
    seed: int = 0,
) -> float:
    """Monte Carlo estimator matching the tie-aware finite law."""

    if trials < 1:
        raise ValueError("trials must be positive")
    utility_arr = _as_1d_float(utilities, "utilities")
    score_arr = utility_arr if scores is None else _as_1d_float(scores, "scores")
    if score_arr.shape != utility_arr.shape:
        raise ValueError("utilities and scores must have the same length")
    if n < 1:
        raise ValueError("n must be at least 1")

    rng = np.random.default_rng(seed)
    m = utility_arr.size
    samples = rng.integers(0, m, size=(trials, n))
    sampled_scores = score_arr[samples]
    max_scores = np.max(sampled_scores, axis=1, keepdims=True)
    tie_mask = sampled_scores == max_scores
    tie_random = rng.random(size=(trials, n))
    tie_random[~tie_mask] = -1.0
    chosen_positions = np.argmax(tie_random, axis=1)
    chosen_indices = samples[np.arange(trials), chosen_positions]
    return float(np.mean(utility_arr[chosen_indices]))


def oracle_score_check(utilities: Iterable[float], n: int = 8) -> dict[str, float | bool]:
    """Check that using true utility as score improves over random selection."""

    utility_arr = _as_1d_float(utilities, "utilities")
    random_mean = float(np.mean(utility_arr))
    oracle_mean = exact_best_of_n_expected_utility(utility_arr, utility_arr, n=n)
    return {
        "random_mean": random_mean,
        "oracle_mean": oracle_mean,
        "passes": bool(oracle_mean >= random_mean - 1e-12),
    }


def anti_aligned_score_check(utilities: Iterable[float], n: int = 8) -> dict[str, float | bool]:
    """Check the failure case where the score is anti-aligned with utility."""

    utility_arr = _as_1d_float(utilities, "utilities")
    random_mean = float(np.mean(utility_arr))
    anti_mean = exact_best_of_n_expected_utility(utility_arr, -utility_arr, n=n)
    return {
        "random_mean": random_mean,
        "anti_aligned_mean": anti_mean,
        "passes": bool(anti_mean <= random_mean + 1e-12),
    }


def law_validation_row(
    utilities: Iterable[float],
    scores: Iterable[float],
    n: int,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    """Return one empirical-vs-exact validation row."""

    predicted = exact_best_of_n_expected_utility(utilities, scores, n)
    empirical = monte_carlo_best_of_n(utilities, scores, n=n, trials=trials, seed=seed)
    return {
        "N": int(n),
        "predicted_selected_utility": float(predicted),
        "empirical_selected_utility": float(empirical),
        "absolute_error": float(abs(predicted - empirical)),
        "trials": int(trials),
        "seed": int(seed),
    }
