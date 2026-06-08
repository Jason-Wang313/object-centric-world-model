"""Best-of-N selectors for object-centric candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from .envs import ObjectScene
from .object_model import Candidate
from .repair import (
    combined_repair_score,
    identity_repaired_score,
    property_calibrated_score,
    targeted_diagnostic_probe,
)


def _select_max(
    candidates: list[Candidate],
    key: Callable[[Candidate], float],
    seed: int = 0,
    label: str = "raw",
) -> Candidate:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    scores = np.asarray([key(candidate) for candidate in candidates], dtype=float)
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score))
    rng = np.random.default_rng(seed)
    chosen = candidates[int(rng.choice(tied))]
    diagnostics = dict(chosen.diagnostics)
    diagnostics["selector_score_label"] = label
    diagnostics["selector_score"] = max_score
    return replace(chosen, score=max_score, diagnostics=diagnostics)


def select_raw(candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0) -> Candidate:
    return _select_max(candidates, lambda candidate: candidate.score, seed=seed, label="raw")


def select_random(candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0) -> Candidate:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    rng = np.random.default_rng(seed)
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    diagnostics = dict(chosen.diagnostics)
    diagnostics["selector_score_label"] = "random"
    diagnostics["selector_score"] = chosen.score
    return replace(chosen, diagnostics=diagnostics)


def select_oracle(candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0) -> Candidate:
    return _select_max(candidates, lambda candidate: candidate.real_utility, seed=seed, label="oracle")


def select_identity_consistent(
    candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0
) -> Candidate:
    return _select_max(candidates, identity_repaired_score, seed=seed, label="identity_consistent")


def select_property_calibrated(
    candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0
) -> Candidate:
    return _select_max(candidates, property_calibrated_score, seed=seed, label="property_calibrated")


def select_targeted_probe(
    candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0
) -> Candidate:
    if scene is None:
        raise ValueError("scene is required for targeted probing")
    probed = [targeted_diagnostic_probe(candidate, scene, seed=seed, action="push") for candidate in candidates]
    return _select_max(probed, lambda candidate: candidate.score, seed=seed, label="targeted_probe")


def select_combined_repair(
    candidates: list[Candidate], scene: ObjectScene | None = None, seed: int = 0
) -> Candidate:
    if scene is None:
        raise ValueError("scene is required for combined repair")
    return _select_max(
        candidates,
        lambda candidate: combined_repair_score(candidate, scene, seed=seed),
        seed=seed,
        label="combined_repair",
    )


SELECTORS: dict[str, Callable[[list[Candidate], ObjectScene | None, int], Candidate]] = {
    "raw": select_raw,
    "identity_consistent": select_identity_consistent,
    "property_calibrated": select_property_calibrated,
    "targeted_probe": select_targeted_probe,
    "combined_repair": select_combined_repair,
    "random": select_random,
    "oracle": select_oracle,
}
