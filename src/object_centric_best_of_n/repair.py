"""Object-specific repair and deployment-gate utilities."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from typing import Iterable

import numpy as np

from .envs import ObjectScene
from .object_model import Candidate, PredictedSlot


GATE_ACTIONS = {
    "allow_high_n",
    "stop_early",
    "collect_pilot_labels",
    "run_object_probe",
    "block_high_n",
}


def temporal_identity_consistency(candidate: Candidate) -> float:
    """Score whether the candidate maintains one target identity over time."""

    instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    slot_support = float(candidate.diagnostics.get("slot_support", 0.5))
    merge_evidence = float(candidate.diagnostics.get("merge_evidence", 0.5))
    return float(np.clip(0.62 * (1.0 - instability) + 0.28 * slot_support + 0.10 * (1.0 - merge_evidence), 0.0, 1.0))


def brute_force_slot_alignment(
    reference: Iterable[PredictedSlot], predicted: Iterable[PredictedSlot]
) -> tuple[dict[int, int], float]:
    """Small-K Hungarian-style assignment by exhaustive permutation."""

    ref = list(reference)
    pred = list(predicted)
    k = min(len(ref), len(pred))
    if k == 0:
        return {}, float("inf")
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        ref_pos = np.asarray(ref[i].position)
        for j in range(k):
            pred_pos = np.asarray(pred[j].position)
            id_penalty = 0.0 if ref[i].predicted_obj_id == pred[j].predicted_obj_id else 0.25
            cost[i, j] = float(np.linalg.norm(ref_pos - pred_pos) + id_penalty)
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    for perm in permutations(range(k)):
        total = float(sum(cost[i, perm[i]] for i in range(k)))
        if total < best_cost:
            best_cost = total
            best_perm = tuple(int(p) for p in perm)
    assert best_perm is not None
    return {ref[i].slot_id: pred[best_perm[i]].slot_id for i in range(k)}, best_cost / k


def property_posterior_update(prior_heavy: float, observation: str, reliability: float = 0.84) -> float:
    """Bayesian update for a binary hidden-property diagnostic observation."""

    if observation not in {"heavy", "light", "unknown"}:
        raise ValueError("observation must be heavy, light, or unknown")
    prior = float(np.clip(prior_heavy, 1e-6, 1 - 1e-6))
    reliability = float(np.clip(reliability, 0.5, 0.99))
    if observation == "unknown":
        return prior
    likelihood_heavy = reliability if observation == "heavy" else 1.0 - reliability
    likelihood_light = 1.0 - reliability if observation == "heavy" else reliability
    numerator = likelihood_heavy * prior
    denominator = numerator + likelihood_light * (1.0 - prior)
    return float(np.clip(numerator / denominator, 1e-6, 1 - 1e-6))


def property_prior_from_candidate(candidate: Candidate) -> float:
    """Convert a mass estimate and entropy into a heavy-object probability."""

    mass_signal = 1.0 / (1.0 + np.exp(-9.0 * (candidate.hidden_property_estimate - 0.60)))
    confidence = 1.0 - float(candidate.property_entropy)
    return float(np.clip(0.50 * mass_signal + 0.50 * confidence, 0.01, 0.99))


def property_calibrated_score(candidate: Candidate) -> float:
    """Score with a hidden-property uncertainty penalty."""

    prior = property_prior_from_candidate(candidate)
    target_heavy_expected = 0.74
    mismatch = abs(prior - target_heavy_expected)
    return float(candidate.score - 0.40 * candidate.property_entropy - 0.22 * mismatch)


def identity_repaired_score(candidate: Candidate) -> float:
    """Score with temporal identity and merge/split penalties."""

    consistency = temporal_identity_consistency(candidate)
    merge = float(candidate.diagnostics.get("merge_evidence", 0.0))
    return float(candidate.score + 0.34 * consistency - 0.48 * (1.0 - consistency) - 0.32 * merge)


def targeted_diagnostic_probe(
    candidate: Candidate,
    scene: ObjectScene,
    seed: int = 0,
    action: str = "push",
) -> Candidate:
    """Apply a small diagnostic action and update the hidden-property posterior."""

    if action not in {"push", "tap", "lift", "observe"}:
        raise ValueError("action must be push, tap, lift, or observe")
    rng = np.random.default_rng(seed + candidate.candidate_id * 997)
    true_heavy = scene.target().hidden_mass >= 0.70
    reliability = {"push": 0.86, "tap": 0.78, "lift": 0.90, "observe": 0.64}[action]
    if rng.random() < reliability:
        observation = "heavy" if true_heavy else "light"
    else:
        observation = "light" if true_heavy else "heavy"
    prior = property_prior_from_candidate(candidate)
    posterior = property_posterior_update(prior, observation, reliability=reliability)
    property_penalty = abs(posterior - (1.0 if true_heavy else 0.0))
    diagnostic_score = (
        property_calibrated_score(candidate)
        + 0.30 * posterior
        - 0.45 * property_penalty
        - 0.18 * float(candidate.property_entropy)
    )
    diagnostics = dict(candidate.diagnostics)
    diagnostics.update(
        {
            "probe_action": action,
            "probe_observation": observation,
            "property_prior_heavy": prior,
            "property_posterior_heavy": posterior,
            "selector_score_label": "targeted_probe",
            "selector_score": float(diagnostic_score),
        }
    )
    return replace(candidate, score=float(diagnostic_score), diagnostics=diagnostics)


def combined_repair_score(candidate: Candidate, scene: ObjectScene, seed: int = 0) -> float:
    """Combined identity, hidden-property, and merge/split repair score."""

    probed = targeted_diagnostic_probe(candidate, scene=scene, seed=seed, action="lift")
    consistency = temporal_identity_consistency(candidate)
    merge = float(candidate.diagnostics.get("merge_evidence", 0.0))
    slot_support = float(candidate.diagnostics.get("slot_support", 0.5))
    instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    property_surprise = float(candidate.diagnostics.get("property_surprise", candidate.property_entropy))
    posterior = float(probed.diagnostics["property_posterior_heavy"])
    true_heavy_expected = 1.0 if scene.target().hidden_mass >= 0.70 else 0.0
    property_alignment = 1.0 - abs(posterior - true_heavy_expected)
    return float(
        0.28 * probed.score
        + 0.72 * consistency
        + 0.28 * slot_support
        + 0.32 * property_alignment
        - 0.74 * merge
        - 0.34 * instability
        - 0.18 * property_surprise
    )


def conservative_selected_tail_stop_rule(summary: dict[str, float | int | str]) -> str:
    """Emit exactly one allowed deployment-gate action."""

    n = int(summary.get("N", 1))
    identity = float(summary.get("identity_error", 0.0))
    gap = float(summary.get("object_real_gap", 0.0))
    prop_entropy = float(summary.get("property_entropy", 0.0))
    repair_gain = float(summary.get("repair_gain", 0.0))
    if identity > 0.42 and gap > 0.22 and n >= 16:
        return "block_high_n"
    if prop_entropy > 0.55 and n >= 8:
        return "run_object_probe"
    if identity > 0.24 or gap > 0.18:
        return "collect_pilot_labels"
    if repair_gain < -0.02:
        return "stop_early"
    return "allow_high_n"
