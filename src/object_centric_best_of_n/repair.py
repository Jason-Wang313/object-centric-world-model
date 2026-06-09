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

PILOT_CALIBRATION_FEATURES = (
    "raw_object_score",
    "identity_consistency",
    "slot_support",
    "target_slot_confidence",
    "predicted_target_matches_query",
    "one_minus_identity_instability",
    "one_minus_merge_evidence",
    "one_minus_property_entropy",
    "one_minus_property_surprise",
    "hidden_mass_estimate",
    "property_prior_heavy",
    "slot_count_scaled",
)


def temporal_identity_consistency(candidate: Candidate) -> float:
    """Score whether the candidate maintains one target identity over time."""

    instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    slot_support = float(candidate.diagnostics.get("slot_support", 0.5))
    merge_evidence = float(candidate.diagnostics.get("merge_evidence", 0.5))
    return float(np.clip(0.62 * (1.0 - instability) + 0.28 * slot_support + 0.10 * (1.0 - merge_evidence), 0.0, 1.0))


def pilot_calibration_features(candidate: Candidate) -> np.ndarray:
    """Observable candidate features used by the pilot-label calibrator."""

    target_confidences = [
        float(slot.confidence)
        for slot in candidate.slots
        if slot.predicted_obj_id == candidate.target_id and not slot.merged_ids
    ]
    target_slot_confidence = max(target_confidences) if target_confidences else 0.0
    identity_instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    merge_evidence = float(candidate.diagnostics.get("merge_evidence", 0.5))
    property_surprise = float(candidate.diagnostics.get("property_surprise", candidate.property_entropy))
    slot_support = float(candidate.diagnostics.get("slot_support", 0.5))
    return np.asarray(
        [
            float(candidate.score),
            temporal_identity_consistency(candidate),
            slot_support,
            target_slot_confidence,
            float(candidate.predicted_target_id == candidate.target_id),
            1.0 - identity_instability,
            1.0 - merge_evidence,
            1.0 - float(candidate.property_entropy),
            1.0 - property_surprise,
            float(candidate.hidden_property_estimate),
            property_prior_from_candidate(candidate),
            min(1.0, len(candidate.slots) / 8.0),
        ],
        dtype=float,
    )


def fit_pilot_calibrator(
    candidates: Iterable[Candidate],
    ridge: float = 1e-3,
) -> dict[str, object]:
    """Fit a tiny ridge utility calibrator from pilot-labeled candidates."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("pilot candidates must be non-empty")
    x = np.vstack([pilot_calibration_features(candidate) for candidate in candidate_list])
    y = np.asarray([candidate.real_utility for candidate in candidate_list], dtype=float)
    feature_mean = np.mean(x, axis=0)
    feature_scale = np.std(x, axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    z = (x - feature_mean) / feature_scale
    design = np.column_stack([np.ones(z.shape[0]), z])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    train_pred = np.clip(design @ weights, 0.0, 1.0)
    train_corr = 0.0
    if np.std(train_pred) > 1e-12 and np.std(y) > 1e-12:
        train_corr = float(np.corrcoef(train_pred, y)[0, 1])
    return {
        "feature_names": list(PILOT_CALIBRATION_FEATURES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "weights": weights.tolist(),
        "ridge": float(ridge),
        "n_train_candidates": int(len(candidate_list)),
        "train_mae": float(np.mean(np.abs(train_pred - y))),
        "train_correlation": train_corr,
    }


def empty_pilot_calibrator() -> dict[str, object]:
    """Return a deterministic no-label baseline calibrator for budget-zero rows."""

    return {
        "feature_names": list(PILOT_CALIBRATION_FEATURES),
        "feature_mean": [0.0] * len(PILOT_CALIBRATION_FEATURES),
        "feature_scale": [1.0] * len(PILOT_CALIBRATION_FEATURES),
        "weights": [0.5] + [0.0] * len(PILOT_CALIBRATION_FEATURES),
        "ridge": 0.0,
        "n_train_candidates": 0,
        "train_mae": 0.0,
        "train_correlation": 0.0,
        "no_label_baseline": True,
    }


def pilot_calibrated_score(candidate: Candidate, calibrator: dict[str, object]) -> float:
    """Predict real utility from pilot-calibrated observable features."""

    weights = np.asarray(calibrator["weights"], dtype=float)
    feature_mean = np.asarray(calibrator["feature_mean"], dtype=float)
    feature_scale = np.asarray(calibrator["feature_scale"], dtype=float)
    x = pilot_calibration_features(candidate)
    z = (x - feature_mean) / feature_scale
    score = float(weights[0] + np.dot(weights[1:], z))
    return float(np.clip(score, 0.0, 1.0))


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
    consistency = temporal_identity_consistency(candidate)
    merge = float(candidate.diagnostics.get("merge_evidence", 0.0))
    instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    diagnostic_score = (
        property_calibrated_score(candidate)
        + 0.30 * posterior
        + 0.30 * consistency
        - 0.45 * property_penalty
        - 0.18 * float(candidate.property_entropy)
        - 0.25 * merge
        - 0.20 * instability
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


def observable_repair_score(candidate: Candidate, scene: ObjectScene, seed: int = 0) -> float:
    """Repair score using observable slot diagnostics and probe posterior only.

    The scene is used to simulate the diagnostic probe observation, but this
    score does not compare the posterior with the scene's true hidden property.
    """

    probed = targeted_diagnostic_probe(candidate, scene=scene, seed=seed, action="lift")
    consistency = temporal_identity_consistency(candidate)
    merge = float(candidate.diagnostics.get("merge_evidence", 0.0))
    slot_support = float(candidate.diagnostics.get("slot_support", 0.5))
    instability = float(candidate.diagnostics.get("identity_instability", 0.5))
    property_surprise = float(candidate.diagnostics.get("property_surprise", candidate.property_entropy))
    posterior = float(probed.diagnostics["property_posterior_heavy"])
    posterior_confidence = max(posterior, 1.0 - posterior)
    return float(
        0.22 * candidate.score
        + 0.76 * consistency
        + 0.32 * slot_support
        + 0.18 * posterior_confidence
        + 0.12 * (1.0 - candidate.property_entropy)
        - 0.78 * merge
        - 0.38 * instability
        - 0.20 * property_surprise
    )


def conservative_selected_tail_stop_rule(summary: dict[str, float | int | str]) -> str:
    """Emit exactly one allowed deployment-gate action."""

    if bool(summary.get("hidden_mode_unidentifiable", False)) or str(summary.get("gate_reason", "")) in {
        "hidden_mode_unidentifiable",
        "tail_rank_failure",
    }:
        return "block_high_n"
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


def observable_feature_signature(candidate: Candidate) -> tuple[float, ...]:
    """Rounded deployable signature used to detect indistinguishable candidates."""

    return tuple(float(np.round(value, 6)) for value in pilot_calibration_features(candidate))


def hidden_mode_unidentifiable_gate(candidates: Iterable[Candidate], n: int) -> dict[str, float | int | str]:
    """Block high-N when observable candidate signatures collapse to one mode."""

    candidate_list = list(candidates)
    if not candidate_list:
        return {"gate_action": "block_high_n", "gate_reason": "empty_candidate_set", "observable_feature_collision_rate": 1.0}
    signatures = [observable_feature_signature(candidate) for candidate in candidate_list]
    unique = len(set(signatures))
    collision_rate = 1.0 - unique / max(1, len(signatures))
    if int(n) >= 16 and unique <= 1:
        return {
            "gate_action": "block_high_n",
            "gate_reason": "hidden_mode_unidentifiable",
            "observable_feature_collision_rate": float(collision_rate),
        }
    if int(n) >= 16 and collision_rate >= 0.75:
        return {
            "gate_action": "block_high_n",
            "gate_reason": "tail_rank_failure",
            "observable_feature_collision_rate": float(collision_rate),
        }
    return {
        "gate_action": "allow_high_n",
        "gate_reason": "observable_modes_separable",
        "observable_feature_collision_rate": float(collision_rate),
    }
