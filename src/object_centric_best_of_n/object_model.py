"""Object-centric future generator used by the controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .envs import ObjectScene, observe_slots, real_action_utility


@dataclass(frozen=True)
class PredictedSlot:
    slot_id: int
    predicted_obj_id: int | None
    position: tuple[float, float]
    confidence: float
    trajectory: tuple[tuple[float, float], ...]
    hidden_mass_estimate: float
    merged_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    scenario: str
    predicted_target_id: int | None
    target_id: int
    score: float
    real_utility: float
    identity_error: float
    swap: float
    merge_split: float
    occlusion_error: float
    property_error: float
    hidden_property_true: float
    hidden_property_estimate: float
    property_entropy: float
    object_real_gap: float
    slots: tuple[PredictedSlot, ...]
    diagnostics: dict[str, float | int | str]

    def with_score(self, score: float, label: str) -> "Candidate":
        diagnostics = dict(self.diagnostics)
        diagnostics["selector_score_label"] = label
        diagnostics["selector_score"] = float(score)
        return replace(self, score=float(score), diagnostics=diagnostics)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scenario": self.scenario,
            "predicted_target_id": self.predicted_target_id,
            "target_id": self.target_id,
            "object_score": self.score,
            "real_utility": self.real_utility,
            "identity_error": self.identity_error,
            "swap_rate": self.swap,
            "merge_split_rate": self.merge_split,
            "occlusion_error": self.occlusion_error,
            "property_error": self.property_error,
            "property_entropy": self.property_entropy,
            "object_real_gap": self.object_real_gap,
            "slot_count": len(self.slots),
            "selector_score_label": self.diagnostics.get("selector_score_label", "raw"),
        }


SCENARIO_PARAMS: dict[str, dict[str, float]] = {
    "good": {
        "identity_base": 0.02,
        "identity_tail": 0.04,
        "merge_base": 0.01,
        "merge_tail": 0.03,
        "property_base": 0.04,
        "property_tail": 0.05,
        "artifact_bonus": 0.05,
    },
    "swap": {
        "identity_base": 0.10,
        "identity_tail": 0.56,
        "merge_base": 0.02,
        "merge_tail": 0.08,
        "property_base": 0.08,
        "property_tail": 0.14,
        "artifact_bonus": 0.58,
    },
    "merge_split": {
        "identity_base": 0.05,
        "identity_tail": 0.18,
        "merge_base": 0.10,
        "merge_tail": 0.55,
        "property_base": 0.08,
        "property_tail": 0.15,
        "artifact_bonus": 0.50,
    },
    "occlusion": {
        "identity_base": 0.12,
        "identity_tail": 0.48,
        "merge_base": 0.04,
        "merge_tail": 0.15,
        "property_base": 0.10,
        "property_tail": 0.18,
        "artifact_bonus": 0.52,
    },
    "hidden_property": {
        "identity_base": 0.05,
        "identity_tail": 0.13,
        "merge_base": 0.03,
        "merge_tail": 0.09,
        "property_base": 0.16,
        "property_tail": 0.58,
        "artifact_bonus": 0.48,
    },
    "raw": {
        "identity_base": 0.09,
        "identity_tail": 0.42,
        "merge_base": 0.06,
        "merge_tail": 0.32,
        "property_base": 0.15,
        "property_tail": 0.42,
        "artifact_bonus": 0.56,
    },
}


class ObjectCentricFutureGenerator:
    """Generate object slots, identities, properties, trajectories, and scores."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate_candidates(
        self,
        scene: ObjectScene,
        n: int,
        scenario: str = "raw",
        seed: int | None = None,
    ) -> list[Candidate]:
        if n < 1:
            raise ValueError("n must be at least 1")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        return [self.generate_candidate(scene, i, scenario, rng) for i in range(n)]

    def generate_candidate(
        self,
        scene: ObjectScene,
        candidate_id: int,
        scenario: str,
        rng: np.random.Generator,
    ) -> Candidate:
        params = SCENARIO_PARAMS.get(scenario, SCENARIO_PARAMS["raw"])
        ambition = float(rng.beta(2.4, 1.8))
        tail_pressure = ambition**1.7

        identity_prob = params["identity_base"] + params["identity_tail"] * tail_pressure
        merge_prob = params["merge_base"] + params["merge_tail"] * tail_pressure
        property_prob = params["property_base"] + params["property_tail"] * tail_pressure

        if scene.occlusion_band is not None:
            identity_prob += 0.10
        if scene.crossing:
            identity_prob += 0.08

        identity_error = float(rng.random() < min(0.92, identity_prob))
        merge_split = float(rng.random() < min(0.88, merge_prob))
        property_error = float(rng.random() < min(0.90, property_prob))
        swap = float(identity_error and rng.random() < 0.80)
        target_id = scene.target_id
        predicted_target_id = 1 if identity_error else target_id
        if merge_split and rng.random() < 0.20:
            predicted_target_id = None

        target = scene.target()
        hidden_true = float(target.hidden_mass)
        if property_error:
            hidden_estimate = float(np.clip(hidden_true + rng.normal(-0.34, 0.13), 0.05, 1.0))
        else:
            hidden_estimate = float(np.clip(hidden_true + rng.normal(0.0, 0.06), 0.05, 1.0))
        property_entropy = float(np.clip(0.18 + 0.55 * property_error + rng.normal(0, 0.04), 0.02, 0.95))
        action_strength = float(np.clip(0.20 + 0.72 * ambition, 0.05, 1.0))
        real_utility = real_action_utility(
            scene,
            predicted_target_id,
            hidden_estimate,
            action_strength,
            merge_split=bool(merge_split),
        )

        artifact_load = 0.52 * identity_error + 0.40 * merge_split + 0.32 * property_error
        score = (
            0.12
            + 0.55 * ambition
            + params["artifact_bonus"] * artifact_load
            + 0.10 * (1.0 - property_entropy)
            + rng.normal(0.0, 0.045)
        )
        score = float(np.clip(score, 0.0, 1.35))
        object_real_gap = float(score - real_utility)

        slots, bookkeeping = observe_slots(
            scene,
            rng,
            slot_dropout=0.04 + 0.10 * tail_pressure,
            merge_split=0.05 + 0.50 * merge_split,
            identity_swap=0.05 + 0.68 * swap,
        )
        pred_slots: list[PredictedSlot] = []
        for slot in slots:
            traj = []
            base = np.asarray(slot.position)
            for step in range(5):
                drift = np.array([0.035 * ambition * step, 0.025 * ambition * step])
                if slot.bound_obj_id != target_id and slot.bound_obj_id is not None:
                    drift *= 0.55
                pos = np.clip(base + drift + rng.normal(0.0, 0.006, size=2), 0.0, 1.0)
                traj.append((float(pos[0]), float(pos[1])))
            pred_slots.append(
                PredictedSlot(
                    slot_id=slot.slot_id,
                    predicted_obj_id=slot.bound_obj_id,
                    position=slot.position,
                    confidence=slot.confidence,
                    trajectory=tuple(traj),
                    hidden_mass_estimate=hidden_estimate if slot.bound_obj_id == target_id else float(rng.uniform(0.2, 0.8)),
                    merged_ids=slot.merged_ids,
                )
            )

        occlusion_error = float(scene.occlusion_band is not None and identity_error)
        diagnostics: dict[str, float | int | str] = {
            "ambition": ambition,
            "tail_pressure": tail_pressure,
            "identity_instability": float(np.clip(0.18 + 0.70 * identity_error + 0.16 * merge_split, 0, 1)),
            "slot_support": float(np.clip(1.0 - 0.20 * bookkeeping["dropped"] - 0.35 * merge_split, 0, 1)),
            "merge_evidence": float(np.clip(0.12 + 0.80 * merge_split, 0, 1)),
            "property_surprise": float(np.clip(property_entropy + 0.20 * property_error, 0, 1)),
            "bookkeeping_dropped": int(bookkeeping["dropped"]),
            "bookkeeping_merged": int(bookkeeping["merged"]),
            "bookkeeping_split": int(bookkeeping["split"]),
            "bookkeeping_swapped": int(bookkeeping["swapped"]),
        }
        return Candidate(
            candidate_id=int(candidate_id),
            scenario=scenario,
            predicted_target_id=predicted_target_id,
            target_id=target_id,
            score=score,
            real_utility=real_utility,
            identity_error=identity_error,
            swap=swap,
            merge_split=merge_split,
            occlusion_error=occlusion_error,
            property_error=property_error,
            hidden_property_true=hidden_true,
            hidden_property_estimate=hidden_estimate,
            property_entropy=property_entropy,
            object_real_gap=object_real_gap,
            slots=tuple(pred_slots),
            diagnostics=diagnostics,
        )


def candidates_to_records(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [candidate.to_record() for candidate in candidates]
