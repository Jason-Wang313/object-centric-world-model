"""Synthetic object-centric manipulation scenes.

The scenes are deliberately small and inspectable. They expose slots, object
identities, occlusion, hidden physical properties, identity crossings, dropout,
and merge/split artifacts without claiming real-robot coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("cube", "sphere", "cylinder")


@dataclass(frozen=True)
class ObjectState:
    obj_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    color: str
    shape: str
    hidden_mass: float
    hidden_friction: float
    is_target: bool = False
    occluded: bool = False


@dataclass(frozen=True)
class SlotObservation:
    slot_id: int
    bound_obj_id: int | None
    position: tuple[float, float]
    confidence: float
    visible: bool
    dropped: bool = False
    merged_ids: tuple[int, ...] = ()
    split_from: int | None = None


@dataclass(frozen=True)
class ObjectScene:
    scene_id: int
    objects: tuple[ObjectState, ...]
    target_id: int
    goal_position: tuple[float, float]
    occlusion_band: tuple[float, float] | None = None
    crossing: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def target(self) -> ObjectState:
        return next(obj for obj in self.objects if obj.obj_id == self.target_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "target_id": self.target_id,
            "goal_position": self.goal_position,
            "occlusion_band": self.occlusion_band,
            "crossing": self.crossing,
            "notes": list(self.notes),
            "objects": [obj.__dict__ for obj in self.objects],
        }


def make_scene(
    seed: int = 0,
    n_objects: int = 4,
    occlusion: bool = False,
    hidden_property: bool = False,
    crossing: bool = False,
) -> ObjectScene:
    """Create a reproducible object manipulation scene."""

    if n_objects < 2:
        raise ValueError("n_objects must be at least 2")
    rng = np.random.default_rng(seed)
    target_id = 0
    objects: list[ObjectState] = []
    for obj_id in range(n_objects):
        if obj_id == 0:
            color, shape = "red", "cube"
            base_position = np.array([0.18, 0.28])
        elif obj_id == 1:
            color, shape = "red", "cube"
            base_position = np.array([0.25, 0.31])
        else:
            color = COLORS[obj_id % len(COLORS)]
            shape = SHAPES[obj_id % len(SHAPES)]
            base_position = rng.uniform(0.15, 0.85, size=2)

        jitter = rng.normal(0.0, 0.015, size=2)
        velocity = rng.normal(0.0, 0.04, size=2)
        if crossing and obj_id in (0, 1):
            velocity = np.array([0.10 if obj_id == 0 else -0.10, 0.0])
            base_position = np.array([0.20 if obj_id == 0 else 0.80, 0.50])

        hidden_mass_value = 0.82 if obj_id == 0 else float(rng.uniform(0.25, 0.65))
        if not hidden_property:
            hidden_mass_value = 0.55 if obj_id == 0 else float(rng.uniform(0.35, 0.62))

        objects.append(
            ObjectState(
                obj_id=obj_id,
                position=tuple(np.clip(base_position + jitter, 0.05, 0.95)),
                velocity=tuple(velocity),
                color=color,
                shape=shape,
                hidden_mass=float(hidden_mass_value),
                hidden_friction=float(rng.uniform(0.15, 0.55)),
                is_target=obj_id == target_id,
                occluded=bool(occlusion and obj_id in (0, 1)),
            )
        )

    notes: list[str] = ["visually similar target/distractor"]
    if occlusion:
        notes.append("target and distractor pass through occlusion band")
    if hidden_property:
        notes.append("target mass is hidden until probing")
    if crossing:
        notes.append("target/distractor identities cross in image space")
    return ObjectScene(
        scene_id=int(seed),
        objects=tuple(objects),
        target_id=target_id,
        goal_position=(0.82, 0.76),
        occlusion_band=(0.44, 0.56) if occlusion else None,
        crossing=crossing,
        notes=tuple(notes),
    )


def trajectory(scene: ObjectScene, steps: int = 8) -> dict[int, list[tuple[float, float]]]:
    """Generate simple linear object trajectories with optional crossing."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    out: dict[int, list[tuple[float, float]]] = {}
    for obj in scene.objects:
        positions: list[tuple[float, float]] = []
        start = np.asarray(obj.position)
        vel = np.asarray(obj.velocity)
        for t in range(steps):
            pos = np.clip(start + vel * (t / max(1, steps - 1)), 0.0, 1.0)
            positions.append((float(pos[0]), float(pos[1])))
        out[obj.obj_id] = positions
    return out


def is_occluded(scene: ObjectScene, position: tuple[float, float]) -> bool:
    """Return whether a position lies inside the scene occlusion band."""

    if scene.occlusion_band is None:
        return False
    low, high = scene.occlusion_band
    return bool(low <= position[0] <= high)


def observe_slots(
    scene: ObjectScene,
    rng: np.random.Generator | None = None,
    slot_dropout: float = 0.0,
    merge_split: float = 0.0,
    identity_swap: float = 0.0,
) -> tuple[list[SlotObservation], dict[str, int]]:
    """Emit slot observations and bookkeeping for dropout, merge/split, swaps."""

    rng = np.random.default_rng() if rng is None else rng
    slots: list[SlotObservation] = []
    bookkeeping = {"dropped": 0, "merged": 0, "split": 0, "swapped": 0}
    bound_ids = [obj.obj_id for obj in scene.objects]
    if len(bound_ids) >= 2 and rng.random() < identity_swap:
        bound_ids[0], bound_ids[1] = bound_ids[1], bound_ids[0]
        bookkeeping["swapped"] = 1

    slot_id = 0
    used_in_merge: set[int] = set()
    if len(scene.objects) >= 2 and rng.random() < merge_split:
        a, b = scene.objects[0], scene.objects[1]
        avg = tuple(((np.asarray(a.position) + np.asarray(b.position)) / 2).tolist())
        slots.append(
            SlotObservation(
                slot_id=slot_id,
                bound_obj_id=None,
                position=(float(avg[0]), float(avg[1])),
                confidence=0.42,
                visible=not (a.occluded or b.occluded),
                merged_ids=(a.obj_id, b.obj_id),
            )
        )
        slot_id += 1
        used_in_merge.update({a.obj_id, b.obj_id})
        bookkeeping["merged"] = 1
        if rng.random() < 0.5:
            slots.append(
                SlotObservation(
                    slot_id=slot_id,
                    bound_obj_id=a.obj_id,
                    position=(float(a.position[0] + 0.018), float(a.position[1] - 0.018)),
                    confidence=0.36,
                    visible=not a.occluded,
                    split_from=a.obj_id,
                )
            )
            slot_id += 1
            bookkeeping["split"] = 1

    for obj, bound_id in zip(scene.objects, bound_ids):
        if obj.obj_id in used_in_merge:
            continue
        if rng.random() < slot_dropout:
            slots.append(
                SlotObservation(
                    slot_id=slot_id,
                    bound_obj_id=obj.obj_id,
                    position=obj.position,
                    confidence=0.0,
                    visible=False,
                    dropped=True,
                )
            )
            slot_id += 1
            bookkeeping["dropped"] += 1
            continue
        noise = rng.normal(0.0, 0.012, size=2)
        pos = np.clip(np.asarray(obj.position) + noise, 0.0, 1.0)
        occluded = obj.occluded or is_occluded(scene, (float(pos[0]), float(pos[1])))
        slots.append(
            SlotObservation(
                slot_id=slot_id,
                bound_obj_id=bound_id,
                position=(float(pos[0]), float(pos[1])),
                confidence=float(0.88 if not occluded else 0.48),
                visible=not occluded,
            )
        )
        slot_id += 1
    return slots, bookkeeping


def real_action_utility(
    scene: ObjectScene,
    predicted_obj_id: int | None,
    hidden_mass_estimate: float,
    action_strength: float,
    merge_split: bool = False,
) -> float:
    """Utility for pushing the true target to the goal under hidden mass."""

    target = scene.target()
    identity_ok = predicted_obj_id == target.obj_id
    mass_mismatch = abs(hidden_mass_estimate - target.hidden_mass)
    strength_mismatch = abs(action_strength - target.hidden_mass)
    utility = 0.94
    utility -= 0.62 * (not identity_ok)
    utility -= 0.45 * min(1.0, mass_mismatch / 0.7)
    utility -= 0.25 * min(1.0, strength_mismatch / 0.7)
    utility -= 0.34 * merge_split
    if target.occluded:
        utility -= 0.10 * (not identity_ok)
    return float(np.clip(utility, 0.0, 1.0))
