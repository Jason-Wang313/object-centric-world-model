"""A small NumPy semi-learned object-centric model.

This is not a benchmark model. It is a CPU-only learned artifact that trains
slot-level linear predictors on generated object trajectories, so claims about
learned evidence remain limited to this controlled synthetic setting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .envs import make_scene


@dataclass
class LearnedMetrics:
    property_accuracy: float
    random_property_accuracy: float
    identity_alignment_accuracy: float
    random_identity_alignment_accuracy: float
    transition_mse: float
    constant_transition_mse: float
    reward_correlation: float
    n_train_slots: int
    n_test_slots: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "property_accuracy": float(self.property_accuracy),
            "random_property_accuracy": float(self.random_property_accuracy),
            "identity_alignment_accuracy": float(self.identity_alignment_accuracy),
            "random_identity_alignment_accuracy": float(self.random_identity_alignment_accuracy),
            "transition_mse": float(self.transition_mse),
            "constant_transition_mse": float(self.constant_transition_mse),
            "reward_correlation": float(self.reward_correlation),
            "n_train_slots": int(self.n_train_slots),
            "n_test_slots": int(self.n_test_slots),
        }


class NumpyObjectCentricModel:
    """Linear slot encoder, transition predictor, property posterior, reward scorer."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = float(ridge)
        self.transition_w: np.ndarray | None = None
        self.property_w: np.ndarray | None = None
        self.reward_w: np.ndarray | None = None
        self.identity_w: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y_transition: np.ndarray,
        y_property: np.ndarray,
        y_reward: np.ndarray,
        identity_x: np.ndarray | None = None,
        identity_y: np.ndarray | None = None,
    ) -> "NumpyObjectCentricModel":
        design = _add_bias(x)
        reg = self.ridge * np.eye(design.shape[1])
        self.transition_w = np.linalg.solve(design.T @ design + reg, design.T @ y_transition)
        self.property_w = np.linalg.solve(design.T @ design + reg, design.T @ y_property)
        self.reward_w = np.linalg.solve(design.T @ design + reg, design.T @ y_reward)
        if identity_x is not None and identity_y is not None:
            identity_design = _add_bias(identity_x)
            identity_reg = self.ridge * np.eye(identity_design.shape[1])
            self.identity_w = np.linalg.solve(
                identity_design.T @ identity_design + identity_reg,
                identity_design.T @ identity_y,
            )
        return self

    def predict_transition(self, x: np.ndarray) -> np.ndarray:
        if self.transition_w is None:
            raise RuntimeError("model is not fit")
        return _add_bias(x) @ self.transition_w

    def predict_property_proba(self, x: np.ndarray) -> np.ndarray:
        if self.property_w is None:
            raise RuntimeError("model is not fit")
        logits = _add_bias(x) @ self.property_w
        return np.clip(logits, 0.0, 1.0)

    def predict_reward(self, x: np.ndarray) -> np.ndarray:
        if self.reward_w is None:
            raise RuntimeError("model is not fit")
        return np.clip(_add_bias(x) @ self.reward_w, 0.0, 1.0)

    def predict_identity_alignment(self, identity_x: np.ndarray) -> np.ndarray:
        if self.identity_w is None:
            raise RuntimeError("identity alignment model is not fit")
        return np.clip(_add_bias(identity_x) @ self.identity_w, 0.0, 1.0)

    def encode_slots(self, seed: int = 0, n_scenes: int = 4) -> np.ndarray:
        rows = []
        for offset in range(n_scenes):
            scene = make_scene(seed + offset, hidden_property=True, crossing=bool(offset % 2))
            for obj in scene.objects:
                rows.append(_object_features(obj.position, obj.velocity, obj.hidden_mass, obj.color, obj.shape, obj.occluded))
        return np.asarray(rows, dtype=float)


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(x.shape[0]), x]


def _object_features(
    position: tuple[float, float],
    velocity: tuple[float, float],
    hidden_mass: float,
    color: str,
    shape: str,
    occluded: bool,
) -> list[float]:
    color_red = 1.0 if color == "red" else 0.0
    shape_cube = 1.0 if shape == "cube" else 0.0
    noisy_mass_sensor = hidden_mass + (0.10 if not occluded else -0.04)
    return [
        float(position[0]),
        float(position[1]),
        float(velocity[0]),
        float(velocity[1]),
        color_red,
        shape_cube,
        float(occluded),
        float(noisy_mass_sensor),
    ]


def make_synthetic_trajectory_dataset(
    n_scenes: int = 240,
    seed: int = 0,
    n_objects: int | tuple[int, int] = 4,
    occlusion_mode: str = "periodic",
    crossing_mode: str = "periodic",
    action_scale: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[list[float]] = []
    transition_targets: list[list[float]] = []
    property_targets: list[float] = []
    reward_targets: list[float] = []
    for scene_idx in range(n_scenes):
        if isinstance(n_objects, tuple):
            low, high = n_objects
            scene_n_objects = int(rng.integers(low, high + 1))
        else:
            scene_n_objects = int(n_objects)
        if occlusion_mode == "always":
            occlusion = True
        elif occlusion_mode == "never":
            occlusion = False
        elif occlusion_mode == "random":
            occlusion = bool(rng.random() < 0.70)
        else:
            occlusion = bool(scene_idx % 3 == 0)
        if crossing_mode == "always":
            crossing = True
        elif crossing_mode == "never":
            crossing = False
        elif crossing_mode == "random":
            crossing = bool(rng.random() < 0.70)
        else:
            crossing = bool(scene_idx % 4 == 0)
        scene = make_scene(
            seed=int(rng.integers(0, 1_000_000)),
            n_objects=scene_n_objects,
            hidden_property=True,
            occlusion=occlusion,
            crossing=crossing,
        )
        for obj in scene.objects:
            action = rng.normal(0.0, action_scale, size=2)
            x = _object_features(obj.position, obj.velocity, obj.hidden_mass, obj.color, obj.shape, obj.occluded)
            x[2] += float(action[0])
            x[3] += float(action[1])
            drag = 0.10 + 0.45 * obj.hidden_mass + 0.18 * obj.hidden_friction
            delta = np.asarray(obj.velocity) + action * (1.0 - drag)
            next_pos = np.clip(np.asarray(obj.position) + delta, 0.0, 1.0)
            reward = 1.0 - min(1.0, float(np.linalg.norm(next_pos - np.asarray(scene.goal_position))))
            reward += 0.10 if obj.is_target else -0.08
            features.append(x)
            transition_targets.append([float(next_pos[0]), float(next_pos[1])])
            property_targets.append(float(obj.hidden_mass >= 0.70))
            reward_targets.append(float(np.clip(reward, 0.0, 1.0)))
    return (
        np.asarray(features, dtype=float),
        np.asarray(transition_targets, dtype=float),
        np.asarray(property_targets, dtype=float),
        np.asarray(reward_targets, dtype=float),
    )


def _next_object_features(obj, action: np.ndarray) -> list[float]:
    drag = 0.10 + 0.45 * obj.hidden_mass + 0.18 * obj.hidden_friction
    delta = np.asarray(obj.velocity) + action * (1.0 - drag)
    next_pos = np.clip(np.asarray(obj.position) + delta, 0.0, 1.0)
    next_velocity = tuple((np.asarray(obj.velocity) + action * (1.0 - drag)).tolist())
    return _object_features(
        (float(next_pos[0]), float(next_pos[1])),
        next_velocity,
        obj.hidden_mass,
        obj.color,
        obj.shape,
        obj.occluded,
    )


def _pair_features(left: list[float], right: list[float]) -> list[float]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    diff = np.abs(left_arr - right_arr)
    return np.r_[diff, left_arr[:4] * right_arr[:4], [1.0 - diff[4], 1.0 - diff[5]]].astype(float).tolist()


def make_identity_alignment_dataset(
    n_scenes: int = 240,
    seed: int = 0,
    n_objects: int | tuple[int, int] = 4,
    occlusion_mode: str = "periodic",
    crossing_mode: str = "always",
    action_scale: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Make positive/negative slot-pair examples for identity persistence."""

    rng = np.random.default_rng(seed + 31_337)
    pair_rows: list[list[float]] = []
    labels: list[float] = []
    for scene_idx in range(n_scenes):
        if isinstance(n_objects, tuple):
            low, high = n_objects
            scene_n_objects = int(rng.integers(low, high + 1))
        else:
            scene_n_objects = int(n_objects)
        if occlusion_mode == "always":
            occlusion = True
        elif occlusion_mode == "never":
            occlusion = False
        elif occlusion_mode == "random":
            occlusion = bool(rng.random() < 0.70)
        else:
            occlusion = bool(scene_idx % 2 == 0)
        if crossing_mode == "always":
            crossing = True
        elif crossing_mode == "never":
            crossing = False
        elif crossing_mode == "random":
            crossing = bool(rng.random() < 0.70)
        else:
            crossing = bool(scene_idx % 2 == 0)
        scene = make_scene(
            seed=int(rng.integers(0, 1_000_000)),
            n_objects=scene_n_objects,
            hidden_property=True,
            occlusion=occlusion,
            crossing=crossing,
        )
        current = []
        future = []
        for obj in scene.objects:
            action = rng.normal(0.0, action_scale, size=2)
            current.append(_object_features(obj.position, obj.velocity, obj.hidden_mass, obj.color, obj.shape, obj.occluded))
            future.append(_next_object_features(obj, action))
        for obj_idx in range(len(scene.objects)):
            pair_rows.append(_pair_features(current[obj_idx], future[obj_idx]))
            labels.append(1.0)
            negative_idx = (obj_idx + int(rng.integers(1, len(scene.objects)))) % len(scene.objects)
            pair_rows.append(_pair_features(current[obj_idx], future[negative_idx]))
            labels.append(0.0)
    return np.asarray(pair_rows, dtype=float), np.asarray(labels, dtype=float)


def _metrics_from_predictions(
    model: NumpyObjectCentricModel,
    x_test: np.ndarray,
    y_trans_test: np.ndarray,
    y_prop_test: np.ndarray,
    y_reward_test: np.ndarray,
    identity_x_test: np.ndarray,
    identity_y_test: np.ndarray,
    transition_baseline: np.ndarray,
    n_train_slots: int,
) -> LearnedMetrics:
    pred_trans = model.predict_transition(x_test)
    pred_prop = model.predict_property_proba(x_test)
    pred_reward = model.predict_reward(x_test)
    pred_identity = model.predict_identity_alignment(identity_x_test)

    transition_mse = float(np.mean((pred_trans - y_trans_test) ** 2))
    constant_transition = np.tile(transition_baseline, (y_trans_test.shape[0], 1))
    constant_transition_mse = float(np.mean((constant_transition - y_trans_test) ** 2))
    property_accuracy = float(np.mean((pred_prop >= 0.5).astype(float) == y_prop_test))
    random_property = max(float(np.mean(y_prop_test)), 1.0 - float(np.mean(y_prop_test)))
    identity_accuracy = float(np.mean((pred_identity >= 0.5).astype(float) == identity_y_test))
    random_identity = max(float(np.mean(identity_y_test)), 1.0 - float(np.mean(identity_y_test)))
    reward_correlation = 0.0
    if np.std(pred_reward) > 1e-12 and np.std(y_reward_test) > 1e-12:
        reward_correlation = float(np.corrcoef(pred_reward, y_reward_test)[0, 1])
    return LearnedMetrics(
        property_accuracy=property_accuracy,
        random_property_accuracy=random_property,
        identity_alignment_accuracy=identity_accuracy,
        random_identity_alignment_accuracy=random_identity,
        transition_mse=transition_mse,
        constant_transition_mse=constant_transition_mse,
        reward_correlation=reward_correlation,
        n_train_slots=int(n_train_slots),
        n_test_slots=int(x_test.shape[0]),
    )


def _evaluate_for_train_size(
    seed: int,
    n_train_scenes: int,
    n_test_scenes: int,
) -> tuple[LearnedMetrics, NumpyObjectCentricModel]:
    x_train, y_trans_train, y_prop_train, y_reward_train = make_synthetic_trajectory_dataset(n_train_scenes, seed)
    x_test, y_trans_test, y_prop_test, y_reward_test = make_synthetic_trajectory_dataset(n_test_scenes, seed + 10_000)
    identity_x_train, identity_y_train = make_identity_alignment_dataset(n_train_scenes, seed)
    identity_x_test, identity_y_test = make_identity_alignment_dataset(n_test_scenes, seed + 10_000)
    model = NumpyObjectCentricModel().fit(
        x_train,
        y_trans_train,
        y_prop_train,
        y_reward_train,
        identity_x=identity_x_train,
        identity_y=identity_y_train,
    )

    return (
        _metrics_from_predictions(
            model,
            x_test,
            y_trans_test,
            y_prop_test,
            y_reward_test,
            identity_x_test,
            identity_y_test,
            transition_baseline=np.mean(y_trans_train, axis=0),
            n_train_slots=int(x_train.shape[0]),
        ),
        model,
    )


def _evaluate_with_masks(
    seed: int,
    n_train_scenes: int,
    n_test_scenes: int,
    feature_mask: np.ndarray,
    identity_mask: np.ndarray,
) -> LearnedMetrics:
    x_train, y_trans_train, y_prop_train, y_reward_train = make_synthetic_trajectory_dataset(n_train_scenes, seed)
    x_test, y_trans_test, y_prop_test, y_reward_test = make_synthetic_trajectory_dataset(n_test_scenes, seed + 10_000)
    identity_x_train, identity_y_train = make_identity_alignment_dataset(n_train_scenes, seed)
    identity_x_test, identity_y_test = make_identity_alignment_dataset(n_test_scenes, seed + 10_000)
    model = NumpyObjectCentricModel().fit(
        x_train[:, feature_mask],
        y_trans_train,
        y_prop_train,
        y_reward_train,
        identity_x=identity_x_train[:, identity_mask],
        identity_y=identity_y_train,
    )
    pred_trans = model.predict_transition(x_test[:, feature_mask])
    pred_prop = model.predict_property_proba(x_test[:, feature_mask])
    pred_reward = model.predict_reward(x_test[:, feature_mask])
    pred_identity = model.predict_identity_alignment(identity_x_test[:, identity_mask])
    transition_mse = float(np.mean((pred_trans - y_trans_test) ** 2))
    constant_transition = np.tile(np.mean(y_trans_train, axis=0), (y_trans_test.shape[0], 1))
    constant_transition_mse = float(np.mean((constant_transition - y_trans_test) ** 2))
    property_accuracy = float(np.mean((pred_prop >= 0.5).astype(float) == y_prop_test))
    random_property = max(float(np.mean(y_prop_test)), 1.0 - float(np.mean(y_prop_test)))
    identity_accuracy = float(np.mean((pred_identity >= 0.5).astype(float) == identity_y_test))
    random_identity = max(float(np.mean(identity_y_test)), 1.0 - float(np.mean(identity_y_test)))
    reward_correlation = 0.0
    if np.std(pred_reward) > 1e-12 and np.std(y_reward_test) > 1e-12:
        reward_correlation = float(np.corrcoef(pred_reward, y_reward_test)[0, 1])
    return LearnedMetrics(
        property_accuracy=property_accuracy,
        random_property_accuracy=random_property,
        identity_alignment_accuracy=identity_accuracy,
        random_identity_alignment_accuracy=random_identity,
        transition_mse=transition_mse,
        constant_transition_mse=constant_transition_mse,
        reward_correlation=reward_correlation,
        n_train_slots=int(x_train.shape[0]),
        n_test_slots=int(x_test.shape[0]),
    )


def learned_ablation_rows(seed: int, n_train_scenes: int, n_test_scenes: int) -> list[dict[str, float | int | str]]:
    """Evaluate feature ablations for learned object-centric evidence."""

    full_features = np.arange(8)
    full_identity = np.arange(14)
    ablations = [
        ("full_object_features", full_features, full_identity),
        ("no_mass_sensor", np.array([0, 1, 2, 3, 4, 5, 6]), full_identity),
        ("kinematic_only_slots", np.array([0, 1, 2, 3]), full_identity),
        ("appearance_only_slots", np.array([4, 5, 6, 7]), full_identity),
        ("kinematic_pair_identity", full_features, np.array([0, 1, 2, 3, 8, 9, 10, 11])),
    ]
    rows: list[dict[str, float | int | str]] = []
    for name, feature_mask, identity_mask in ablations:
        metrics = _evaluate_with_masks(seed, n_train_scenes, n_test_scenes, feature_mask, identity_mask)
        row = metrics.as_dict()
        row["ablation"] = name
        row["n_slot_features"] = int(len(feature_mask))
        row["n_identity_features"] = int(len(identity_mask))
        rows.append(row)
    full = next(row for row in rows if row["ablation"] == "full_object_features")
    for row in rows:
        row["full_minus_property_accuracy"] = float(full["property_accuracy"] - row["property_accuracy"])
        row["full_minus_identity_alignment_accuracy"] = float(
            full["identity_alignment_accuracy"] - row["identity_alignment_accuracy"]
        )
        row["full_minus_reward_correlation"] = float(full["reward_correlation"] - row["reward_correlation"])
    return rows


def learned_domain_shift_rows(
    model: NumpyObjectCentricModel,
    seed: int,
    n_train_scenes: int,
    n_test_scenes: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate the learned model on held-out synthetic distribution shifts."""

    x_train, y_trans_train, _, _ = make_synthetic_trajectory_dataset(n_train_scenes, seed)
    transition_baseline = np.mean(y_trans_train, axis=0)
    variants: list[tuple[str, dict[str, object]]] = [
        ("standard_test", {}),
        (
            "dense6_test",
            {
                "n_objects": 6,
                "occlusion_mode": "periodic",
                "crossing_mode": "periodic",
                "action_scale": 0.08,
            },
        ),
        (
            "dense8_occluded_crossing",
            {
                "n_objects": 8,
                "occlusion_mode": "always",
                "crossing_mode": "always",
                "action_scale": 0.09,
            },
        ),
        (
            "random_count_mixed",
            {
                "n_objects": (5, 9),
                "occlusion_mode": "random",
                "crossing_mode": "random",
                "action_scale": 0.10,
            },
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    for idx, (variant, kwargs) in enumerate(variants):
        x_test, y_trans_test, y_prop_test, y_reward_test = make_synthetic_trajectory_dataset(
            n_test_scenes,
            seed + 70_000 + idx * 997,
            **kwargs,
        )
        identity_x_test, identity_y_test = make_identity_alignment_dataset(
            n_test_scenes,
            seed + 70_000 + idx * 997,
            n_objects=kwargs.get("n_objects", 4),
            occlusion_mode=str(kwargs.get("occlusion_mode", "periodic")),
            crossing_mode=str(kwargs.get("crossing_mode", "always")),
            action_scale=float(kwargs.get("action_scale", 0.08)),
        )
        metrics = _metrics_from_predictions(
            model,
            x_test,
            y_trans_test,
            y_prop_test,
            y_reward_test,
            identity_x_test,
            identity_y_test,
            transition_baseline=transition_baseline,
            n_train_slots=int(x_train.shape[0]),
        )
        row = metrics.as_dict()
        row["variant"] = variant
        row["property_margin"] = float(row["property_accuracy"] - row["random_property_accuracy"])
        row["identity_margin"] = float(row["identity_alignment_accuracy"] - row["random_identity_alignment_accuracy"])
        row["transition_mse_ratio"] = float(row["transition_mse"] / row["constant_transition_mse"])
        row["n_objects_spec"] = str(kwargs.get("n_objects", 4))
        row["occlusion_mode"] = str(kwargs.get("occlusion_mode", "periodic"))
        row["crossing_mode"] = str(kwargs.get("crossing_mode", "periodic"))
        row["action_scale"] = float(kwargs.get("action_scale", 0.08))
        rows.append(row)
    return rows


def train_and_evaluate(
    output_dir: str | Path | None = None,
    seed: int = 0,
    n_train_scenes: int = 260,
    n_test_scenes: int = 120,
) -> tuple[LearnedMetrics, NumpyObjectCentricModel]:
    metrics, model = _evaluate_for_train_size(seed, n_train_scenes, n_test_scenes)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        learning_sizes = [32, 64, 128, n_train_scenes]
        learning_rows = []
        for size in learning_sizes:
            size_metrics, _ = _evaluate_for_train_size(seed + size, size, n_test_scenes)
            row = size_metrics.as_dict()
            row["train_scenes"] = int(size)
            learning_rows.append(row)
        learning_curve = pd.DataFrame(learning_rows)
        learning_curve.to_csv(out / "tables" / "learned_learning_curve.csv", index=False)
        ablation_table = pd.DataFrame(learned_ablation_rows(seed, n_train_scenes, n_test_scenes))
        ablation_table.to_csv(out / "tables" / "learned_ablation.csv", index=False)
        shift_table = pd.DataFrame(learned_domain_shift_rows(model, seed, n_train_scenes, n_test_scenes))
        shift_table.to_csv(out / "tables" / "learned_domain_shift.csv", index=False)
        shifted = shift_table[shift_table["variant"] != "standard_test"]
        summary = {
            "model": "numpy_linear_object_slot_model_with_identity_alignment",
            "claim_scope": "controlled synthetic trajectories only",
            "metrics": metrics.as_dict(),
            "learning_curve_rows": int(learning_curve.shape[0]),
            "ablation_rows": int(ablation_table.shape[0]),
            "domain_shift_rows": int(shift_table.shape[0]),
            "domain_shift_min_property_margin": float(shifted["property_margin"].min()),
            "domain_shift_min_identity_margin": float(shifted["identity_margin"].min()),
            "domain_shift_max_transition_ratio": float(shifted["transition_mse_ratio"].max()),
            "domain_shift_min_reward_correlation": float(shifted["reward_correlation"].min()),
            "passes_minimum_learned_artifact_checks": bool(
                metrics.property_accuracy >= metrics.random_property_accuracy + 0.15
                and metrics.identity_alignment_accuracy >= metrics.random_identity_alignment_accuracy + 0.15
                and metrics.transition_mse <= 0.25 * metrics.constant_transition_mse
                and metrics.reward_correlation > 0.75
                and float(ablation_table.loc[ablation_table["ablation"] == "no_mass_sensor", "full_minus_property_accuracy"].iloc[0]) >= 0.10
                and float(ablation_table.loc[ablation_table["ablation"] == "kinematic_pair_identity", "full_minus_identity_alignment_accuracy"].iloc[0]) >= 0.05
                and float(shifted["property_margin"].min()) >= 0.12
                and float(shifted["identity_margin"].min()) >= 0.15
                and float(shifted["transition_mse_ratio"].max()) <= 0.30
                and float(shifted["reward_correlation"].min()) >= 0.70
            ),
        }
        (out / "learned_object_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame([metrics.as_dict()]).to_csv(out / "tables" / "learned_metrics.csv", index=False)
    return metrics, model
