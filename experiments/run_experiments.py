"""Run controlled object-centric Best-of-N experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from object_centric_best_of_n.audit import write_claim_status, write_final_audit
from object_centric_best_of_n.envs import make_scene, retarget_scene
from object_centric_best_of_n.learned_model import learned_candidate_scores, train_and_evaluate
from object_centric_best_of_n.metrics import (
    add_repair_metadata,
    aggregate_seed_metrics,
    calibration_diagnostics_summary,
    counterfactual_target_summary,
    deployment_gate_from_metrics,
    deployment_policy_summary,
    domain_randomization_summary,
    exact_law_prediction_error,
    extreme_object_count_summary,
    learned_repair_policy_summary,
    learned_selection_summary,
    model_family_proxy_summary,
    negative_control_summary,
    observable_repair_summary,
    ood_summary,
    paired_selector_effects,
    pilot_calibration_summary,
    pilot_budget_summary,
    noisy_probe_summary,
    probe_cost_summary,
    repair_ablation_summary,
    repair_robustness_by_split,
    score_calibration_table,
    selection_record,
    seed_block_robustness,
    sensitivity_summary,
    statistical_audit,
    stress_summary,
    synthetic_benchmark_summary,
    target_identity_sweep_summary,
)
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator
from object_centric_best_of_n.plotting import write_all_figures
from object_centric_best_of_n.repair import (
    combined_repair_score,
    conservative_selected_tail_stop_rule,
    empty_pilot_calibrator,
    fit_pilot_calibrator,
    hidden_mode_unidentifiable_gate,
    observable_repair_score,
    pilot_calibration_features,
    pilot_calibrated_score,
    property_posterior_update,
    property_prior_from_candidate,
    temporal_identity_consistency,
)
from object_centric_best_of_n.selection import SELECTORS
from object_centric_best_of_n.theory import law_validation_row


SCENARIOS = ["good", "swap", "merge_split", "occlusion", "hidden_property", "raw"]
SELECTOR_ORDER = [
    "raw",
    "identity_consistent",
    "property_calibrated",
    "targeted_probe",
    "combined_repair",
    "observable_repair",
    "random",
    "oracle",
]
STRESS_SCENARIOS = ["raw", "occlusion", "hidden_property", "swap", "merge_split"]
STRESS_SELECTORS = ["raw", "identity_consistent", "targeted_probe", "combined_repair", "observable_repair", "random", "oracle"]
SENSITIVITY_NOISE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]
OOD_VARIANTS = [
    ("dense6_good", 6, False, False, False, "good"),
    ("dense6_raw", 6, True, True, True, "raw"),
    ("dense8_occlusion", 8, True, True, True, "occlusion"),
    ("dense8_hidden", 8, False, True, False, "hidden_property"),
]
OOD_SELECTORS = ["raw", "combined_repair", "observable_repair", "random", "oracle"]
EXTREME_OBJECT_VARIANTS = [
    ("extreme10_good", 10, False, False, False, "good"),
    ("extreme10_raw", 10, True, True, True, "raw"),
    ("extreme12_occlusion", 12, True, True, True, "occlusion"),
    ("extreme12_hidden", 12, False, True, False, "hidden_property"),
]
EXTREME_OBJECT_SELECTORS = ["raw", "observable_repair", "combined_repair", "random", "oracle"]
MODEL_FAMILY_SCENARIOS = ["raw", "occlusion", "hidden_property", "swap", "merge_split"]
MODEL_FAMILY_SELECTORS = [
    "raw",
    "latent_global_proxy",
    "relational_slot_proxy",
    "diffusion_score_proxy",
    "combined_repair",
    "oracle",
]
DOMAIN_RANDOMIZATION_SELECTORS = ["raw", "observable_repair", "combined_repair", "random", "oracle"]
COUNTERFACTUAL_TARGET_SELECTORS = ["raw", "observable_repair", "combined_repair", "random", "oracle"]
TARGET_SWEEP_IDS = [0, 1, 2, 3, 4, 5]
TARGET_SWEEP_SELECTORS = ["raw", "observable_repair", "combined_repair", "random", "oracle"]
LEARNED_SELECTION_TARGET_IDS = [0, 1, 2, 3, 4, 5]
LEARNED_SELECTION_SELECTORS = [
    "raw",
    "learned_reward",
    "learned_identity_reward",
    "observable_repair",
    "combined_repair",
    "oracle",
]
LEARNED_REPAIR_POLICY_SELECTORS = [
    "raw",
    "learned_reward",
    "learned_identity_reward",
    "pilot_calibrated",
    "learned_repair_policy",
    "observable_repair",
    "combined_repair",
    "random",
    "oracle",
]
SYNTHETIC_BENCHMARK_VARIANTS = [
    ("dense_clutter_raw", 9, True, True, True, "raw", 0),
    ("retarget_green_hidden", 7, True, True, True, "raw", 2),
    ("crossing_swap", 6, False, False, True, "swap", 0),
    ("occlusion_corridor", 6, True, False, True, "occlusion", 0),
    ("hidden_mass_probe", 6, False, True, False, "hidden_property", 0),
    ("merge_split_clutter", 7, False, False, False, "merge_split", 0),
    ("mixed_medium_raw", 5, True, True, True, "raw", 3),
]
SYNTHETIC_BENCHMARK_SELECTORS = ["raw", "observable_repair", "combined_repair", "random", "oracle"]
PILOT_CALIBRATION_SELECTORS = ["raw", "pilot_calibrated", "observable_repair", "combined_repair", "random", "oracle"]
PILOT_BUDGETS = [0, 8, 32, 128]
NOISY_PROBE_RELIABILITIES = [0.55, 0.65, 0.75, 0.85, 0.90]
NOISY_PROBE_SELECTORS = ["raw", "noisy_probe_repair", "observable_repair", "combined_repair", "random", "oracle"]
PROBE_COSTS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
PROBE_COST_SCENARIOS = ["hidden_property", "raw"]
PROBE_COST_SELECTORS = ["raw", "targeted_probe", "observable_repair", "combined_repair", "oracle"]
PROBE_USING_SELECTORS = {"targeted_probe", "observable_repair", "combined_repair"}
DEPLOYMENT_POLICY_SELECTOR_FOR_ACTION = {
    "allow_high_n": "raw",
    "stop_early": "raw",
    "collect_pilot_labels": "observable_repair",
    "run_object_probe": "targeted_probe",
    "block_high_n": "combined_repair",
}


def _parse_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _scene_for_scenario(seed: int, scenario: str):
    return make_scene(
        seed=seed,
        occlusion=scenario in {"occlusion", "raw"},
        hidden_property=scenario in {"hidden_property", "raw"},
        crossing=scenario in {"occlusion", "swap", "raw"},
    )


def _run_stress_panel(
    generator: ObjectCentricFutureGenerator,
    stress_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in stress_seeds:
        for scenario in STRESS_SCENARIOS:
            scene = _scene_for_scenario(50_000 + seed, scenario)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=97_531 + seed * 173 + len(scenario),
            )
            for selector_name in STRESS_SELECTORS:
                selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                rows.append(selection_record("F_high_n_stress", scenario, selector_name, n, seed, selected, candidates))
    return pd.DataFrame(rows)


def _choose_by_scores(candidates, scores: np.ndarray, seed: int):
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score))
    rng = np.random.default_rng(seed)
    return candidates[int(rng.choice(tied))]


def _select_by_scores_with_label(candidates, scores: np.ndarray, seed: int, label: str):
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score))
    rng = np.random.default_rng(seed)
    chosen = candidates[int(rng.choice(tied))]
    return chosen.with_score(max_score, label)


def _noisy_probe_repair_score(candidate, scene, reliability: float, seed: int) -> float:
    rng = np.random.default_rng(seed + candidate.candidate_id * 997 + int(reliability * 10_000))
    true_heavy = scene.target().hidden_mass >= 0.70
    if rng.random() < reliability:
        observation = "heavy" if true_heavy else "light"
    else:
        observation = "light" if true_heavy else "heavy"
    prior = property_prior_from_candidate(candidate)
    posterior = property_posterior_update(prior, observation, reliability=reliability)
    consistency = temporal_identity_consistency(candidate)
    merge = _diagnostic(candidate, "merge_evidence")
    slot_support = _diagnostic(candidate, "slot_support", 0.5)
    instability = _diagnostic(candidate, "identity_instability", 0.5)
    property_surprise = _diagnostic(candidate, "property_surprise", candidate.property_entropy)
    posterior_confidence = max(posterior, 1.0 - posterior)
    return float(
        0.20 * candidate.score
        + 0.82 * consistency
        + 0.34 * slot_support
        + 0.16 * posterior_confidence
        + 0.10 * (1.0 - candidate.property_entropy)
        - 0.82 * merge
        - 0.40 * instability
        - 0.22 * property_surprise
    )


def _select_noisy_probe_repair(candidates, scene, reliability: float, seed: int):
    scores = np.asarray(
        [_noisy_probe_repair_score(candidate, scene, reliability=reliability, seed=seed) for candidate in candidates],
        dtype=float,
    )
    return _select_by_scores_with_label(candidates, scores, seed=seed, label="noisy_probe_repair")


def _diagnostic(candidate, name: str, default: float = 0.0) -> float:
    return float(candidate.diagnostics.get(name, default))


def _model_family_proxy_score(candidate, selector_name: str) -> float:
    """Toy proxy scores for reviewer diagnostics, not external benchmarks."""

    ambition = _diagnostic(candidate, "ambition")
    identity_instability = _diagnostic(candidate, "identity_instability")
    slot_support = _diagnostic(candidate, "slot_support", 1.0)
    merge_evidence = _diagnostic(candidate, "merge_evidence")
    property_surprise = _diagnostic(candidate, "property_surprise")
    if selector_name == "latent_global_proxy":
        return float(candidate.score + 0.05 * ambition - 0.08 * candidate.property_entropy)
    if selector_name == "relational_slot_proxy":
        return float(candidate.score - 0.35 * identity_instability - 0.45 * merge_evidence + 0.16 * slot_support)
    if selector_name == "diffusion_score_proxy":
        return float(candidate.score + 0.12 * ambition + 0.05 * property_surprise)
    raise ValueError(f"unknown proxy selector: {selector_name}")


def _select_model_family_proxy(candidates, selector_name: str, seed: int):
    scores = np.asarray([_model_family_proxy_score(candidate, selector_name) for candidate in candidates], dtype=float)
    selected = _choose_by_scores(candidates, scores, seed=seed)
    return selected.with_score(float(np.max(scores)), selector_name)


def _run_sensitivity_panel(
    generator: ObjectCentricFutureGenerator,
    sensitivity_seeds: list[int],
    n: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int]] = []
    reps = 3 if mode == "smoke" else 12
    for seed in sensitivity_seeds:
        scene = _scene_for_scenario(80_000 + seed, "raw")
        candidates = generator.generate_candidates(
            scene,
            n=n,
            scenario="raw",
            seed=123_457 + seed * 313,
        )
        raw_scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
        repair_scores = np.asarray([combined_repair_score(candidate, scene, seed=seed) for candidate in candidates], dtype=float)
        for candidate in candidates:
            candidate_rows.append(
                {
                    "seed": int(seed),
                    "candidate_id": int(candidate.candidate_id),
                    "raw_object_score": float(candidate.score),
                    "real_utility": float(candidate.real_utility),
                    "identity_error": float(candidate.identity_error),
                    "merge_split": float(candidate.merge_split),
                    "property_error": float(candidate.property_error),
                    "object_real_gap": float(candidate.object_real_gap),
                }
            )
        for noise in SENSITIVITY_NOISE:
            for rep in range(reps):
                rng = np.random.default_rng(900_000 + seed * 1009 + rep * 137 + int(noise * 1000))
                raw_selected = _choose_by_scores(candidates, raw_scores + rng.normal(0.0, noise, size=len(candidates)), seed + rep)
                repair_selected = _choose_by_scores(
                    candidates,
                    repair_scores + rng.normal(0.0, noise, size=len(candidates)),
                    seed + rep + 17,
                )
                for selector, selected in [("raw_noisy", raw_selected), ("combined_repair_noisy", repair_selected)]:
                    rows.append(
                        {
                            "seed": int(seed),
                            "rep": int(rep),
                            "selector": selector,
                            "score_noise": float(noise),
                            "selected_candidate_id": int(selected.candidate_id),
                            "selected_real_utility": float(selected.real_utility),
                            "identity_error": float(selected.identity_error),
                            "merge_split": float(selected.merge_split),
                            "property_error": float(selected.property_error),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def _run_ood_panel(
    generator: ObjectCentricFutureGenerator,
    ood_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in ood_seeds:
        for variant_name, n_objects, occlusion, hidden_property, crossing, scenario in OOD_VARIANTS:
            scene = make_scene(
                seed=120_000 + seed,
                n_objects=n_objects,
                occlusion=occlusion,
                hidden_property=hidden_property,
                crossing=crossing,
            )
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=77_000 + seed * 313 + len(variant_name),
            )
            for selector_name in OOD_SELECTORS:
                selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                rows.append(selection_record("K_ood_object_count", variant_name, selector_name, n, seed, selected, candidates))
    return pd.DataFrame(rows)


def _run_extreme_object_count_panel(
    generator: ObjectCentricFutureGenerator,
    extreme_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in extreme_seeds:
        for variant_name, n_objects, occlusion, hidden_property, crossing, scenario in EXTREME_OBJECT_VARIANTS:
            scene = make_scene(
                seed=420_000 + seed,
                n_objects=n_objects,
                occlusion=occlusion,
                hidden_property=hidden_property,
                crossing=crossing,
            )
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=421_337 + seed * 421 + n_objects + len(variant_name),
            )
            for selector_name in EXTREME_OBJECT_SELECTORS:
                selected = SELECTORS[selector_name](candidates, scene, seed=seed + n + n_objects)
                record = selection_record(
                    "U_extreme_object_count",
                    variant_name,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "n_objects": int(n_objects),
                        "occlusion_flag": int(occlusion),
                        "hidden_property_flag": int(hidden_property),
                        "crossing_flag": int(crossing),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def _run_model_family_panel(
    generator: ObjectCentricFutureGenerator,
    family_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in family_seeds:
        for scenario in MODEL_FAMILY_SCENARIOS:
            scene = _scene_for_scenario(160_000 + seed, scenario)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=141_421 + seed * 911 + len(scenario),
            )
            for selector_name in MODEL_FAMILY_SELECTORS:
                if selector_name in SELECTORS:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                else:
                    selected = _select_model_family_proxy(candidates, selector_name, seed=seed + n)
                rows.append(
                    selection_record(
                        "L_model_family_proxies",
                        scenario,
                        selector_name,
                        n,
                        seed,
                        selected,
                        candidates,
                    )
                )
    return pd.DataFrame(rows)


def _domain_randomized_scene(seed: int):
    rng = np.random.default_rng(220_000 + seed)
    n_objects = int(rng.integers(3, 10))
    occlusion = bool(rng.random() < 0.65)
    hidden_property = bool(rng.random() < 0.70)
    crossing = bool(rng.random() < 0.65)
    if not (occlusion or hidden_property or crossing):
        hidden_property = True
    if hidden_property and (occlusion or crossing):
        scenario = "raw"
    elif occlusion:
        scenario = "occlusion"
    elif hidden_property:
        scenario = "hidden_property"
    elif crossing:
        scenario = "swap"
    else:
        scenario = "merge_split"
    scene = make_scene(
        seed=220_000 + seed,
        n_objects=n_objects,
        occlusion=occlusion,
        hidden_property=hidden_property,
        crossing=crossing,
    )
    return scene, scenario, n_objects, occlusion, hidden_property, crossing


def _run_domain_randomization_panel(
    generator: ObjectCentricFutureGenerator,
    domain_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in domain_seeds:
        scene, generator_scenario, n_objects, occlusion, hidden_property, crossing = _domain_randomized_scene(seed)
        candidates = generator.generate_candidates(
            scene,
            n=n,
            scenario=generator_scenario,
            seed=199_999 + seed * 613 + n_objects,
        )
        for selector_name in DOMAIN_RANDOMIZATION_SELECTORS:
            selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
            record = selection_record(
                "P_domain_randomized_stress",
                "domain_randomized",
                selector_name,
                n,
                seed,
                selected,
                candidates,
            )
            record.update(
                {
                    "n_objects": n_objects,
                    "occlusion_flag": int(occlusion),
                    "hidden_property_flag": int(hidden_property),
                    "crossing_flag": int(crossing),
                    "generator_scenario": generator_scenario,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _run_counterfactual_target_panel(
    generator: ObjectCentricFutureGenerator,
    counter_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in counter_seeds:
        base_scene = make_scene(
            seed=260_000 + seed,
            n_objects=4,
            occlusion=True,
            hidden_property=True,
            crossing=True,
        )
        scene = retarget_scene(base_scene, target_id=1)
        candidates = generator.generate_candidates(
            scene,
            n=n,
            scenario="raw",
            seed=277_213 + seed * 701 + n,
        )
        for selector_name in COUNTERFACTUAL_TARGET_SELECTORS:
            selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
            record = selection_record(
                "Q_counterfactual_target",
                "target_id_1",
                selector_name,
                n,
                seed,
                selected,
                candidates,
            )
            record.update(
                {
                    "original_target_id": int(base_scene.target_id),
                    "counterfactual_target_id": int(scene.target_id),
                    "n_objects": int(len(scene.objects)),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _run_target_identity_sweep_panel(
    generator: ObjectCentricFutureGenerator,
    target_sweep_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in target_sweep_seeds:
        for target_id in TARGET_SWEEP_IDS:
            base_scene = make_scene(
                seed=560_000 + seed * 31 + target_id,
                n_objects=6,
                occlusion=True,
                hidden_property=True,
                crossing=True,
            )
            scene = retarget_scene(base_scene, target_id=target_id)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario="raw",
                seed=571_111 + seed * 997 + target_id * 37,
            )
            for selector_name in TARGET_SWEEP_SELECTORS:
                selected = SELECTORS[selector_name](candidates, scene, seed=seed + n + target_id * 13)
                record = selection_record(
                    "X_target_identity_sweep",
                    "target_identity_sweep",
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "original_target_id": int(base_scene.target_id),
                        "target_id": int(scene.target_id),
                        "n_objects": int(len(scene.objects)),
                        "generator_scenario": "raw",
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def _run_learned_selection_panel(
    generator: ObjectCentricFutureGenerator,
    learned_model,
    learned_selection_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in learned_selection_seeds:
        eval_cases = [
            (
                "raw_heldout",
                _scene_for_scenario(610_000 + seed, "raw"),
                "raw",
                {"target_id": 0, "n_objects": 4},
            )
        ]
        base_scene = make_scene(
            seed=620_000 + seed,
            n_objects=6,
            occlusion=True,
            hidden_property=True,
            crossing=True,
        )
        for target_id in LEARNED_SELECTION_TARGET_IDS:
            scene = retarget_scene(base_scene, target_id=target_id)
            eval_cases.append(
                (
                    f"target_{target_id}",
                    scene,
                    "raw",
                    {"target_id": int(target_id), "n_objects": int(len(scene.objects))},
                )
            )
        for scenario_label, scene, generator_scenario, flags in eval_cases:
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=generator_scenario,
                seed=631_000 + seed * 997 + len(scenario_label),
            )
            learned_scores = learned_candidate_scores(learned_model, candidates, scene)
            for selector_name in LEARNED_SELECTION_SELECTORS:
                if selector_name in {"learned_reward", "learned_identity_reward"}:
                    selected = _select_by_scores_with_label(
                        candidates,
                        learned_scores[selector_name],
                        seed=seed + n + len(scenario_label),
                        label=selector_name,
                    )
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "Y_learned_selection_transfer",
                    scenario_label,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "generator_scenario": generator_scenario,
                        "learned_reward_mean": float(np.mean(learned_scores["learned_reward"])),
                        "learned_identity_alignment_mean": float(
                            np.mean(learned_scores["learned_identity_alignment"])
                        ),
                        "learned_property_confidence_mean": float(
                            np.mean(learned_scores["learned_property_confidence"])
                        ),
                        **flags,
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


LEARNED_REPAIR_POLICY_FEATURE_NAMES = [
    *[f"pilot_{name}" for name in [
        "raw_score",
        "identity_consistency",
        "slot_support",
        "target_slot_confidence",
        "target_id_match",
        "one_minus_identity_instability",
        "one_minus_merge_evidence",
        "one_minus_property_entropy",
        "one_minus_property_surprise",
        "hidden_mass_estimate",
        "property_prior_heavy",
        "slot_count_scaled",
    ]],
    "learned_reward",
    "learned_identity_alignment",
    "learned_property_confidence",
    "observable_repair_score",
    "learned_identity_times_observable",
]


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)


def _learned_repair_policy_features(
    candidate,
    learned_scores: dict[str, np.ndarray],
    candidate_index: int,
    observable_score: float,
) -> np.ndarray:
    learned_reward = float(learned_scores["learned_reward"][candidate_index])
    learned_identity = float(learned_scores["learned_identity_alignment"][candidate_index])
    learned_property = float(learned_scores["learned_property_confidence"][candidate_index])
    return np.concatenate(
        [
            pilot_calibration_features(candidate),
            np.asarray(
                [
                    learned_reward,
                    learned_identity,
                    learned_property,
                    float(observable_score),
                    learned_identity * float(observable_score),
                ],
                dtype=float,
            ),
        ]
    )


def _fit_learned_repair_policy(
    generator: ObjectCentricFutureGenerator,
    learned_model,
    train_seeds: list[int],
    n: int,
) -> dict[str, object]:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for seed in train_seeds:
        for scenario in STRESS_SCENARIOS:
            scene = _scene_for_scenario(690_000 + seed, scenario)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=701_111 + seed * 977 + len(scenario),
            )
            learned_scores = learned_candidate_scores(learned_model, candidates, scene)
            observable_scores = np.asarray(
                [
                    observable_repair_score(candidate, scene, seed=seed + n)
                    for candidate in candidates
                ],
                dtype=float,
            )
            for idx, candidate in enumerate(candidates):
                rows.append(
                    _learned_repair_policy_features(
                        candidate,
                        learned_scores,
                        idx,
                        float(observable_scores[idx]),
                    )
                )
                labels.append(float(candidate.real_utility))
    x = np.vstack(rows)
    y = np.asarray(labels, dtype=float)
    feature_mean = np.mean(x, axis=0)
    feature_scale = np.std(x, axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    z = (x - feature_mean) / feature_scale
    design = np.column_stack([np.ones(z.shape[0]), z])
    ridge = 4e-3
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    train_pred = np.clip(design @ weights, 0.0, 1.0)
    train_corr = 0.0
    if np.std(train_pred) > 1e-12 and np.std(y) > 1e-12:
        train_corr = float(np.corrcoef(train_pred, y)[0, 1])
    return {
        "feature_names": LEARNED_REPAIR_POLICY_FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "weights": weights.tolist(),
        "selection_score_blend": {
            "ridge_utility": 0.65,
            "learned_identity_reward": 0.15,
            "normalized_observable_repair": 0.20,
        },
        "ridge": ridge,
        "n_train_candidates": int(len(labels)),
        "train_mae": float(np.mean(np.abs(train_pred - y))),
        "train_correlation": train_corr,
    }


def _predict_learned_repair_policy(
    candidates,
    learned_scores: dict[str, np.ndarray],
    observable_scores: np.ndarray,
    policy: dict[str, object],
) -> np.ndarray:
    weights = np.asarray(policy["weights"], dtype=float)
    feature_mean = np.asarray(policy["feature_mean"], dtype=float)
    feature_scale = np.asarray(policy["feature_scale"], dtype=float)
    rows = [
        _learned_repair_policy_features(candidate, learned_scores, idx, float(observable_scores[idx]))
        for idx, candidate in enumerate(candidates)
    ]
    x = np.vstack(rows)
    z = (x - feature_mean) / feature_scale
    ridge_utility = np.clip(weights[0] + z @ weights[1:], 0.0, 1.0)
    blend = policy.get("selection_score_blend", {})
    ridge_weight = float(blend.get("ridge_utility", 1.0))
    identity_weight = float(blend.get("learned_identity_reward", 0.0))
    observable_weight = float(blend.get("normalized_observable_repair", 0.0))
    score = (
        ridge_weight * ridge_utility
        + identity_weight * np.asarray(learned_scores["learned_identity_reward"], dtype=float)
        + observable_weight * _minmax_normalize(observable_scores)
    )
    return np.clip(score, 0.0, 1.0)


def _run_learned_repair_policy_panel(
    generator: ObjectCentricFutureGenerator,
    learned_model,
    pilot_calibrator: dict[str, object],
    train_seeds: list[int],
    eval_seeds: list[int],
    n: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    policy = _fit_learned_repair_policy(generator, learned_model, train_seeds=train_seeds, n=n)
    rows: list[dict[str, float | int | str]] = []
    for seed in eval_seeds:
        for (
            variant_label,
            n_objects,
            occlusion,
            hidden_property,
            crossing,
            generator_scenario,
            target_id,
        ) in SYNTHETIC_BENCHMARK_VARIANTS:
            scene = make_scene(
                seed=720_000 + seed * 2_003 + target_id * 37 + n_objects,
                n_objects=n_objects,
                occlusion=occlusion,
                hidden_property=hidden_property,
                crossing=crossing,
            )
            if target_id != scene.target_id:
                scene = retarget_scene(scene, target_id=target_id)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=generator_scenario,
                seed=731_000 + seed * 991 + len(variant_label),
            )
            learned_scores = learned_candidate_scores(learned_model, candidates, scene)
            observable_scores = np.asarray(
                [
                    observable_repair_score(candidate, scene, seed=seed + n)
                    for candidate in candidates
                ],
                dtype=float,
            )
            policy_scores = _predict_learned_repair_policy(candidates, learned_scores, observable_scores, policy)
            pilot_scores = np.asarray(
                [pilot_calibrated_score(candidate, pilot_calibrator) for candidate in candidates],
                dtype=float,
            )
            for selector_name in LEARNED_REPAIR_POLICY_SELECTORS:
                if selector_name in {"learned_reward", "learned_identity_reward"}:
                    selected = _select_by_scores_with_label(
                        candidates,
                        learned_scores[selector_name],
                        seed=seed + n + len(variant_label),
                        label=selector_name,
                    )
                elif selector_name == "pilot_calibrated":
                    selected = _select_by_scores_with_label(
                        candidates,
                        pilot_scores,
                        seed=seed + n + len(variant_label),
                        label=selector_name,
                    )
                elif selector_name == "learned_repair_policy":
                    selected = _select_by_scores_with_label(
                        candidates,
                        policy_scores,
                        seed=seed + n + len(variant_label),
                        label=selector_name,
                    )
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "AB_learned_repair_policy_transfer",
                    variant_label,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "suite_variant": variant_label,
                        "n_objects": int(n_objects),
                        "occlusion_flag": int(occlusion),
                        "hidden_property_flag": int(hidden_property),
                        "crossing_flag": int(crossing),
                        "generator_scenario": generator_scenario,
                        "target_id": int(scene.target_id),
                        "learned_reward_mean": float(np.mean(learned_scores["learned_reward"])),
                        "learned_identity_alignment_mean": float(
                            np.mean(learned_scores["learned_identity_alignment"])
                        ),
                        "learned_property_confidence_mean": float(
                            np.mean(learned_scores["learned_property_confidence"])
                        ),
                        "learned_repair_policy_score_mean": float(np.mean(policy_scores)),
                        "learned_repair_policy_train_candidates": int(policy["n_train_candidates"]),
                        "learned_repair_policy_train_mae": float(policy["train_mae"]),
                        "learned_repair_policy_train_correlation": float(policy["train_correlation"]),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows), policy


def _run_synthetic_benchmark_panel(
    generator: ObjectCentricFutureGenerator,
    benchmark_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in benchmark_seeds:
        for (
            variant_label,
            n_objects,
            occlusion,
            hidden_property,
            crossing,
            generator_scenario,
            target_id,
        ) in SYNTHETIC_BENCHMARK_VARIANTS:
            scene = make_scene(
                seed=700_000 + seed * 37 + len(variant_label),
                n_objects=n_objects,
                occlusion=occlusion,
                hidden_property=hidden_property,
                crossing=crossing,
            )
            if target_id != scene.target_id:
                scene = retarget_scene(scene, target_id=target_id)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=generator_scenario,
                seed=711_000 + seed * 997 + len(variant_label),
            )
            for selector_name in SYNTHETIC_BENCHMARK_SELECTORS:
                selected = SELECTORS[selector_name](candidates, scene, seed=seed + n + len(variant_label))
                record = selection_record(
                    "Z_synthetic_task_suite",
                    variant_label,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "suite_variant": variant_label,
                        "n_objects": int(n_objects),
                        "occlusion_flag": int(occlusion),
                        "hidden_property_flag": int(hidden_property),
                        "crossing_flag": int(crossing),
                        "generator_scenario": generator_scenario,
                        "target_id": int(scene.target_id),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def _record_from_existing(
    source: pd.Series,
    selector: str,
    high_n: int,
    selected_n: int,
    gate_action: str,
    delegated_selector: str,
    gate_summary: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    metric_cols = [
        "selected_candidate_id",
        "selected_object_score",
        "selected_real_utility",
        "identity_error",
        "swap_rate",
        "merge_split_rate",
        "property_error",
        "property_entropy",
        "occlusion_error",
        "object_real_gap",
        "candidate_mean_score",
        "candidate_mean_real_utility",
        "candidate_best_real_utility",
        "regret",
        "oracle_gap",
        "upper_tail_rank_correlation",
    ]
    row: dict[str, float | int | str] = {
        "experiment": "AA_deployment_gate_policy",
        "scenario": str(source["scenario"]),
        "selector": selector,
        "N": int(high_n),
        "seed": int(source["seed"]),
        "selected_N": int(selected_n),
        "gate_action": gate_action,
        "delegated_selector": delegated_selector,
        "gate_identity_error": float(gate_summary["identity_error"]),
        "gate_object_real_gap": float(gate_summary["object_real_gap"]),
        "gate_property_entropy": float(gate_summary["property_entropy"]),
        "gate_repair_gain": float(gate_summary["repair_gain"]),
    }
    for col in metric_cols:
        value = source[col]
        row[col] = int(value) if col == "selected_candidate_id" else float(value)
    return row


def _run_deployment_policy_panel(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if seed_df.empty:
        return pd.DataFrame(rows)
    high_n = int(seed_df["N"].max())
    ns = sorted(int(n) for n in seed_df["N"].unique())
    early_candidates = [n for n in ns if n <= 4]
    early_n = max(early_candidates) if early_candidates else ns[0]
    for (scenario, seed), group in seed_df.groupby(["scenario", "seed"], sort=True):
        raw_high = group[(group["selector"] == "raw") & (group["N"] == high_n)]
        combined_high = group[(group["selector"] == "combined_repair") & (group["N"] == high_n)]
        raw_early = group[(group["selector"] == "raw") & (group["N"] == early_n)]
        oracle_high = group[(group["selector"] == "oracle") & (group["N"] == high_n)]
        if raw_high.empty or combined_high.empty or raw_early.empty or oracle_high.empty:
            continue
        raw_high_row = raw_high.iloc[0]
        combined_high_row = combined_high.iloc[0]
        gate_summary = {
            "N": high_n,
            "identity_error": float(raw_high_row["identity_error"]),
            "object_real_gap": float(raw_high_row["object_real_gap"]),
            "property_entropy": float(raw_high_row["property_entropy"]),
            "repair_gain": float(
                combined_high_row["selected_real_utility"] - raw_high_row["selected_real_utility"]
            ),
        }
        action = conservative_selected_tail_stop_rule(gate_summary)
        delegated_selector = DEPLOYMENT_POLICY_SELECTOR_FOR_ACTION[action]
        selected_n = early_n if action == "stop_early" else high_n
        delegated = group[(group["selector"] == delegated_selector) & (group["N"] == selected_n)]
        if delegated.empty:
            continue
        rows.append(
            _record_from_existing(
                raw_high_row,
                "raw_high_n",
                high_n,
                high_n,
                "baseline_raw_high_n",
                "raw",
                gate_summary,
            )
        )
        rows.append(
            _record_from_existing(
                raw_early.iloc[0],
                "stop_early_raw",
                high_n,
                early_n,
                "stop_early",
                "raw",
                gate_summary,
            )
        )
        rows.append(
            _record_from_existing(
                delegated.iloc[0],
                "gate_policy",
                high_n,
                selected_n,
                action,
                delegated_selector,
                gate_summary,
            )
        )
        rows.append(
            _record_from_existing(
                oracle_high.iloc[0],
                "oracle",
                high_n,
                high_n,
                "oracle_reference",
                "oracle",
                gate_summary,
            )
        )
    return pd.DataFrame(rows)


def _pilot_training_candidates(
    generator: ObjectCentricFutureGenerator,
    train_seeds: list[int],
    n: int,
) -> list:
    candidates = []
    for seed in train_seeds:
        for scenario in STRESS_SCENARIOS:
            scene = _scene_for_scenario(300_000 + seed, scenario)
            candidates.extend(
                generator.generate_candidates(
                    scene,
                    n=n,
                    scenario=scenario,
                    seed=331_777 + seed * 997 + len(scenario),
                )
            )
    return candidates


def _pilot_eval_specs(seed: int):
    domain_scene, domain_generator_scenario, n_objects, occlusion, hidden_property, crossing = _domain_randomized_scene(
        40_000 + seed
    )
    return [
        (
            "raw_heldout",
            _scene_for_scenario(340_000 + seed, "raw"),
            "raw",
            {"n_objects": 4, "occlusion_flag": 1, "hidden_property_flag": 1, "crossing_flag": 1},
        ),
        (
            "domain_randomized_heldout",
            domain_scene,
            domain_generator_scenario,
            {
                "n_objects": n_objects,
                "occlusion_flag": int(occlusion),
                "hidden_property_flag": int(hidden_property),
                "crossing_flag": int(crossing),
            },
        ),
        (
            "target_id_1_heldout",
            retarget_scene(
                make_scene(
                    seed=360_000 + seed,
                    n_objects=4,
                    occlusion=True,
                    hidden_property=True,
                    crossing=True,
                ),
                target_id=1,
            ),
            "raw",
            {"n_objects": 4, "occlusion_flag": 1, "hidden_property_flag": 1, "crossing_flag": 1},
        ),
    ]


def _run_pilot_calibration_panel(
    generator: ObjectCentricFutureGenerator,
    train_seeds: list[int],
    eval_seeds: list[int],
    n: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_candidates = _pilot_training_candidates(generator, train_seeds=train_seeds, n=n)
    calibrator = fit_pilot_calibrator(train_candidates, ridge=2e-3)
    rows: list[dict[str, float | int | str]] = []
    for seed in eval_seeds:
        for scenario_label, scene, generator_scenario, flags in _pilot_eval_specs(seed):
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=generator_scenario,
                seed=349_999 + seed * 881 + len(scenario_label),
            )
            pilot_scores = np.asarray([pilot_calibrated_score(candidate, calibrator) for candidate in candidates])
            for selector_name in PILOT_CALIBRATION_SELECTORS:
                if selector_name == "pilot_calibrated":
                    selected = _select_by_scores_with_label(candidates, pilot_scores, seed=seed + n, label=selector_name)
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "R_pilot_label_calibration",
                    scenario_label,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "pilot_train_candidates": int(calibrator["n_train_candidates"]),
                        "pilot_train_mae": float(calibrator["train_mae"]),
                        "pilot_train_correlation": float(calibrator["train_correlation"]),
                        "generator_scenario": generator_scenario,
                        **flags,
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows), calibrator


def _run_pilot_budget_panel(
    generator: ObjectCentricFutureGenerator,
    train_seeds: list[int],
    eval_seeds: list[int],
    n: int,
    budgets: list[int],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_candidates = _pilot_training_candidates(generator, train_seeds=train_seeds, n=n)
    rng = np.random.default_rng(540_001)
    order = rng.permutation(len(train_candidates))
    rows: list[dict[str, float | int | str]] = []
    calibrators: list[dict[str, object]] = []
    eval_cases = []
    for seed in eval_seeds:
        for scenario_label, scene, generator_scenario, flags in _pilot_eval_specs(80_000 + seed):
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=generator_scenario,
                seed=547_999 + seed * 907 + len(scenario_label),
            )
            eval_cases.append((seed, scenario_label, scene, generator_scenario, flags, candidates))
    for budget in budgets:
        budget_n = int(min(budget, len(train_candidates)))
        subset = [train_candidates[int(idx)] for idx in order[:budget_n]]
        calibrator = empty_pilot_calibrator() if budget_n == 0 else fit_pilot_calibrator(subset, ridge=5e-3)
        calibrators.append({"pilot_label_budget": budget_n, "calibrator": calibrator})
        for seed, scenario_label, scene, generator_scenario, flags, candidates in eval_cases:
            pilot_scores = np.asarray([pilot_calibrated_score(candidate, calibrator) for candidate in candidates])
            for selector_name in PILOT_CALIBRATION_SELECTORS:
                if selector_name == "pilot_calibrated":
                    selected = _select_by_scores_with_label(
                        candidates,
                        pilot_scores,
                        seed=seed + n + budget_n,
                        label=selector_name,
                    )
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "W_pilot_label_budget",
                    scenario_label,
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "pilot_label_budget": budget_n,
                        "pilot_train_candidates": int(calibrator["n_train_candidates"]),
                        "pilot_train_mae": float(calibrator["train_mae"]),
                        "pilot_train_correlation": float(calibrator["train_correlation"]),
                        "generator_scenario": generator_scenario,
                        **flags,
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows), calibrators


def _run_leave_one_failure_out_panel(
    generator: ObjectCentricFutureGenerator,
    train_seeds: list[int],
    eval_seeds: list[int],
    n: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, float | int | str]] = []
    calibrator_summaries: list[dict[str, object]] = []
    for heldout_scenario in STRESS_SCENARIOS:
        train_scenarios = [scenario for scenario in STRESS_SCENARIOS if scenario != heldout_scenario]
        train_candidates = []
        for seed in train_seeds:
            for scenario in train_scenarios:
                scene = _scene_for_scenario(390_000 + seed, scenario)
                train_candidates.extend(
                    generator.generate_candidates(
                        scene,
                        n=n,
                        scenario=scenario,
                        seed=407_311 + seed * 977 + len(scenario),
                    )
                )
        calibrator = fit_pilot_calibrator(train_candidates, ridge=2e-3)
        calibrator_summaries.append(
            {
                "heldout_scenario": heldout_scenario,
                "train_scenarios": train_scenarios,
                "calibrator": calibrator,
            }
        )
        for seed in eval_seeds:
            scene = _scene_for_scenario(430_000 + seed, heldout_scenario)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=heldout_scenario,
                seed=419_999 + seed * 887 + len(heldout_scenario),
            )
            pilot_scores = np.asarray([pilot_calibrated_score(candidate, calibrator) for candidate in candidates])
            for selector_name in PILOT_CALIBRATION_SELECTORS:
                if selector_name == "pilot_calibrated":
                    selected = _select_by_scores_with_label(candidates, pilot_scores, seed=seed + n, label=selector_name)
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "S_leave_one_failure_out",
                    f"heldout_{heldout_scenario}",
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "heldout_scenario": heldout_scenario,
                        "train_scenarios": "|".join(train_scenarios),
                        "pilot_train_candidates": int(calibrator["n_train_candidates"]),
                        "pilot_train_mae": float(calibrator["train_mae"]),
                        "pilot_train_correlation": float(calibrator["train_correlation"]),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows), calibrator_summaries


def _run_noisy_probe_panel(
    generator: ObjectCentricFutureGenerator,
    probe_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in probe_seeds:
        scene = _scene_for_scenario(470_000 + seed, "raw")
        candidates = generator.generate_candidates(
            scene,
            n=n,
            scenario="raw",
            seed=463_997 + seed * 971 + n,
        )
        for reliability in NOISY_PROBE_RELIABILITIES:
            for selector_name in NOISY_PROBE_SELECTORS:
                if selector_name == "noisy_probe_repair":
                    selected = _select_noisy_probe_repair(
                        candidates,
                        scene,
                        reliability=reliability,
                        seed=seed + n + int(reliability * 1000),
                    )
                else:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                record = selection_record(
                    "T_noisy_probe_reliability",
                    "noisy_probe",
                    selector_name,
                    n,
                    seed,
                    selected,
                    candidates,
                )
                record.update(
                    {
                        "probe_reliability": float(reliability),
                        "probe_noise_rate": float(1.0 - reliability),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def _run_probe_cost_panel(
    generator: ObjectCentricFutureGenerator,
    probe_cost_seeds: list[int],
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in probe_cost_seeds:
        for scenario in PROBE_COST_SCENARIOS:
            scene = _scene_for_scenario(510_000 + seed, scenario)
            candidates = generator.generate_candidates(
                scene,
                n=n,
                scenario=scenario,
                seed=509_003 + seed * 991 + len(scenario),
            )
            selected_by_selector = {
                selector_name: SELECTORS[selector_name](candidates, scene, seed=seed + n)
                for selector_name in PROBE_COST_SELECTORS
            }
            for probe_cost in PROBE_COSTS:
                for selector_name, selected in selected_by_selector.items():
                    record = selection_record(
                        "V_probe_cost_sensitivity",
                        scenario,
                        selector_name,
                        n,
                        seed,
                        selected,
                        candidates,
                    )
                    gross_utility = float(record["selected_real_utility"])
                    incurred_cost = float(probe_cost) if selector_name in PROBE_USING_SELECTORS else 0.0
                    net_utility = float(np.clip(gross_utility - incurred_cost, 0.0, 1.0))
                    record.update(
                        {
                            "gross_selected_real_utility": gross_utility,
                            "probe_cost": float(probe_cost),
                            "incurred_probe_cost": incurred_cost,
                            "selected_real_utility": net_utility,
                            "object_real_gap": float(record["selected_object_score"] - net_utility),
                            "regret": float(record["candidate_best_real_utility"] - net_utility),
                            "oracle_gap": float(record["candidate_best_real_utility"] - net_utility),
                            "probe_cost_applied": int(selector_name in PROBE_USING_SELECTORS),
                        }
                    )
                    rows.append(record)
    return pd.DataFrame(rows)


REPAIR_SPLIT_SEEDS = [0, 1, 2, 3, 4]
REPAIR_GRID = [
    {
        "config_id": "ridge_lcb_no_penalty",
        "ridge": 2e-3,
        "uncertainty_penalty": 0.0,
        "conformal_confidence": 0.90,
        "gate_threshold": 0.0,
        "feature_set": "pilot_observable_v1",
    },
    {
        "config_id": "ridge_lcb_mild_penalty",
        "ridge": 2e-3,
        "uncertainty_penalty": 0.18,
        "conformal_confidence": 0.90,
        "gate_threshold": 0.0,
        "feature_set": "pilot_observable_v1",
    },
    {
        "config_id": "ridge_lcb_strong_penalty",
        "ridge": 5e-3,
        "uncertainty_penalty": 0.32,
        "conformal_confidence": 0.90,
        "gate_threshold": 0.0,
        "feature_set": "pilot_observable_v1",
    },
]
SUPPORT_SELECTOR_GRID = ["observable_repair", "targeted_probe"]


def _repair_condition_catalog(mode: str) -> list[dict[str, int | str]]:
    condition_seeds = list(range(4)) if mode == "smoke" else list(range(12))
    rows: list[dict[str, int | str]] = []
    for condition_seed in condition_seeds:
        for scenario_idx, scenario in enumerate(STRESS_SCENARIOS):
            scene_seed = 610_000 + condition_seed * 31 + scenario_idx
            rows.append(
                {
                    "condition_id": f"{scenario}:{condition_seed}",
                    "condition_seed": int(condition_seed),
                    "scenario": scenario,
                    "scene_seed": int(scene_seed),
                }
            )
    return rows


def _repair_condition_splits(mode: str, split_seeds: list[int]) -> pd.DataFrame:
    conditions = _repair_condition_catalog(mode)
    split_names = ["pilot_train", "pilot_calibration", "dev", "final_test"]
    rows: list[dict[str, int | str]] = []
    for split_seed in split_seeds:
        rng = np.random.default_rng(620_000 + split_seed)
        order = rng.permutation(len(conditions))
        n_total = len(conditions)
        n_train = max(1, int(round(0.25 * n_total)))
        n_cal = max(1, int(round(0.20 * n_total)))
        n_dev = max(1, int(round(0.20 * n_total)))
        boundaries = {
            "pilot_train": set(int(idx) for idx in order[:n_train]),
            "pilot_calibration": set(int(idx) for idx in order[n_train : n_train + n_cal]),
            "dev": set(int(idx) for idx in order[n_train + n_cal : n_train + n_cal + n_dev]),
            "final_test": set(int(idx) for idx in order[n_train + n_cal + n_dev :]),
        }
        for split_name in split_names:
            for idx in sorted(boundaries[split_name]):
                rows.append({"split_seed": int(split_seed), "split": split_name, **conditions[idx]})
    return pd.DataFrame(rows)


def _scene_for_repair_condition(row: pd.Series):
    return _scene_for_scenario(int(row["scene_seed"]), str(row["scenario"]))


def _condition_candidates(
    generator: ObjectCentricFutureGenerator,
    condition: pd.Series,
    n: int,
    cache: dict[tuple[int, str], tuple[object, list[object]]] | None = None,
) -> tuple[object, list[object]]:
    key = (int(n), str(condition["condition_id"]))
    if cache is not None and key in cache:
        return cache[key]
    scenario = str(condition["scenario"])
    scene = _scene_for_repair_condition(condition)
    seed = 631_000 + int(condition["condition_seed"]) * 997 + n * 37 + len(scenario)
    candidates = generator.generate_candidates(scene, n=n, scenario=scenario, seed=seed)
    if cache is not None:
        cache[key] = (scene, candidates)
    return scene, candidates


def _candidate_uncertainty(candidate) -> float:
    return float(
        np.clip(
            0.36 * _diagnostic(candidate, "identity_instability", 0.5)
            + 0.34 * _diagnostic(candidate, "merge_evidence")
            + 0.30 * float(candidate.property_entropy),
            0.0,
            1.0,
        )
    )


def _pilot_train_subset(
    generator: ObjectCentricFutureGenerator,
    split_conditions: pd.DataFrame,
    n: int,
    budget: int,
    ridge: float,
    split_seed: int,
    cache: dict[tuple[int, str], tuple[object, list[object]]] | None = None,
) -> dict[str, object]:
    if budget <= 0:
        return empty_pilot_calibrator()
    candidates = []
    for _, condition in split_conditions[split_conditions["split"] == "pilot_train"].iterrows():
        _, condition_candidates = _condition_candidates(generator, condition, n=n, cache=cache)
        candidates.extend(condition_candidates)
    rng = np.random.default_rng(640_000 + split_seed + int(budget))
    order = rng.permutation(len(candidates))
    subset = [candidates[int(idx)] for idx in order[: min(int(budget), len(candidates))]]
    return fit_pilot_calibrator(subset, ridge=ridge) if subset else empty_pilot_calibrator()


def _conformal_residual_quantile(
    generator: ObjectCentricFutureGenerator,
    split_conditions: pd.DataFrame,
    n: int,
    calibrator: dict[str, object],
    confidence: float,
    cache: dict[tuple[int, str], tuple[object, list[object]]] | None = None,
) -> float:
    residuals: list[float] = []
    for _, condition in split_conditions[split_conditions["split"] == "pilot_calibration"].iterrows():
        _, candidates = _condition_candidates(generator, condition, n=n, cache=cache)
        for candidate in candidates:
            residuals.append(max(0.0, pilot_calibrated_score(candidate, calibrator) - float(candidate.real_utility)))
    if not residuals:
        return 0.0
    return float(np.quantile(np.asarray(residuals, dtype=float), float(confidence)))


def _pilot_lcb_scores(
    candidates,
    calibrator: dict[str, object],
    residual_q: float,
    uncertainty_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray([pilot_calibrated_score(candidate, calibrator) for candidate in candidates], dtype=float)
    uncertainty = np.asarray([_candidate_uncertainty(candidate) for candidate in candidates], dtype=float)
    lcbs = np.clip(predictions - residual_q - float(uncertainty_penalty) * uncertainty, 0.0, 1.0)
    return predictions, lcbs, uncertainty


def _mean_dev_gap_closure(
    generator: ObjectCentricFutureGenerator,
    split_conditions: pd.DataFrame,
    n: int,
    selector_name: str,
    split_seed: int,
    calibrator: dict[str, object] | None = None,
    residual_q: float = 0.0,
    uncertainty_penalty: float = 0.0,
    cache: dict[tuple[int, str], tuple[object, list[object]]] | None = None,
) -> float:
    values: list[float] = []
    dev = split_conditions[split_conditions["split"] == "dev"]
    for _, condition in dev.iterrows():
        scene, candidates = _condition_candidates(generator, condition, n=n, cache=cache)
        raw = SELECTORS["raw"](candidates, scene, seed=split_seed + n)
        oracle = SELECTORS["oracle"](candidates, scene, seed=split_seed + n)
        if selector_name == "pilot_calibrated":
            assert calibrator is not None
            _, lcbs, _ = _pilot_lcb_scores(
                candidates,
                calibrator=calibrator,
                residual_q=residual_q,
                uncertainty_penalty=uncertainty_penalty,
            )
            selected = _select_by_scores_with_label(candidates, lcbs, seed=split_seed + n, label="pilot_calibrated")
        else:
            selected = SELECTORS[selector_name](candidates, scene, seed=split_seed + n)
        denominator = float(oracle.real_utility - raw.real_utility)
        if abs(denominator) > 1e-12:
            values.append(float((selected.real_utility - raw.real_utility) / denominator))
    return float(np.mean(values)) if values else float("-inf")


def _run_final_test_repair_panel(
    generator: ObjectCentricFutureGenerator,
    mode: str,
    ns: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    max_n = max(ns)
    eval_ns = [n for n in [32, 64] if n <= max_n]
    if not eval_ns:
        eval_ns = [max_n]
    split_df = _repair_condition_splits(mode, REPAIR_SPLIT_SEEDS)
    final_rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, object]] = []
    selected_configs: list[dict[str, object]] = []
    candidate_cache: dict[tuple[int, str], tuple[object, list[object]]] = {}
    for split_seed in REPAIR_SPLIT_SEEDS:
        split_conditions = split_df[split_df["split_seed"] == split_seed].copy()
        for n in eval_ns:
            for budget in PILOT_BUDGETS:
                dev_scores: list[tuple[float, dict[str, object], dict[str, object], float]] = []
                for config in REPAIR_GRID:
                    calibrator = _pilot_train_subset(
                        generator,
                        split_conditions,
                        n=n,
                        budget=budget,
                        ridge=float(config["ridge"]),
                        split_seed=split_seed,
                        cache=candidate_cache,
                    )
                    residual_q = _conformal_residual_quantile(
                        generator,
                        split_conditions,
                        n=n,
                        calibrator=calibrator,
                        confidence=float(config["conformal_confidence"]),
                        cache=candidate_cache,
                    )
                    dev_score = _mean_dev_gap_closure(
                        generator,
                        split_conditions,
                        n=n,
                        selector_name="pilot_calibrated",
                        split_seed=split_seed,
                        calibrator=calibrator,
                        residual_q=residual_q,
                        uncertainty_penalty=float(config["uncertainty_penalty"]),
                        cache=candidate_cache,
                    )
                    dev_scores.append((dev_score, config, calibrator, residual_q))
                    selection_rows.append(
                        {
                            "split_seed": int(split_seed),
                            "N": int(n),
                            "repair_budget": int(budget),
                            "selector": "pilot_calibrated",
                            "repair_tier": "deployable_no_leak",
                            "config_id": str(config["config_id"]),
                            "ridge": float(config["ridge"]),
                            "uncertainty_penalty": float(config["uncertainty_penalty"]),
                            "conformal_confidence": float(config["conformal_confidence"]),
                            "gate_threshold": float(config["gate_threshold"]),
                            "feature_set": str(config["feature_set"]),
                            "dev_gap_closure_mean": float(dev_score),
                            "selected": 0,
                            "hyperparameter_source": "dev_condition_grid",
                            "final_test": 0,
                        }
                    )
                best_score, best_config, best_calibrator, best_residual_q = max(dev_scores, key=lambda item: item[0])
                for row in selection_rows:
                    if (
                        row["split_seed"] == split_seed
                        and row["N"] == n
                        and row["repair_budget"] == budget
                        and row["selector"] == "pilot_calibrated"
                        and row["config_id"] == best_config["config_id"]
                    ):
                        row["selected"] = 1
                support_scores = [
                    (
                        _mean_dev_gap_closure(
                            generator,
                            split_conditions,
                            n=n,
                            selector_name=support_selector,
                            split_seed=split_seed,
                            cache=candidate_cache,
                        ),
                        support_selector,
                    )
                    for support_selector in SUPPORT_SELECTOR_GRID
                ]
                support_dev_score, support_selector = max(support_scores, key=lambda item: item[0])
                selection_rows.append(
                    {
                        "split_seed": int(split_seed),
                        "N": int(n),
                        "repair_budget": int(budget),
                        "selector": support_selector,
                        "repair_tier": "support_covered",
                        "config_id": f"support_{support_selector}",
                        "ridge": float("nan"),
                        "uncertainty_penalty": float("nan"),
                        "conformal_confidence": float("nan"),
                        "gate_threshold": float("nan"),
                        "feature_set": "support_probe_proxy",
                        "dev_gap_closure_mean": float(support_dev_score),
                        "selected": 1,
                        "hyperparameter_source": "dev_condition_grid",
                        "final_test": 0,
                    }
                )
                selected_configs.append(
                    {
                        "split_seed": int(split_seed),
                        "N": int(n),
                        "repair_budget": int(budget),
                        "deployable_config": dict(best_config),
                        "deployable_dev_gap_closure_mean": float(best_score),
                        "support_selector": support_selector,
                        "support_dev_gap_closure_mean": float(support_dev_score),
                    }
                )
                final_conditions = split_conditions[split_conditions["split"] == "final_test"]
                for _, condition in final_conditions.iterrows():
                    scene, candidates = _condition_candidates(generator, condition, n=n, cache=candidate_cache)
                    raw = SELECTORS["raw"](candidates, scene, seed=split_seed + n)
                    oracle = SELECTORS["oracle"](candidates, scene, seed=split_seed + n)
                    predictions, lcbs, uncertainty = _pilot_lcb_scores(
                        candidates,
                        calibrator=best_calibrator,
                        residual_q=best_residual_q,
                        uncertainty_penalty=float(best_config["uncertainty_penalty"]),
                    )
                    pilot_selected = _select_by_scores_with_label(
                        candidates,
                        lcbs,
                        seed=split_seed + n + budget,
                        label="pilot_calibrated",
                    )
                    support_selected = SELECTORS[support_selector](candidates, scene, seed=split_seed + n)
                    oracle_feature_selected = SELECTORS["combined_repair"](candidates, scene, seed=split_seed + n)
                    all_labeled_selected = SELECTORS["oracle"](candidates, scene, seed=split_seed + n)
                    selected_ids = {int(pilot_selected.candidate_id)}
                    q_edges = np.quantile(uncertainty, [1 / 3, 2 / 3]) if len(uncertainty) >= 3 else [0.33, 0.66]
                    for candidate, prediction, lcb, unc in zip(candidates, predictions, lcbs, uncertainty):
                        if unc <= q_edges[0]:
                            risk_bin = "low"
                        elif unc <= q_edges[1]:
                            risk_bin = "medium"
                        else:
                            risk_bin = "high"
                        calibration_rows.append(
                            {
                                "split_seed": int(split_seed),
                                "condition_id": str(condition["condition_id"]),
                                "scenario": str(condition["scenario"]),
                                "N": int(n),
                                "repair_budget": int(budget),
                                "selector": "pilot_calibrated",
                                "repair_tier": "deployable_no_leak",
                                "candidate_id": int(candidate.candidate_id),
                                "predicted_utility": float(prediction),
                                "lcb_utility": float(lcb),
                                "real_utility": float(candidate.real_utility),
                                "lcb_covered": int(float(candidate.real_utility) >= float(lcb)),
                                "selected_tail": int(int(candidate.candidate_id) in selected_ids),
                                "risk_bin": risk_bin,
                                "uncertainty": float(unc),
                                "violation": int(float(candidate.real_utility) < float(lcb)),
                                "adaptive_gate_regret": 0.0,
                                "block_accuracy": float("nan"),
                                "final_test": 1,
                                "uses_real_utility_features": False,
                                "uses_hidden_features": False,
                                "hyperparameter_source": "dev_condition_grid",
                            }
                        )
                    selector_records = [
                        ("raw", raw),
                        ("pilot_calibrated", pilot_selected),
                        (support_selector, support_selected),
                        ("combined_repair", oracle_feature_selected),
                        ("repair_all_candidates_labeled_oracle", all_labeled_selected),
                    ]
                    for selector_name, selected in selector_records:
                        record = selection_record(
                            "AC_nested_final_test_repair",
                            str(condition["scenario"]),
                            selector_name,
                            n,
                            int(condition["condition_seed"]),
                            selected,
                            candidates,
                        )
                        record.update(
                            {
                                "split_seed": int(split_seed),
                                "condition_id": str(condition["condition_id"]),
                                "condition_split": "final_test",
                                "repair_budget": int(budget),
                                "raw_selected_real_utility": float(raw.real_utility),
                                "oracle_selected_real_utility": float(oracle.real_utility),
                                "gap_closure": float(
                                    (selected.real_utility - raw.real_utility) / (oracle.real_utility - raw.real_utility)
                                )
                                if abs(float(oracle.real_utility - raw.real_utility)) > 1e-12
                                else float("nan"),
                                "deployable_config_id": str(best_config["config_id"]),
                                "support_selector": support_selector,
                                "conformal_residual_q": float(best_residual_q),
                                "conformal_confidence": float(best_config["conformal_confidence"]),
                                "final_test": 1,
                            }
                        )
                        final_rows.append(record)
    final_df = add_repair_metadata(pd.DataFrame(final_rows), final_test=True)
    robustness_df = repair_robustness_by_split(final_df)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_summary = calibration_diagnostics_summary(calibration_df)
    selection_df = pd.DataFrame(selection_rows)
    return final_df, robustness_df, calibration_summary, split_df, selection_df, selected_configs


def _run_unidentifiable_negative_control(
    generator: ObjectCentricFutureGenerator,
    n: int,
    split_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split_seed in split_seeds:
        scene = _scene_for_scenario(690_000 + split_seed, "raw")
        base = generator.generate_candidates(scene, n=1, scenario="raw", seed=691_000 + split_seed)[0]
        candidates = []
        for idx in range(n):
            utility = 0.92 if idx % 2 else 0.08
            candidates.append(
                replace(
                    base,
                    candidate_id=idx,
                    score=1.0,
                    real_utility=float(utility),
                    hidden_property_true=float(utility),
                    object_real_gap=float(1.0 - utility),
                )
            )
        raw = SELECTORS["raw"](candidates, scene, seed=split_seed + n)
        oracle = SELECTORS["oracle"](candidates, scene, seed=split_seed + n)
        gate = hidden_mode_unidentifiable_gate(candidates, n=n)
        action = conservative_selected_tail_stop_rule({"N": n, **gate})
        block_accuracy = float(action == "block_high_n")
        rows.append(
            {
                "contrast": "hidden_mode_unidentifiable",
                "split_seed": int(split_seed),
                "N": int(n),
                "selector": "adaptive_gate",
                "repair_tier": "deployable_no_leak",
                "uses_real_utility_features": False,
                "uses_hidden_features": False,
                "hyperparameter_source": "fixed_predeclared",
                "selected_real_utility_mean": float(raw.real_utility),
                "oracle_selected_real_utility_mean": float(oracle.real_utility),
                "oracle_gap_mean": float(oracle.real_utility - raw.real_utility),
                "identity_error_mean": float(raw.identity_error),
                "gate_action": action,
                "gate_reason": str(gate["gate_reason"]),
                "observable_feature_collision_rate": float(gate["observable_feature_collision_rate"]),
                "block_accuracy": block_accuracy,
                "tail_rank_failure": int(action == "block_high_n"),
                "final_test": 1,
            }
        )
    return pd.DataFrame(rows)


def _learned_generalization_diagnostics(
    learned_row: dict[str, float],
    learned_repair_policy_metrics: pd.DataFrame,
    repair_robustness: pd.DataFrame,
) -> pd.DataFrame:
    learned_policy = (
        learned_repair_policy_metrics[learned_repair_policy_metrics["selector"] == "learned_repair_policy"]
        if not learned_repair_policy_metrics.empty
        else pd.DataFrame()
    )
    deployable_repair = (
        repair_robustness[
            (repair_robustness["repair_tier"] == "deployable_no_leak")
            & (repair_robustness["selector"] == "pilot_calibrated")
        ]
        if not repair_robustness.empty
        else pd.DataFrame()
    )
    transition_mse = float(learned_row.get("transition_mse", float("nan")))
    constant_transition_mse = float(learned_row.get("constant_transition_mse", float("nan")))
    return pd.DataFrame(
        [
            {
                "evaluation_split": "heldout_final_test",
                "trajectory_mse": transition_mse,
                "final_state_error": float(np.sqrt(max(transition_mse, 0.0))) if np.isfinite(transition_mse) else float("nan"),
                "denoising_proxy_loss": transition_mse,
                "baseline_trajectory_mse": constant_transition_mse,
                "sample_diversity": float(learned_policy["selected_real_utility_mean"].std(ddof=0))
                if not learned_policy.empty
                else float("nan"),
                "rank_correlation": float(learned_row.get("reward_correlation", float("nan"))),
                "selected_tail_calibration_error": float(learned_policy["learned_repair_policy_oracle_gap_mean"].mean())
                if not learned_policy.empty and "learned_repair_policy_oracle_gap_mean" in learned_policy
                else float("nan"),
                "repair_gap_closure": float(deployable_repair["mean_gap_closure_across_splits"].mean())
                if not deployable_repair.empty
                else float("nan"),
                "repair_tier": "deployable_no_leak",
                "final_test": 1,
            }
        ]
    )


def run(root: Path, mode: str, ns: list[int], seeds: list[int]) -> dict[str, object]:
    start = time.time()
    results = root / "results"
    tables = results / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    progress_log = results / "run_progress.log"

    def mark(stage: str) -> None:
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.time() - start:.3f}\t{mode}\t{stage}\n")

    progress_log.write_text("", encoding="utf-8")
    mark("start")

    seed_rows: list[dict[str, float | int | str]] = []
    law_rows: list[dict[str, float | int | str]] = []
    generator = ObjectCentricFutureGenerator(seed=17)
    trials = 1200 if mode == "smoke" else 15_000

    for seed in seeds:
        for scenario in SCENARIOS:
            scene = _scene_for_scenario(10_000 + seed, scenario)
            for n in ns:
                candidates = generator.generate_candidates(
                    scene,
                    n=n,
                    scenario=scenario,
                    seed=seed * 1009 + n * 37 + len(scenario),
                )
                for selector_name in SELECTOR_ORDER:
                    selected = SELECTORS[selector_name](candidates, scene, seed=seed + n)
                    seed_rows.append(selection_record("A_controlled_object_binding", scenario, selector_name, n, seed, selected, candidates))
                if scenario == "raw":
                    law_rows.append(
                        {
                            "scenario": scenario,
                            **law_validation_row(
                                [candidate.real_utility for candidate in candidates],
                                [candidate.score for candidate in candidates],
                                n=n,
                                trials=trials,
                                seed=seed + 42,
                            ),
                        }
                    )

    seed_df = pd.DataFrame(seed_rows)
    main = aggregate_seed_metrics(seed_df)
    mark("main_controlled_complete")
    repair_metrics = main[
        main["selector"].isin(
            [
                "raw",
                "identity_consistent",
                "property_calibrated",
                "targeted_probe",
                "combined_repair",
                "observable_repair",
                "random",
                "oracle",
            ]
        )
    ].copy()
    paired_effects = paired_selector_effects(seed_df)
    deployment_policy_seed_df = _run_deployment_policy_panel(seed_df)
    deployment_policy_metrics = deployment_policy_summary(deployment_policy_seed_df)
    mark("deployment_policy_complete")
    law_df = pd.DataFrame(law_rows)
    stress_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    stress_seed_df = _run_stress_panel(generator, stress_seeds=stress_seeds, n=max(ns))
    stress_metrics = stress_summary(stress_seed_df)
    sensitivity_seeds = list(range(4)) if mode == "smoke" else list(range(24))
    sensitivity_seed_df, calibration_candidate_df = _run_sensitivity_panel(generator, sensitivity_seeds, n=max(ns), mode=mode)
    sensitivity_metrics = sensitivity_summary(sensitivity_seed_df)
    calibration_metrics = score_calibration_table(calibration_candidate_df)
    mark("stress_sensitivity_complete")
    ood_seeds = list(range(4)) if mode == "smoke" else list(range(16))
    ood_seed_df = _run_ood_panel(generator, ood_seeds, n=max(ns))
    ood_metrics = ood_summary(ood_seed_df)
    extreme_object_seeds = list(range(4)) if mode == "smoke" else list(range(24))
    extreme_object_seed_df = _run_extreme_object_count_panel(generator, extreme_object_seeds, n=max(ns))
    extreme_object_metrics = extreme_object_count_summary(extreme_object_seed_df)
    family_seeds = list(range(4)) if mode == "smoke" else list(range(16))
    family_seed_df = _run_model_family_panel(generator, family_seeds, n=max(ns))
    family_metrics = model_family_proxy_summary(family_seed_df)
    domain_seeds = list(range(6)) if mode == "smoke" else list(range(48))
    domain_seed_df = _run_domain_randomization_panel(generator, domain_seeds, n=max(ns))
    domain_metrics = domain_randomization_summary(domain_seed_df)
    counter_seeds = list(range(6)) if mode == "smoke" else list(range(48))
    counter_seed_df = _run_counterfactual_target_panel(generator, counter_seeds, n=max(ns))
    counter_metrics = counterfactual_target_summary(counter_seed_df)
    target_sweep_seeds = list(range(4)) if mode == "smoke" else list(range(48))
    target_sweep_seed_df = _run_target_identity_sweep_panel(generator, target_sweep_seeds, n=max(ns))
    target_sweep_metrics = target_identity_sweep_summary(target_sweep_seed_df)
    mark("ood_extreme_family_domain_counter_target_complete")
    pilot_train_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    pilot_eval_seeds = list(range(4)) if mode == "smoke" else list(range(48))
    pilot_seed_df, pilot_calibrator = _run_pilot_calibration_panel(
        generator,
        train_seeds=pilot_train_seeds,
        eval_seeds=pilot_eval_seeds,
        n=max(ns),
    )
    pilot_metrics = pilot_calibration_summary(pilot_seed_df)
    pilot_budget_train_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    pilot_budget_eval_seeds = list(range(4)) if mode == "smoke" else list(range(48))
    pilot_budget_seed_df, pilot_budget_calibrators = _run_pilot_budget_panel(
        generator,
        train_seeds=pilot_budget_train_seeds,
        eval_seeds=pilot_budget_eval_seeds,
        n=max(ns),
        budgets=PILOT_BUDGETS,
    )
    pilot_budget_metrics = pilot_budget_summary(pilot_budget_seed_df)
    loso_train_seeds = list(range(3)) if mode == "smoke" else list(range(24))
    loso_eval_seeds = list(range(3)) if mode == "smoke" else list(range(40))
    loso_seed_df, loso_calibrators = _run_leave_one_failure_out_panel(
        generator,
        train_seeds=loso_train_seeds,
        eval_seeds=loso_eval_seeds,
        n=max(ns),
    )
    loso_metrics = pilot_calibration_summary(loso_seed_df)
    probe_seeds = list(range(4)) if mode == "smoke" else list(range(48))
    noisy_probe_seed_df = _run_noisy_probe_panel(generator, probe_seeds=probe_seeds, n=max(ns))
    noisy_probe_metrics = noisy_probe_summary(noisy_probe_seed_df)
    probe_cost_seeds = list(range(4)) if mode == "smoke" else list(range(48))
    probe_cost_seed_df = _run_probe_cost_panel(generator, probe_cost_seeds=probe_cost_seeds, n=max(ns))
    probe_cost_metrics = probe_cost_summary(probe_cost_seed_df)
    mark("pilot_probe_panels_complete")
    learned_metrics, learned_model = train_and_evaluate(results, seed=123 if mode == "smoke" else 456)
    learned_row = learned_metrics.as_dict()
    pd.DataFrame([learned_row]).to_csv(tables / "learned_metrics.csv", index=False)
    mark("learned_model_complete")
    learned_curve = pd.read_csv(tables / "learned_learning_curve.csv")
    learned_ablation = pd.read_csv(tables / "learned_ablation.csv")
    learned_domain_shift = pd.read_csv(tables / "learned_domain_shift.csv")
    learned_selection_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    learned_selection_seed_df = _run_learned_selection_panel(
        generator,
        learned_model,
        learned_selection_seeds=learned_selection_seeds,
        n=max(ns),
    )
    learned_selection_metrics = learned_selection_summary(learned_selection_seed_df)
    learned_repair_policy_train_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    learned_repair_policy_eval_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    learned_repair_policy_seed_df, learned_repair_policy = _run_learned_repair_policy_panel(
        generator,
        learned_model,
        pilot_calibrator,
        train_seeds=learned_repair_policy_train_seeds,
        eval_seeds=learned_repair_policy_eval_seeds,
        n=max(ns),
    )
    learned_repair_policy_metrics = learned_repair_policy_summary(learned_repair_policy_seed_df)
    mark("learned_selection_repair_complete")
    synthetic_benchmark_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    synthetic_benchmark_seed_df = _run_synthetic_benchmark_panel(
        generator,
        benchmark_seeds=synthetic_benchmark_seeds,
        n=max(ns),
    )
    synthetic_benchmark_metrics = synthetic_benchmark_summary(synthetic_benchmark_seed_df)
    mark("synthetic_benchmark_complete")
    (
        repair_final_test_df,
        repair_robustness_split_df,
        calibration_diagnostics,
        repair_condition_splits,
        repair_model_selection,
        selected_repair_configs,
    ) = _run_final_test_repair_panel(generator, mode=mode, ns=ns)
    unidentifiable_negative_control = _run_unidentifiable_negative_control(
        generator,
        n=max(ns),
        split_seeds=REPAIR_SPLIT_SEEDS,
    )
    learned_generalization = _learned_generalization_diagnostics(
        learned_row,
        learned_repair_policy_metrics,
        repair_robustness_split_df,
    )
    mark("nested_repair_complete")
    bootstrap_reps = 400 if mode == "smoke" else 2000
    statistical_metrics = statistical_audit(
        seed_df,
        ood_seed_df=ood_seed_df,
        extreme_object_seed_df=extreme_object_seed_df,
        family_seed_df=family_seed_df,
        counterfactual_seed_df=counter_seed_df,
        target_sweep_seed_df=target_sweep_seed_df,
        pilot_seed_df=pilot_seed_df,
        pilot_budget_seed_df=pilot_budget_seed_df,
        leave_one_failure_seed_df=loso_seed_df,
        noisy_probe_seed_df=noisy_probe_seed_df,
        probe_cost_seed_df=probe_cost_seed_df,
        learned_selection_seed_df=learned_selection_seed_df,
        learned_repair_policy_seed_df=learned_repair_policy_seed_df,
        synthetic_benchmark_seed_df=synthetic_benchmark_seed_df,
        deployment_policy_seed_df=deployment_policy_seed_df,
        bootstrap_reps=bootstrap_reps,
        seed=240_001,
    )
    mark("statistical_audit_complete")
    ablation_metrics = repair_ablation_summary(main, paired_effects)
    observable_metrics = observable_repair_summary(main, paired_effects)
    robustness_metrics = seed_block_robustness(seed_df, block_size=2 if mode == "smoke" else 4)
    negative_control = negative_control_summary(main)

    seed_df = add_repair_metadata(seed_df)
    main = add_repair_metadata(main)
    repair_metrics = add_repair_metadata(repair_metrics)
    paired_effects = add_repair_metadata(paired_effects)
    deployment_policy_seed_df = add_repair_metadata(deployment_policy_seed_df)
    deployment_policy_metrics = add_repair_metadata(deployment_policy_metrics)
    stress_seed_df = add_repair_metadata(stress_seed_df)
    stress_metrics = add_repair_metadata(stress_metrics)
    sensitivity_seed_df = add_repair_metadata(sensitivity_seed_df)
    sensitivity_metrics = add_repair_metadata(sensitivity_metrics)
    ood_seed_df = add_repair_metadata(ood_seed_df)
    ood_metrics = add_repair_metadata(ood_metrics)
    extreme_object_seed_df = add_repair_metadata(extreme_object_seed_df)
    extreme_object_metrics = add_repair_metadata(extreme_object_metrics)
    family_seed_df = add_repair_metadata(family_seed_df)
    family_metrics = add_repair_metadata(family_metrics)
    domain_seed_df = add_repair_metadata(domain_seed_df)
    domain_metrics = add_repair_metadata(domain_metrics)
    counter_seed_df = add_repair_metadata(counter_seed_df)
    counter_metrics = add_repair_metadata(counter_metrics)
    target_sweep_seed_df = add_repair_metadata(target_sweep_seed_df)
    target_sweep_metrics = add_repair_metadata(target_sweep_metrics)
    pilot_seed_df = add_repair_metadata(pilot_seed_df)
    pilot_metrics = add_repair_metadata(pilot_metrics)
    pilot_budget_seed_df = add_repair_metadata(pilot_budget_seed_df)
    pilot_budget_metrics = add_repair_metadata(pilot_budget_metrics)
    loso_seed_df = add_repair_metadata(loso_seed_df)
    loso_metrics = add_repair_metadata(loso_metrics)
    noisy_probe_seed_df = add_repair_metadata(noisy_probe_seed_df)
    noisy_probe_metrics = add_repair_metadata(noisy_probe_metrics)
    probe_cost_seed_df = add_repair_metadata(probe_cost_seed_df)
    probe_cost_metrics = add_repair_metadata(probe_cost_metrics)
    learned_selection_seed_df = add_repair_metadata(learned_selection_seed_df)
    learned_selection_metrics = add_repair_metadata(learned_selection_metrics)
    learned_repair_policy_seed_df = add_repair_metadata(learned_repair_policy_seed_df)
    learned_repair_policy_metrics = add_repair_metadata(learned_repair_policy_metrics)
    synthetic_benchmark_seed_df = add_repair_metadata(synthetic_benchmark_seed_df)
    synthetic_benchmark_metrics = add_repair_metadata(synthetic_benchmark_metrics)

    seed_df.to_csv(tables / "seed_metrics.csv", index=False)
    main.to_csv(tables / "main_metrics.csv", index=False)
    repair_metrics.to_csv(tables / "repair_metrics.csv", index=False)
    paired_effects.to_csv(tables / "paired_effects.csv", index=False)
    deployment_policy_seed_df.to_csv(tables / "deployment_policy_seed_metrics.csv", index=False)
    deployment_policy_metrics.to_csv(tables / "deployment_policy_metrics.csv", index=False)
    ablation_metrics.to_csv(tables / "repair_ablation.csv", index=False)
    observable_metrics.to_csv(tables / "observable_repair_metrics.csv", index=False)
    robustness_metrics.to_csv(tables / "seed_block_robustness.csv", index=False)
    negative_control.to_csv(tables / "negative_control.csv", index=False)
    law_df.to_csv(tables / "exact_law_validation.csv", index=False)
    stress_seed_df.to_csv(tables / "stress_seed_metrics.csv", index=False)
    stress_metrics.to_csv(tables / "stress_metrics.csv", index=False)
    sensitivity_seed_df.to_csv(tables / "sensitivity_seed_metrics.csv", index=False)
    sensitivity_metrics.to_csv(tables / "sensitivity_metrics.csv", index=False)
    calibration_candidate_df.to_csv(tables / "score_calibration_candidates.csv", index=False)
    calibration_metrics.to_csv(tables / "score_calibration.csv", index=False)
    ood_seed_df.to_csv(tables / "ood_seed_metrics.csv", index=False)
    ood_metrics.to_csv(tables / "ood_metrics.csv", index=False)
    extreme_object_seed_df.to_csv(tables / "extreme_object_count_seed_metrics.csv", index=False)
    extreme_object_metrics.to_csv(tables / "extreme_object_count_metrics.csv", index=False)
    family_seed_df.to_csv(tables / "model_family_proxy_seed_metrics.csv", index=False)
    family_metrics.to_csv(tables / "model_family_proxy_metrics.csv", index=False)
    domain_seed_df.to_csv(tables / "domain_randomization_seed_metrics.csv", index=False)
    domain_metrics.to_csv(tables / "domain_randomization_metrics.csv", index=False)
    counter_seed_df.to_csv(tables / "counterfactual_target_seed_metrics.csv", index=False)
    counter_metrics.to_csv(tables / "counterfactual_target_metrics.csv", index=False)
    target_sweep_seed_df.to_csv(tables / "target_identity_sweep_seed_metrics.csv", index=False)
    target_sweep_metrics.to_csv(tables / "target_identity_sweep_metrics.csv", index=False)
    pilot_seed_df.to_csv(tables / "pilot_calibration_seed_metrics.csv", index=False)
    pilot_metrics.to_csv(tables / "pilot_calibration_metrics.csv", index=False)
    pilot_budget_seed_df.to_csv(tables / "pilot_budget_seed_metrics.csv", index=False)
    pilot_budget_metrics.to_csv(tables / "pilot_budget_metrics.csv", index=False)
    loso_seed_df.to_csv(tables / "leave_one_failure_seed_metrics.csv", index=False)
    loso_metrics.to_csv(tables / "leave_one_failure_metrics.csv", index=False)
    noisy_probe_seed_df.to_csv(tables / "noisy_probe_seed_metrics.csv", index=False)
    noisy_probe_metrics.to_csv(tables / "noisy_probe_metrics.csv", index=False)
    probe_cost_seed_df.to_csv(tables / "probe_cost_seed_metrics.csv", index=False)
    probe_cost_metrics.to_csv(tables / "probe_cost_metrics.csv", index=False)
    learned_selection_seed_df.to_csv(tables / "learned_selection_seed_metrics.csv", index=False)
    learned_selection_metrics.to_csv(tables / "learned_selection_metrics.csv", index=False)
    learned_repair_policy_seed_df.to_csv(tables / "learned_repair_policy_seed_metrics.csv", index=False)
    learned_repair_policy_metrics.to_csv(tables / "learned_repair_policy_metrics.csv", index=False)
    synthetic_benchmark_seed_df.to_csv(tables / "synthetic_benchmark_seed_metrics.csv", index=False)
    synthetic_benchmark_metrics.to_csv(tables / "synthetic_benchmark_metrics.csv", index=False)
    repair_final_test_df.to_csv(tables / "repair_final_test_metrics.csv", index=False)
    repair_robustness_split_df.to_csv(tables / "repair_robustness_by_split.csv", index=False)
    calibration_diagnostics.to_csv(tables / "calibration_diagnostics.csv", index=False)
    repair_condition_splits.to_csv(tables / "repair_condition_splits.csv", index=False)
    repair_model_selection.to_csv(tables / "repair_model_selection.csv", index=False)
    unidentifiable_negative_control.to_csv(tables / "unidentifiable_negative_control.csv", index=False)
    learned_generalization.to_csv(tables / "learned_generalization_diagnostics.csv", index=False)
    (results / "repair_model_selection.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "split_seeds": REPAIR_SPLIT_SEEDS,
                "pilot_budgets": PILOT_BUDGETS,
                "grid": REPAIR_GRID,
                "support_selector_grid": SUPPORT_SELECTOR_GRID,
                "selected_configs": selected_repair_configs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mark("tables_written")
    learned_repair_policy_summary_payload = {
        "mode": mode,
        "train_seeds": learned_repair_policy_train_seeds,
        "eval_seeds": learned_repair_policy_eval_seeds,
        "n_eval_rows": int(learned_repair_policy_seed_df.shape[0]),
        "policy": learned_repair_policy,
    }
    (results / "learned_repair_policy_summary.json").write_text(
        json.dumps(learned_repair_policy_summary_payload, indent=2),
        encoding="utf-8",
    )
    pilot_summary = {
        "mode": mode,
        "train_seeds": pilot_train_seeds,
        "eval_seeds": pilot_eval_seeds,
        "n_eval_rows": int(pilot_seed_df.shape[0]),
        "calibrator": pilot_calibrator,
    }
    (results / "pilot_calibration_summary.json").write_text(json.dumps(pilot_summary, indent=2), encoding="utf-8")
    pilot_budget_summary_payload = {
        "mode": mode,
        "train_seeds": pilot_budget_train_seeds,
        "eval_seeds": pilot_budget_eval_seeds,
        "budgets": PILOT_BUDGETS,
        "n_eval_rows": int(pilot_budget_seed_df.shape[0]),
        "calibrators": pilot_budget_calibrators,
    }
    (results / "pilot_budget_summary.json").write_text(
        json.dumps(pilot_budget_summary_payload, indent=2),
        encoding="utf-8",
    )
    loso_summary = {
        "mode": mode,
        "train_seeds": loso_train_seeds,
        "eval_seeds": loso_eval_seeds,
        "n_eval_rows": int(loso_seed_df.shape[0]),
        "calibrators": loso_calibrators,
    }
    (results / "leave_one_failure_summary.json").write_text(json.dumps(loso_summary, indent=2), encoding="utf-8")
    statistical_metrics.to_csv(tables / "statistical_audit.csv", index=False)

    write_all_figures(
        main,
        seed_df,
        law_df,
        figures,
        stress_df=stress_metrics,
        learned_curve=learned_curve,
        ablation_df=ablation_metrics,
        robustness_df=robustness_metrics,
        calibration_df=calibration_metrics,
        sensitivity_df=sensitivity_metrics,
        negative_df=negative_control,
        learned_ablation_df=learned_ablation,
        ood_df=ood_metrics,
        extreme_object_df=extreme_object_metrics,
        family_df=family_metrics,
        statistical_df=statistical_metrics,
        observable_df=observable_metrics,
        domain_df=domain_metrics,
        counterfactual_df=counter_metrics,
        target_sweep_df=target_sweep_metrics,
        pilot_df=pilot_metrics,
        pilot_budget_df=pilot_budget_metrics,
        leave_one_failure_df=loso_metrics,
        noisy_probe_df=noisy_probe_metrics,
        probe_cost_df=probe_cost_metrics,
        learned_domain_shift_df=learned_domain_shift,
        learned_selection_df=learned_selection_metrics,
        learned_repair_policy_df=learned_repair_policy_metrics,
        synthetic_benchmark_df=synthetic_benchmark_metrics,
        deployment_policy_df=deployment_policy_metrics,
        repair_robustness_df=repair_robustness_split_df,
    )
    mark("figures_written")
    gate = deployment_gate_from_metrics(main)
    raw_tail = main[(main["scenario"] == "raw") & (main["selector"] == "raw")].sort_values("N")
    raw_tail_score_gain = float(raw_tail["selected_object_score_mean"].iloc[-1] - raw_tail["selected_object_score_mean"].iloc[0])
    raw_tail_utility_drop = float(raw_tail["selected_real_utility_mean"].iloc[0] - raw_tail["selected_real_utility_mean"].iloc[-1])
    raw_combined_nmax = paired_effects[
        (paired_effects["scenario"] == "raw")
        & (paired_effects["selector"] == "combined_repair")
        & (paired_effects["N"] == max(ns))
    ]
    stress_combined = stress_metrics[
        (stress_metrics["selector"] == "combined_repair") & (stress_metrics["scenario"].isin(STRESS_SCENARIOS))
    ]
    raw_ablation = ablation_metrics[ablation_metrics["scenario"] == "raw"]
    raw_observable = observable_metrics[observable_metrics["scenario"] == "raw"]
    deployment_policy_gate = deployment_policy_metrics[deployment_policy_metrics["selector"] == "gate_policy"]
    deployment_policy_raw = deployment_policy_metrics[deployment_policy_metrics["selector"] == "raw_high_n"]
    deployment_policy_early = deployment_policy_metrics[deployment_policy_metrics["selector"] == "stop_early_raw"]
    deployment_policy_oracle = deployment_policy_metrics[deployment_policy_metrics["selector"] == "oracle"]
    deployment_policy_corrupted = deployment_policy_gate[
        deployment_policy_gate["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"])
    ]
    deployment_policy_corrupted_raw = deployment_policy_raw[
        deployment_policy_raw["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"])
    ]
    deployment_policy_corrupted_early = deployment_policy_early[
        deployment_policy_early["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"])
    ]
    deployment_policy_actions = (
        "|".join(
            f"{key}:{int(value)}"
            for key, value in deployment_policy_seed_df[
                deployment_policy_seed_df["selector"] == "gate_policy"
            ]["gate_action"].value_counts().sort_index().items()
        )
        if not deployment_policy_seed_df.empty and "gate_action" in deployment_policy_seed_df
        else None
    )
    robustness_pass_rate = float(
        np.mean(
            (robustness_metrics["raw_tail_score_gain"] >= 0.30)
            & (robustness_metrics["raw_tail_utility_drop"] >= 0.10)
            & (robustness_metrics["raw_tail_identity_error"] >= 0.75)
            & (robustness_metrics["combined_raw_nmax_gain"] >= 0.55)
            & (robustness_metrics["combined_raw_nmax_win_rate"] >= 0.75)
        )
    ) if not robustness_metrics.empty else None
    top_calibration = calibration_metrics.sort_values("score_bin").iloc[-1] if not calibration_metrics.empty else None
    good_control = negative_control[negative_control["contrast"] == "good_control"]
    corrupted_mean = negative_control[negative_control["contrast"] == "corrupted_mean"]
    good_minus_corrupted = negative_control[negative_control["contrast"] == "good_minus_corrupted"]
    no_mass_ablation = learned_ablation[learned_ablation["ablation"] == "no_mass_sensor"]
    kinematic_pair_ablation = learned_ablation[learned_ablation["ablation"] == "kinematic_pair_identity"]
    shifted_learned = learned_domain_shift[learned_domain_shift["variant"] != "standard_test"]
    learned_selection_raw = learned_selection_metrics[learned_selection_metrics["selector"] == "raw"]
    learned_selection_reward = learned_selection_metrics[learned_selection_metrics["selector"] == "learned_reward"]
    learned_selection_identity = learned_selection_metrics[
        learned_selection_metrics["selector"] == "learned_identity_reward"
    ]
    learned_repair_policy_raw = learned_repair_policy_metrics[learned_repair_policy_metrics["selector"] == "raw"]
    learned_repair_policy_learned_identity = learned_repair_policy_metrics[
        learned_repair_policy_metrics["selector"] == "learned_identity_reward"
    ]
    learned_repair_policy_policy = learned_repair_policy_metrics[
        learned_repair_policy_metrics["selector"] == "learned_repair_policy"
    ]
    learned_repair_policy_pilot = learned_repair_policy_metrics[
        learned_repair_policy_metrics["selector"] == "pilot_calibrated"
    ]
    synthetic_benchmark_raw = synthetic_benchmark_metrics[synthetic_benchmark_metrics["selector"] == "raw"]
    synthetic_benchmark_combined = synthetic_benchmark_metrics[
        synthetic_benchmark_metrics["selector"] == "combined_repair"
    ]
    synthetic_benchmark_observable = synthetic_benchmark_metrics[
        synthetic_benchmark_metrics["selector"] == "observable_repair"
    ]
    ood_combined = ood_metrics[
        (ood_metrics["selector"] == "combined_repair")
        & (ood_metrics["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
    ]
    ood_observable = ood_metrics[
        (ood_metrics["selector"] == "observable_repair")
        & (ood_metrics["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
    ]
    ood_raw = ood_metrics[
        (ood_metrics["selector"] == "raw")
        & (ood_metrics["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
    ]
    ood_good = ood_metrics[(ood_metrics["selector"] == "raw") & (ood_metrics["scenario"] == "dense6_good")]
    extreme_corrupted = ["extreme10_raw", "extreme12_occlusion", "extreme12_hidden"]
    extreme_combined = extreme_object_metrics[
        (extreme_object_metrics["selector"] == "combined_repair")
        & (extreme_object_metrics["scenario"].isin(extreme_corrupted))
    ]
    extreme_observable = extreme_object_metrics[
        (extreme_object_metrics["selector"] == "observable_repair")
        & (extreme_object_metrics["scenario"].isin(extreme_corrupted))
    ]
    extreme_raw = extreme_object_metrics[
        (extreme_object_metrics["selector"] == "raw")
        & (extreme_object_metrics["scenario"].isin(extreme_corrupted))
    ]
    extreme_good = extreme_object_metrics[
        (extreme_object_metrics["selector"] == "raw")
        & (extreme_object_metrics["scenario"] == "extreme10_good")
    ]
    family_combined = family_metrics[
        (family_metrics["selector"] == "combined_repair")
        & (family_metrics["scenario"].isin(MODEL_FAMILY_SCENARIOS))
    ]
    domain_combined = domain_metrics[domain_metrics["selector"] == "combined_repair"]
    domain_observable = domain_metrics[domain_metrics["selector"] == "observable_repair"]
    domain_raw = domain_metrics[domain_metrics["selector"] == "raw"]
    counter_combined = counter_metrics[counter_metrics["selector"] == "combined_repair"]
    counter_observable = counter_metrics[counter_metrics["selector"] == "observable_repair"]
    counter_raw = counter_metrics[counter_metrics["selector"] == "raw"]
    target_sweep_combined = target_sweep_metrics[target_sweep_metrics["selector"] == "combined_repair"]
    target_sweep_observable = target_sweep_metrics[target_sweep_metrics["selector"] == "observable_repair"]
    target_sweep_raw = target_sweep_metrics[target_sweep_metrics["selector"] == "raw"]
    pilot_calibrated = pilot_metrics[pilot_metrics["selector"] == "pilot_calibrated"]
    pilot_raw = pilot_metrics[pilot_metrics["selector"] == "raw"]
    mature_pilot_budget = pilot_budget_metrics[
        (pilot_budget_metrics["selector"] == "pilot_calibrated")
        & (pilot_budget_metrics["pilot_label_budget"] >= 128)
    ]
    smallest_pilot_budget = pilot_budget_metrics[
        (pilot_budget_metrics["selector"] == "pilot_calibrated")
        & (pilot_budget_metrics["pilot_label_budget"] == min(PILOT_BUDGETS))
    ]
    largest_actual_pilot_budget = (
        int(pilot_budget_metrics["pilot_label_budget"].max()) if not pilot_budget_metrics.empty else max(PILOT_BUDGETS)
    )
    largest_pilot_budget = pilot_budget_metrics[
        (pilot_budget_metrics["selector"] == "pilot_calibrated")
        & (pilot_budget_metrics["pilot_label_budget"] == largest_actual_pilot_budget)
    ]
    loso_pilot = loso_metrics[loso_metrics["selector"] == "pilot_calibrated"]
    loso_raw = loso_metrics[loso_metrics["selector"] == "raw"]
    noisy_probe_repair = noisy_probe_metrics[noisy_probe_metrics["selector"] == "noisy_probe_repair"]
    noisy_probe_focus = noisy_probe_repair[noisy_probe_repair["probe_reliability"] >= 0.75]
    probe_cost_focus = probe_cost_metrics[
        (probe_cost_metrics["probe_cost"] <= 0.10)
        & (probe_cost_metrics["scenario"].isin(PROBE_COST_SCENARIOS))
    ]
    probe_cost_combined = probe_cost_focus[probe_cost_focus["selector"] == "combined_repair"]
    probe_cost_observable = probe_cost_focus[probe_cost_focus["selector"] == "observable_repair"]
    probe_cost_targeted = probe_cost_focus[
        (probe_cost_focus["selector"] == "targeted_probe")
        & (probe_cost_focus["scenario"] == "hidden_property")
    ]
    probe_cost_raw = probe_cost_focus[probe_cost_focus["selector"] == "raw"]
    probe_cost_max = probe_cost_metrics[probe_cost_metrics["probe_cost"] == max(PROBE_COSTS)]
    probe_cost_combined_max = probe_cost_max[probe_cost_max["selector"] == "combined_repair"]
    probe_cost_observable_max = probe_cost_max[probe_cost_max["selector"] == "observable_repair"]
    statistical_pass_margin = None
    if not statistical_metrics.empty:
        statistical_pass_margin = float(
            (statistical_metrics["bootstrap_ci_low"] - statistical_metrics["threshold"]).min()
        )
    sensitivity_low_noise = sensitivity_metrics[sensitivity_metrics["score_noise"] <= 0.10]
    combined_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "combined_repair_noisy"]
    raw_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "raw_noisy"]
    sensitivity_margin = None
    if not combined_sensitivity.empty and not raw_sensitivity.empty:
        sensitivity_margin = float(
            combined_sensitivity["selected_real_utility_mean"].mean()
            - raw_sensitivity["selected_real_utility_mean"].mean()
        )
    robustness_n = int(repair_robustness_split_df["N"].max()) if not repair_robustness_split_df.empty else max(ns)

    def _robust_closure(selector: str, budget: int | None = None) -> float | None:
        if repair_robustness_split_df.empty:
            return None
        focus = repair_robustness_split_df[
            (repair_robustness_split_df["selector"] == selector)
            & (repair_robustness_split_df["N"] == robustness_n)
        ]
        if budget is not None and "repair_budget" in focus:
            focus = focus[focus["repair_budget"] == budget]
        if focus.empty:
            return None
        return float(focus["mean_gap_closure_across_splits"].iloc[0])

    def _robust_worst_quartile(selector: str, budget: int | None = None) -> float | None:
        if repair_robustness_split_df.empty:
            return None
        focus = repair_robustness_split_df[
            (repair_robustness_split_df["selector"] == selector)
            & (repair_robustness_split_df["N"] == robustness_n)
        ]
        if budget is not None and "repair_budget" in focus:
            focus = focus[focus["repair_budget"] == budget]
        if focus.empty:
            return None
        return float(focus["worst_quartile_gap_closure"].iloc[0])

    calibration_overall = (
        calibration_diagnostics[calibration_diagnostics["metric_scope"] == "overall"]
        if not calibration_diagnostics.empty
        else pd.DataFrame()
    )
    hidden_control_block_rate = (
        float((unidentifiable_negative_control["gate_action"] == "block_high_n").mean())
        if not unidentifiable_negative_control.empty
        else None
    )
    summary = {
        "mode": mode,
        "ns": ns,
        "seeds": seeds,
        "stress_seeds": stress_seeds,
        "repair_split_seeds": REPAIR_SPLIT_SEEDS,
        "n_seed_rows": int(seed_df.shape[0]),
        "n_main_rows": int(main.shape[0]),
        "n_deployment_policy_rows": int(deployment_policy_seed_df.shape[0]),
        "n_stress_rows": int(stress_seed_df.shape[0]),
        "n_ood_rows": int(ood_seed_df.shape[0]),
        "n_extreme_object_count_rows": int(extreme_object_seed_df.shape[0]),
        "n_model_family_proxy_rows": int(family_seed_df.shape[0]),
        "n_domain_randomization_rows": int(domain_seed_df.shape[0]),
        "n_counterfactual_target_rows": int(counter_seed_df.shape[0]),
        "n_target_identity_sweep_rows": int(target_sweep_seed_df.shape[0]),
        "n_pilot_calibration_rows": int(pilot_seed_df.shape[0]),
        "n_pilot_budget_rows": int(pilot_budget_seed_df.shape[0]),
        "n_leave_one_failure_rows": int(loso_seed_df.shape[0]),
        "n_noisy_probe_rows": int(noisy_probe_seed_df.shape[0]),
        "n_probe_cost_rows": int(probe_cost_seed_df.shape[0]),
        "n_learned_selection_rows": int(learned_selection_seed_df.shape[0]),
        "n_learned_repair_policy_rows": int(learned_repair_policy_seed_df.shape[0]),
        "n_synthetic_benchmark_rows": int(synthetic_benchmark_seed_df.shape[0]),
        "n_repair_final_test_rows": int(repair_final_test_df.shape[0]),
        "n_repair_split_seed_rows": int(repair_robustness_split_df.shape[0]),
        "deployment_gate": gate,
        "exact_law_mean_absolute_error": exact_law_prediction_error(law_df),
        "raw_tail_score_gain": raw_tail_score_gain,
        "raw_tail_utility_drop": raw_tail_utility_drop,
        "combined_repair_raw_nmax_mean_gain": float(raw_combined_nmax["mean_gain"].iloc[0]) if not raw_combined_nmax.empty else None,
        "combined_repair_raw_nmax_win_rate": float(raw_combined_nmax["win_rate"].iloc[0]) if not raw_combined_nmax.empty else None,
        "combined_repair_raw_ablation_dominance": float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) if not raw_ablation.empty else None,
        "observable_repair_raw_nmax_utility": float(raw_observable["observable_repair_utility"].iloc[0]) if not raw_observable.empty else None,
        "observable_repair_raw_nmax_gain": float(raw_observable["observable_vs_raw_gain"].iloc[0]) if not raw_observable.empty else None,
        "observable_repair_combined_gap": float(raw_observable["combined_minus_observable_gap"].iloc[0]) if not raw_observable.empty else None,
        "deployment_policy_mean_utility": float(deployment_policy_gate["selected_real_utility_mean"].mean()) if not deployment_policy_gate.empty else None,
        "deployment_policy_raw_mean_utility": float(deployment_policy_raw["selected_real_utility_mean"].mean()) if not deployment_policy_raw.empty else None,
        "deployment_policy_raw_high_mean_utility": float(deployment_policy_raw["selected_real_utility_mean"].mean()) if not deployment_policy_raw.empty else None,
        "deployment_policy_stop_early_mean_utility": float(deployment_policy_early["selected_real_utility_mean"].mean()) if not deployment_policy_early.empty else None,
        "deployment_policy_oracle_mean_utility": float(deployment_policy_oracle["selected_real_utility_mean"].mean()) if not deployment_policy_oracle.empty else None,
        "deployment_policy_vs_raw_gain": float(deployment_policy_gate["deployment_policy_vs_raw_gain_mean"].mean()) if not deployment_policy_gate.empty else None,
        "deployment_policy_corrupted_vs_raw_gain": float(deployment_policy_corrupted["deployment_policy_vs_raw_gain_mean"].mean()) if not deployment_policy_corrupted.empty else None,
        "deployment_policy_vs_stop_early_gain": float(deployment_policy_gate["deployment_policy_vs_stop_early_gain_mean"].mean()) if not deployment_policy_gate.empty else None,
        "deployment_policy_corrupted_vs_stop_early_gain": float(deployment_policy_corrupted["deployment_policy_vs_stop_early_gain_mean"].mean()) if not deployment_policy_corrupted.empty else None,
        "deployment_policy_min_corrupted_utility": float(deployment_policy_corrupted["selected_real_utility_mean"].min()) if not deployment_policy_corrupted.empty else None,
        "deployment_policy_min_win_rate": float(deployment_policy_corrupted["deployment_policy_win_rate"].min()) if not deployment_policy_corrupted.empty else None,
        "deployment_policy_all_scenario_min_win_rate": float(deployment_policy_gate["deployment_policy_win_rate"].min()) if not deployment_policy_gate.empty else None,
        "deployment_policy_corrupted_raw_mean_utility": float(deployment_policy_corrupted_raw["selected_real_utility_mean"].mean()) if not deployment_policy_corrupted_raw.empty else None,
        "deployment_policy_corrupted_stop_early_mean_utility": float(deployment_policy_corrupted_early["selected_real_utility_mean"].mean()) if not deployment_policy_corrupted_early.empty else None,
        "deployment_policy_oracle_gap": float(deployment_policy_gate["deployment_policy_oracle_gap_mean"].mean()) if not deployment_policy_gate.empty else None,
        "deployment_policy_actions": deployment_policy_actions,
        "stress_combined_mean_selected_utility": float(stress_combined["selected_real_utility_mean"].mean()) if not stress_combined.empty else None,
        "seed_block_robustness_pass_rate": robustness_pass_rate,
        "raw_score_top_bin_object_real_gap": float(top_calibration["object_real_gap"]) if top_calibration is not None else None,
        "raw_score_top_bin_identity_error": float(top_calibration["identity_error_rate"]) if top_calibration is not None else None,
        "good_control_raw_nmax_utility": float(good_control["selected_real_utility_mean"].iloc[0]) if not good_control.empty else None,
        "good_control_raw_nmax_identity_error": float(good_control["identity_error_mean"].iloc[0]) if not good_control.empty else None,
        "good_minus_corrupted_raw_nmax_utility": float(good_minus_corrupted["selected_real_utility_mean"].iloc[0]) if not good_minus_corrupted.empty else None,
        "corrupted_mean_raw_nmax_utility": float(corrupted_mean["selected_real_utility_mean"].iloc[0]) if not corrupted_mean.empty else None,
        "combined_repair_min_low_noise_utility": float(combined_sensitivity["selected_real_utility_mean"].min()) if not combined_sensitivity.empty else None,
        "combined_vs_raw_low_noise_sensitivity_margin": sensitivity_margin,
        "deployable_no_leak_budget32_gap_closure": _robust_closure("pilot_calibrated", 32),
        "deployable_no_leak_budget128_gap_closure": _robust_closure("pilot_calibrated", 128),
        "deployable_no_leak_budget32_worst_quartile_gap_closure": _robust_worst_quartile("pilot_calibrated", 32),
        "support_covered_budget32_gap_closure": _robust_closure("observable_repair", 32),
        "support_covered_budget128_gap_closure": _robust_closure("observable_repair", 128),
        "oracle_upper_bound_gap_closure": _robust_closure("repair_all_candidates_labeled_oracle", 32),
        "calibration_lcb_coverage_overall": float(calibration_overall["lcb_coverage"].iloc[0]) if not calibration_overall.empty else None,
        "calibration_violation_rate_overall": float(calibration_overall["violation_rate"].iloc[0]) if not calibration_overall.empty else None,
        "hidden_mode_unidentifiable_block_rate": hidden_control_block_rate,
        "learned_full_minus_no_mass_property_accuracy": float(no_mass_ablation["full_minus_property_accuracy"].iloc[0]) if not no_mass_ablation.empty else None,
        "learned_full_minus_kinematic_pair_identity_accuracy": float(kinematic_pair_ablation["full_minus_identity_alignment_accuracy"].iloc[0]) if not kinematic_pair_ablation.empty else None,
        "learned_shift_min_property_margin": float(shifted_learned["property_margin"].min()) if not shifted_learned.empty else None,
        "learned_shift_min_identity_margin": float(shifted_learned["identity_margin"].min()) if not shifted_learned.empty else None,
        "learned_shift_max_transition_ratio": float(shifted_learned["transition_mse_ratio"].max()) if not shifted_learned.empty else None,
        "learned_shift_min_reward_correlation": float(shifted_learned["reward_correlation"].min()) if not shifted_learned.empty else None,
        "learned_selection_raw_mean_utility": float(learned_selection_raw["selected_real_utility_mean"].mean()) if not learned_selection_raw.empty else None,
        "learned_selection_reward_mean_utility": float(learned_selection_reward["selected_real_utility_mean"].mean()) if not learned_selection_reward.empty else None,
        "learned_selection_identity_mean_utility": float(learned_selection_identity["selected_real_utility_mean"].mean()) if not learned_selection_identity.empty else None,
        "learned_selection_identity_min_scenario_utility": float(learned_selection_identity["selected_real_utility_mean"].min()) if not learned_selection_identity.empty else None,
        "learned_selection_identity_vs_raw_gain": float(learned_selection_identity["learned_identity_vs_raw_gain_mean"].mean()) if not learned_selection_identity.empty else None,
        "learned_selection_identity_vs_reward_gain": float(learned_selection_identity["learned_identity_vs_reward_gain_mean"].mean()) if not learned_selection_identity.empty else None,
        "learned_selection_identity_min_win_rate": float(learned_selection_identity["learned_identity_win_rate"].min()) if not learned_selection_identity.empty else None,
        "learned_repair_policy_raw_mean_utility": float(learned_repair_policy_raw["selected_real_utility_mean"].mean()) if not learned_repair_policy_raw.empty else None,
        "learned_repair_policy_learned_identity_mean_utility": float(learned_repair_policy_learned_identity["selected_real_utility_mean"].mean()) if not learned_repair_policy_learned_identity.empty else None,
        "learned_repair_policy_mean_utility": float(learned_repair_policy_policy["selected_real_utility_mean"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_min_variant_utility": float(learned_repair_policy_policy["selected_real_utility_mean"].min()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_vs_raw_gain": float(learned_repair_policy_policy["learned_repair_policy_vs_raw_gain_mean"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_vs_learned_identity_gain": float(learned_repair_policy_policy["learned_repair_policy_vs_learned_identity_gain_mean"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_vs_pilot_gain": float(learned_repair_policy_policy["learned_repair_policy_vs_pilot_gain_mean"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_min_win_rate": float(learned_repair_policy_policy["learned_repair_policy_win_rate"].min()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_mean_learned_identity_win_rate": float(learned_repair_policy_policy["learned_repair_policy_over_learned_identity_win_rate"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_min_learned_identity_win_rate": float(learned_repair_policy_policy["learned_repair_policy_over_learned_identity_win_rate"].min()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_mean_learned_identity_nonloss_rate": float(learned_repair_policy_policy["learned_repair_policy_over_learned_identity_nonloss_rate"].mean()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_min_learned_identity_nonloss_rate": float(learned_repair_policy_policy["learned_repair_policy_over_learned_identity_nonloss_rate"].min()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_max_learned_identity_loss": float(learned_repair_policy_policy["learned_repair_policy_worst_learned_identity_loss"].max()) if not learned_repair_policy_policy.empty else None,
        "learned_repair_policy_pilot_mean_utility": float(learned_repair_policy_pilot["selected_real_utility_mean"].mean()) if not learned_repair_policy_pilot.empty else None,
        "learned_repair_policy_train_correlation": float(learned_repair_policy["train_correlation"]),
        "learned_repair_policy_train_mae": float(learned_repair_policy["train_mae"]),
        "synthetic_benchmark_raw_mean_utility": float(synthetic_benchmark_raw["selected_real_utility_mean"].mean()) if not synthetic_benchmark_raw.empty else None,
        "synthetic_benchmark_raw_mean_identity_error": float(synthetic_benchmark_raw["identity_error_mean"].mean()) if not synthetic_benchmark_raw.empty else None,
        "synthetic_benchmark_combined_mean_utility": float(synthetic_benchmark_combined["selected_real_utility_mean"].mean()) if not synthetic_benchmark_combined.empty else None,
        "synthetic_benchmark_observable_mean_utility": float(synthetic_benchmark_observable["selected_real_utility_mean"].mean()) if not synthetic_benchmark_observable.empty else None,
        "synthetic_benchmark_combined_min_variant_utility": float(synthetic_benchmark_combined["selected_real_utility_mean"].min()) if not synthetic_benchmark_combined.empty else None,
        "synthetic_benchmark_observable_min_variant_utility": float(synthetic_benchmark_observable["selected_real_utility_mean"].min()) if not synthetic_benchmark_observable.empty else None,
        "synthetic_benchmark_combined_vs_raw_gain": float(synthetic_benchmark_combined["synthetic_benchmark_combined_vs_raw_gain_mean"].mean()) if not synthetic_benchmark_combined.empty else None,
        "synthetic_benchmark_observable_vs_raw_gain": float(synthetic_benchmark_observable["synthetic_benchmark_observable_vs_raw_gain_mean"].mean()) if not synthetic_benchmark_observable.empty else None,
        "synthetic_benchmark_combined_min_win_rate": float(synthetic_benchmark_combined["synthetic_benchmark_combined_win_rate"].min()) if not synthetic_benchmark_combined.empty else None,
        "ood_combined_mean_selected_utility": float(ood_combined["selected_real_utility_mean"].mean()) if not ood_combined.empty else None,
        "ood_raw_mean_selected_utility": float(ood_raw["selected_real_utility_mean"].mean()) if not ood_raw.empty else None,
        "ood_combined_vs_raw_gain": float(ood_combined["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) if not ood_combined.empty and not ood_raw.empty else None,
        "ood_good_control_raw_utility": float(ood_good["selected_real_utility_mean"].iloc[0]) if not ood_good.empty else None,
        "ood_observable_mean_selected_utility": float(ood_observable["selected_real_utility_mean"].mean()) if not ood_observable.empty else None,
        "ood_observable_vs_raw_gain": float(ood_observable["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) if not ood_observable.empty and not ood_raw.empty else None,
        "extreme_object_count_raw_mean_utility": float(extreme_raw["selected_real_utility_mean"].mean()) if not extreme_raw.empty else None,
        "extreme_object_count_combined_mean_utility": float(extreme_combined["selected_real_utility_mean"].mean()) if not extreme_combined.empty else None,
        "extreme_object_count_observable_mean_utility": float(extreme_observable["selected_real_utility_mean"].mean()) if not extreme_observable.empty else None,
        "extreme_object_count_combined_vs_raw_gain": float(extreme_combined["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) if not extreme_combined.empty and not extreme_raw.empty else None,
        "extreme_object_count_observable_vs_raw_gain": float(extreme_observable["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) if not extreme_observable.empty and not extreme_raw.empty else None,
        "extreme_object_count_good_control_raw_utility": float(extreme_good["selected_real_utility_mean"].iloc[0]) if not extreme_good.empty else None,
        "model_family_combined_vs_best_proxy_gain": float(family_combined["combined_vs_best_proxy_gain_mean"].mean()) if not family_combined.empty else None,
        "model_family_min_combined_vs_best_proxy_gain": float(family_combined["combined_vs_best_proxy_gain_mean"].min()) if not family_combined.empty else None,
        "model_family_max_combined_oracle_gap": float(family_combined["combined_oracle_gap_mean"].max()) if not family_combined.empty else None,
        "domain_randomized_raw_utility": float(domain_raw["selected_real_utility_mean"].iloc[0]) if not domain_raw.empty else None,
        "domain_randomized_combined_utility": float(domain_combined["selected_real_utility_mean"].iloc[0]) if not domain_combined.empty else None,
        "domain_randomized_observable_utility": float(domain_observable["selected_real_utility_mean"].iloc[0]) if not domain_observable.empty else None,
        "domain_randomized_combined_vs_raw_gain": float(domain_combined["domain_combined_vs_raw_gain_mean"].iloc[0]) if not domain_combined.empty else None,
        "domain_randomized_observable_vs_raw_gain": float(domain_observable["domain_observable_vs_raw_gain_mean"].iloc[0]) if not domain_observable.empty else None,
        "domain_randomized_combined_win_rate": float(domain_combined["domain_combined_win_rate"].iloc[0]) if not domain_combined.empty else None,
        "counterfactual_raw_utility": float(counter_raw["selected_real_utility_mean"].iloc[0]) if not counter_raw.empty else None,
        "counterfactual_combined_utility": float(counter_combined["selected_real_utility_mean"].iloc[0]) if not counter_combined.empty else None,
        "counterfactual_observable_utility": float(counter_observable["selected_real_utility_mean"].iloc[0]) if not counter_observable.empty else None,
        "counterfactual_combined_vs_raw_gain": float(counter_combined["counterfactual_combined_vs_raw_gain_mean"].iloc[0]) if not counter_combined.empty else None,
        "counterfactual_observable_vs_raw_gain": float(counter_observable["counterfactual_observable_vs_raw_gain_mean"].iloc[0]) if not counter_observable.empty else None,
        "counterfactual_combined_win_rate": float(counter_combined["counterfactual_combined_win_rate"].iloc[0]) if not counter_combined.empty else None,
        "target_sweep_raw_mean_utility": float(target_sweep_raw["selected_real_utility_mean"].mean()) if not target_sweep_raw.empty else None,
        "target_sweep_raw_mean_identity_error": float(target_sweep_raw["identity_error_mean"].mean()) if not target_sweep_raw.empty else None,
        "target_sweep_combined_mean_utility": float(target_sweep_combined["selected_real_utility_mean"].mean()) if not target_sweep_combined.empty else None,
        "target_sweep_observable_mean_utility": float(target_sweep_observable["selected_real_utility_mean"].mean()) if not target_sweep_observable.empty else None,
        "target_sweep_combined_min_target_utility": float(target_sweep_combined["selected_real_utility_mean"].min()) if not target_sweep_combined.empty else None,
        "target_sweep_observable_min_target_utility": float(target_sweep_observable["selected_real_utility_mean"].min()) if not target_sweep_observable.empty else None,
        "target_sweep_combined_vs_raw_gain": float(target_sweep_combined["target_sweep_combined_vs_raw_gain_mean"].mean()) if not target_sweep_combined.empty else None,
        "target_sweep_observable_vs_raw_gain": float(target_sweep_observable["target_sweep_observable_vs_raw_gain_mean"].mean()) if not target_sweep_observable.empty else None,
        "target_sweep_combined_min_win_rate": float(target_sweep_combined["target_sweep_combined_win_rate"].min()) if not target_sweep_combined.empty else None,
        "pilot_raw_mean_utility": float(pilot_raw["selected_real_utility_mean"].mean()) if not pilot_raw.empty else None,
        "pilot_calibrated_mean_utility": float(pilot_calibrated["selected_real_utility_mean"].mean()) if not pilot_calibrated.empty else None,
        "pilot_calibrated_vs_raw_gain": float(pilot_calibrated["pilot_vs_raw_gain_mean"].mean()) if not pilot_calibrated.empty else None,
        "pilot_calibrated_min_win_rate": float(pilot_calibrated["pilot_win_rate"].min()) if not pilot_calibrated.empty else None,
        "pilot_calibrated_max_oracle_gap": float(pilot_calibrated["pilot_oracle_gap_mean"].max()) if not pilot_calibrated.empty else None,
        "pilot_calibration_train_mae": float(pilot_calibrator["train_mae"]),
        "pilot_calibration_train_correlation": float(pilot_calibrator["train_correlation"]),
        "pilot_budget_mature_mean_utility": float(mature_pilot_budget["selected_real_utility_mean"].mean()) if not mature_pilot_budget.empty else None,
        "pilot_budget_mature_vs_raw_gain": float(mature_pilot_budget["pilot_budget_vs_raw_gain_mean"].mean()) if not mature_pilot_budget.empty else None,
        "pilot_budget_mature_min_win_rate": float(mature_pilot_budget["pilot_budget_win_rate"].min()) if not mature_pilot_budget.empty else None,
        "pilot_budget_smallest_vs_raw_gain": float(smallest_pilot_budget["pilot_budget_vs_raw_gain_mean"].mean()) if not smallest_pilot_budget.empty else None,
        "pilot_budget_largest_vs_raw_gain": float(largest_pilot_budget["pilot_budget_vs_raw_gain_mean"].mean()) if not largest_pilot_budget.empty else None,
        "pilot_budget_largest_max_oracle_gap": float(largest_pilot_budget["pilot_budget_oracle_gap_mean"].max()) if not largest_pilot_budget.empty else None,
        "leave_one_failure_raw_mean_utility": float(loso_raw["selected_real_utility_mean"].mean()) if not loso_raw.empty else None,
        "leave_one_failure_pilot_mean_utility": float(loso_pilot["selected_real_utility_mean"].mean()) if not loso_pilot.empty else None,
        "leave_one_failure_pilot_vs_raw_gain": float(loso_pilot["pilot_vs_raw_gain_mean"].mean()) if not loso_pilot.empty else None,
        "leave_one_failure_pilot_min_win_rate": float(loso_pilot["pilot_win_rate"].min()) if not loso_pilot.empty else None,
        "leave_one_failure_pilot_max_oracle_gap": float(loso_pilot["pilot_oracle_gap_mean"].max()) if not loso_pilot.empty else None,
        "leave_one_failure_min_train_correlation": float(min(item["calibrator"]["train_correlation"] for item in loso_calibrators)) if loso_calibrators else None,
        "noisy_probe_min_reliable_utility": float(noisy_probe_focus["selected_real_utility_mean"].min()) if not noisy_probe_focus.empty else None,
        "noisy_probe_mean_reliable_gain": float(noisy_probe_focus["noisy_probe_vs_raw_gain_mean"].mean()) if not noisy_probe_focus.empty else None,
        "noisy_probe_min_reliable_win_rate": float(noisy_probe_focus["noisy_probe_win_rate"].min()) if not noisy_probe_focus.empty else None,
        "noisy_probe_max_reliable_oracle_gap": float(noisy_probe_focus["noisy_probe_oracle_gap_mean"].max()) if not noisy_probe_focus.empty else None,
        "probe_cost_low_cost_combined_mean_utility": float(probe_cost_combined["selected_real_utility_mean"].mean()) if not probe_cost_combined.empty else None,
        "probe_cost_low_cost_combined_vs_raw_gain": float(probe_cost_combined["probe_cost_combined_vs_raw_gain_mean"].mean()) if not probe_cost_combined.empty else None,
        "probe_cost_low_cost_observable_vs_raw_gain": float(probe_cost_observable["probe_cost_observable_vs_raw_gain_mean"].mean()) if not probe_cost_observable.empty else None,
        "probe_cost_low_cost_targeted_vs_raw_gain": float(probe_cost_targeted["probe_cost_targeted_vs_raw_gain_mean"].mean()) if not probe_cost_targeted.empty else None,
        "probe_cost_max_cost_combined_vs_raw_gain": float(probe_cost_combined_max["probe_cost_combined_vs_raw_gain_mean"].mean()) if not probe_cost_combined_max.empty else None,
        "probe_cost_max_cost_observable_vs_raw_gain": float(probe_cost_observable_max["probe_cost_observable_vs_raw_gain_mean"].mean()) if not probe_cost_observable_max.empty else None,
        "probe_cost_low_cost_raw_mean_utility": float(probe_cost_raw["selected_real_utility_mean"].mean()) if not probe_cost_raw.empty else None,
        "statistical_audit_all_pass": bool(statistical_metrics["passes"].all()) if not statistical_metrics.empty else None,
        "statistical_audit_min_ci_margin": statistical_pass_margin,
        "learned_metrics": learned_row,
        "passes_claim_audit": False,
        "runtime_seconds": round(time.time() - start, 3),
    }
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mark("summary_written")
    verification_log = results / "verification_log.json"
    if not verification_log.exists():
        verification_log.write_text(
            json.dumps(
                {
                    "command_results": {
                        f"experiments --mode {mode}": "pass (generated by experiments/run_experiments.py)"
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    claims = write_claim_status(root)
    summary["passes_claim_audit"] = claims["passes_claim_audit"]
    summary["runtime_seconds"] = round(time.time() - start, 3)
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_claim_status(root)
    write_final_audit(root, command_results={f"experiments --mode {mode}": "pass"})
    mark("claim_audit_written")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--ns", default=None)
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    default_ns = [1, 4, 16] if args.mode == "smoke" else [1, 2, 4, 8, 16, 32, 64]
    default_seeds = [0, 1] if args.mode == "smoke" else list(range(16))
    summary = run(Path(args.root), args.mode, _parse_ints(args.ns, default_ns), _parse_ints(args.seeds, default_seeds))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
