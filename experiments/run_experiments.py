"""Run controlled object-centric Best-of-N experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from object_centric_best_of_n.audit import write_claim_status, write_final_audit
from object_centric_best_of_n.envs import make_scene, retarget_scene
from object_centric_best_of_n.learned_model import learned_candidate_scores, train_and_evaluate
from object_centric_best_of_n.metrics import (
    aggregate_seed_metrics,
    counterfactual_target_summary,
    deployment_gate_from_metrics,
    domain_randomization_summary,
    exact_law_prediction_error,
    extreme_object_count_summary,
    model_family_proxy_summary,
    negative_control_summary,
    learned_selection_summary,
    observable_repair_summary,
    ood_summary,
    paired_selector_effects,
    pilot_calibration_summary,
    pilot_budget_summary,
    noisy_probe_summary,
    probe_cost_summary,
    repair_ablation_summary,
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
    fit_pilot_calibrator,
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
PILOT_BUDGETS = [16, 32, 64, 128, 256, 512]
NOISY_PROBE_RELIABILITIES = [0.55, 0.65, 0.75, 0.85, 0.90]
NOISY_PROBE_SELECTORS = ["raw", "noisy_probe_repair", "observable_repair", "combined_repair", "random", "oracle"]
PROBE_COSTS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
PROBE_COST_SCENARIOS = ["hidden_property", "raw"]
PROBE_COST_SELECTORS = ["raw", "targeted_probe", "observable_repair", "combined_repair", "oracle"]
PROBE_USING_SELECTORS = {"targeted_probe", "observable_repair", "combined_repair"}


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
        calibrator = fit_pilot_calibrator(subset, ridge=5e-3)
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


def run(root: Path, mode: str, ns: list[int], seeds: list[int]) -> dict[str, object]:
    start = time.time()
    results = root / "results"
    tables = results / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

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
    law_df = pd.DataFrame(law_rows)
    stress_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    stress_seed_df = _run_stress_panel(generator, stress_seeds=stress_seeds, n=max(ns))
    stress_metrics = stress_summary(stress_seed_df)
    sensitivity_seeds = list(range(4)) if mode == "smoke" else list(range(24))
    sensitivity_seed_df, calibration_candidate_df = _run_sensitivity_panel(generator, sensitivity_seeds, n=max(ns), mode=mode)
    sensitivity_metrics = sensitivity_summary(sensitivity_seed_df)
    calibration_metrics = score_calibration_table(calibration_candidate_df)
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
    learned_metrics, learned_model = train_and_evaluate(results, seed=123 if mode == "smoke" else 456)
    learned_row = learned_metrics.as_dict()
    pd.DataFrame([learned_row]).to_csv(tables / "learned_metrics.csv", index=False)
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
    synthetic_benchmark_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    synthetic_benchmark_seed_df = _run_synthetic_benchmark_panel(
        generator,
        benchmark_seeds=synthetic_benchmark_seeds,
        n=max(ns),
    )
    synthetic_benchmark_metrics = synthetic_benchmark_summary(synthetic_benchmark_seed_df)
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
        synthetic_benchmark_seed_df=synthetic_benchmark_seed_df,
        bootstrap_reps=bootstrap_reps,
        seed=240_001,
    )
    ablation_metrics = repair_ablation_summary(main, paired_effects)
    observable_metrics = observable_repair_summary(main, paired_effects)
    robustness_metrics = seed_block_robustness(seed_df, block_size=2 if mode == "smoke" else 4)
    negative_control = negative_control_summary(main)

    seed_df.to_csv(tables / "seed_metrics.csv", index=False)
    main.to_csv(tables / "main_metrics.csv", index=False)
    repair_metrics.to_csv(tables / "repair_metrics.csv", index=False)
    paired_effects.to_csv(tables / "paired_effects.csv", index=False)
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
    synthetic_benchmark_seed_df.to_csv(tables / "synthetic_benchmark_seed_metrics.csv", index=False)
    synthetic_benchmark_metrics.to_csv(tables / "synthetic_benchmark_metrics.csv", index=False)
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
        synthetic_benchmark_df=synthetic_benchmark_metrics,
    )
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
    summary = {
        "mode": mode,
        "ns": ns,
        "seeds": seeds,
        "stress_seeds": stress_seeds,
        "n_seed_rows": int(seed_df.shape[0]),
        "n_main_rows": int(main.shape[0]),
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
        "n_synthetic_benchmark_rows": int(synthetic_benchmark_seed_df.shape[0]),
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
    claims = write_claim_status(root)
    summary["passes_claim_audit"] = claims["passes_claim_audit"]
    summary["runtime_seconds"] = round(time.time() - start, 3)
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_claim_status(root)
    write_final_audit(root, command_results={f"experiments --mode {mode}": "pass"})
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
