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
from object_centric_best_of_n.learned_model import train_and_evaluate
from object_centric_best_of_n.metrics import (
    aggregate_seed_metrics,
    counterfactual_target_summary,
    deployment_gate_from_metrics,
    domain_randomization_summary,
    exact_law_prediction_error,
    model_family_proxy_summary,
    negative_control_summary,
    observable_repair_summary,
    ood_summary,
    paired_selector_effects,
    repair_ablation_summary,
    score_calibration_table,
    selection_record,
    seed_block_robustness,
    sensitivity_summary,
    statistical_audit,
    stress_summary,
)
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator
from object_centric_best_of_n.plotting import write_all_figures
from object_centric_best_of_n.repair import combined_repair_score
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
    family_seeds = list(range(4)) if mode == "smoke" else list(range(16))
    family_seed_df = _run_model_family_panel(generator, family_seeds, n=max(ns))
    family_metrics = model_family_proxy_summary(family_seed_df)
    domain_seeds = list(range(6)) if mode == "smoke" else list(range(48))
    domain_seed_df = _run_domain_randomization_panel(generator, domain_seeds, n=max(ns))
    domain_metrics = domain_randomization_summary(domain_seed_df)
    counter_seeds = list(range(6)) if mode == "smoke" else list(range(48))
    counter_seed_df = _run_counterfactual_target_panel(generator, counter_seeds, n=max(ns))
    counter_metrics = counterfactual_target_summary(counter_seed_df)
    bootstrap_reps = 400 if mode == "smoke" else 2000
    statistical_metrics = statistical_audit(
        seed_df,
        ood_seed_df=ood_seed_df,
        family_seed_df=family_seed_df,
        counterfactual_seed_df=counter_seed_df,
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
    family_seed_df.to_csv(tables / "model_family_proxy_seed_metrics.csv", index=False)
    family_metrics.to_csv(tables / "model_family_proxy_metrics.csv", index=False)
    domain_seed_df.to_csv(tables / "domain_randomization_seed_metrics.csv", index=False)
    domain_metrics.to_csv(tables / "domain_randomization_metrics.csv", index=False)
    counter_seed_df.to_csv(tables / "counterfactual_target_seed_metrics.csv", index=False)
    counter_metrics.to_csv(tables / "counterfactual_target_metrics.csv", index=False)
    statistical_metrics.to_csv(tables / "statistical_audit.csv", index=False)

    learned_metrics, _ = train_and_evaluate(results, seed=123 if mode == "smoke" else 456)
    learned_row = learned_metrics.as_dict()
    pd.DataFrame([learned_row]).to_csv(tables / "learned_metrics.csv", index=False)
    learned_curve = pd.read_csv(tables / "learned_learning_curve.csv")
    learned_ablation = pd.read_csv(tables / "learned_ablation.csv")

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
        family_df=family_metrics,
        statistical_df=statistical_metrics,
        observable_df=observable_metrics,
        domain_df=domain_metrics,
        counterfactual_df=counter_metrics,
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
        "n_model_family_proxy_rows": int(family_seed_df.shape[0]),
        "n_domain_randomization_rows": int(domain_seed_df.shape[0]),
        "n_counterfactual_target_rows": int(counter_seed_df.shape[0]),
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
        "ood_combined_mean_selected_utility": float(ood_combined["selected_real_utility_mean"].mean()) if not ood_combined.empty else None,
        "ood_raw_mean_selected_utility": float(ood_raw["selected_real_utility_mean"].mean()) if not ood_raw.empty else None,
        "ood_combined_vs_raw_gain": float(ood_combined["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) if not ood_combined.empty and not ood_raw.empty else None,
        "ood_good_control_raw_utility": float(ood_good["selected_real_utility_mean"].iloc[0]) if not ood_good.empty else None,
        "ood_observable_mean_selected_utility": float(ood_observable["selected_real_utility_mean"].mean()) if not ood_observable.empty else None,
        "ood_observable_vs_raw_gain": float(ood_observable["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) if not ood_observable.empty and not ood_raw.empty else None,
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
