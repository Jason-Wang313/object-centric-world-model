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
from object_centric_best_of_n.envs import make_scene
from object_centric_best_of_n.learned_model import train_and_evaluate
from object_centric_best_of_n.metrics import (
    aggregate_seed_metrics,
    deployment_gate_from_metrics,
    exact_law_prediction_error,
    negative_control_summary,
    paired_selector_effects,
    repair_ablation_summary,
    score_calibration_table,
    selection_record,
    seed_block_robustness,
    sensitivity_summary,
    stress_summary,
)
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator
from object_centric_best_of_n.plotting import write_all_figures
from object_centric_best_of_n.repair import combined_repair_score
from object_centric_best_of_n.selection import SELECTORS
from object_centric_best_of_n.theory import law_validation_row


SCENARIOS = ["good", "swap", "merge_split", "occlusion", "hidden_property", "raw"]
SELECTOR_ORDER = ["raw", "identity_consistent", "property_calibrated", "targeted_probe", "combined_repair", "random", "oracle"]
STRESS_SCENARIOS = ["raw", "occlusion", "hidden_property", "swap", "merge_split"]
STRESS_SELECTORS = ["raw", "identity_consistent", "targeted_probe", "combined_repair", "random", "oracle"]
SENSITIVITY_NOISE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]


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
    repair_metrics = main[main["selector"].isin(["raw", "identity_consistent", "property_calibrated", "targeted_probe", "combined_repair", "random", "oracle"])].copy()
    paired_effects = paired_selector_effects(seed_df)
    law_df = pd.DataFrame(law_rows)
    stress_seeds = list(range(4)) if mode == "smoke" else list(range(32))
    stress_seed_df = _run_stress_panel(generator, stress_seeds=stress_seeds, n=max(ns))
    stress_metrics = stress_summary(stress_seed_df)
    sensitivity_seeds = list(range(4)) if mode == "smoke" else list(range(24))
    sensitivity_seed_df, calibration_candidate_df = _run_sensitivity_panel(generator, sensitivity_seeds, n=max(ns), mode=mode)
    sensitivity_metrics = sensitivity_summary(sensitivity_seed_df)
    calibration_metrics = score_calibration_table(calibration_candidate_df)
    ablation_metrics = repair_ablation_summary(main, paired_effects)
    robustness_metrics = seed_block_robustness(seed_df, block_size=2 if mode == "smoke" else 4)
    negative_control = negative_control_summary(main)

    seed_df.to_csv(tables / "seed_metrics.csv", index=False)
    main.to_csv(tables / "main_metrics.csv", index=False)
    repair_metrics.to_csv(tables / "repair_metrics.csv", index=False)
    paired_effects.to_csv(tables / "paired_effects.csv", index=False)
    ablation_metrics.to_csv(tables / "repair_ablation.csv", index=False)
    robustness_metrics.to_csv(tables / "seed_block_robustness.csv", index=False)
    negative_control.to_csv(tables / "negative_control.csv", index=False)
    law_df.to_csv(tables / "exact_law_validation.csv", index=False)
    stress_seed_df.to_csv(tables / "stress_seed_metrics.csv", index=False)
    stress_metrics.to_csv(tables / "stress_metrics.csv", index=False)
    sensitivity_seed_df.to_csv(tables / "sensitivity_seed_metrics.csv", index=False)
    sensitivity_metrics.to_csv(tables / "sensitivity_metrics.csv", index=False)
    calibration_candidate_df.to_csv(tables / "score_calibration_candidates.csv", index=False)
    calibration_metrics.to_csv(tables / "score_calibration.csv", index=False)

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
        "deployment_gate": gate,
        "exact_law_mean_absolute_error": exact_law_prediction_error(law_df),
        "raw_tail_score_gain": raw_tail_score_gain,
        "raw_tail_utility_drop": raw_tail_utility_drop,
        "combined_repair_raw_nmax_mean_gain": float(raw_combined_nmax["mean_gain"].iloc[0]) if not raw_combined_nmax.empty else None,
        "combined_repair_raw_nmax_win_rate": float(raw_combined_nmax["win_rate"].iloc[0]) if not raw_combined_nmax.empty else None,
        "combined_repair_raw_ablation_dominance": float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) if not raw_ablation.empty else None,
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
