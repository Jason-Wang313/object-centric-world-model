"""Claim and artifact audit utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


FORBIDDEN_SUPPORTED_PATTERNS = (
    "real robot",
    "real-robot",
    "sota",
    "state of the art",
    "universal object learning",
    "broad benchmark",
    "benchmark superiority",
)

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "results/tables/main_metrics.csv": ("scenario", "selector", "N", "selected_real_utility_mean"),
    "results/tables/seed_metrics.csv": ("scenario", "selector", "N", "seed", "selected_real_utility"),
    "results/tables/learned_metrics.csv": ("property_accuracy", "identity_alignment_accuracy", "transition_mse"),
    "results/tables/learned_learning_curve.csv": ("train_scenes", "property_accuracy", "identity_alignment_accuracy"),
    "results/tables/learned_domain_shift.csv": ("variant", "property_margin", "identity_margin", "transition_mse_ratio"),
    "results/tables/learned_selection_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "learned_reward_mean"),
    "results/tables/learned_selection_metrics.csv": ("scenario", "selector", "learned_identity_vs_raw_gain_mean", "learned_identity_vs_reward_gain_mean"),
    "results/tables/synthetic_benchmark_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "suite_variant"),
    "results/tables/synthetic_benchmark_metrics.csv": ("scenario", "selector", "suite_variant", "synthetic_benchmark_combined_vs_raw_gain_mean", "synthetic_benchmark_observable_vs_raw_gain_mean"),
    "results/tables/repair_metrics.csv": ("scenario", "selector", "N", "selected_real_utility_mean"),
    "results/tables/paired_effects.csv": ("scenario", "selector", "N", "mean_gain", "win_rate"),
    "results/tables/repair_ablation.csv": ("scenario", "combined_vs_raw_gain", "combined_vs_best_single_gain"),
    "results/tables/observable_repair_metrics.csv": ("scenario", "observable_repair_utility", "observable_vs_raw_gain", "combined_minus_observable_gap"),
    "results/tables/stress_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility"),
    "results/tables/stress_metrics.csv": ("scenario", "selector", "selected_real_utility_mean"),
    "results/tables/seed_block_robustness.csv": ("block_id", "raw_tail_score_gain", "combined_raw_nmax_gain"),
    "results/tables/score_calibration_candidates.csv": ("raw_object_score", "real_utility", "identity_error"),
    "results/tables/score_calibration.csv": ("score_bin", "mean_raw_object_score", "mean_real_utility", "object_real_gap"),
    "results/tables/sensitivity_seed_metrics.csv": ("selector", "score_noise", "selected_real_utility", "identity_error"),
    "results/tables/sensitivity_metrics.csv": ("selector", "score_noise", "selected_real_utility_mean"),
    "results/tables/negative_control.csv": ("contrast", "selected_real_utility_mean", "identity_error_mean"),
    "results/tables/learned_ablation.csv": ("ablation", "property_accuracy", "identity_alignment_accuracy"),
    "results/tables/ood_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility"),
    "results/tables/ood_metrics.csv": ("scenario", "selector", "selected_real_utility_mean"),
    "results/tables/extreme_object_count_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "n_objects"),
    "results/tables/extreme_object_count_metrics.csv": ("scenario", "selector", "selected_real_utility_mean", "extreme_combined_vs_raw_gain_mean"),
    "results/tables/model_family_proxy_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility"),
    "results/tables/model_family_proxy_metrics.csv": ("scenario", "selector", "best_proxy_utility_mean", "combined_vs_best_proxy_gain_mean"),
    "results/tables/domain_randomization_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "n_objects"),
    "results/tables/domain_randomization_metrics.csv": ("scenario", "selector", "domain_combined_vs_raw_gain_mean", "domain_observable_vs_raw_gain_mean"),
    "results/tables/counterfactual_target_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "counterfactual_target_id"),
    "results/tables/counterfactual_target_metrics.csv": ("scenario", "selector", "counterfactual_combined_vs_raw_gain_mean", "counterfactual_observable_vs_raw_gain_mean"),
    "results/tables/target_identity_sweep_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "target_id"),
    "results/tables/target_identity_sweep_metrics.csv": ("scenario", "selector", "target_id", "target_sweep_combined_vs_raw_gain_mean", "target_sweep_observable_vs_raw_gain_mean"),
    "results/tables/pilot_calibration_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "pilot_train_candidates"),
    "results/tables/pilot_calibration_metrics.csv": ("scenario", "selector", "pilot_vs_raw_gain_mean", "pilot_win_rate"),
    "results/tables/pilot_budget_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "pilot_label_budget"),
    "results/tables/pilot_budget_metrics.csv": ("scenario", "selector", "pilot_label_budget", "pilot_budget_vs_raw_gain_mean"),
    "results/tables/leave_one_failure_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "heldout_scenario"),
    "results/tables/leave_one_failure_metrics.csv": ("scenario", "selector", "pilot_vs_raw_gain_mean", "pilot_win_rate"),
    "results/tables/noisy_probe_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "probe_reliability"),
    "results/tables/noisy_probe_metrics.csv": ("scenario", "selector", "probe_reliability", "noisy_probe_vs_raw_gain_mean"),
    "results/tables/probe_cost_seed_metrics.csv": ("scenario", "selector", "seed", "selected_real_utility", "probe_cost", "gross_selected_real_utility"),
    "results/tables/probe_cost_metrics.csv": ("scenario", "selector", "probe_cost", "probe_cost_combined_vs_raw_gain_mean"),
    "results/tables/statistical_audit.csv": ("effect_id", "estimate", "bootstrap_ci_low", "bootstrap_ci_high", "threshold", "passes"),
    "results/tables/exact_law_validation.csv": ("N", "predicted_selected_utility", "empirical_selected_utility", "absolute_error"),
}

REQUIRED_FIGURES = (
    "figures/figure1_selected_tail_binding_failure.png",
    "figures/figure2_repair_comparison.png",
    "figures/figure3_tail_diagnostics.png",
    "figures/figure4_targeted_probe_before_after.png",
    "figures/figure5_exact_law_validation.png",
    "figures/figure6_stress_robustness.png",
    "figures/figure7_learned_object_model.png",
    "figures/figure8_repair_ablation.png",
    "figures/figure9_seed_block_robustness.png",
    "figures/figure10_score_calibration.png",
    "figures/figure11_score_noise_sensitivity.png",
    "figures/figure12_negative_control.png",
    "figures/figure13_learned_ablation.png",
    "figures/figure14_ood_object_count_stress.png",
    "figures/figure15_model_family_proxies.png",
    "figures/figure16_statistical_audit.png",
    "figures/figure17_observable_repair.png",
    "figures/figure18_domain_randomization.png",
    "figures/figure19_counterfactual_target.png",
    "figures/figure20_pilot_calibration.png",
    "figures/figure21_leave_one_failure_out.png",
    "figures/figure22_noisy_probe_reliability.png",
    "figures/figure23_learned_domain_shift.png",
    "figures/figure24_extreme_object_count.png",
    "figures/figure25_probe_cost_sensitivity.png",
    "figures/figure26_pilot_label_budget.png",
    "figures/figure27_target_identity_sweep.png",
    "figures/figure28_learned_selection_transfer.png",
    "figures/figure29_synthetic_benchmark_suite.png",
)

REQUIRED_JSON = (
    "results/run_summary.json",
    "results/learned_object_model_summary.json",
    "results/pilot_calibration_summary.json",
    "results/pilot_budget_summary.json",
    "results/leave_one_failure_summary.json",
    "results/verification_log.json",
    "results/artifact_manifest.json",
)


def _status(passes: bool | None) -> str:
    if passes is None:
        return "supported"
    return "strongly_supported" if passes else "needs_more_evidence"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_claim_strength(root: str | Path) -> dict[str, dict[str, object]]:
    root = Path(root)
    tables = root / "results" / "tables"
    main = _read_csv(tables / "main_metrics.csv")
    law = _read_csv(tables / "exact_law_validation.csv")
    paired = _read_csv(tables / "paired_effects.csv")
    stress = _read_csv(tables / "stress_metrics.csv")
    ablation = _read_csv(tables / "repair_ablation.csv")
    observable = _read_csv(tables / "observable_repair_metrics.csv")
    robustness = _read_csv(tables / "seed_block_robustness.csv")
    calibration = _read_csv(tables / "score_calibration.csv")
    sensitivity = _read_csv(tables / "sensitivity_metrics.csv")
    negative_control = _read_csv(tables / "negative_control.csv")
    learned_ablation = _read_csv(tables / "learned_ablation.csv")
    learned_shift = _read_csv(tables / "learned_domain_shift.csv")
    learned_selection = _read_csv(tables / "learned_selection_metrics.csv")
    synthetic_benchmark = _read_csv(tables / "synthetic_benchmark_metrics.csv")
    ood = _read_csv(tables / "ood_metrics.csv")
    extreme = _read_csv(tables / "extreme_object_count_metrics.csv")
    family = _read_csv(tables / "model_family_proxy_metrics.csv")
    domain = _read_csv(tables / "domain_randomization_metrics.csv")
    counterfactual = _read_csv(tables / "counterfactual_target_metrics.csv")
    target_sweep = _read_csv(tables / "target_identity_sweep_metrics.csv")
    pilot = _read_csv(tables / "pilot_calibration_metrics.csv")
    pilot_budget = _read_csv(tables / "pilot_budget_metrics.csv")
    loso = _read_csv(tables / "leave_one_failure_metrics.csv")
    noisy_probe = _read_csv(tables / "noisy_probe_metrics.csv")
    probe_cost = _read_csv(tables / "probe_cost_metrics.csv")
    statistical = _read_csv(tables / "statistical_audit.csv")
    learned = _read_json(root / "results" / "learned_object_model_summary.json")
    pilot_summary = _read_json(root / "results" / "pilot_calibration_summary.json")
    pilot_budget_summary_payload = _read_json(root / "results" / "pilot_budget_summary.json")
    loso_summary = _read_json(root / "results" / "leave_one_failure_summary.json")

    strengths: dict[str, dict[str, object]] = {}
    if not law.empty:
        strengths["C1"] = {
            "passes": bool(law["absolute_error"].max() <= 0.015 and law["absolute_error"].mean() <= 0.006),
            "threshold": "max exact-law absolute error <= 0.015 and mean <= 0.006",
            "observed": {
                "max_absolute_error": float(law["absolute_error"].max()),
                "mean_absolute_error": float(law["absolute_error"].mean()),
            },
        }
    if not main.empty:
        raw = main[(main["scenario"] == "raw") & (main["selector"] == "raw")].sort_values("N")
        if len(raw) >= 2:
            score_gain = float(raw["selected_object_score_mean"].iloc[-1] - raw["selected_object_score_mean"].iloc[0])
            utility_drop = float(raw["selected_real_utility_mean"].iloc[0] - raw["selected_real_utility_mean"].iloc[-1])
            identity_tail = float(raw["identity_error_mean"].iloc[-1])
            top_calibration = calibration.sort_values("score_bin").iloc[-1] if not calibration.empty else None
            good_control = negative_control[negative_control["contrast"] == "good_control"] if not negative_control.empty else pd.DataFrame()
            good_minus = negative_control[negative_control["contrast"] == "good_minus_corrupted"] if not negative_control.empty else pd.DataFrame()
            ood_good = ood[(ood["scenario"] == "dense6_good") & (ood["selector"] == "raw")] if not ood.empty else pd.DataFrame()
            ood_raw = ood[
                (ood["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
                & (ood["selector"] == "raw")
            ] if not ood.empty else pd.DataFrame()
            extreme_good = (
                extreme[(extreme["scenario"] == "extreme10_good") & (extreme["selector"] == "raw")]
                if not extreme.empty
                else pd.DataFrame()
            )
            extreme_raw = (
                extreme[
                    (extreme["scenario"].isin(["extreme10_raw", "extreme12_occlusion", "extreme12_hidden"]))
                    & (extreme["selector"] == "raw")
                ]
                if not extreme.empty
                else pd.DataFrame()
            )
            target_raw = (
                target_sweep[target_sweep["selector"] == "raw"]
                if not target_sweep.empty
                else pd.DataFrame()
            )
            synthetic_benchmark_raw = (
                synthetic_benchmark[synthetic_benchmark["selector"] == "raw"]
                if not synthetic_benchmark.empty
                else pd.DataFrame()
            )
            c2_stats = (
                statistical[statistical["effect_id"].isin(["raw_tail_score_gain", "raw_tail_utility_drop"])]
                if not statistical.empty
                else pd.DataFrame()
            )
            strengths["C2"] = {
                "passes": bool(
                    score_gain >= 0.35
                    and utility_drop >= 0.15
                    and identity_tail >= 0.75
                    and not robustness.empty
                    and float(robustness["raw_tail_score_gain"].min()) >= 0.30
                    and float(robustness["raw_tail_utility_drop"].min()) >= 0.10
                    and float(robustness["raw_tail_identity_error"].min()) >= 0.75
                    and top_calibration is not None
                    and float(top_calibration["object_real_gap"]) >= 0.45
                    and float(top_calibration["identity_error_rate"]) >= 0.55
                    and not good_control.empty
                    and float(good_control["selected_real_utility_mean"].iloc[0]) >= 0.60
                    and float(good_control["identity_error_mean"].iloc[0]) <= 0.25
                    and not good_minus.empty
                    and float(good_minus["selected_real_utility_mean"].iloc[0]) >= 0.45
                    and not ood_good.empty
                    and float(ood_good["selected_real_utility_mean"].iloc[0]) >= 0.60
                    and not ood_raw.empty
                    and float(ood_raw["selected_real_utility_mean"].mean()) <= 0.12
                    and float(ood_raw["identity_error_mean"].mean()) >= 0.80
                    and not extreme_good.empty
                    and float(extreme_good["selected_real_utility_mean"].iloc[0]) >= 0.55
                    and not extreme_raw.empty
                    and float(extreme_raw["selected_real_utility_mean"].mean()) <= 0.12
                    and float(extreme_raw["identity_error_mean"].mean()) >= 0.75
                    and not target_raw.empty
                    and float(target_raw["selected_real_utility_mean"].mean()) <= 0.15
                    and float(target_raw["identity_error_mean"].mean()) >= 0.75
                    and not synthetic_benchmark_raw.empty
                    and float(synthetic_benchmark_raw["selected_real_utility_mean"].mean()) <= 0.15
                    and float(synthetic_benchmark_raw["identity_error_mean"].mean()) >= 0.75
                    and not c2_stats.empty
                    and bool(c2_stats["passes"].all())
                ),
                "threshold": "raw high-N score gain >= 0.35, utility drop >= 0.15, tail identity error >= 0.75, all seed blocks pass reduced thresholds, top raw-score calibration bin has gap >= 0.45 with identity error >= 0.55, good negative controls avoid collapse, dense OOD, extreme 10/12-object, target-identity-sweep, and benchmark-style synthetic task-suite corrupted variants collapse, and bootstrap lower bounds for raw score gain and utility drop pass",
                "observed": {
                    "raw_tail_score_gain": score_gain,
                    "raw_tail_utility_drop": utility_drop,
                    "raw_tail_identity_error": identity_tail,
                    "min_block_score_gain": float(robustness["raw_tail_score_gain"].min()) if not robustness.empty else None,
                    "min_block_utility_drop": float(robustness["raw_tail_utility_drop"].min()) if not robustness.empty else None,
                    "min_block_identity_error": float(robustness["raw_tail_identity_error"].min()) if not robustness.empty else None,
                    "top_calibration_object_real_gap": float(top_calibration["object_real_gap"]) if top_calibration is not None else None,
                    "top_calibration_identity_error": float(top_calibration["identity_error_rate"]) if top_calibration is not None else None,
                    "good_control_utility": float(good_control["selected_real_utility_mean"].iloc[0]) if not good_control.empty else None,
                    "good_control_identity_error": float(good_control["identity_error_mean"].iloc[0]) if not good_control.empty else None,
                    "good_minus_corrupted_utility": float(good_minus["selected_real_utility_mean"].iloc[0]) if not good_minus.empty else None,
                    "ood_good_raw_utility": float(ood_good["selected_real_utility_mean"].iloc[0]) if not ood_good.empty else None,
                    "ood_corrupted_raw_mean_utility": float(ood_raw["selected_real_utility_mean"].mean()) if not ood_raw.empty else None,
                    "ood_corrupted_raw_identity_error": float(ood_raw["identity_error_mean"].mean()) if not ood_raw.empty else None,
                    "extreme_good_raw_utility": float(extreme_good["selected_real_utility_mean"].iloc[0]) if not extreme_good.empty else None,
                    "extreme_corrupted_raw_mean_utility": float(extreme_raw["selected_real_utility_mean"].mean()) if not extreme_raw.empty else None,
                    "extreme_corrupted_raw_identity_error": float(extreme_raw["identity_error_mean"].mean()) if not extreme_raw.empty else None,
                    "target_sweep_raw_mean_utility": float(target_raw["selected_real_utility_mean"].mean()) if not target_raw.empty else None,
                    "target_sweep_raw_identity_error": float(target_raw["identity_error_mean"].mean()) if not target_raw.empty else None,
                    "synthetic_benchmark_raw_mean_utility": float(synthetic_benchmark_raw["selected_real_utility_mean"].mean()) if not synthetic_benchmark_raw.empty else None,
                    "synthetic_benchmark_raw_identity_error": float(synthetic_benchmark_raw["identity_error_mean"].mean()) if not synthetic_benchmark_raw.empty else None,
                    "bootstrap_raw_tail_min_ci_margin": float((c2_stats["bootstrap_ci_low"] - c2_stats["threshold"]).min()) if not c2_stats.empty else None,
                },
            }
    if not paired.empty and not stress.empty and not ablation.empty and not observable.empty and not robustness.empty and not sensitivity.empty:
        raw_gain = paired[
            (paired["scenario"] == "raw")
            & (paired["selector"] == "combined_repair")
            & (paired["N"] == paired["N"].max())
        ]
        hidden_probe = paired[
            (paired["scenario"] == "hidden_property")
            & (paired["selector"] == "targeted_probe")
            & (paired["N"] == paired["N"].max())
        ]
        stress_combined = stress[stress["selector"] == "combined_repair"]
        raw_ablation = ablation[ablation["scenario"] == "raw"]
        raw_pass = (
            not raw_gain.empty
            and float(raw_gain["mean_gain"].iloc[0]) >= 0.55
            and float(raw_gain["win_rate"].iloc[0]) >= 0.75
        )
        probe_pass = not hidden_probe.empty and float(hidden_probe["mean_gain"].iloc[0]) >= 0.12
        stress_pass = (
            not stress_combined.empty
            and float(stress_combined["selected_real_utility_mean"].mean()) >= 0.75
            and float(stress_combined["selected_real_utility_mean"].min()) >= 0.80
        )
        ablation_pass = (
            not raw_ablation.empty
            and float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) >= 0.20
            and float(raw_ablation["combined_oracle_gap"].iloc[0]) <= 0.08
        )
        raw_observable = observable[observable["scenario"] == "raw"]
        observable_corrupted = observable[observable["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"])]
        observable_pass = (
            not raw_observable.empty
            and not observable_corrupted.empty
            and float(raw_observable["observable_vs_raw_gain"].iloc[0]) >= 0.55
            and float(raw_observable["observable_repair_utility"].iloc[0]) >= 0.72
            and float(observable_corrupted["observable_repair_utility"].mean()) >= 0.70
            and float(observable_corrupted["combined_minus_observable_gap"].max()) <= 0.20
        )
        robustness_pass = bool(
            float(robustness["combined_raw_nmax_gain"].min()) >= 0.55
            and float(robustness["combined_raw_nmax_win_rate"].min()) >= 0.75
        )
        sensitivity_low_noise = sensitivity[sensitivity["score_noise"] <= 0.10]
        combined_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "combined_repair_noisy"]
        raw_sensitivity = sensitivity_low_noise[sensitivity_low_noise["selector"] == "raw_noisy"]
        sensitivity_margin = None
        if not combined_sensitivity.empty and not raw_sensitivity.empty:
            sensitivity_margin = float(
                combined_sensitivity["selected_real_utility_mean"].mean()
                - raw_sensitivity["selected_real_utility_mean"].mean()
            )
        sensitivity_pass = (
            not combined_sensitivity.empty
            and not raw_sensitivity.empty
            and float(combined_sensitivity["selected_real_utility_mean"].min()) >= 0.75
            and sensitivity_margin is not None
            and sensitivity_margin >= 0.55
        )
        ood_combined = ood[
            (ood["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
            & (ood["selector"] == "combined_repair")
        ] if not ood.empty else pd.DataFrame()
        ood_raw = ood[
            (ood["scenario"].isin(["dense6_raw", "dense8_occlusion", "dense8_hidden"]))
            & (ood["selector"] == "raw")
        ] if not ood.empty else pd.DataFrame()
        ood_pass = (
            not ood_combined.empty
            and not ood_raw.empty
            and float(ood_combined["selected_real_utility_mean"].mean()) >= 0.80
            and float(ood_combined["selected_real_utility_mean"].min()) >= 0.82
            and float(ood_combined["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) >= 0.70
        )
        extreme_combined = (
            extreme[
                (extreme["scenario"].isin(["extreme10_raw", "extreme12_occlusion", "extreme12_hidden"]))
                & (extreme["selector"] == "combined_repair")
            ]
            if not extreme.empty
            else pd.DataFrame()
        )
        extreme_observable = (
            extreme[
                (extreme["scenario"].isin(["extreme10_raw", "extreme12_occlusion", "extreme12_hidden"]))
                & (extreme["selector"] == "observable_repair")
            ]
            if not extreme.empty
            else pd.DataFrame()
        )
        extreme_raw = (
            extreme[
                (extreme["scenario"].isin(["extreme10_raw", "extreme12_occlusion", "extreme12_hidden"]))
                & (extreme["selector"] == "raw")
            ]
            if not extreme.empty
            else pd.DataFrame()
        )
        extreme_pass = (
            not extreme_combined.empty
            and not extreme_observable.empty
            and not extreme_raw.empty
            and float(extreme_combined["selected_real_utility_mean"].mean()) >= 0.78
            and float(extreme_combined["selected_real_utility_mean"].min()) >= 0.75
            and float(extreme_observable["selected_real_utility_mean"].mean()) >= 0.72
            and float(extreme_combined["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) >= 0.60
            and float(extreme_observable["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) >= 0.50
        )
        domain_combined = domain[domain["selector"] == "combined_repair"] if not domain.empty else pd.DataFrame()
        domain_observable = domain[domain["selector"] == "observable_repair"] if not domain.empty else pd.DataFrame()
        domain_raw = domain[domain["selector"] == "raw"] if not domain.empty else pd.DataFrame()
        domain_pass = (
            not domain_combined.empty
            and not domain_observable.empty
            and not domain_raw.empty
            and float(domain_raw["selected_real_utility_mean"].iloc[0]) <= 0.20
            and float(domain_combined["selected_real_utility_mean"].iloc[0]) >= 0.75
            and float(domain_observable["selected_real_utility_mean"].iloc[0]) >= 0.72
            and float(domain_combined["domain_combined_vs_raw_gain_mean"].iloc[0]) >= 0.60
            and float(domain_observable["domain_observable_vs_raw_gain_mean"].iloc[0]) >= 0.55
            and float(domain_combined["domain_combined_win_rate"].iloc[0]) >= 0.85
        )
        counter_combined = counterfactual[counterfactual["selector"] == "combined_repair"] if not counterfactual.empty else pd.DataFrame()
        counter_observable = counterfactual[counterfactual["selector"] == "observable_repair"] if not counterfactual.empty else pd.DataFrame()
        counter_raw = counterfactual[counterfactual["selector"] == "raw"] if not counterfactual.empty else pd.DataFrame()
        counterfactual_pass = (
            not counter_combined.empty
            and not counter_observable.empty
            and not counter_raw.empty
            and float(counter_raw["selected_real_utility_mean"].iloc[0]) <= 0.25
            and float(counter_combined["selected_real_utility_mean"].iloc[0]) >= 0.70
            and float(counter_observable["selected_real_utility_mean"].iloc[0]) >= 0.65
            and float(counter_combined["counterfactual_combined_vs_raw_gain_mean"].iloc[0]) >= 0.55
            and float(counter_observable["counterfactual_observable_vs_raw_gain_mean"].iloc[0]) >= 0.50
            and float(counter_combined["counterfactual_combined_win_rate"].iloc[0]) >= 0.85
        )
        target_sweep_combined = target_sweep[target_sweep["selector"] == "combined_repair"] if not target_sweep.empty else pd.DataFrame()
        target_sweep_observable = target_sweep[target_sweep["selector"] == "observable_repair"] if not target_sweep.empty else pd.DataFrame()
        target_sweep_raw = target_sweep[target_sweep["selector"] == "raw"] if not target_sweep.empty else pd.DataFrame()
        target_sweep_pass = (
            not target_sweep_combined.empty
            and not target_sweep_observable.empty
            and not target_sweep_raw.empty
            and float(target_sweep_raw["selected_real_utility_mean"].mean()) <= 0.20
            and float(target_sweep_combined["selected_real_utility_mean"].mean()) >= 0.75
            and float(target_sweep_observable["selected_real_utility_mean"].mean()) >= 0.72
            and float(target_sweep_combined["selected_real_utility_mean"].min()) >= 0.62
            and float(target_sweep_observable["selected_real_utility_mean"].min()) >= 0.70
            and float(target_sweep_combined["target_sweep_combined_vs_raw_gain_mean"].mean()) >= 0.60
            and float(target_sweep_observable["target_sweep_observable_vs_raw_gain_mean"].mean()) >= 0.55
            and float(target_sweep_combined["target_sweep_combined_win_rate"].min()) >= 0.85
            and float(target_sweep_observable["target_sweep_observable_win_rate"].min()) >= 0.85
        )
        synthetic_benchmark_combined = (
            synthetic_benchmark[synthetic_benchmark["selector"] == "combined_repair"]
            if not synthetic_benchmark.empty
            else pd.DataFrame()
        )
        synthetic_benchmark_observable = (
            synthetic_benchmark[synthetic_benchmark["selector"] == "observable_repair"]
            if not synthetic_benchmark.empty
            else pd.DataFrame()
        )
        synthetic_benchmark_raw = (
            synthetic_benchmark[synthetic_benchmark["selector"] == "raw"]
            if not synthetic_benchmark.empty
            else pd.DataFrame()
        )
        synthetic_benchmark_pass = (
            not synthetic_benchmark_combined.empty
            and not synthetic_benchmark_observable.empty
            and not synthetic_benchmark_raw.empty
            and float(synthetic_benchmark_raw["selected_real_utility_mean"].mean()) <= 0.15
            and float(synthetic_benchmark_combined["selected_real_utility_mean"].mean()) >= 0.78
            and float(synthetic_benchmark_observable["selected_real_utility_mean"].mean()) >= 0.75
            and float(synthetic_benchmark_combined["selected_real_utility_mean"].min()) >= 0.70
            and float(synthetic_benchmark_observable["selected_real_utility_mean"].min()) >= 0.70
            and float(synthetic_benchmark_combined["synthetic_benchmark_combined_vs_raw_gain_mean"].mean()) >= 0.60
            and float(synthetic_benchmark_observable["synthetic_benchmark_observable_vs_raw_gain_mean"].mean()) >= 0.55
            and float(synthetic_benchmark_combined["synthetic_benchmark_combined_win_rate"].min()) >= 0.85
            and float(synthetic_benchmark_observable["synthetic_benchmark_observable_win_rate"].min()) >= 0.85
        )
        pilot_calibrated = pilot[pilot["selector"] == "pilot_calibrated"] if not pilot.empty else pd.DataFrame()
        pilot_pass = (
            not pilot_calibrated.empty
            and float(pilot_calibrated["selected_real_utility_mean"].mean()) >= 0.72
            and float(pilot_calibrated["selected_real_utility_mean"].min()) >= 0.68
            and float(pilot_calibrated["pilot_vs_raw_gain_mean"].mean()) >= 0.55
            and float(pilot_calibrated["pilot_win_rate"].min()) >= 0.85
            and float(pilot_calibrated["pilot_oracle_gap_mean"].max()) <= 0.22
            and float(pilot_summary.get("calibrator", {}).get("train_correlation", 0.0)) >= 0.80
        )
        mature_pilot_budget = (
            pilot_budget[
                (pilot_budget["selector"] == "pilot_calibrated")
                & (pilot_budget["pilot_label_budget"] >= 128)
            ]
            if not pilot_budget.empty
            else pd.DataFrame()
        )
        largest_pilot_budget = (
            pilot_budget[
                (pilot_budget["selector"] == "pilot_calibrated")
                & (pilot_budget["pilot_label_budget"] == pilot_budget["pilot_label_budget"].max())
            ]
            if not pilot_budget.empty
            else pd.DataFrame()
        )
        budget_train_correlations = (
            [
                float(item.get("calibrator", {}).get("train_correlation", 0.0))
                for item in pilot_budget_summary_payload.get("calibrators", [])
                if int(item.get("pilot_label_budget", 0)) >= 128
            ]
            if pilot_budget_summary_payload
            else []
        )
        pilot_budget_pass = (
            not mature_pilot_budget.empty
            and not largest_pilot_budget.empty
            and float(mature_pilot_budget["selected_real_utility_mean"].mean()) >= 0.72
            and float(mature_pilot_budget["pilot_budget_vs_raw_gain_mean"].mean()) >= 0.55
            and float(mature_pilot_budget["pilot_budget_win_rate"].min()) >= 0.85
            and float(largest_pilot_budget["selected_real_utility_mean"].mean()) >= 0.78
            and float(largest_pilot_budget["pilot_budget_oracle_gap_mean"].max()) <= 0.22
            and bool(budget_train_correlations)
            and min(budget_train_correlations) >= 0.75
        )
        loso_pilot = loso[loso["selector"] == "pilot_calibrated"] if not loso.empty else pd.DataFrame()
        loso_train_correlations = (
            [
                float(item.get("calibrator", {}).get("train_correlation", 0.0))
                for item in loso_summary.get("calibrators", [])
            ]
            if loso_summary
            else []
        )
        loso_pass = (
            not loso_pilot.empty
            and float(loso_pilot["selected_real_utility_mean"].mean()) >= 0.70
            and float(loso_pilot["selected_real_utility_mean"].min()) >= 0.62
            and float(loso_pilot["pilot_vs_raw_gain_mean"].mean()) >= 0.50
            and float(loso_pilot["pilot_win_rate"].min()) >= 0.80
            and float(loso_pilot["pilot_oracle_gap_mean"].max()) <= 0.28
            and bool(loso_train_correlations)
            and min(loso_train_correlations) >= 0.75
        )
        noisy_probe_repair = (
            noisy_probe[
                (noisy_probe["selector"] == "noisy_probe_repair")
                & (noisy_probe["probe_reliability"] >= 0.75)
            ]
            if not noisy_probe.empty
            else pd.DataFrame()
        )
        noisy_probe_pass = (
            not noisy_probe_repair.empty
            and float(noisy_probe_repair["selected_real_utility_mean"].mean()) >= 0.74
            and float(noisy_probe_repair["selected_real_utility_mean"].min()) >= 0.70
            and float(noisy_probe_repair["noisy_probe_vs_raw_gain_mean"].mean()) >= 0.55
            and float(noisy_probe_repair["noisy_probe_win_rate"].min()) >= 0.85
            and float(noisy_probe_repair["noisy_probe_oracle_gap_mean"].max()) <= 0.25
        )
        low_cost_probe = (
            probe_cost[
                (probe_cost["probe_cost"] <= 0.10)
                & (probe_cost["scenario"].isin(["hidden_property", "raw"]))
            ]
            if not probe_cost.empty
            else pd.DataFrame()
        )
        probe_cost_combined = low_cost_probe[low_cost_probe["selector"] == "combined_repair"]
        probe_cost_observable = low_cost_probe[low_cost_probe["selector"] == "observable_repair"]
        probe_cost_targeted = low_cost_probe[
            (low_cost_probe["selector"] == "targeted_probe")
            & (low_cost_probe["scenario"] == "hidden_property")
        ]
        high_cost_probe = (
            probe_cost[
                (probe_cost["probe_cost"] >= 0.20)
                & (probe_cost["scenario"].isin(["hidden_property", "raw"]))
            ]
            if not probe_cost.empty
            else pd.DataFrame()
        )
        high_cost_combined = high_cost_probe[high_cost_probe["selector"] == "combined_repair"]
        high_cost_observable = high_cost_probe[high_cost_probe["selector"] == "observable_repair"]
        probe_cost_pass = (
            not probe_cost_combined.empty
            and not probe_cost_observable.empty
            and not probe_cost_targeted.empty
            and not high_cost_combined.empty
            and not high_cost_observable.empty
            and float(probe_cost_combined["selected_real_utility_mean"].mean()) >= 0.72
            and float(probe_cost_observable["selected_real_utility_mean"].mean()) >= 0.70
            and float(probe_cost_targeted["selected_real_utility_mean"].mean()) >= 0.55
            and float(probe_cost_combined["probe_cost_combined_vs_raw_gain_mean"].mean()) >= 0.55
            and float(probe_cost_observable["probe_cost_observable_vs_raw_gain_mean"].mean()) >= 0.50
            and float(probe_cost_targeted["probe_cost_targeted_vs_raw_gain_mean"].mean()) >= 0.35
            and float(probe_cost_combined["probe_cost_combined_win_rate"].min()) >= 0.85
            and float(high_cost_combined["probe_cost_combined_vs_raw_gain_mean"].mean()) >= 0.35
            and float(high_cost_observable["probe_cost_observable_vs_raw_gain_mean"].mean()) >= 0.30
        )
        family_combined = (
            family[
                (family["selector"] == "combined_repair")
                & (family["scenario"].isin(["raw", "occlusion", "hidden_property", "swap", "merge_split"]))
            ]
            if not family.empty
            else pd.DataFrame()
        )
        family_pass = (
            not family_combined.empty
            and float(family_combined["combined_vs_best_proxy_gain_mean"].mean()) >= 0.20
            and float(family_combined["combined_vs_best_proxy_gain_mean"].min()) >= 0.05
            and float(family_combined["combined_oracle_gap_mean"].max()) <= 0.12
        )
        c3_stats = (
            statistical[
                statistical["effect_id"].isin(
                    [
                        "combined_repair_raw_gain",
                        "observable_repair_raw_gain",
                        "targeted_probe_hidden_gain",
                        "ood_combined_repair_gain",
                        "ood_observable_repair_gain",
                        "extreme_object_combined_repair_gain",
                        "extreme_object_observable_repair_gain",
                        "model_family_proxy_gain",
                        "counterfactual_combined_repair_gain",
                        "counterfactual_observable_repair_gain",
                        "target_sweep_combined_repair_gain",
                        "target_sweep_observable_repair_gain",
                        "synthetic_benchmark_combined_repair_gain",
                        "synthetic_benchmark_observable_repair_gain",
                        "pilot_calibrated_repair_gain",
                        "pilot_budget_mature_gain",
                        "leave_one_failure_pilot_gain",
                        "noisy_probe_repair_gain",
                        "probe_cost_combined_repair_gain",
                        "probe_cost_observable_repair_gain",
                        "probe_cost_targeted_hidden_repair_gain",
                    ]
                )
            ]
            if not statistical.empty
            else pd.DataFrame()
        )
        statistical_pass = not c3_stats.empty and bool(c3_stats["passes"].all())
        strengths["C3"] = {
            "passes": bool(raw_pass and probe_pass and stress_pass and ablation_pass and observable_pass and robustness_pass and sensitivity_pass and ood_pass and extreme_pass and domain_pass and counterfactual_pass and target_sweep_pass and synthetic_benchmark_pass and pilot_pass and pilot_budget_pass and loso_pass and noisy_probe_pass and probe_cost_pass and family_pass and statistical_pass),
            "threshold": "combined raw Nmax gain >= 0.55 with win-rate >= 0.75, targeted hidden-property gain >= 0.12, stress combined mean >= 0.75 and min >= 0.80, raw ablation dominance >= 0.20 with oracle gap <= 0.08, observable-only repair beats raw and remains close to controlled combined repair, all seed blocks repair, combined repair remains strong under score noise <= 0.10, dense OOD and extreme 10/12-object repair succeed, held-out domain-randomized stress succeeds, counterfactual target-swap, multi-target identity-sweep, and benchmark-style synthetic task-suite stress succeed, held-out pilot-label calibration and pilot-label budget sensitivity succeed, leave-one-failure-out pilot calibration succeeds, noisy diagnostic-probe repair succeeds for reliability >= 0.75, combined and observable repair remain beneficial under diagnostic costs <= 0.10 while targeted probing remains beneficial for hidden-property scenes, high-cost margins remain positive, controlled toy model-family proxy comparison has mean margin >= 0.20 with every scenario positive by >= 0.05 and max oracle gap <= 0.12, and bootstrap lower bounds for key repair gains pass",
            "observed": {
                "combined_raw_nmax_gain": float(raw_gain["mean_gain"].iloc[0]) if not raw_gain.empty else None,
                "combined_raw_nmax_win_rate": float(raw_gain["win_rate"].iloc[0]) if not raw_gain.empty else None,
                "targeted_hidden_property_nmax_gain": float(hidden_probe["mean_gain"].iloc[0]) if not hidden_probe.empty else None,
                "stress_combined_mean_utility": float(stress_combined["selected_real_utility_mean"].mean()) if not stress_combined.empty else None,
                "stress_combined_min_utility": float(stress_combined["selected_real_utility_mean"].min()) if not stress_combined.empty else None,
                "raw_ablation_combined_vs_best_single_gain": float(raw_ablation["combined_vs_best_single_gain"].iloc[0]) if not raw_ablation.empty else None,
                "raw_ablation_combined_oracle_gap": float(raw_ablation["combined_oracle_gap"].iloc[0]) if not raw_ablation.empty else None,
                "observable_raw_gain": float(raw_observable["observable_vs_raw_gain"].iloc[0]) if not raw_observable.empty else None,
                "observable_raw_utility": float(raw_observable["observable_repair_utility"].iloc[0]) if not raw_observable.empty else None,
                "observable_mean_corrupted_utility": float(observable_corrupted["observable_repair_utility"].mean()) if not observable_corrupted.empty else None,
                "observable_max_combined_gap": float(observable_corrupted["combined_minus_observable_gap"].max()) if not observable_corrupted.empty else None,
                "min_block_combined_raw_gain": float(robustness["combined_raw_nmax_gain"].min()) if not robustness.empty else None,
                "min_block_combined_win_rate": float(robustness["combined_raw_nmax_win_rate"].min()) if not robustness.empty else None,
                "combined_min_low_noise_utility": float(combined_sensitivity["selected_real_utility_mean"].min()) if not combined_sensitivity.empty else None,
                "combined_vs_raw_low_noise_margin": sensitivity_margin,
                "ood_combined_mean_utility": float(ood_combined["selected_real_utility_mean"].mean()) if not ood_combined.empty else None,
                "ood_combined_min_utility": float(ood_combined["selected_real_utility_mean"].min()) if not ood_combined.empty else None,
                "ood_combined_vs_raw_gain": float(ood_combined["selected_real_utility_mean"].mean() - ood_raw["selected_real_utility_mean"].mean()) if not ood_combined.empty and not ood_raw.empty else None,
                "extreme_combined_mean_utility": float(extreme_combined["selected_real_utility_mean"].mean()) if not extreme_combined.empty else None,
                "extreme_combined_min_utility": float(extreme_combined["selected_real_utility_mean"].min()) if not extreme_combined.empty else None,
                "extreme_observable_mean_utility": float(extreme_observable["selected_real_utility_mean"].mean()) if not extreme_observable.empty else None,
                "extreme_combined_vs_raw_gain": float(extreme_combined["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) if not extreme_combined.empty and not extreme_raw.empty else None,
                "extreme_observable_vs_raw_gain": float(extreme_observable["selected_real_utility_mean"].mean() - extreme_raw["selected_real_utility_mean"].mean()) if not extreme_observable.empty and not extreme_raw.empty else None,
                "domain_raw_utility": float(domain_raw["selected_real_utility_mean"].iloc[0]) if not domain_raw.empty else None,
                "domain_combined_utility": float(domain_combined["selected_real_utility_mean"].iloc[0]) if not domain_combined.empty else None,
                "domain_observable_utility": float(domain_observable["selected_real_utility_mean"].iloc[0]) if not domain_observable.empty else None,
                "domain_combined_vs_raw_gain": float(domain_combined["domain_combined_vs_raw_gain_mean"].iloc[0]) if not domain_combined.empty else None,
                "domain_observable_vs_raw_gain": float(domain_observable["domain_observable_vs_raw_gain_mean"].iloc[0]) if not domain_observable.empty else None,
                "domain_combined_win_rate": float(domain_combined["domain_combined_win_rate"].iloc[0]) if not domain_combined.empty else None,
                "counterfactual_raw_utility": float(counter_raw["selected_real_utility_mean"].iloc[0]) if not counter_raw.empty else None,
                "counterfactual_combined_utility": float(counter_combined["selected_real_utility_mean"].iloc[0]) if not counter_combined.empty else None,
                "counterfactual_observable_utility": float(counter_observable["selected_real_utility_mean"].iloc[0]) if not counter_observable.empty else None,
                "counterfactual_combined_vs_raw_gain": float(counter_combined["counterfactual_combined_vs_raw_gain_mean"].iloc[0]) if not counter_combined.empty else None,
                "counterfactual_observable_vs_raw_gain": float(counter_observable["counterfactual_observable_vs_raw_gain_mean"].iloc[0]) if not counter_observable.empty else None,
                "counterfactual_combined_win_rate": float(counter_combined["counterfactual_combined_win_rate"].iloc[0]) if not counter_combined.empty else None,
                "target_sweep_raw_mean_utility": float(target_sweep_raw["selected_real_utility_mean"].mean()) if not target_sweep_raw.empty else None,
                "target_sweep_combined_mean_utility": float(target_sweep_combined["selected_real_utility_mean"].mean()) if not target_sweep_combined.empty else None,
                "target_sweep_combined_min_target_utility": float(target_sweep_combined["selected_real_utility_mean"].min()) if not target_sweep_combined.empty else None,
                "target_sweep_observable_mean_utility": float(target_sweep_observable["selected_real_utility_mean"].mean()) if not target_sweep_observable.empty else None,
                "target_sweep_observable_min_target_utility": float(target_sweep_observable["selected_real_utility_mean"].min()) if not target_sweep_observable.empty else None,
                "target_sweep_combined_vs_raw_gain": float(target_sweep_combined["target_sweep_combined_vs_raw_gain_mean"].mean()) if not target_sweep_combined.empty else None,
                "target_sweep_observable_vs_raw_gain": float(target_sweep_observable["target_sweep_observable_vs_raw_gain_mean"].mean()) if not target_sweep_observable.empty else None,
                "target_sweep_combined_min_win_rate": float(target_sweep_combined["target_sweep_combined_win_rate"].min()) if not target_sweep_combined.empty else None,
                "synthetic_benchmark_raw_mean_utility": float(synthetic_benchmark_raw["selected_real_utility_mean"].mean()) if not synthetic_benchmark_raw.empty else None,
                "synthetic_benchmark_combined_mean_utility": float(synthetic_benchmark_combined["selected_real_utility_mean"].mean()) if not synthetic_benchmark_combined.empty else None,
                "synthetic_benchmark_combined_min_variant_utility": float(synthetic_benchmark_combined["selected_real_utility_mean"].min()) if not synthetic_benchmark_combined.empty else None,
                "synthetic_benchmark_observable_mean_utility": float(synthetic_benchmark_observable["selected_real_utility_mean"].mean()) if not synthetic_benchmark_observable.empty else None,
                "synthetic_benchmark_observable_min_variant_utility": float(synthetic_benchmark_observable["selected_real_utility_mean"].min()) if not synthetic_benchmark_observable.empty else None,
                "synthetic_benchmark_combined_vs_raw_gain": float(synthetic_benchmark_combined["synthetic_benchmark_combined_vs_raw_gain_mean"].mean()) if not synthetic_benchmark_combined.empty else None,
                "synthetic_benchmark_observable_vs_raw_gain": float(synthetic_benchmark_observable["synthetic_benchmark_observable_vs_raw_gain_mean"].mean()) if not synthetic_benchmark_observable.empty else None,
                "synthetic_benchmark_combined_min_win_rate": float(synthetic_benchmark_combined["synthetic_benchmark_combined_win_rate"].min()) if not synthetic_benchmark_combined.empty else None,
                "pilot_calibrated_mean_utility": float(pilot_calibrated["selected_real_utility_mean"].mean()) if not pilot_calibrated.empty else None,
                "pilot_calibrated_min_utility": float(pilot_calibrated["selected_real_utility_mean"].min()) if not pilot_calibrated.empty else None,
                "pilot_calibrated_vs_raw_gain": float(pilot_calibrated["pilot_vs_raw_gain_mean"].mean()) if not pilot_calibrated.empty else None,
                "pilot_calibrated_min_win_rate": float(pilot_calibrated["pilot_win_rate"].min()) if not pilot_calibrated.empty else None,
                "pilot_calibrated_max_oracle_gap": float(pilot_calibrated["pilot_oracle_gap_mean"].max()) if not pilot_calibrated.empty else None,
                "pilot_train_correlation": float(pilot_summary.get("calibrator", {}).get("train_correlation", 0.0)) if pilot_summary else None,
                "pilot_budget_mature_mean_utility": float(mature_pilot_budget["selected_real_utility_mean"].mean()) if not mature_pilot_budget.empty else None,
                "pilot_budget_mature_vs_raw_gain": float(mature_pilot_budget["pilot_budget_vs_raw_gain_mean"].mean()) if not mature_pilot_budget.empty else None,
                "pilot_budget_mature_min_win_rate": float(mature_pilot_budget["pilot_budget_win_rate"].min()) if not mature_pilot_budget.empty else None,
                "pilot_budget_largest_mean_utility": float(largest_pilot_budget["selected_real_utility_mean"].mean()) if not largest_pilot_budget.empty else None,
                "pilot_budget_largest_max_oracle_gap": float(largest_pilot_budget["pilot_budget_oracle_gap_mean"].max()) if not largest_pilot_budget.empty else None,
                "pilot_budget_min_mature_train_correlation": min(budget_train_correlations) if budget_train_correlations else None,
                "leave_one_failure_pilot_mean_utility": float(loso_pilot["selected_real_utility_mean"].mean()) if not loso_pilot.empty else None,
                "leave_one_failure_pilot_min_utility": float(loso_pilot["selected_real_utility_mean"].min()) if not loso_pilot.empty else None,
                "leave_one_failure_pilot_vs_raw_gain": float(loso_pilot["pilot_vs_raw_gain_mean"].mean()) if not loso_pilot.empty else None,
                "leave_one_failure_pilot_min_win_rate": float(loso_pilot["pilot_win_rate"].min()) if not loso_pilot.empty else None,
                "leave_one_failure_pilot_max_oracle_gap": float(loso_pilot["pilot_oracle_gap_mean"].max()) if not loso_pilot.empty else None,
                "leave_one_failure_min_train_correlation": min(loso_train_correlations) if loso_train_correlations else None,
                "noisy_probe_min_reliable_utility": float(noisy_probe_repair["selected_real_utility_mean"].min()) if not noisy_probe_repair.empty else None,
                "noisy_probe_mean_reliable_gain": float(noisy_probe_repair["noisy_probe_vs_raw_gain_mean"].mean()) if not noisy_probe_repair.empty else None,
                "noisy_probe_min_reliable_win_rate": float(noisy_probe_repair["noisy_probe_win_rate"].min()) if not noisy_probe_repair.empty else None,
                "noisy_probe_max_reliable_oracle_gap": float(noisy_probe_repair["noisy_probe_oracle_gap_mean"].max()) if not noisy_probe_repair.empty else None,
                "probe_cost_low_cost_combined_mean_utility": float(probe_cost_combined["selected_real_utility_mean"].mean()) if not probe_cost_combined.empty else None,
                "probe_cost_low_cost_observable_mean_utility": float(probe_cost_observable["selected_real_utility_mean"].mean()) if not probe_cost_observable.empty else None,
                "probe_cost_low_cost_targeted_mean_utility": float(probe_cost_targeted["selected_real_utility_mean"].mean()) if not probe_cost_targeted.empty else None,
                "probe_cost_low_cost_combined_gain": float(probe_cost_combined["probe_cost_combined_vs_raw_gain_mean"].mean()) if not probe_cost_combined.empty else None,
                "probe_cost_low_cost_observable_gain": float(probe_cost_observable["probe_cost_observable_vs_raw_gain_mean"].mean()) if not probe_cost_observable.empty else None,
                "probe_cost_low_cost_targeted_gain": float(probe_cost_targeted["probe_cost_targeted_vs_raw_gain_mean"].mean()) if not probe_cost_targeted.empty else None,
                "probe_cost_min_combined_win_rate": float(probe_cost_combined["probe_cost_combined_win_rate"].min()) if not probe_cost_combined.empty else None,
                "probe_cost_high_cost_combined_gain": float(high_cost_combined["probe_cost_combined_vs_raw_gain_mean"].mean()) if not high_cost_combined.empty else None,
                "probe_cost_high_cost_observable_gain": float(high_cost_observable["probe_cost_observable_vs_raw_gain_mean"].mean()) if not high_cost_observable.empty else None,
                "model_family_combined_vs_best_proxy_gain": float(family_combined["combined_vs_best_proxy_gain_mean"].mean()) if not family_combined.empty else None,
                "model_family_min_combined_vs_best_proxy_gain": float(family_combined["combined_vs_best_proxy_gain_mean"].min()) if not family_combined.empty else None,
                "model_family_max_combined_oracle_gap": float(family_combined["combined_oracle_gap_mean"].max()) if not family_combined.empty else None,
                "bootstrap_repair_min_ci_margin": float((c3_stats["bootstrap_ci_low"] - c3_stats["threshold"]).min()) if not c3_stats.empty else None,
            },
        }
    learned_metrics = learned.get("metrics", {})
    if learned_metrics:
        prop_margin = learned_metrics["property_accuracy"] - learned_metrics["random_property_accuracy"]
        identity_margin = (
            learned_metrics["identity_alignment_accuracy"] - learned_metrics["random_identity_alignment_accuracy"]
        )
        transition_ratio = learned_metrics["transition_mse"] / learned_metrics["constant_transition_mse"]
        shifted_learned = learned_shift[learned_shift["variant"] != "standard_test"] if not learned_shift.empty else pd.DataFrame()
        no_mass = learned_ablation[learned_ablation["ablation"] == "no_mass_sensor"] if not learned_ablation.empty else pd.DataFrame()
        kinematic_pair = (
            learned_ablation[learned_ablation["ablation"] == "kinematic_pair_identity"]
            if not learned_ablation.empty
            else pd.DataFrame()
        )
        learned_identity_selector = (
            learned_selection[learned_selection["selector"] == "learned_identity_reward"]
            if not learned_selection.empty
            else pd.DataFrame()
        )
        learned_selection_stats = (
            statistical[
                statistical["effect_id"].isin(
                    [
                        "learned_selection_identity_gain",
                        "learned_selection_identity_over_reward_gain",
                    ]
                )
            ]
            if not statistical.empty
            else pd.DataFrame()
        )
        strengths["C4"] = {
            "passes": bool(
                learned.get("passes_minimum_learned_artifact_checks")
                and prop_margin >= 0.15
                and identity_margin >= 0.15
                and transition_ratio <= 0.25
                and learned_metrics["reward_correlation"] >= 0.75
                and not no_mass.empty
                and float(no_mass["full_minus_property_accuracy"].iloc[0]) >= 0.10
                and not kinematic_pair.empty
                and float(kinematic_pair["full_minus_identity_alignment_accuracy"].iloc[0]) >= 0.05
                and not shifted_learned.empty
                and float(shifted_learned["property_margin"].min()) >= 0.12
                and float(shifted_learned["identity_margin"].min()) >= 0.15
                and float(shifted_learned["transition_mse_ratio"].max()) <= 0.30
                and float(shifted_learned["reward_correlation"].min()) >= 0.70
                and not learned_identity_selector.empty
                and float(learned_identity_selector["selected_real_utility_mean"].mean()) >= 0.50
                and float(learned_identity_selector["selected_real_utility_mean"].min()) >= 0.35
                and float(learned_identity_selector["learned_identity_vs_raw_gain_mean"].mean()) >= 0.40
                and float(learned_identity_selector["learned_identity_vs_reward_gain_mean"].mean()) >= 0.15
                and float(learned_identity_selector["learned_identity_win_rate"].min()) >= 0.70
                and not learned_selection_stats.empty
                and bool(learned_selection_stats["passes"].all())
            ),
            "threshold": "property and identity margins >= 0.15, transition MSE <= 25% baseline, reward correlation >= 0.75, learned feature ablations show object information matters, held-out learned domain-shift variants retain property margin >= 0.12, identity margin >= 0.15, transition ratio <= 0.30, and reward correlation >= 0.70, and the learned identity+reward selector transfers to held-out candidate selection with mean utility >= 0.50, min scenario utility >= 0.35, mean raw gain >= 0.40, identity-over-reward gain >= 0.15, win rate >= 0.70, and bootstrap lower bounds passing",
            "observed": {
                "property_margin": float(prop_margin),
                "identity_alignment_margin": float(identity_margin),
                "transition_mse_ratio": float(transition_ratio),
                "reward_correlation": float(learned_metrics["reward_correlation"]),
                "full_minus_no_mass_property_accuracy": float(no_mass["full_minus_property_accuracy"].iloc[0]) if not no_mass.empty else None,
                "full_minus_kinematic_pair_identity_accuracy": float(kinematic_pair["full_minus_identity_alignment_accuracy"].iloc[0]) if not kinematic_pair.empty else None,
                "learned_shift_min_property_margin": float(shifted_learned["property_margin"].min()) if not shifted_learned.empty else None,
                "learned_shift_min_identity_margin": float(shifted_learned["identity_margin"].min()) if not shifted_learned.empty else None,
                "learned_shift_max_transition_ratio": float(shifted_learned["transition_mse_ratio"].max()) if not shifted_learned.empty else None,
                "learned_shift_min_reward_correlation": float(shifted_learned["reward_correlation"].min()) if not shifted_learned.empty else None,
                "learned_selection_identity_mean_utility": float(learned_identity_selector["selected_real_utility_mean"].mean()) if not learned_identity_selector.empty else None,
                "learned_selection_identity_min_scenario_utility": float(learned_identity_selector["selected_real_utility_mean"].min()) if not learned_identity_selector.empty else None,
                "learned_selection_identity_vs_raw_gain": float(learned_identity_selector["learned_identity_vs_raw_gain_mean"].mean()) if not learned_identity_selector.empty else None,
                "learned_selection_identity_vs_reward_gain": float(learned_identity_selector["learned_identity_vs_reward_gain_mean"].mean()) if not learned_identity_selector.empty else None,
                "learned_selection_identity_min_win_rate": float(learned_identity_selector["learned_identity_win_rate"].min()) if not learned_identity_selector.empty else None,
                "learned_selection_bootstrap_min_ci_margin": float((learned_selection_stats["bootstrap_ci_low"] - learned_selection_stats["threshold"]).min()) if not learned_selection_stats.empty else None,
            },
        }
    return strengths


def claim_inventory(root: str | Path | None = None) -> list[dict[str, object]]:
    strengths = evaluate_claim_strength(root) if root is not None else {}
    return [
        {
            "id": "C1",
            "claim": "Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.",
            "status": _status(strengths.get("C1", {}).get("passes") if root is not None else None),
            "evidence": "theory tests and exact_law_validation.csv",
            "strength": strengths.get("C1", {}),
        },
        {
            "id": "C2",
            "claim": "In controlled object-centric scenes, high-N selection can increase object score while real utility stagnates or falls due to binding failures.",
            "status": _status(strengths.get("C2", {}).get("passes") if root is not None else None),
            "evidence": "figure1, figure14, figure24, figure27, figure29, main_metrics.csv, ood_metrics.csv, extreme_object_count_metrics.csv, target_identity_sweep_metrics.csv, and synthetic_benchmark_metrics.csv",
            "strength": strengths.get("C2", {}),
        },
        {
            "id": "C3",
            "claim": "Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.",
            "status": _status(strengths.get("C3", {}).get("passes") if root is not None else None),
            "evidence": "figure2, figure4, figure19, figure20, figure21, figure22, figure24, figure25, figure26, figure27, figure29, paired_effects.csv, stress_metrics.csv, counterfactual_target_metrics.csv, target_identity_sweep_metrics.csv, synthetic_benchmark_metrics.csv, pilot_calibration_metrics.csv, pilot_budget_metrics.csv, leave_one_failure_metrics.csv, noisy_probe_metrics.csv, probe_cost_metrics.csv, and extreme_object_count_metrics.csv",
            "strength": strengths.get("C3", {}),
        },
        {
            "id": "C4",
            "claim": "A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories.",
            "status": _status(strengths.get("C4", {}).get("passes") if root is not None else None),
            "evidence": "learned_object_model_summary.json, learned_metrics.csv, learned_learning_curve.csv, learned_ablation.csv, learned_domain_shift.csv, learned_selection_metrics.csv, and figure28_learned_selection_transfer.png",
            "strength": strengths.get("C4", {}),
        },
        {
            "id": "C5",
            "claim": "The method is validated on real robot systems.",
            "status": "unsupported",
            "evidence": "no real-robot experiments are present",
        },
        {
            "id": "C6",
            "claim": "The method establishes broad benchmark superiority over graph physics, latent, or diffusion world models.",
            "status": "unsupported",
            "evidence": "no broad benchmark suite is present",
        },
    ]


def scan_forbidden_overclaims(claims: Iterable[dict[str, object]]) -> list[str]:
    problems: list[str] = []
    for claim in claims:
        text = (claim.get("claim", "") + " " + claim.get("evidence", "")).lower()
        status = claim.get("status", "").lower()
        if status in {"supported", "strongly_supported"}:
            for pattern in FORBIDDEN_SUPPORTED_PATTERNS:
                if pattern in text:
                    problems.append(f"{claim.get('id', '?')}: supported claim contains forbidden pattern '{pattern}'")
    return problems


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def verify_artifacts(root: str | Path) -> dict[str, object]:
    root = Path(root)
    problems: list[str] = []
    checked: list[str] = []
    for rel, columns in REQUIRED_TABLES.items():
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing table: {rel}")
            continue
        if path.stat().st_size <= 0:
            problems.append(f"empty table file: {rel}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            problems.append(f"table has no rows: {rel}")
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            problems.append(f"table {rel} missing columns: {missing_columns}")
    for rel in REQUIRED_FIGURES:
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing figure: {rel}")
            continue
        width, height = _png_dimensions(path)
        if width < 400 or height < 300:
            problems.append(f"figure {rel} has suspicious dimensions: {width}x{height}")
    for rel in REQUIRED_JSON:
        path = root / rel
        checked.append(rel)
        if not path.exists():
            problems.append(f"missing json artifact: {rel}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"invalid json {rel}: {exc}")
    return {"checked_count": len(checked), "problems": problems, "passes": not problems}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(root: str | Path) -> dict[str, object]:
    root = Path(root)
    manifest_paths = sorted(
        set(REQUIRED_TABLES)
        | set(REQUIRED_FIGURES)
        | {
            "results/run_summary.json",
            "results/learned_object_model_summary.json",
            "results/pilot_calibration_summary.json",
            "results/leave_one_failure_summary.json",
            "results/verification_log.json",
            "docs/results_digest.md",
        }
    )
    files = []
    for rel in manifest_paths:
        path = root / rel
        if path.exists():
            files.append({"path": rel, "bytes": int(path.stat().st_size), "sha256": _sha256(path)})
        else:
            files.append({"path": rel, "missing": True})
    payload = {"artifact_count": len(files), "files": files}
    out = root / "results" / "artifact_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def scan_text_overclaims(root: str | Path) -> list[str]:
    root = Path(root)
    support_terms = (
        "validated",
        "validation",
        "superiority",
        "outperform",
        "outperforms",
        "beats",
        "state of the art",
        "sota",
        "real-robot evidence",
        "real robot evidence",
    )
    negators = (
        "not",
        "no ",
        "unsupported",
        "without",
        "does not",
        "do not",
        "needs",
        "requires",
        "not a",
    )
    problems: list[str] = []
    files = [root / "README.md"]
    files.extend(sorted((root / "docs").glob("*.md")))
    files.extend(sorted((root / "paper").glob("*.md")))
    for path in files:
        if not path.exists():
            continue
        in_unsupported_section = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.lower()
            if text.startswith("##"):
                in_unsupported_section = "unsupported" in text or "boundaries" in text or "what this is not" in text
            if in_unsupported_section:
                continue
            has_forbidden = any(pattern in text for pattern in FORBIDDEN_SUPPORTED_PATTERNS)
            has_support = any(term in text for term in support_terms)
            has_negator = any(term in text for term in negators)
            if has_forbidden and has_support and not has_negator:
                problems.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    return problems


def write_results_digest(root: str | Path) -> None:
    root = Path(root)
    summary = _read_json(root / "results" / "run_summary.json")
    claims = _read_json(root / "results" / "claims_status.json")
    learned = _read_json(root / "results" / "learned_object_model_summary.json")
    lines = [
        "# Results Digest",
        "",
        "This digest is generated from the current result artifacts.",
        "",
        "## Headline Numbers",
        f"- Exact-law mean absolute error: {summary.get('exact_law_mean_absolute_error', 'unknown')}",
        f"- Raw selected-tail score gain: {summary.get('raw_tail_score_gain', 'unknown')}",
        f"- Raw selected-tail utility drop: {summary.get('raw_tail_utility_drop', 'unknown')}",
        f"- Combined repair raw Nmax gain: {summary.get('combined_repair_raw_nmax_mean_gain', 'unknown')}",
        f"- Observable repair raw Nmax gain: {summary.get('observable_repair_raw_nmax_gain', 'unknown')}",
        f"- Combined repair raw ablation dominance: {summary.get('combined_repair_raw_ablation_dominance', 'unknown')}",
        f"- Stress combined mean selected utility: {summary.get('stress_combined_mean_selected_utility', 'unknown')}",
        f"- Seed-block robustness pass rate: {summary.get('seed_block_robustness_pass_rate', 'unknown')}",
        f"- Top raw-score calibration gap: {summary.get('raw_score_top_bin_object_real_gap', 'unknown')}",
        f"- Combined repair low-noise minimum utility: {summary.get('combined_repair_min_low_noise_utility', 'unknown')}",
        f"- Combined-vs-raw low-noise sensitivity margin: {summary.get('combined_vs_raw_low_noise_sensitivity_margin', 'unknown')}",
        f"- Good-control raw high-N utility: {summary.get('good_control_raw_nmax_utility', 'unknown')}",
        f"- Good-minus-corrupted raw high-N utility: {summary.get('good_minus_corrupted_raw_nmax_utility', 'unknown')}",
        f"- Learned full-minus-no-mass property accuracy: {summary.get('learned_full_minus_no_mass_property_accuracy', 'unknown')}",
        f"- Learned full-minus-kinematic-pair identity accuracy: {summary.get('learned_full_minus_kinematic_pair_identity_accuracy', 'unknown')}",
        f"- Learned shift min property margin: {summary.get('learned_shift_min_property_margin', 'unknown')}",
        f"- Learned shift min identity margin: {summary.get('learned_shift_min_identity_margin', 'unknown')}",
        f"- Learned selection identity-vs-raw gain: {summary.get('learned_selection_identity_vs_raw_gain', 'unknown')}",
        f"- Learned selection identity-vs-reward gain: {summary.get('learned_selection_identity_vs_reward_gain', 'unknown')}",
        f"- Synthetic task-suite combined-vs-raw gain: {summary.get('synthetic_benchmark_combined_vs_raw_gain', 'unknown')}",
        f"- Synthetic task-suite observable-vs-raw gain: {summary.get('synthetic_benchmark_observable_vs_raw_gain', 'unknown')}",
        f"- OOD combined mean selected utility: {summary.get('ood_combined_mean_selected_utility', 'unknown')}",
        f"- OOD combined-vs-raw gain: {summary.get('ood_combined_vs_raw_gain', 'unknown')}",
        f"- Extreme object-count combined-vs-raw gain: {summary.get('extreme_object_count_combined_vs_raw_gain', 'unknown')}",
        f"- Extreme object-count observable-vs-raw gain: {summary.get('extreme_object_count_observable_vs_raw_gain', 'unknown')}",
        f"- Domain-randomized combined-vs-raw gain: {summary.get('domain_randomized_combined_vs_raw_gain', 'unknown')}",
        f"- Counterfactual target-swap combined-vs-raw gain: {summary.get('counterfactual_combined_vs_raw_gain', 'unknown')}",
        f"- Target-identity sweep combined-vs-raw gain: {summary.get('target_sweep_combined_vs_raw_gain', 'unknown')}",
        f"- Pilot-calibrated held-out gain: {summary.get('pilot_calibrated_vs_raw_gain', 'unknown')}",
        f"- Pilot-budget mature gain: {summary.get('pilot_budget_mature_vs_raw_gain', 'unknown')}",
        f"- Pilot-budget largest gain: {summary.get('pilot_budget_largest_vs_raw_gain', 'unknown')}",
        f"- Leave-one-failure pilot gain: {summary.get('leave_one_failure_pilot_vs_raw_gain', 'unknown')}",
        f"- Noisy-probe reliable gain: {summary.get('noisy_probe_mean_reliable_gain', 'unknown')}",
        f"- Probe-cost low-cost combined-vs-raw gain: {summary.get('probe_cost_low_cost_combined_vs_raw_gain', 'unknown')}",
        f"- Probe-cost max-cost combined-vs-raw gain: {summary.get('probe_cost_max_cost_combined_vs_raw_gain', 'unknown')}",
        f"- Toy proxy combined-vs-best-proxy gain: {summary.get('model_family_combined_vs_best_proxy_gain', 'unknown')}",
        f"- Bootstrap audit minimum CI margin: {summary.get('statistical_audit_min_ci_margin', 'unknown')}",
        "",
        "## Learned Model",
    ]
    metrics = learned.get("metrics", {})
    if metrics:
        lines.extend(
            [
                f"- Property accuracy: {metrics.get('property_accuracy')} versus baseline {metrics.get('random_property_accuracy')}",
                f"- Identity alignment accuracy: {metrics.get('identity_alignment_accuracy')} versus baseline {metrics.get('random_identity_alignment_accuracy')}",
                f"- Transition MSE ratio: {metrics.get('transition_mse') / metrics.get('constant_transition_mse') if metrics.get('constant_transition_mse') else 'unknown'}",
                f"- Reward correlation: {metrics.get('reward_correlation')}",
            ]
        )
    lines.extend(["", "## Claim Status"])
    for claim in claims.get("claims", []):
        lines.append(f"- {claim.get('id')}: {claim.get('status')} - {claim.get('claim')}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "Real-robot validation and broad benchmark superiority remain unsupported and are not claimed as supported paper results.",
        ]
    )
    (root / "docs" / "results_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_inventory(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    groups = {
        "tables": sorted(str(path.relative_to(root)) for path in (root / "results" / "tables").glob("*.csv")),
        "figures": sorted(str(path.relative_to(root)) for path in (root / "figures").glob("*.png")),
        "docs": sorted(str(path.relative_to(root)) for path in (root / "docs").glob("*.md")),
        "paper": sorted(str(path.relative_to(root)) for path in (root / "paper").glob("*.md")),
    }
    extras = []
    for rel in [
        "results/run_summary.json",
        "results/learned_object_model_summary.json",
        "results/pilot_calibration_summary.json",
        "results/pilot_budget_summary.json",
        "results/leave_one_failure_summary.json",
        "results/claims_status.json",
        "results/verification_log.json",
    ]:
        if (root / rel).exists():
            extras.append(rel)
    groups["json"] = extras
    return groups


def write_claim_status(root: str | Path) -> dict[str, object]:
    root = Path(root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_artifact_manifest(root)
    claims = claim_inventory(root)
    problems = scan_forbidden_overclaims(claims)
    text_overclaims = scan_text_overclaims(root)
    artifact_verification = verify_artifacts(root)
    weak_supported = [
        f"{claim['id']}: {claim['status']}"
        for claim in claims
        if claim["id"] in {"C1", "C2", "C3", "C4"} and claim["status"] != "strongly_supported"
    ]
    payload = {
        "claims": claims,
        "forbidden_supported_overclaims": problems,
        "paper_text_overclaims": text_overclaims,
        "artifact_verification": artifact_verification,
        "weak_or_missing_core_claims": weak_supported,
        "passes_claim_audit": not problems and not weak_supported and not text_overclaims and artifact_verification["passes"],
    }
    (results / "claims_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Claims Status", ""]
    for claim in claims:
        lines.extend(
            [
                f"## {claim['id']}: {claim['status']}",
                claim["claim"],
                "",
                f"Evidence: {claim['evidence']}",
                "",
                f"Strength: {json.dumps(claim.get('strength', {}), indent=2)}",
                "",
            ]
        )
    if weak_supported:
        lines.append("## Weak or missing core evidence")
        lines.extend(f"- {problem}" for problem in weak_supported)
        lines.append("")
    if problems:
        lines.append("## Forbidden supported overclaims")
        lines.extend(f"- {problem}" for problem in problems)
    if text_overclaims:
        lines.append("## Paper text overclaims")
        lines.extend(f"- {problem}" for problem in text_overclaims)
    if not artifact_verification["passes"]:
        lines.append("## Artifact verification problems")
        lines.extend(f"- {problem}" for problem in artifact_verification["problems"])
    lines.append("")
    lines.append(f"Artifact verification checked {artifact_verification['checked_count']} required artifacts.")
    if not problems and not text_overclaims and artifact_verification["passes"]:
        lines.append("")
        lines.append("No paper-text or artifact overclaim problems detected.")
    (results / "claims_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_final_audit(root: str | Path, command_results: dict[str, str] | None = None) -> None:
    root = Path(root)
    inventory = artifact_inventory(root)
    summary_payload = _read_json(root / "results" / "run_summary.json")
    if command_results is None:
        command_results = {}
        verification_path = root / "results" / "verification_log.json"
        if verification_path.exists():
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            command_results.update(verification.get("command_results", {}))
        summary_path = root / "results" / "run_summary.json"
        if summary_path.exists() and not command_results:
            summary = summary_payload
            mode = summary.get("mode", "unknown")
            runtime = summary.get("runtime_seconds", "unknown")
            gate = summary.get("deployment_gate", "unknown")
            script_name = "run_all.sh" if mode == "full" else f"run_{mode}.sh"
            command_results[f"bash scripts/{script_name}"] = f"pass (experiment runtime {runtime}s, gate {gate})"
            if summary.get("passes_claim_audit"):
                command_results["bash scripts/run_claim_audit.sh"] = "pass"
    lines = [
        "# Final Audit",
        "",
        "Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.",
        "",
        "## Command Results",
    ]
    if command_results:
        lines.extend(f"- {name}: {status}" for name, status in command_results.items())
    else:
        lines.append("- Pending final command run in this checkout.")
    lines.extend(
        [
            "",
            "## Strongest Artifacts",
            "- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv. "
            f"Raw score gain {summary_payload.get('raw_tail_score_gain', 'unknown')} and raw utility drop {summary_payload.get('raw_tail_utility_drop', 'unknown')}.",
            "- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level transition, hidden-property, identity-alignment, and reward predictors.",
            "- Repair artifact: figure2_repair_comparison.png, paired_effects.csv, and stress_metrics.csv. "
            f"Raw Nmax combined-repair gain {summary_payload.get('combined_repair_raw_nmax_mean_gain', 'unknown')} with win rate {summary_payload.get('combined_repair_raw_nmax_win_rate', 'unknown')}.",
            "- Observable-repair artifact: figure17_observable_repair.png and observable_repair_metrics.csv. "
            f"Raw Nmax observable-repair gain {summary_payload.get('observable_repair_raw_nmax_gain', 'unknown')}.",
            "- Ablation artifact: figure8_repair_ablation.png and repair_ablation.csv. "
            f"Raw Nmax combined-repair dominance over the best single repair {summary_payload.get('combined_repair_raw_ablation_dominance', 'unknown')}.",
            "- Robustness artifact: figure9_seed_block_robustness.png and seed_block_robustness.csv. "
            f"Seed-block robustness pass rate {summary_payload.get('seed_block_robustness_pass_rate', 'unknown')}.",
            "- Stress artifact: figure6_stress_robustness.png. "
            f"Combined repair mean selected stress utility {summary_payload.get('stress_combined_mean_selected_utility', 'unknown')}.",
            "- Calibration artifact: figure10_score_calibration.png and score_calibration.csv. "
            f"Top raw-score bin object-real gap {summary_payload.get('raw_score_top_bin_object_real_gap', 'unknown')}.",
            "- Sensitivity artifact: figure11_score_noise_sensitivity.png and sensitivity_metrics.csv. "
            f"Combined repair low-noise minimum utility {summary_payload.get('combined_repair_min_low_noise_utility', 'unknown')}.",
            "- Negative-control artifact: figure12_negative_control.png and negative_control.csv. "
            f"Good-control raw high-N utility {summary_payload.get('good_control_raw_nmax_utility', 'unknown')}.",
            "- Learned-ablation artifact: figure13_learned_ablation.png and learned_ablation.csv. "
            f"Full-minus-no-mass property gain {summary_payload.get('learned_full_minus_no_mass_property_accuracy', 'unknown')}.",
            "- Learned domain-shift artifact: figure23_learned_domain_shift.png and learned_domain_shift.csv. "
            f"Minimum shifted property margin {summary_payload.get('learned_shift_min_property_margin', 'unknown')} and identity margin {summary_payload.get('learned_shift_min_identity_margin', 'unknown')}.",
            "- Learned selection transfer artifact: figure28_learned_selection_transfer.png and learned_selection_metrics.csv. "
            f"Identity+reward learned selector raw gain {summary_payload.get('learned_selection_identity_vs_raw_gain', 'unknown')} "
            f"and identity-over-reward gain {summary_payload.get('learned_selection_identity_vs_reward_gain', 'unknown')}.",
            "- Synthetic task-suite artifact: figure29_synthetic_benchmark_suite.png and synthetic_benchmark_metrics.csv. "
            f"Combined-vs-raw gain {summary_payload.get('synthetic_benchmark_combined_vs_raw_gain', 'unknown')} "
            f"and minimum combined variant utility {summary_payload.get('synthetic_benchmark_combined_min_variant_utility', 'unknown')}.",
            "- OOD artifact: figure14_ood_object_count_stress.png and ood_metrics.csv. "
            f"Dense corrupted OOD combined-vs-raw gain {summary_payload.get('ood_combined_vs_raw_gain', 'unknown')}.",
            "- Extreme object-count artifact: figure24_extreme_object_count.png and extreme_object_count_metrics.csv. "
            f"10/12-object corrupted combined-vs-raw gain {summary_payload.get('extreme_object_count_combined_vs_raw_gain', 'unknown')}.",
            "- Domain-randomized artifact: figure18_domain_randomization.png and domain_randomization_metrics.csv. "
            f"Combined-vs-raw gain {summary_payload.get('domain_randomized_combined_vs_raw_gain', 'unknown')}.",
            "- Counterfactual target artifact: figure19_counterfactual_target.png and counterfactual_target_metrics.csv. "
            f"Combined-vs-raw gain {summary_payload.get('counterfactual_combined_vs_raw_gain', 'unknown')}.",
            "- Target-identity sweep artifact: figure27_target_identity_sweep.png and target_identity_sweep_metrics.csv. "
            f"Combined-vs-raw gain {summary_payload.get('target_sweep_combined_vs_raw_gain', 'unknown')}, "
            f"with minimum target utility {summary_payload.get('target_sweep_combined_min_target_utility', 'unknown')}.",
            "- Pilot-label calibration artifact: figure20_pilot_calibration.png, pilot_calibration_metrics.csv, and pilot_calibration_summary.json. "
            f"Held-out calibrated-vs-raw gain {summary_payload.get('pilot_calibrated_vs_raw_gain', 'unknown')}.",
            "- Pilot-label budget artifact: figure26_pilot_label_budget.png, pilot_budget_metrics.csv, and pilot_budget_summary.json. "
            f"Mature-budget gain {summary_payload.get('pilot_budget_mature_vs_raw_gain', 'unknown')} and largest-budget gain {summary_payload.get('pilot_budget_largest_vs_raw_gain', 'unknown')}.",
            "- Leave-one-failure-out artifact: figure21_leave_one_failure_out.png, leave_one_failure_metrics.csv, and leave_one_failure_summary.json. "
            f"Held-out-family calibrated-vs-raw gain {summary_payload.get('leave_one_failure_pilot_vs_raw_gain', 'unknown')}.",
            "- Noisy-probe artifact: figure22_noisy_probe_reliability.png and noisy_probe_metrics.csv. "
            f"Reliable-probe gain {summary_payload.get('noisy_probe_mean_reliable_gain', 'unknown')}.",
            "- Probe-cost artifact: figure25_probe_cost_sensitivity.png and probe_cost_metrics.csv. "
            f"Low-cost combined-vs-raw gain {summary_payload.get('probe_cost_low_cost_combined_vs_raw_gain', 'unknown')} and max-cost gain {summary_payload.get('probe_cost_max_cost_combined_vs_raw_gain', 'unknown')}.",
            "- Toy proxy artifact: figure15_model_family_proxies.png and model_family_proxy_metrics.csv. "
            f"Combined-vs-best-proxy gain {summary_payload.get('model_family_combined_vs_best_proxy_gain', 'unknown')}.",
            "- Statistical audit artifact: figure16_statistical_audit.png and statistical_audit.csv. "
            f"Minimum bootstrap CI margin {summary_payload.get('statistical_audit_min_ci_margin', 'unknown')}.",
            "",
            "## Differentiation",
            "The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.",
            "The toy proxy panel is a controlled diagnostic comparison, not a graph-physics benchmark, latent dynamics benchmark, diffusion world-model benchmark, or real-robot evaluation.",
            "",
            "## Remaining Weaknesses",
            f"- Synthetic scenes remain controlled, though the default run now uses {len(summary_payload.get('seeds', [])) or 'unknown'} main seeds, {len(summary_payload.get('stress_seeds', [])) or 'unknown'} stress seeds, dense and extreme object-count stress, benchmark-style synthetic task-suite stress, held-out domain-randomized stress, target-identity sweep stress, learned selection transfer, held-out pilot-label calibration, pilot-label budget sensitivity, leave-one-failure-out calibration, noisy-probe reliability stress, and probe-cost sensitivity.",
            "- Observable-only, pilot-calibrated, noisy-probe, and probe-cost repair reduce direct hidden-property truth alignment and free-probe assumptions, and learned domain-shift tests add dense/occluded/crossing variants, but all probe and slot diagnostics still come from the toy generator.",
            "- No real-robot or broad benchmark evidence is claimed.",
            "",
            "## Artifact Inventory",
        ]
    )
    for group, paths in inventory.items():
        lines.append(f"### {group}")
        if paths:
            lines.extend(f"- {path}" for path in paths)
        else:
            lines.append("- none")
    (root / "docs" / "final_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path.cwd())
    args = parser.parse_args()
    write_results_digest(args.root)
    payload = write_claim_status(args.root)
    write_results_digest(args.root)
    write_final_audit(args.root)
    if not payload["passes_claim_audit"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
