import pandas as pd

from object_centric_best_of_n.metrics import (
    aggregate_seed_metrics,
    counterfactual_target_summary,
    deployment_policy_summary,
    domain_randomization_summary,
    extreme_object_count_summary,
    learned_repair_policy_summary,
    learned_selection_summary,
    model_family_proxy_summary,
    negative_control_summary,
    noisy_probe_summary,
    observable_repair_summary,
    ood_summary,
    paired_selector_effects,
    pilot_calibration_summary,
    pilot_budget_summary,
    probe_cost_summary,
    repair_ablation_summary,
    score_calibration_table,
    seed_block_robustness,
    sensitivity_summary,
    statistical_audit,
    stress_summary,
    synthetic_benchmark_summary,
    target_identity_sweep_summary,
)


def test_paired_effects_and_stress_summary_are_computed():
    rows = []
    for seed in range(4):
        rows.append(
            {
                "experiment": "toy",
                "scenario": "raw",
                "selector": "raw",
                "N": 8,
                "seed": seed,
                "selected_real_utility": 0.2,
                "selected_object_score": 1.0,
                "identity_error": 1.0,
                "swap_rate": 1.0,
                "merge_split_rate": 1.0,
                "property_error": 1.0,
                "property_entropy": 0.8,
                "occlusion_error": 1.0,
                "object_real_gap": 0.8,
                "regret": 0.7,
                "oracle_gap": 0.7,
                "upper_tail_rank_correlation": -0.5,
            }
        )
        rows.append({**rows[-1], "selector": "identity_consistent", "selected_real_utility": 0.6, "identity_error": 0.0})
        rows.append({**rows[-1], "selector": "targeted_probe", "selected_real_utility": 0.78, "identity_error": 0.0})
        rows.append({**rows[-1], "selector": "combined_repair", "selected_real_utility": 0.9, "identity_error": 0.0})
        rows.append({**rows[-1], "selector": "observable_repair", "selected_real_utility": 0.85, "identity_error": 0.0})
        rows.append({**rows[-1], "selector": "pilot_calibrated", "selected_real_utility": 0.82, "identity_error": 0.0})
        rows.append({**rows[-1], "selector": "oracle", "selected_real_utility": 0.95, "identity_error": 0.0})
    df = pd.DataFrame(rows)
    paired = paired_selector_effects(df)
    assert paired.loc[paired["selector"] == "combined_repair", "mean_gain"].iloc[0] == 0.7
    stress = stress_summary(df)
    assert "combined_vs_raw_gain_mean" in stress.columns
    main = aggregate_seed_metrics(df)
    ablation = repair_ablation_summary(main, paired)
    assert "combined_vs_best_single_gain" in ablation.columns
    observable = observable_repair_summary(main, paired)
    assert observable["observable_vs_raw_gain"].iloc[0] > 0.0
    robustness = seed_block_robustness(df, block_size=2)
    assert robustness["combined_raw_nmax_gain"].min() == 0.7
    calibration = score_calibration_table(
        pd.DataFrame(
            {
                "raw_object_score": [0.1, 0.2, 0.9, 1.0],
                "real_utility": [0.8, 0.7, 0.1, 0.0],
                "identity_error": [0, 0, 1, 1],
                "merge_split": [0, 0, 1, 1],
                "property_error": [0, 0, 1, 1],
            }
        ),
        bins=2,
    )
    assert calibration.iloc[-1]["object_real_gap"] > 0.0
    sensitivity = sensitivity_summary(
        pd.DataFrame(
            {
                "selector": ["raw_noisy", "raw_noisy", "combined_repair_noisy", "combined_repair_noisy"],
                "score_noise": [0.0, 0.1, 0.0, 0.1],
                "selected_real_utility": [0.1, 0.2, 0.9, 0.85],
                "identity_error": [1, 1, 0, 0],
            }
        )
    )
    assert set(sensitivity["selector"]) == {"raw_noisy", "combined_repair_noisy"}
    negative = negative_control_summary(main)
    assert "corrupted_mean" in set(negative["contrast"])
    ood = ood_summary(df)
    assert "combined_vs_raw_gain_mean" in ood.columns
    extreme_df = pd.DataFrame(
        [
            {**row, "experiment": "U_extreme_object_count", "scenario": "extreme10_raw", "n_objects": 10}
            for row in rows
        ]
    )
    extreme = extreme_object_count_summary(extreme_df)
    assert "extreme_combined_vs_raw_gain_mean" in extreme.columns
    assert extreme[extreme["selector"] == "combined_repair"]["extreme_combined_win_rate"].iloc[0] == 1.0
    domain = domain_randomization_summary(df)
    assert "domain_combined_vs_raw_gain_mean" in domain.columns
    counter = counterfactual_target_summary(df)
    assert "counterfactual_combined_vs_raw_gain_mean" in counter.columns
    assert counter[counter["selector"] == "combined_repair"]["counterfactual_combined_win_rate"].iloc[0] == 1.0
    target_sweep_df = pd.DataFrame(
        [
            {**row, "experiment": "X_target_identity_sweep", "target_id": row["seed"] % 2}
            for row in rows
        ]
    )
    target_sweep = target_identity_sweep_summary(target_sweep_df)
    assert "target_sweep_combined_vs_raw_gain_mean" in target_sweep.columns
    assert target_sweep[target_sweep["selector"] == "combined_repair"]["target_sweep_combined_win_rate"].min() == 1.0
    learned_selection_df = pd.DataFrame(
        [
            {
                **row,
                "experiment": "Y_learned_selection_transfer",
                "learned_reward_mean": 0.4,
                "learned_identity_alignment_mean": 0.6,
                "learned_property_confidence_mean": 0.8,
                "target_id": row["seed"] % 2,
            }
            for row in rows
            if row["selector"] in {"raw", "observable_repair", "combined_repair", "oracle"}
        ]
        + [
            {
                **rows[0],
                "experiment": "Y_learned_selection_transfer",
                "selector": "learned_reward",
                "selected_real_utility": 0.45,
                "learned_reward_mean": 0.4,
                "learned_identity_alignment_mean": 0.6,
                "learned_property_confidence_mean": 0.8,
                "target_id": 0,
            },
            {
                **rows[0],
                "experiment": "Y_learned_selection_transfer",
                "selector": "learned_identity_reward",
                "selected_real_utility": 0.72,
                "learned_reward_mean": 0.4,
                "learned_identity_alignment_mean": 0.6,
                "learned_property_confidence_mean": 0.8,
                "target_id": 0,
            },
        ]
    )
    learned_selection = learned_selection_summary(learned_selection_df)
    assert "learned_identity_vs_raw_gain_mean" in learned_selection.columns
    assert learned_selection[
        learned_selection["selector"] == "learned_identity_reward"
    ]["learned_identity_vs_reward_gain_mean"].iloc[0] > 0.0
    learned_repair_policy_df = pd.DataFrame(
        [
            {
                **row,
                "experiment": "AB_learned_repair_policy_transfer",
                "selector": selector,
                "selected_real_utility": utility,
                "learned_reward_mean": 0.4,
                "learned_identity_alignment_mean": 0.6,
                "learned_property_confidence_mean": 0.8,
                "learned_repair_policy_score_mean": 0.75,
                "learned_repair_policy_train_candidates": 64,
                "learned_repair_policy_train_mae": 0.05,
                "learned_repair_policy_train_correlation": 0.9,
                "suite_variant": row["scenario"],
                "target_id": row["seed"] % 2,
            }
            for row in rows
            if row["selector"] == "raw"
            for selector, utility in [
                ("raw", 0.2),
                ("learned_reward", 0.42),
                ("learned_identity_reward", 0.55),
                ("pilot_calibrated", 0.82),
                ("learned_repair_policy", 0.9),
                ("observable_repair", 0.88),
                ("combined_repair", 0.92),
                ("random", 0.25),
                ("oracle", 0.95),
            ]
        ]
    )
    learned_repair_policy = learned_repair_policy_summary(learned_repair_policy_df)
    assert "learned_repair_policy_vs_raw_gain_mean" in learned_repair_policy.columns
    assert learned_repair_policy[
        learned_repair_policy["selector"] == "learned_repair_policy"
    ]["learned_repair_policy_over_learned_identity_win_rate"].iloc[0] == 1.0
    assert learned_repair_policy[
        learned_repair_policy["selector"] == "learned_repair_policy"
    ]["learned_repair_policy_over_learned_identity_nonloss_rate"].iloc[0] == 1.0
    assert learned_repair_policy[
        learned_repair_policy["selector"] == "learned_repair_policy"
    ]["learned_repair_policy_worst_learned_identity_loss"].iloc[0] == 0.0
    synthetic_benchmark_df = pd.DataFrame(
        [
            {
                **row,
                "experiment": "Z_synthetic_task_suite",
                "suite_variant": row["scenario"],
                "n_objects": 6,
                "occlusion_flag": 1,
                "hidden_property_flag": 1,
                "crossing_flag": 1,
                "generator_scenario": "raw",
                "target_id": 0,
            }
            for row in rows
            if row["selector"] in {"raw", "observable_repair", "combined_repair", "random", "oracle"}
        ]
    )
    synthetic_benchmark = synthetic_benchmark_summary(synthetic_benchmark_df)
    assert "synthetic_benchmark_combined_vs_raw_gain_mean" in synthetic_benchmark.columns
    assert synthetic_benchmark[
        synthetic_benchmark["selector"] == "combined_repair"
    ]["synthetic_benchmark_combined_win_rate"].iloc[0] == 1.0
    deployment_policy_df = pd.DataFrame(
        [
            {
                **row,
                "experiment": "AA_deployment_gate_policy",
                "selector": selector,
                "selected_real_utility": utility,
                "selected_N": 8 if selector != "stop_early_raw" else 4,
                "gate_action": "block_high_n" if selector == "gate_policy" else f"baseline_{selector}",
                "delegated_selector": "combined_repair" if selector == "gate_policy" else selector,
                "gate_identity_error": 1.0,
                "gate_object_real_gap": 0.8,
                "gate_property_entropy": 0.7,
                "gate_repair_gain": 0.7,
            }
            for row in rows
            if row["selector"] == "raw"
            for selector, utility in [
                ("raw_high_n", 0.2),
                ("stop_early_raw", 0.3),
                ("gate_policy", 0.9),
                ("oracle", 0.95),
            ]
        ]
    )
    deployment_policy = deployment_policy_summary(deployment_policy_df)
    assert "deployment_policy_vs_raw_gain_mean" in deployment_policy.columns
    assert deployment_policy[
        deployment_policy["selector"] == "gate_policy"
    ]["deployment_policy_win_rate"].iloc[0] == 1.0
    pilot = pilot_calibration_summary(df)
    assert "pilot_vs_raw_gain_mean" in pilot.columns
    assert pilot[pilot["selector"] == "pilot_calibrated"]["pilot_win_rate"].iloc[0] == 1.0
    pilot_budget_df = pd.DataFrame(
        [
            {
                **row,
                "experiment": "W_pilot_label_budget",
                "pilot_label_budget": 128,
                "pilot_train_mae": 0.1,
                "pilot_train_correlation": 0.9,
            }
            for row in rows
            if row["selector"] in {"raw", "pilot_calibrated", "observable_repair", "combined_repair", "random", "oracle"}
        ]
    )
    pilot_budget = pilot_budget_summary(pilot_budget_df)
    assert "pilot_budget_vs_raw_gain_mean" in pilot_budget.columns
    assert pilot_budget[pilot_budget["selector"] == "pilot_calibrated"]["pilot_budget_win_rate"].iloc[0] == 1.0
    noisy_probe = noisy_probe_summary(
        pd.DataFrame(
            [
                {**row, "probe_reliability": 0.75, "probe_noise_rate": 0.25}
                for row in rows
            ]
        )
    )
    assert "noisy_probe_vs_raw_gain_mean" in noisy_probe.columns
    assert noisy_probe[noisy_probe["selector"] == "noisy_probe_repair"].empty
    noisy_rows = rows + [
        {**rows[0], "selector": "noisy_probe_repair", "selected_real_utility": 0.84}
        for _ in range(4)
    ]
    noisy_df = pd.DataFrame([{**row, "probe_reliability": 0.75, "probe_noise_rate": 0.25} for row in noisy_rows])
    noisy_probe = noisy_probe_summary(noisy_df)
    assert noisy_probe[noisy_probe["selector"] == "noisy_probe_repair"]["noisy_probe_win_rate"].iloc[0] == 1.0
    probe_cost_df = pd.DataFrame(
        [
            {
                **row,
                "probe_cost": 0.10,
                "gross_selected_real_utility": row["selected_real_utility"] + (0.10 if row["selector"] in {"targeted_probe", "observable_repair", "combined_repair"} else 0.0),
                "incurred_probe_cost": 0.10 if row["selector"] in {"targeted_probe", "observable_repair", "combined_repair"} else 0.0,
                "probe_cost_applied": int(row["selector"] in {"targeted_probe", "observable_repair", "combined_repair"}),
            }
            for row in rows
            if row["selector"] in {"raw", "targeted_probe", "observable_repair", "combined_repair", "oracle"}
        ]
    )
    probe_cost = probe_cost_summary(probe_cost_df)
    assert "probe_cost_combined_vs_raw_gain_mean" in probe_cost.columns
    assert probe_cost[probe_cost["selector"] == "combined_repair"]["probe_cost_combined_win_rate"].iloc[0] == 1.0
    family = model_family_proxy_summary(
        pd.DataFrame(
            rows
            + [
                {**rows[0], "selector": "latent_global_proxy", "selected_real_utility": 0.35},
                {**rows[0], "selector": "relational_slot_proxy", "selected_real_utility": 0.45},
                {**rows[0], "selector": "diffusion_score_proxy", "selected_real_utility": 0.25},
            ]
        )
    )
    combined = family[family["selector"] == "combined_repair"]
    assert "combined_vs_best_proxy_gain_mean" in family.columns
    assert combined["combined_vs_best_proxy_gain_mean"].iloc[0] > 0.0
    stats = statistical_audit(
        df,
        ood_seed_df=df,
        extreme_object_seed_df=extreme_df,
        family_seed_df=pd.DataFrame(
            rows
            + [
                {**rows[0], "selector": "latent_global_proxy", "selected_real_utility": 0.35},
                {**rows[0], "selector": "relational_slot_proxy", "selected_real_utility": 0.45},
                {**rows[0], "selector": "diffusion_score_proxy", "selected_real_utility": 0.25},
            ]
        ),
        target_sweep_seed_df=target_sweep_df,
        pilot_seed_df=df,
        pilot_budget_seed_df=pilot_budget_df,
        leave_one_failure_seed_df=df,
        noisy_probe_seed_df=noisy_df,
        probe_cost_seed_df=probe_cost_df,
        learned_selection_seed_df=learned_selection_df,
        learned_repair_policy_seed_df=learned_repair_policy_df,
        synthetic_benchmark_seed_df=synthetic_benchmark_df,
        deployment_policy_seed_df=deployment_policy_df,
        bootstrap_reps=50,
    )
    assert {"effect_id", "bootstrap_ci_low", "passes"}.issubset(stats.columns)
    assert "combined_repair_raw_gain" in set(stats["effect_id"])
    assert "pilot_calibrated_repair_gain" in set(stats["effect_id"])
    assert "pilot_budget_mature_gain" in set(stats["effect_id"])
    assert "leave_one_failure_pilot_gain" in set(stats["effect_id"])
    assert "noisy_probe_repair_gain" in set(stats["effect_id"])
    assert "extreme_object_combined_repair_gain" in set(stats["effect_id"])
    assert "target_sweep_combined_repair_gain" in set(stats["effect_id"])
    assert "target_sweep_observable_repair_gain" in set(stats["effect_id"])
    assert "learned_selection_identity_gain" in set(stats["effect_id"])
    assert "learned_selection_identity_over_reward_gain" in set(stats["effect_id"])
    assert "learned_repair_policy_gain" in set(stats["effect_id"])
    assert "learned_repair_policy_over_learned_identity_gain" in set(stats["effect_id"])
    assert "synthetic_benchmark_combined_repair_gain" in set(stats["effect_id"])
    assert "synthetic_benchmark_observable_repair_gain" in set(stats["effect_id"])
    assert "deployment_policy_gate_gain" in set(stats["effect_id"])
    assert "deployment_policy_gate_over_stop_early_gain" in set(stats["effect_id"])
    assert "probe_cost_combined_repair_gain" in set(stats["effect_id"])
    assert "probe_cost_targeted_hidden_repair_gain" in set(stats["effect_id"])
