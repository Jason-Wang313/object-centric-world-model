import pandas as pd

from object_centric_best_of_n.metrics import (
    aggregate_seed_metrics,
    counterfactual_target_summary,
    domain_randomization_summary,
    model_family_proxy_summary,
    negative_control_summary,
    observable_repair_summary,
    ood_summary,
    paired_selector_effects,
    pilot_calibration_summary,
    repair_ablation_summary,
    score_calibration_table,
    seed_block_robustness,
    sensitivity_summary,
    statistical_audit,
    stress_summary,
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
    domain = domain_randomization_summary(df)
    assert "domain_combined_vs_raw_gain_mean" in domain.columns
    counter = counterfactual_target_summary(df)
    assert "counterfactual_combined_vs_raw_gain_mean" in counter.columns
    assert counter[counter["selector"] == "combined_repair"]["counterfactual_combined_win_rate"].iloc[0] == 1.0
    pilot = pilot_calibration_summary(df)
    assert "pilot_vs_raw_gain_mean" in pilot.columns
    assert pilot[pilot["selector"] == "pilot_calibrated"]["pilot_win_rate"].iloc[0] == 1.0
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
        family_seed_df=pd.DataFrame(
            rows
            + [
                {**rows[0], "selector": "latent_global_proxy", "selected_real_utility": 0.35},
                {**rows[0], "selector": "relational_slot_proxy", "selected_real_utility": 0.45},
                {**rows[0], "selector": "diffusion_score_proxy", "selected_real_utility": 0.25},
            ]
        ),
        pilot_seed_df=df,
        leave_one_failure_seed_df=df,
        bootstrap_reps=50,
    )
    assert {"effect_id", "bootstrap_ci_low", "passes"}.issubset(stats.columns)
    assert "combined_repair_raw_gain" in set(stats["effect_id"])
    assert "pilot_calibrated_repair_gain" in set(stats["effect_id"])
    assert "leave_one_failure_pilot_gain" in set(stats["effect_id"])
