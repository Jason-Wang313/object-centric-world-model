import pandas as pd

from object_centric_best_of_n.metrics import paired_selector_effects, stress_summary


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
        rows.append({**rows[-1], "selector": "combined_repair", "selected_real_utility": 0.9, "identity_error": 0.0})
    df = pd.DataFrame(rows)
    paired = paired_selector_effects(df)
    assert paired.loc[paired["selector"] == "combined_repair", "mean_gain"].iloc[0] == 0.7
    stress = stress_summary(df)
    assert "combined_vs_raw_gain_mean" in stress.columns
