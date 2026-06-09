# Claims Status

## C1: strongly_supported
Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.

Evidence: theory tests and exact_law_validation.csv

Strength: {
  "passes": true,
  "threshold": "max exact-law absolute error <= 0.015 and mean <= 0.006",
  "observed": {
    "max_absolute_error": 0.006141488371171,
    "mean_absolute_error": 0.00046291189986127505
  }
}

## C2: strongly_supported
In controlled object-centric scenes, high-N selection can increase object score while real utility stagnates or falls due to binding failures.

Evidence: figure1 and main_metrics.csv for the raw scenario

Strength: {
  "passes": true,
  "threshold": "raw high-N score gain >= 0.35, utility drop >= 0.15, tail identity error >= 0.75, all seed blocks pass reduced thresholds, top raw-score calibration bin has gap >= 0.45 with identity error >= 0.55, good negative controls avoid collapse, dense OOD corrupted variants collapse, and bootstrap lower bounds for raw score gain and utility drop pass",
  "observed": {
    "raw_tail_score_gain": 0.5759192453426587,
    "raw_tail_utility_drop": 0.3639708878079679,
    "raw_tail_identity_error": 1.0,
    "min_block_score_gain": 0.418084184937237,
    "min_block_utility_drop": 0.252396546523533,
    "min_block_identity_error": 1.0,
    "top_calibration_object_real_gap": 1.095467459764002,
    "top_calibration_identity_error": 0.94140625,
    "good_control_utility": 0.6554531451148605,
    "good_control_identity_error": 0.125,
    "good_minus_corrupted_utility": 0.608449149414918,
    "ood_good_raw_utility": 0.7549937089444607,
    "ood_corrupted_raw_mean_utility": 0.03568903305390833,
    "ood_corrupted_raw_identity_error": 0.9166666666666666,
    "bootstrap_raw_tail_min_ci_margin": 0.0829473068486257
  }
}

## C3: strongly_supported
Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.

Evidence: figure2, figure4, figure19, figure20, figure21, paired_effects.csv, stress_metrics.csv, counterfactual_target_metrics.csv, pilot_calibration_metrics.csv, and leave_one_failure_metrics.csv

Strength: {
  "passes": true,
  "threshold": "combined raw Nmax gain >= 0.55 with win-rate >= 0.75, targeted hidden-property gain >= 0.12, stress combined mean >= 0.75 and min >= 0.80, raw ablation dominance >= 0.20 with oracle gap <= 0.08, observable-only repair beats raw and remains close to controlled combined repair, all seed blocks repair, combined repair remains strong under score noise <= 0.10, dense OOD repair succeeds, held-out domain-randomized stress succeeds, counterfactual target-swap stress succeeds, held-out pilot-label calibration succeeds, leave-one-failure-out pilot calibration succeeds, controlled toy model-family proxy comparison has mean margin >= 0.20 with every scenario positive by >= 0.05 and max oracle gap <= 0.12, and bootstrap lower bounds for key repair gains pass",
  "observed": {
    "combined_raw_nmax_gain": 0.8803086375224858,
    "combined_raw_nmax_win_rate": 1.0,
    "targeted_hidden_property_nmax_gain": 0.7747901941902325,
    "stress_combined_mean_utility": 0.8494153088926296,
    "stress_combined_min_utility": 0.8252742847310228,
    "raw_ablation_combined_vs_best_single_gain": 0.2770953565422955,
    "raw_ablation_combined_oracle_gap": 0.0244701492605839,
    "observable_raw_gain": 0.8803086375224858,
    "observable_raw_utility": 0.8803086375224858,
    "observable_mean_corrupted_utility": 0.8502469933665043,
    "observable_max_combined_gap": 0.0107309785170239,
    "min_block_combined_raw_gain": 0.8602071898170613,
    "min_block_combined_win_rate": 1.0,
    "combined_min_low_noise_utility": 0.8374916544433992,
    "combined_vs_raw_low_noise_margin": 0.8534731753001509,
    "ood_combined_mean_utility": 0.876852645124664,
    "ood_combined_min_utility": 0.8643270511948262,
    "ood_combined_vs_raw_gain": 0.8411636120707556,
    "domain_raw_utility": 0.0241071357890832,
    "domain_combined_utility": 0.8584949932735436,
    "domain_observable_utility": 0.8551763530136266,
    "domain_combined_vs_raw_gain": 0.8343878574844603,
    "domain_observable_vs_raw_gain": 0.8310692172245433,
    "domain_combined_win_rate": 1.0,
    "counterfactual_raw_utility": 0.0,
    "counterfactual_combined_utility": 0.816906396281512,
    "counterfactual_observable_utility": 0.8118514080123783,
    "counterfactual_combined_vs_raw_gain": 0.816906396281512,
    "counterfactual_observable_vs_raw_gain": 0.8118514080123783,
    "counterfactual_combined_win_rate": 1.0,
    "pilot_calibrated_mean_utility": 0.827787343785161,
    "pilot_calibrated_min_utility": 0.7851225988354029,
    "pilot_calibrated_vs_raw_gain": 0.8192029444060406,
    "pilot_calibrated_min_win_rate": 1.0,
    "pilot_calibrated_max_oracle_gap": 0.1327885883263263,
    "pilot_train_correlation": 0.983367982746434,
    "leave_one_failure_pilot_mean_utility": 0.8160878196526353,
    "leave_one_failure_pilot_min_utility": 0.806587797091829,
    "leave_one_failure_pilot_vs_raw_gain": 0.7758011252035516,
    "leave_one_failure_pilot_min_win_rate": 1.0,
    "leave_one_failure_pilot_max_oracle_gap": 0.1150950379779143,
    "leave_one_failure_min_train_correlation": 0.9818478546301981,
    "model_family_combined_vs_best_proxy_gain": 0.5504614056934154,
    "model_family_min_combined_vs_best_proxy_gain": 0.3399964978763068,
    "model_family_max_combined_oracle_gap": 0.1058960383796072,
    "bootstrap_repair_min_ci_margin": 0.21608402224147516
  }
}

## C4: strongly_supported
A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories.

Evidence: learned_object_model_summary.json, learned_metrics.csv, and learned_learning_curve.csv

Strength: {
  "passes": true,
  "threshold": "property and identity margins >= 0.15, transition MSE <= 25% baseline, reward correlation >= 0.75, and learned feature ablations show object information matters",
  "observed": {
    "property_margin": 0.24583333333333335,
    "identity_alignment_margin": 0.48750000000000004,
    "transition_mse_ratio": 0.007032264292782088,
    "reward_correlation": 0.953061460933608,
    "full_minus_no_mass_property_accuracy": 0.1229166666666666,
    "full_minus_kinematic_pair_identity_accuracy": 0.0760416666666666
  }
}

## C5: unsupported
The method is validated on real robot systems.

Evidence: no real-robot experiments are present

Strength: {}

## C6: unsupported
The method establishes broad benchmark superiority over graph physics, latent, or diffusion world models.

Evidence: no broad benchmark suite is present

Strength: {}


Artifact verification checked 58 required artifacts.

No paper-text or artifact overclaim problems detected.
