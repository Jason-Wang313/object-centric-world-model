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
  "threshold": "raw high-N score gain >= 0.35, utility drop >= 0.15, tail identity error >= 0.75",
  "observed": {
    "raw_tail_score_gain": 0.5759192453426587,
    "raw_tail_utility_drop": 0.3639708878079679,
    "raw_tail_identity_error": 1.0
  }
}

## C3: strongly_supported
Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.

Evidence: figure2, figure4, paired_effects.csv, and stress_metrics.csv

Strength: {
  "passes": true,
  "threshold": "combined raw Nmax gain >= 0.55 with win-rate >= 0.75, targeted hidden-property gain >= 0.12, stress combined mean utility >= 0.75",
  "observed": {
    "combined_raw_nmax_gain": 0.8803086375224858,
    "combined_raw_nmax_win_rate": 1.0,
    "targeted_hidden_property_nmax_gain": 0.7747901941902325,
    "stress_combined_mean_utility": 0.8494153088926296
  }
}

## C4: strongly_supported
A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories.

Evidence: learned_object_model_summary.json, learned_metrics.csv, and learned_learning_curve.csv

Strength: {
  "passes": true,
  "threshold": "property and identity margins >= 0.15, transition MSE <= 25% baseline, reward correlation >= 0.75",
  "observed": {
    "property_margin": 0.24583333333333335,
    "identity_alignment_margin": 0.48750000000000004,
    "transition_mse_ratio": 0.007032264292782088,
    "reward_correlation": 0.953061460933608
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

No forbidden supported overclaims detected.
