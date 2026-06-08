# Reviewer Attacks

## Attack: The law is generic, not object-centric.

Response: Correct. The finite law is generic. The object-centric contribution is defining the score/utility pairs around slots, target identity, occlusion, hidden properties, and repair signals, then showing controlled selected-tail failures.

## Attack: The experiments are synthetic.

Response: Correct. Claims are limited to controlled synthetic and semi-learned CPU evidence. Real-robot and broad benchmark claims are explicitly unsupported.

## Attack: Repairs may use generator diagnostics.

Response: Correct. The repair stack uses controlled diagnostic signals such as identity instability, merge evidence, property entropy, and targeted probe observations. This is appropriate for a v1 mechanism study, not a deployment claim.

## Attack: The repair result could be a seed fluke.

Response: The upgraded run reports paired per-seed gains in `paired_effects.csv`, repair ablations in `repair_ablation.csv`, seed-block robustness in `seed_block_robustness.csv`, and a separate high-N stress panel in `stress_metrics.csv`. The claim audit requires positive paired raw-to-repair gain, high win rate, targeted hidden-property gain, ablation dominance, seed-block robustness, and stress utility before marking the repair claim strongly supported.

## Attack: The raw score might only be miscalibrated under one exact scoring setup.

Response: `score_calibration.csv` bins raw candidate object scores and measures real utility, object-real gap, and identity-error rate. `sensitivity_metrics.csv` perturbs raw and repaired scores with score noise and requires the combined repair to remain strong under low-noise perturbations.

## Attack: Best-of-N may just be bad in every synthetic scene.

Response: `negative_control.csv` compares the good non-corrupted setting with corrupted high-N settings. The audit requires the good control to retain utility and low identity error while corrupted settings collapse.

## Attack: The result may not survive more distractor objects.

Response: `ood_metrics.csv` evaluates dense 6- and 8-object variants, including a dense good control and dense corrupted scenes. The audit requires dense corrupted scenes to collapse under raw selection and recover under combined repair.

## Attack: The paper text might drift beyond the artifact evidence.

Response: `run_claim_audit.sh` scans README, docs, and paper text for supported forbidden overclaims, verifies required tables, figures, and JSON artifacts, and writes `results/artifact_manifest.json` with deterministic hashes. The generated `docs/results_digest.md` records the current evidence boundaries.

## Attack: The learned model is too simple.

Response: The learned artifact is intentionally CPU NumPy and semi-learned. It now includes transition, hidden-property, identity-alignment, reward, and learning-curve evidence. It does not establish modern benchmark performance.

## Attack: The learned evidence may not use object information.

Response: `learned_ablation.csv` removes object-relevant features such as the mass sensor and identity-pair features. The audit requires the full object-feature model to beat these ablations on hidden-property and identity-alignment metrics.

## Attack: Oracle rows make repairs look weak or strong.

Response: Oracle rows are upper bounds for interpreting regret and oracle gap. Repair claims are based on controlled improvements over raw and random selectors, not equality to oracle in every setting.
