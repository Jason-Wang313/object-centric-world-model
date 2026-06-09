# Reviewer Attacks

## Attack: The law is generic, not object-centric.

Response: Correct. The finite law is generic. The object-centric contribution is defining the score/utility pairs around slots, target identity, occlusion, hidden properties, and repair signals, then showing controlled selected-tail failures.

## Attack: The experiments are synthetic.

Response: Correct. Claims are limited to controlled synthetic and semi-learned CPU evidence. Real-robot and broad benchmark claims are explicitly unsupported.

## Attack: Repairs may use generator diagnostics.

Response: Correct for the strongest controlled stack: it uses diagnostic signals such as identity instability, merge evidence, property entropy, and targeted probe observations. The upgraded run also reports `observable_repair_metrics.csv`, where an observable-only repair score uses slot diagnostics and probe posterior information without comparing to the scene's true hidden property. This is still a synthetic mechanism study, not a deployment claim.

## Attack: Diagnostic probes are unrealistically clean.

Response: `noisy_probe_metrics.csv` varies the diagnostic observation reliability from barely better than chance to clean-probe settings. The noisy-probe selector uses observable slot diagnostics and the noisy posterior, not direct hidden-property truth alignment, and the audit requires selected utility, raw gain, win rate, oracle gap, and bootstrap lower-bound checks for reliability at or above 0.75.

## Attack: Diagnostic probes are treated as free.

Response: `probe_cost_metrics.csv` charges diagnostic actions before reporting selected real utility. The audit requires combined and observable repair to keep positive cost-adjusted gains for probe costs up to 0.10, requires targeted probing to remain positive in hidden-property scenes, and requires combined and observable repair to retain positive margins at higher costs.

## Attack: The deployment gate says to collect pilot labels, but labels are not tested.

Response: `pilot_calibration_metrics.csv` evaluates a held-out selector trained from pilot-labeled candidates using observable object features. The train seeds are separate from the held-out raw, randomized-domain, and target-swap evaluation seeds. The upgraded run also writes `leave_one_failure_metrics.csv`, where each raw, occlusion, hidden-property, swap, or merge/split failure family is held out during pilot-calibrator training and then evaluated as the test family. The audit requires calibrated selected utility, paired raw gain, win rate, oracle gap, leave-one-failure transfer, and bootstrap lower-bound checks before counting the evidence.

## Attack: The repair result could be a seed fluke.

Response: The upgraded run reports paired per-seed gains in `paired_effects.csv`, nonparametric bootstrap intervals in `statistical_audit.csv`, repair ablations in `repair_ablation.csv`, seed-block robustness in `seed_block_robustness.csv`, and a separate high-N stress panel in `stress_metrics.csv`. The claim audit requires positive paired raw-to-repair gain, high win rate, targeted hidden-property gain, ablation dominance, seed-block robustness, bootstrap lower-bound checks, and stress utility before marking the repair claim strongly supported.

## Attack: The raw score might only be miscalibrated under one exact scoring setup.

Response: `score_calibration.csv` bins raw candidate object scores and measures real utility, object-real gap, and identity-error rate. `sensitivity_metrics.csv` perturbs raw and repaired scores with score noise and requires the combined repair to remain strong under low-noise perturbations.

## Attack: Best-of-N may just be bad in every synthetic scene.

Response: `negative_control.csv` compares the good non-corrupted setting with corrupted high-N settings. The audit requires the good control to retain utility and low identity error while corrupted settings collapse.

## Attack: The result may not survive more distractor objects.

Response: `ood_metrics.csv` evaluates dense 6- and 8-object variants, including a dense good control and dense corrupted scenes. `extreme_object_count_metrics.csv` separately evaluates 10- and 12-object variants. The audit requires dense and extreme corrupted scenes to collapse under raw selection and recover under combined and observable repair.

## Attack: The scenarios are hand-picked.

Response: `domain_randomization_metrics.csv` evaluates a held-out randomized synthetic domain with varied object counts, occlusion, crossing, and hidden-property flags. The audit requires raw selection to remain unsafe while observable and combined repair recover utility on this randomized panel.

## Attack: The target identity is hard-coded.

Response: `retarget_scene` creates a counterfactual scene where the true target is object 1 rather than object 0, and the generator now chooses wrong identities relative to the scene target rather than a fixed object ID. `counterfactual_target_metrics.csv` and `figure19_counterfactual_target.png` require raw high-N selection to fail while observable and combined repair recover utility on this target-swap panel.

## Attack: Repairs only beat weak selector baselines.

Response: `model_family_proxy_metrics.csv` adds controlled toy proxy selectors with latent-global, relational-slot, and diffusion-score scoring rules. The audit requires combined repair to keep a positive scenario-wise margin over the best proxy while staying close to oracle. This is a diagnostic panel, not evidence for broad benchmark superiority.

## Attack: The paper text might drift beyond the artifact evidence.

Response: `run_claim_audit.sh` scans README, docs, and paper text for supported forbidden overclaims, verifies required tables, figures, and JSON artifacts, and writes `results/artifact_manifest.json` with deterministic hashes. The generated `docs/results_digest.md` records the current evidence boundaries.

## Attack: The learned model is too simple.

Response: The learned artifact is intentionally CPU NumPy and semi-learned. It now includes transition, hidden-property, identity-alignment, reward, learning-curve evidence, feature ablations, and `learned_domain_shift.csv` checks on held-out dense, occluded, crossing, and mixed object-count synthetic variants. It does not establish modern benchmark performance.

## Attack: The learned evidence may not use object information.

Response: `learned_ablation.csv` removes object-relevant features such as the mass sensor and identity-pair features, while `learned_domain_shift.csv` tests held-out denser and more occluded object scenes. The audit requires the full object-feature model to beat these ablations and retain margins on shifted synthetic variants.

## Attack: Oracle rows make repairs look weak or strong.

Response: Oracle rows are upper bounds for interpreting regret and oracle gap. Repair claims are based on controlled improvements over raw and random selectors, not equality to oracle in every setting.
