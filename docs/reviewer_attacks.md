# Reviewer Attacks

## Attack: The law is generic, not object-centric.

Response: Correct. The finite law is generic. The object-centric contribution is defining the score/utility pairs around slots, target identity, occlusion, hidden properties, and repair signals, then showing controlled selected-tail failures.

## Attack: The experiments are synthetic.

Response: Correct. Claims are limited to controlled synthetic and semi-learned CPU evidence. Real-robot and broad benchmark claims are explicitly unsupported.

## Attack: Repairs may use generator diagnostics.

Response: Correct for the stronger controlled and oracle stacks, which is why repair tables now carry `repair_tier`, `uses_real_utility_features`, `uses_hidden_features`, `hyperparameter_source`, `split_seed`, and `final_test`. Deployable no-leak rows may use generated futures, model scores, generated uncertainty, and pilot labels only. Probe/simulator rows are support-covered, and all-candidate labeled or hidden-truth rows are oracle upper bounds, not deployment evidence.

## Attack: The repair result leaks evaluation real utility.

Response: The audit fails if any `deployable_no_leak` row is marked as using real-utility features or hidden features, or if an oracle-like selector is mislabeled as deployable. Pilot labels are restricted to pilot-train/calibration conditions, and final-test selectors cannot access evaluation `real_utility`.

## Attack: The repair hyperparameters were hand-tuned on test.

Response: `repair_condition_splits.csv` records condition-level pilot train, pilot calibration, dev, and final-test splits for each split seed. `repair_model_selection.csv` and `repair_model_selection.json` record the fixed grid and selected configuration. Selection uses dev conditions only; final numbers are reported on held-out final-test conditions.

## Attack: Diagnostic probes are unrealistically clean.

Response: `noisy_probe_metrics.csv` varies the diagnostic observation reliability from barely better than chance to clean-probe settings. The noisy-probe selector uses observable slot diagnostics and the noisy posterior, not direct hidden-property truth alignment, and the audit requires selected utility, raw gain, win rate, oracle gap, and bootstrap lower-bound checks for reliability at or above 0.75.

## Attack: Diagnostic probes are treated as free.

Response: `probe_cost_metrics.csv` charges diagnostic actions before reporting selected real utility. The audit requires combined and observable repair to keep positive cost-adjusted gains for probe costs up to 0.10, requires targeted probing to remain positive in hidden-property scenes, and requires combined and observable repair to retain positive margins at higher costs.

## Attack: The deployment gate says to collect pilot labels, but labels are not tested.

Response: `pilot_calibration_metrics.csv` evaluates a held-out selector trained from pilot-labeled candidates using observable object features. The train seeds are separate from the held-out raw, randomized-domain, and target-swap evaluation seeds. `pilot_budget_metrics.csv` sweeps the number of labeled candidates, and `leave_one_failure_metrics.csv` holds each raw, occlusion, hidden-property, swap, or merge/split failure family out during pilot-calibrator training and then evaluates it as the test family. The audit requires calibrated selected utility, paired raw gain, budget sensitivity, win rate, oracle gap, leave-one-failure transfer, and bootstrap lower-bound checks before counting the evidence.

## Attack: The deployment gate emits actions, but obeying the gate is not evaluated.

Response: `deployment_policy_metrics.csv` and `figure30_deployment_gate_policy.png` simulate a conservative policy that maps gate actions to raw high-N, raw stop-early, observable repair, targeted probing, or combined repair. The audit requires the gate policy to recover corrupted scenarios while beating both raw high-N selection and a raw stop-early fallback with bootstrap lower-bound checks.

## Attack: The repair result could be a seed fluke.

Response: The upgraded run reports paired per-seed gains in `paired_effects.csv`, nonparametric bootstrap intervals in `statistical_audit.csv`, repair ablations in `repair_ablation.csv`, seed-block robustness in `seed_block_robustness.csv`, and a separate high-N stress panel in `stress_metrics.csv`. The claim audit requires positive paired raw-to-repair gain, high win rate, targeted hidden-property gain, ablation dominance, seed-block robustness, bootstrap lower-bound checks, and stress utility before marking the repair claim strongly supported.

## Attack: The raw score might only be miscalibrated under one exact scoring setup.

Response: `score_calibration.csv` bins raw candidate object scores and measures real utility, object-real gap, and identity-error rate. `sensitivity_metrics.csv` perturbs raw and repaired scores with score noise and requires the combined repair to remain strong under low-noise perturbations.

## Attack: score-tail may just be bad in every synthetic scene.

Response: `negative_control.csv` compares the good non-corrupted setting with corrupted high-N settings. The audit requires the good control to retain utility and low identity error while corrupted settings collapse.

## Attack: The result may not survive more distractor objects.

Response: `ood_metrics.csv` evaluates dense 6- and 8-object variants, including a dense good control and dense corrupted scenes. `extreme_object_count_metrics.csv` separately evaluates 10- and 12-object variants. The audit requires dense and extreme corrupted scenes to collapse under raw selection and recover under combined and observable repair.

## Attack: The scenarios are hand-picked.

Response: `domain_randomization_metrics.csv` evaluates a held-out randomized synthetic domain with varied object counts, occlusion, crossing, and hidden-property flags. The audit requires raw selection to remain unsafe while observable and combined repair recover utility on this randomized panel.

## Attack: The stress tests are isolated cases rather than a suite.

Response: `synthetic_benchmark_metrics.csv` adds a benchmark-style controlled synthetic task suite covering dense clutter, retargeted nonzero-object targets, crossing swaps, occlusion corridors, hidden-mass probing, merge/split clutter, and mixed raw scenes. The audit requires raw high-N selection to collapse across the suite while observable and combined repair recover utility with paired win rates and bootstrap lower-bound checks. This is still a synthetic suite, not a broad external benchmark.

## Attack: The target identity is hard-coded.

Response: `retarget_scene` creates a counterfactual scene where the true target is object 1 rather than object 0, and the generator now chooses wrong identities relative to the scene target rather than a fixed object ID. `target_identity_sweep_metrics.csv` broadens this to six possible true target identities in 6-object scenes. `counterfactual_target_metrics.csv`, `target_identity_sweep_metrics.csv`, `figure19_counterfactual_target.png`, and `figure27_target_identity_sweep.png` require raw high-N selection to fail while observable and combined repair recover utility.

## Attack: Repairs only beat weak selector baselines.

Response: `model_family_proxy_metrics.csv` adds controlled toy proxy selectors with latent-global, relational-slot, and diffusion-score scoring rules. The audit requires combined repair to keep a positive scenario-wise margin over the best proxy while staying close to oracle. This is a diagnostic panel, not evidence for broad benchmark superiority.

## Attack: The paper text might drift beyond the artifact evidence.

Response: `run_claim_audit.sh` scans README, docs, and paper text for supported forbidden overclaims, verifies required tables, figures, and JSON artifacts, and writes `results/artifact_manifest.json` with deterministic hashes. The generated `docs/results_digest.md` records the current evidence boundaries.

## Attack: The learned model is too simple.

Response: The learned artifact is intentionally CPU NumPy and semi-learned. It now includes transition, hidden-property, identity-alignment, reward, learning-curve evidence, feature ablations, and `learned_domain_shift.csv` checks on held-out dense, occluded, crossing, and mixed object-count synthetic variants. It does not establish modern benchmark performance.

## Attack: The learned evidence may not use object information.

Response: `learned_ablation.csv` removes object-relevant features such as the mass sensor and identity-pair features, while `learned_domain_shift.csv` tests held-out denser and more occluded object scenes. The audit requires the full object-feature model to beat these ablations and retain margins on shifted synthetic variants.

## Attack: The learned model is disconnected from score-tail selection.

Response: `learned_selection_metrics.csv` uses the CPU NumPy reward and identity-alignment heads to score held-out candidate futures under score-tail selection. The reward-only learned selector is reported separately, and the audit requires the identity+reward learned selector to beat raw selection, beat reward-only selection, satisfy paired win-rate thresholds, and pass bootstrap lower-bound checks. This remains controlled synthetic transfer evidence, not a benchmark claim.

## Attack: The learned model is disconnected from the repair stack.

Response: `learned_repair_policy_metrics.csv` trains a small ridge repair policy from observable candidate diagnostics plus CPU learned reward, identity-alignment, and hidden-property confidence heads, then selects with a conservative blend of ridge utility, learned identity-reward, and normalized observable repair scores on held-out benchmark-style synthetic variants. The audit requires the learned repair policy to beat raw selection, beat the learned identity+reward selector in mean paired utility with bootstrap lower-bound checks, retain a minimum non-loss rate against learned identity, bound the worst seed-level loss, and remain close to oracle.

## Attack: Oracle rows make repairs look weak or strong.

Response: Oracle rows are upper bounds for interpreting regret and oracle gap. `repair_all_candidates_labeled_oracle` and hidden-feature repair rows are explicitly labeled `oracle_upper_bound`, and the audit fails if they are presented as deployable evidence.

## Attack: Some hidden modes are impossible to identify.

Response: `unidentifiable_negative_control.csv` constructs paired candidates with identical observable generated features but different hidden real utility. The adaptive gate must return `block_high_n` with `hidden_mode_unidentifiable` or `tail_rank_failure`. This is why the paper does not claim universal repair or guaranteed 100% recovery.

## Attack: Learned repair could rely on weak raw learned utility ordering.

Response: `learned_generalization_diagnostics.csv` reports held-out trajectory MSE, final-state error, proxy loss, sample diversity, rank correlation, selected-tail calibration error, and repair gap closure. If learned rank correlation is weak, the claim is limited to pilot calibration and uncertainty-assisted selection rather than raw learned utility ordering.
