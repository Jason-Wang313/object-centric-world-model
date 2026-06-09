# When Objects Lie: Best-of-N Inference Laws for Object-Centric World Models

This repository is a CPU-first research scaffold for studying how Best-of-N inference behaves when the scored futures are object-centric: slots, identities, occlusion, hidden properties, binding failures, and object-specific repair all matter.

The core thesis is narrow: in controlled object-centric scenes, selecting the highest-scoring imagined future can amplify object binding errors, so selected object score can rise while selected real utility stagnates or falls. The repo also tests simple repairs: temporal identity consistency, hidden-property calibration, targeted probing, an observable-only repair score, and a combined repair stack.

## Quickstart

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

The full run writes CSV tables under `results/tables/`, figures under `figures/`, and audit files under `results/` and `docs/`.

## What This Is

- A finite tie-aware Best-of-N law implementation for real and binary utility.
- A synthetic 2D object manipulation environment with visually similar target/distractor objects.
- An object-centric future generator with slots, identities, hidden properties, trajectories, and failure diagnostics.
- A NumPy semi-learned object-centric model trained on generated slot trajectories.
- Held-out learned domain-shift checks for dense, occluded, crossing, and mixed object-count synthetic variants.
- Learned selection transfer, where the CPU NumPy reward and identity-alignment heads score held-out candidate futures.
- A repair comparison over raw scoring, identity consistency, property calibration, targeted probing, observable-only repair, combined repair, random selection, and oracle selection.
- Paired per-seed repair statistics, a high-N stress panel, and threshold-based claim auditing.
- Repair ablations, seed-block robustness checks, paper-text overclaim scanning, and artifact verification.
- Raw-score calibration, score-noise sensitivity analysis, and deterministic artifact hashes.
- Negative controls for the non-corrupted setting and learned feature ablations.
- Dense-object OOD synthetic stress for 6- and 8-object scenes.
- Extreme object-count synthetic stress for 10- and 12-object scenes.
- Held-out domain-randomized synthetic stress with varied object counts and corruption flags.
- Counterfactual target-identity stress that retargets the true object away from object zero.
- Multi-target identity sweep over six possible true target identities.
- Held-out pilot-label calibration for a learned utility selector trained from labeled object candidates.
- Pilot-label budget sensitivity for the same learned utility selector.
- Leave-one-failure-out pilot calibration, where each binding failure family is held out during pilot-calibrator training.
- Noisy diagnostic-probe reliability stress for observable repair under imperfect hidden-property observations.
- Diagnostic probe-cost sensitivity, reporting net utility after charging probe actions.
- A controlled toy model-family proxy panel for latent-global, relational-slot, and diffusion-score selectors.
- A bootstrap statistical audit for the main failure, repair, OOD, extreme object-count, counterfactual target, target-identity sweep, learned selection transfer, pilot calibration, leave-one-failure calibration, noisy-probe repair, probe-cost repair, and toy-proxy effects.

## What This Is Not

This is not a graph physics benchmark, not a diffusion world-model benchmark, not a latent dynamics benchmark, and not a real-robot evaluation. The evidence is controlled and synthetic unless the artifact explicitly says otherwise.

The project borrows only the abstract finite Best-of-N law pattern and audit discipline from WAM-style work. It does not reuse WAM environments, failure modes, package names, or claims. Here the scientific object is object binding under object-centric world-model inference.

## Required Outputs

- `results/tables/main_metrics.csv`
- `results/tables/seed_metrics.csv`
- `results/tables/learned_metrics.csv`
- `results/tables/learned_domain_shift.csv`
- `results/tables/learned_selection_seed_metrics.csv`
- `results/tables/learned_selection_metrics.csv`
- `results/tables/repair_metrics.csv`
- `results/tables/paired_effects.csv`
- `results/tables/repair_ablation.csv`
- `results/tables/observable_repair_metrics.csv`
- `results/tables/exact_law_validation.csv`
- `results/tables/stress_seed_metrics.csv`
- `results/tables/stress_metrics.csv`
- `results/tables/seed_block_robustness.csv`
- `results/tables/score_calibration_candidates.csv`
- `results/tables/score_calibration.csv`
- `results/tables/sensitivity_seed_metrics.csv`
- `results/tables/sensitivity_metrics.csv`
- `results/tables/negative_control.csv`
- `results/tables/learned_ablation.csv`
- `results/tables/ood_seed_metrics.csv`
- `results/tables/ood_metrics.csv`
- `results/tables/extreme_object_count_seed_metrics.csv`
- `results/tables/extreme_object_count_metrics.csv`
- `results/tables/domain_randomization_seed_metrics.csv`
- `results/tables/domain_randomization_metrics.csv`
- `results/tables/counterfactual_target_seed_metrics.csv`
- `results/tables/counterfactual_target_metrics.csv`
- `results/tables/target_identity_sweep_seed_metrics.csv`
- `results/tables/target_identity_sweep_metrics.csv`
- `results/tables/pilot_calibration_seed_metrics.csv`
- `results/tables/pilot_calibration_metrics.csv`
- `results/tables/pilot_budget_seed_metrics.csv`
- `results/tables/pilot_budget_metrics.csv`
- `results/tables/leave_one_failure_seed_metrics.csv`
- `results/tables/leave_one_failure_metrics.csv`
- `results/tables/noisy_probe_seed_metrics.csv`
- `results/tables/noisy_probe_metrics.csv`
- `results/tables/probe_cost_seed_metrics.csv`
- `results/tables/probe_cost_metrics.csv`
- `results/tables/model_family_proxy_seed_metrics.csv`
- `results/tables/model_family_proxy_metrics.csv`
- `results/tables/statistical_audit.csv`
- `results/tables/learned_learning_curve.csv`
- `results/run_summary.json`
- `results/learned_object_model_summary.json`
- `results/pilot_calibration_summary.json`
- `results/pilot_budget_summary.json`
- `results/leave_one_failure_summary.json`
- `results/verification_log.json`
- `results/artifact_manifest.json`
- `docs/results_digest.md`
- `figures/figure1_selected_tail_binding_failure.png`
- `figures/figure2_repair_comparison.png`
- `figures/figure3_tail_diagnostics.png`
- `figures/figure4_targeted_probe_before_after.png`
- `figures/figure5_exact_law_validation.png`
- `figures/figure6_stress_robustness.png`
- `figures/figure7_learned_object_model.png`
- `figures/figure8_repair_ablation.png`
- `figures/figure9_seed_block_robustness.png`
- `figures/figure10_score_calibration.png`
- `figures/figure11_score_noise_sensitivity.png`
- `figures/figure12_negative_control.png`
- `figures/figure13_learned_ablation.png`
- `figures/figure14_ood_object_count_stress.png`
- `figures/figure15_model_family_proxies.png`
- `figures/figure16_statistical_audit.png`
- `figures/figure17_observable_repair.png`
- `figures/figure18_domain_randomization.png`
- `figures/figure19_counterfactual_target.png`
- `figures/figure20_pilot_calibration.png`
- `figures/figure21_leave_one_failure_out.png`
- `figures/figure22_noisy_probe_reliability.png`
- `figures/figure23_learned_domain_shift.png`
- `figures/figure24_extreme_object_count.png`
- `figures/figure25_probe_cost_sensitivity.png`
- `figures/figure26_pilot_label_budget.png`
- `figures/figure27_target_identity_sweep.png`
- `figures/figure28_learned_selection_transfer.png`

## Claim Boundaries

Supported claims are limited to exact finite laws, controlled synthetic failure evidence, controlled repair evidence, toy proxy diagnostics, and one CPU NumPy semi-learned object-centric artifact. The claim audit marks core claims as `strongly_supported` only when generated artifacts clear numeric thresholds, required artifacts verify, and paper text avoids supported overclaims. Unsupported claims include real-robot validation, broad benchmark superiority, and universal object learning.

See `results/claims_status.md` and `docs/final_audit.md` after running the scripts.
