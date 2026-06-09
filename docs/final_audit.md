# Final Audit

Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.

## Command Results
- bash scripts/run_smoke.sh: pass (smoke experiment runtime 52.369s; strict claim audit passed; extreme object-count combined-vs-raw gain 0.7403197418948154; learned shift min property margin 0.125; learned shift min identity margin 0.4421874999999999; noisy-probe reliable gain 0.8110300082552486)
- bash scripts/run_all.sh: pass (full experiment runtime 342.967s; 16 main seeds, learned domain-shift panel with min property margin 0.125 and min identity margin 0.4458333333333333, 48 domain-randomized seeds, 48 counterfactual target seeds, 48 pilot calibration eval seeds, 864 pilot calibration rows, 40 leave-one-failure eval seeds per held-out family, 1200 leave-one-failure rows, 48 noisy-probe reliability seeds, 1440 noisy-probe rows, 16 OOD dense-object seeds, 24 extreme object-count seeds, 16 model-family proxy seeds, 24 sensitivity seeds, 32 stress seeds, observable repair panel, noisy-probe reliability panel, bootstrap statistical audit, gate block_high_n)
- bash scripts/run_claim_audit.sh: pass (all core claims strongly_supported; artifact verifier, hashes, paper-text scan, learned domain-shift checks, OOD checks, extreme object-count checks, domain-randomization checks, counterfactual target-swap checks, pilot-calibration checks, leave-one-failure calibration checks, noisy-probe checks, toy proxy checks, observable repair checks, and bootstrap checks passed)
- pytest: pass (16 passed in 10.27s on post-audit run)

## Strongest Artifacts
- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv. Raw score gain 0.5759192453426587 and raw utility drop 0.36397088780796794.
- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level transition, hidden-property, identity-alignment, and reward predictors.
- Repair artifact: figure2_repair_comparison.png, paired_effects.csv, and stress_metrics.csv. Raw Nmax combined-repair gain 0.8803086375224858 with win rate 1.0.
- Observable-repair artifact: figure17_observable_repair.png and observable_repair_metrics.csv. Raw Nmax observable-repair gain 0.8803086375224858.
- Ablation artifact: figure8_repair_ablation.png and repair_ablation.csv. Raw Nmax combined-repair dominance over the best single repair 0.27709535654229556.
- Robustness artifact: figure9_seed_block_robustness.png and seed_block_robustness.csv. Seed-block robustness pass rate 1.0.
- Stress artifact: figure6_stress_robustness.png. Combined repair mean selected stress utility 0.8494153088926296.
- Calibration artifact: figure10_score_calibration.png and score_calibration.csv. Top raw-score bin object-real gap 1.095467459764002.
- Sensitivity artifact: figure11_score_noise_sensitivity.png and sensitivity_metrics.csv. Combined repair low-noise minimum utility 0.8374916544433992.
- Negative-control artifact: figure12_negative_control.png and negative_control.csv. Good-control raw high-N utility 0.6554531451148605.
- Learned-ablation artifact: figure13_learned_ablation.png and learned_ablation.csv. Full-minus-no-mass property gain 0.1229166666666666.
- Learned domain-shift artifact: figure23_learned_domain_shift.png and learned_domain_shift.csv. Minimum shifted property margin 0.125 and identity margin 0.4458333333333333.
- OOD artifact: figure14_ood_object_count_stress.png and ood_metrics.csv. Dense corrupted OOD combined-vs-raw gain 0.8411636120707556.
- Extreme object-count artifact: figure24_extreme_object_count.png and extreme_object_count_metrics.csv. 10/12-object corrupted combined-vs-raw gain 0.8434197833015505.
- Domain-randomized artifact: figure18_domain_randomization.png and domain_randomization_metrics.csv. Combined-vs-raw gain 0.8343878574844603.
- Counterfactual target artifact: figure19_counterfactual_target.png and counterfactual_target_metrics.csv. Combined-vs-raw gain 0.816906396281512.
- Pilot-label calibration artifact: figure20_pilot_calibration.png, pilot_calibration_metrics.csv, and pilot_calibration_summary.json. Held-out calibrated-vs-raw gain 0.8192029444060406.
- Leave-one-failure-out artifact: figure21_leave_one_failure_out.png, leave_one_failure_metrics.csv, and leave_one_failure_summary.json. Held-out-family calibrated-vs-raw gain 0.7758011252035516.
- Noisy-probe artifact: figure22_noisy_probe_reliability.png and noisy_probe_metrics.csv. Reliable-probe gain 0.8667438725697415.
- Toy proxy artifact: figure15_model_family_proxies.png and model_family_proxy_metrics.csv. Combined-vs-best-proxy gain 0.5504614056934154.
- Statistical audit artifact: figure16_statistical_audit.png and statistical_audit.csv. Minimum bootstrap CI margin 0.08294730684862575.

## Differentiation
The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.
The toy proxy panel is a controlled diagnostic comparison, not a graph-physics benchmark, latent dynamics benchmark, diffusion world-model benchmark, or real-robot evaluation.

## Remaining Weaknesses
- Synthetic scenes remain controlled, though the default run now uses 16 main seeds, 32 stress seeds, dense and extreme object-count stress, held-out domain-randomized stress, held-out pilot-label calibration, leave-one-failure-out calibration, and noisy-probe reliability stress.
- Observable-only, pilot-calibrated, and noisy-probe repair reduce direct hidden-property truth alignment, and learned domain-shift tests add dense/occluded/crossing variants, but all probe and slot diagnostics still come from the toy generator.
- No real-robot or broad benchmark evidence is claimed.

## Artifact Inventory
### tables
- results\tables\counterfactual_target_metrics.csv
- results\tables\counterfactual_target_seed_metrics.csv
- results\tables\domain_randomization_metrics.csv
- results\tables\domain_randomization_seed_metrics.csv
- results\tables\exact_law_validation.csv
- results\tables\extreme_object_count_metrics.csv
- results\tables\extreme_object_count_seed_metrics.csv
- results\tables\learned_ablation.csv
- results\tables\learned_domain_shift.csv
- results\tables\learned_learning_curve.csv
- results\tables\learned_metrics.csv
- results\tables\leave_one_failure_metrics.csv
- results\tables\leave_one_failure_seed_metrics.csv
- results\tables\main_metrics.csv
- results\tables\model_family_proxy_metrics.csv
- results\tables\model_family_proxy_seed_metrics.csv
- results\tables\negative_control.csv
- results\tables\noisy_probe_metrics.csv
- results\tables\noisy_probe_seed_metrics.csv
- results\tables\observable_repair_metrics.csv
- results\tables\ood_metrics.csv
- results\tables\ood_seed_metrics.csv
- results\tables\paired_effects.csv
- results\tables\pilot_calibration_metrics.csv
- results\tables\pilot_calibration_seed_metrics.csv
- results\tables\repair_ablation.csv
- results\tables\repair_metrics.csv
- results\tables\score_calibration.csv
- results\tables\score_calibration_candidates.csv
- results\tables\seed_block_robustness.csv
- results\tables\seed_metrics.csv
- results\tables\sensitivity_metrics.csv
- results\tables\sensitivity_seed_metrics.csv
- results\tables\statistical_audit.csv
- results\tables\stress_metrics.csv
- results\tables\stress_seed_metrics.csv
### figures
- figures\figure10_score_calibration.png
- figures\figure11_score_noise_sensitivity.png
- figures\figure12_negative_control.png
- figures\figure13_learned_ablation.png
- figures\figure14_ood_object_count_stress.png
- figures\figure15_model_family_proxies.png
- figures\figure16_statistical_audit.png
- figures\figure17_observable_repair.png
- figures\figure18_domain_randomization.png
- figures\figure19_counterfactual_target.png
- figures\figure1_selected_tail_binding_failure.png
- figures\figure20_pilot_calibration.png
- figures\figure21_leave_one_failure_out.png
- figures\figure22_noisy_probe_reliability.png
- figures\figure23_learned_domain_shift.png
- figures\figure24_extreme_object_count.png
- figures\figure2_repair_comparison.png
- figures\figure3_tail_diagnostics.png
- figures\figure4_targeted_probe_before_after.png
- figures\figure5_exact_law_validation.png
- figures\figure6_stress_robustness.png
- figures\figure7_learned_object_model.png
- figures\figure8_repair_ablation.png
- figures\figure9_seed_block_robustness.png
### docs
- docs\claims.md
- docs\differentiation_from_best_of_n_wam.md
- docs\differentiation_from_prior_projects.md
- docs\final_audit.md
- docs\results_digest.md
- docs\reviewer_attacks.md
- docs\theory.md
### paper
- paper\abstract.md
- paper\checklist.md
- paper\experiments.md
- paper\intro.md
- paper\limitations.md
- paper\method.md
- paper\related_work.md
- paper\theory.md
### json
- results/run_summary.json
- results/learned_object_model_summary.json
- results/pilot_calibration_summary.json
- results/leave_one_failure_summary.json
- results/claims_status.json
- results/verification_log.json
