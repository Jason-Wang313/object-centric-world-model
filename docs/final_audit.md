# Final Audit

Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.

## Command Results
- bash scripts/run_smoke.sh: pass (smoke experiment runtime 77.252s; strict claim audit passed; tightened bootstrap statistical audit passed with min CI margin 0.010058488216230566; conservative learned repair-policy mean utility 0.8489640266219249; learned repair-policy raw gain 0.7376876264430621; learned repair-policy over learned-identity gain 0.2118973030297409; learned repair-policy mean learned-identity win rate 0.5714285714285714; deployment-policy corrupted gain 0.7185166981979354; deployment-policy corrupted stop-early gain 0.7341964238363273; synthetic task-suite combined gain 0.7553141399022321; synthetic task-suite observable gain 0.7476931390088214; learned-selection identity gain 0.6509536400923468; target-sweep combined gain 0.7895870042905537)
- bash scripts/run_all.sh: pass (full experiment runtime 813.495s; 16 main seeds, learned domain-shift panel with min property margin 0.125 and min identity margin 0.4458333333333333, learned-selection transfer with 32 eval seeds and 1344 rows, conservative learned repair-policy transfer with 32 eval seeds and 2016 rows, learned repair-policy mean utility 0.8457892245396742, learned repair-policy min variant utility 0.8194152353962161, learned repair-policy raw gain 0.8234771981724383, learned repair-policy over learned-identity gain 0.2252909624035247, learned repair-policy mean learned-identity win rate 0.6294642857142857, learned repair-policy min learned-identity win rate 0.375, benchmark-style synthetic task suite with 32 seeds and 1120 rows, deployment-policy panel with 384 rows, deployment-policy corrupted gain 0.7883015801574633, deployment-policy corrupted stop-early gain 0.5368058036226959, deployment-policy min corrupted utility 0.7880296839171549, deployment-policy min win rate 0.9375, synthetic task-suite combined gain 0.8161407808398964, synthetic task-suite observable gain 0.8160747948047943, learned-selection identity gain 0.6583388223801652, learned-selection identity-over-reward gain 0.3596642896057789, 48 domain-randomized seeds, 48 counterfactual target seeds, 48 target-identity sweep seeds over 6 target IDs, 1440 target-sweep rows, 48 pilot calibration eval seeds, 864 pilot calibration rows, 48 pilot-budget eval seeds, 5184 pilot-budget rows, 40 leave-one-failure eval seeds per held-out family, 1200 leave-one-failure rows, 48 noisy-probe reliability seeds, 1440 noisy-probe rows, 48 probe-cost seeds, 3360 probe-cost rows, 16 OOD dense-object seeds, 24 extreme object-count seeds, 16 model-family proxy seeds, 24 sensitivity seeds, 32 stress seeds, tightened bootstrap statistical audit passed with min CI margin 0.03294730684862576, target-sweep combined gain 0.8102272022985341, gate block_high_n)
- bash scripts/run_claim_audit.sh: pass (all core claims strongly_supported after conservative learned repair-policy blend and tightened bootstrap thresholds; paper-claim coverage matrix passed with 4/4 positive claims strongly covered, 2/2 boundary nonclaims explicit, and 6/6 cited claim locations verified; artifact verifier, hashes, paper-text scan, learned repair-policy checks, deployment-policy checks, learned domain-shift checks, learned-selection transfer checks, synthetic task-suite checks, OOD checks, extreme object-count checks, domain-randomization checks, counterfactual target-swap checks, target-identity sweep checks, pilot-calibration checks, pilot-budget checks, leave-one-failure calibration checks, noisy-probe checks, probe-cost checks, toy proxy checks, observable repair checks, and bootstrap checks passed)
- pytest: pass (18 passed in 39.04s after tightened statistical audit thresholds and fresh-root audit-output fix)

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
- Learned selection transfer artifact: figure28_learned_selection_transfer.png and learned_selection_metrics.csv. Identity+reward learned selector raw gain 0.6583388223801652 and identity-over-reward gain 0.3596642896057789.
- Learned repair-policy artifact: figure31_learned_repair_policy_transfer.png and learned_repair_policy_metrics.csv. Policy-vs-raw gain 0.8234771981724383 and policy-vs-learned-identity gain 0.2252909624035247; minimum learned-identity non-loss rate 0.65625 and worst learned-identity seed loss 0.1424497166038502.
- Synthetic task-suite artifact: figure29_synthetic_benchmark_suite.png and synthetic_benchmark_metrics.csv. Combined-vs-raw gain 0.8161407808398964 and minimum combined variant utility 0.7999228162650984.
- Deployment-policy artifact: figure30_deployment_gate_policy.png and deployment_policy_metrics.csv. Corrupted gate-vs-raw gain 0.7883015801574633 and corrupted gate-vs-stop-early gain 0.5368058036226959.
- OOD artifact: figure14_ood_object_count_stress.png and ood_metrics.csv. Dense corrupted OOD combined-vs-raw gain 0.8411636120707556.
- Extreme object-count artifact: figure24_extreme_object_count.png and extreme_object_count_metrics.csv. 10/12-object corrupted combined-vs-raw gain 0.8434197833015505.
- Domain-randomized artifact: figure18_domain_randomization.png and domain_randomization_metrics.csv. Combined-vs-raw gain 0.8343878574844603.
- Counterfactual target artifact: figure19_counterfactual_target.png and counterfactual_target_metrics.csv. Combined-vs-raw gain 0.816906396281512.
- Target-identity sweep artifact: figure27_target_identity_sweep.png and target_identity_sweep_metrics.csv. Combined-vs-raw gain 0.8102272022985341, with minimum target utility 0.7955217976711665.
- Pilot-label calibration artifact: figure20_pilot_calibration.png, pilot_calibration_metrics.csv, and pilot_calibration_summary.json. Held-out calibrated-vs-raw gain 0.8192029444060406.
- Pilot-label budget artifact: figure26_pilot_label_budget.png, pilot_budget_metrics.csv, and pilot_budget_summary.json. Mature-budget gain 0.8331483971245207 and largest-budget gain 0.8331483971245207.
- Leave-one-failure-out artifact: figure21_leave_one_failure_out.png, leave_one_failure_metrics.csv, and leave_one_failure_summary.json. Held-out-family calibrated-vs-raw gain 0.7758011252035516.
- Noisy-probe artifact: figure22_noisy_probe_reliability.png and noisy_probe_metrics.csv. Reliable-probe gain 0.8667438725697415.
- Probe-cost artifact: figure25_probe_cost_sensitivity.png and probe_cost_metrics.csv. Low-cost combined-vs-raw gain 0.77494187977645 and max-cost gain 0.51744187977645.
- Toy proxy artifact: figure15_model_family_proxies.png and model_family_proxy_metrics.csv. Combined-vs-best-proxy gain 0.5504614056934154.
- Statistical audit artifact: figure16_statistical_audit.png and statistical_audit.csv. Minimum bootstrap CI margin 0.03294730684862576.
- Paper-claim coverage artifact: docs/paper_claim_coverage.md, results/paper_claim_coverage.json, and paper_claim_coverage.csv. Positive paper claims map to C1-C4; real-robot and broad-benchmark rows are boundary nonclaims.

## Differentiation
The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.
The toy proxy panel is a controlled diagnostic comparison, not a graph-physics benchmark, latent dynamics benchmark, diffusion world-model benchmark, or real-robot evaluation.

## Remaining Weaknesses
- Synthetic scenes remain controlled, though the default run now uses 16 main seeds, 32 stress seeds, dense and extreme object-count stress, benchmark-style synthetic task-suite stress, deployment-gate policy simulation, held-out domain-randomized stress, target-identity sweep stress, learned selection transfer, learned repair-policy transfer, held-out pilot-label calibration, pilot-label budget sensitivity, leave-one-failure-out calibration, noisy-probe reliability stress, and probe-cost sensitivity.
- Observable-only, pilot-calibrated, noisy-probe, and probe-cost repair reduce direct hidden-property truth alignment and free-probe assumptions, and learned domain-shift tests add dense/occluded/crossing variants, but all probe and slot diagnostics still come from the toy generator.
- No real-robot or broad benchmark evidence is claimed.

## Artifact Inventory
### tables
- results\tables\calibration_diagnostics.csv
- results\tables\counterfactual_target_metrics.csv
- results\tables\counterfactual_target_seed_metrics.csv
- results\tables\deployment_policy_metrics.csv
- results\tables\deployment_policy_seed_metrics.csv
- results\tables\domain_randomization_metrics.csv
- results\tables\domain_randomization_seed_metrics.csv
- results\tables\exact_law_validation.csv
- results\tables\extreme_object_count_metrics.csv
- results\tables\extreme_object_count_seed_metrics.csv
- results\tables\learned_ablation.csv
- results\tables\learned_domain_shift.csv
- results\tables\learned_generalization_diagnostics.csv
- results\tables\learned_learning_curve.csv
- results\tables\learned_metrics.csv
- results\tables\learned_repair_policy_metrics.csv
- results\tables\learned_repair_policy_seed_metrics.csv
- results\tables\learned_selection_metrics.csv
- results\tables\learned_selection_seed_metrics.csv
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
- results\tables\paper_claim_coverage.csv
- results\tables\pilot_budget_metrics.csv
- results\tables\pilot_budget_seed_metrics.csv
- results\tables\pilot_calibration_metrics.csv
- results\tables\pilot_calibration_seed_metrics.csv
- results\tables\probe_cost_metrics.csv
- results\tables\probe_cost_seed_metrics.csv
- results\tables\repair_ablation.csv
- results\tables\repair_condition_splits.csv
- results\tables\repair_final_test_metrics.csv
- results\tables\repair_metrics.csv
- results\tables\repair_model_selection.csv
- results\tables\repair_robustness_by_split.csv
- results\tables\score_calibration.csv
- results\tables\score_calibration_candidates.csv
- results\tables\seed_block_robustness.csv
- results\tables\seed_metrics.csv
- results\tables\sensitivity_metrics.csv
- results\tables\sensitivity_seed_metrics.csv
- results\tables\statistical_audit.csv
- results\tables\stress_metrics.csv
- results\tables\stress_seed_metrics.csv
- results\tables\synthetic_benchmark_metrics.csv
- results\tables\synthetic_benchmark_seed_metrics.csv
- results\tables\target_identity_sweep_metrics.csv
- results\tables\target_identity_sweep_seed_metrics.csv
- results\tables\unidentifiable_negative_control.csv
### figures
- figures\figure10_repair_robustness.png
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
- figures\figure25_probe_cost_sensitivity.png
- figures\figure26_pilot_label_budget.png
- figures\figure27_target_identity_sweep.png
- figures\figure28_learned_selection_transfer.png
- figures\figure29_synthetic_benchmark_suite.png
- figures\figure2_repair_comparison.png
- figures\figure30_deployment_gate_policy.png
- figures\figure31_learned_repair_policy_transfer.png
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
- docs\paper_claim_coverage.md
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
- results/learned_repair_policy_summary.json
- results/paper_claim_coverage.json
- results/pilot_calibration_summary.json
- results/pilot_budget_summary.json
- results/leave_one_failure_summary.json
- results/claims_status.json
- results/verification_log.json
