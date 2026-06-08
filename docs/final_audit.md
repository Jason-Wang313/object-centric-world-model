# Final Audit

Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.

## Command Results
- bash scripts/run_smoke.sh: pass (smoke experiment runtime 34.762s; strict claim audit passed)
- bash scripts/run_all.sh: pass (full experiment runtime 123.57s; 16 main seeds, 16 OOD dense-object seeds, 24 sensitivity seeds, 32 stress seeds, gate block_high_n)
- bash scripts/run_claim_audit.sh: pass (all core claims strongly_supported; artifact verifier, hashes, paper-text scan, and OOD checks passed)
- pytest: pass (13 passed in 11.56s on final run)

## Strongest Artifacts
- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv. Raw score gain 0.5759192453426587 and raw utility drop 0.36397088780796794.
- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level transition, hidden-property, identity-alignment, and reward predictors.
- Repair artifact: figure2_repair_comparison.png, paired_effects.csv, and stress_metrics.csv. Raw Nmax combined-repair gain 0.8803086375224858 with win rate 1.0.
- Ablation artifact: figure8_repair_ablation.png and repair_ablation.csv. Raw Nmax combined-repair dominance over the best single repair 0.27709535654229556.
- Robustness artifact: figure9_seed_block_robustness.png and seed_block_robustness.csv. Seed-block robustness pass rate 1.0.
- Stress artifact: figure6_stress_robustness.png. Combined repair mean selected stress utility 0.8494153088926296.
- Calibration artifact: figure10_score_calibration.png and score_calibration.csv. Top raw-score bin object-real gap 1.095467459764002.
- Sensitivity artifact: figure11_score_noise_sensitivity.png and sensitivity_metrics.csv. Combined repair low-noise minimum utility 0.8374916544433992.
- Negative-control artifact: figure12_negative_control.png and negative_control.csv. Good-control raw high-N utility 0.6554531451148605.
- Learned-ablation artifact: figure13_learned_ablation.png and learned_ablation.csv. Full-minus-no-mass property gain 0.1229166666666666.
- OOD artifact: figure14_ood_object_count_stress.png and ood_metrics.csv. Dense corrupted OOD combined-vs-raw gain 0.8411636120707556.

## Differentiation
The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.
It is not a graph-physics benchmark, a latent dynamics benchmark, a diffusion world-model benchmark, or a real-robot evaluation.

## Remaining Weaknesses
- Synthetic scenes remain controlled, though the default run now uses 16 main seeds and 32 stress seeds.
- Repairs use diagnostic signals available in the toy generator.
- No real-robot or broad benchmark evidence is claimed.

## Artifact Inventory
### tables
- results\tables\exact_law_validation.csv
- results\tables\learned_ablation.csv
- results\tables\learned_learning_curve.csv
- results\tables\learned_metrics.csv
- results\tables\main_metrics.csv
- results\tables\negative_control.csv
- results\tables\ood_metrics.csv
- results\tables\ood_seed_metrics.csv
- results\tables\paired_effects.csv
- results\tables\repair_ablation.csv
- results\tables\repair_metrics.csv
- results\tables\score_calibration.csv
- results\tables\score_calibration_candidates.csv
- results\tables\seed_block_robustness.csv
- results\tables\seed_metrics.csv
- results\tables\sensitivity_metrics.csv
- results\tables\sensitivity_seed_metrics.csv
- results\tables\stress_metrics.csv
- results\tables\stress_seed_metrics.csv
### figures
- figures\figure10_score_calibration.png
- figures\figure11_score_noise_sensitivity.png
- figures\figure12_negative_control.png
- figures\figure13_learned_ablation.png
- figures\figure14_ood_object_count_stress.png
- figures\figure1_selected_tail_binding_failure.png
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
- results/claims_status.json
- results/verification_log.json
