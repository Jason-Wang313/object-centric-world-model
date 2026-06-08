# Final Audit

Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.

## Command Results
- bash scripts/run_smoke.sh: pass (smoke experiment runtime 20.641s; strict claim audit passed)
- bash scripts/run_all.sh: pass (full experiment runtime 139.351s; 16 main seeds, 32 stress seeds, gate block_high_n)
- bash scripts/run_claim_audit.sh: pass (all core claims strongly_supported; unsupported claims remain unsupported)
- pytest: pass (13 passed in 11.06s on final run)

## Strongest Artifacts
- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv. Raw score gain 0.5759192453426587 and raw utility drop 0.36397088780796794.
- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level transition, hidden-property, identity-alignment, and reward predictors.
- Repair artifact: figure2_repair_comparison.png, paired_effects.csv, and stress_metrics.csv. Raw Nmax combined-repair gain 0.8803086375224858 with win rate 1.0.
- Stress artifact: figure6_stress_robustness.png. Combined repair mean selected stress utility 0.8494153088926296.

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
- results\tables\learned_learning_curve.csv
- results\tables\learned_metrics.csv
- results\tables\main_metrics.csv
- results\tables\paired_effects.csv
- results\tables\repair_metrics.csv
- results\tables\seed_metrics.csv
- results\tables\stress_metrics.csv
- results\tables\stress_seed_metrics.csv
### figures
- figures\figure1_selected_tail_binding_failure.png
- figures\figure2_repair_comparison.png
- figures\figure3_tail_diagnostics.png
- figures\figure4_targeted_probe_before_after.png
- figures\figure5_exact_law_validation.png
- figures\figure6_stress_robustness.png
- figures\figure7_learned_object_model.png
### docs
- docs\claims.md
- docs\differentiation_from_best_of_n_wam.md
- docs\differentiation_from_prior_projects.md
- docs\final_audit.md
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
