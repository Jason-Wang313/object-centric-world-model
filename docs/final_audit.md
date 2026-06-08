# Final Audit

Paper-readiness judgment: paper-worthy v1 for controlled synthetic evidence; needs benchmark validation for broader claims.

## Command Results
- bash scripts/run_smoke.sh: pass (smoke experiment runtime 12.771s; generated required artifact set)
- bash scripts/run_all.sh: pass (full experiment runtime 66.563s; gate block_high_n)
- bash scripts/run_claim_audit.sh: pass (standalone audit rerun passed)
- pytest: pass (12 passed in 12.43s on final run)

## Strongest Artifacts
- Failure artifact: figure1_selected_tail_binding_failure.png and raw high-N rows in main_metrics.csv.
- Learned artifact: learned_object_model_summary.json with CPU NumPy slot-level predictors.
- Repair artifact: figure2_repair_comparison.png and repair_metrics.csv.

## Differentiation
The repo reuses the finite Best-of-N law pattern only. It changes the scientific object to object-centric slots, identity persistence, occlusion, hidden properties, and object-level repair.
It is not a graph-physics benchmark, a latent dynamics benchmark, a diffusion world-model benchmark, or a real-robot evaluation.

## Remaining Weaknesses
- Synthetic scenes are intentionally controlled and small.
- Repairs use diagnostic signals available in the toy generator.
- No real-robot or broad benchmark evidence is claimed.

## Artifact Inventory
### tables
- results\tables\exact_law_validation.csv
- results\tables\learned_metrics.csv
- results\tables\main_metrics.csv
- results\tables\repair_metrics.csv
- results\tables\seed_metrics.csv
### figures
- figures\figure1_selected_tail_binding_failure.png
- figures\figure2_repair_comparison.png
- figures\figure3_tail_diagnostics.png
- figures\figure4_targeted_probe_before_after.png
- figures\figure5_exact_law_validation.png
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
