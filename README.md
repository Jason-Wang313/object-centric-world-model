# When Objects Lie: Best-of-N Inference Laws for Object-Centric World Models

This repository is a CPU-first research scaffold for studying how Best-of-N inference behaves when the scored futures are object-centric: slots, identities, occlusion, hidden properties, binding failures, and object-specific repair all matter.

The core thesis is narrow: in controlled object-centric scenes, selecting the highest-scoring imagined future can amplify object binding errors, so selected object score can rise while selected real utility stagnates or falls. The repo also tests simple repairs: temporal identity consistency, hidden-property calibration, targeted probing, and a combined repair stack.

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
- A repair comparison over raw scoring, identity consistency, property calibration, targeted probing, combined repair, random selection, and oracle selection.

## What This Is Not

This is not a graph physics benchmark, not a diffusion world-model benchmark, not a latent dynamics benchmark, and not a real-robot evaluation. The evidence is controlled and synthetic unless the artifact explicitly says otherwise.

The project borrows only the abstract finite Best-of-N law pattern and audit discipline from WAM-style work. It does not reuse WAM environments, failure modes, package names, or claims. Here the scientific object is object binding under object-centric world-model inference.

## Required Outputs

- `results/tables/main_metrics.csv`
- `results/tables/seed_metrics.csv`
- `results/tables/learned_metrics.csv`
- `results/tables/repair_metrics.csv`
- `results/tables/exact_law_validation.csv`
- `results/run_summary.json`
- `results/learned_object_model_summary.json`
- `figures/figure1_selected_tail_binding_failure.png`
- `figures/figure2_repair_comparison.png`
- `figures/figure3_tail_diagnostics.png`
- `figures/figure4_targeted_probe_before_after.png`
- `figures/figure5_exact_law_validation.png`

## Claim Boundaries

Supported claims are limited to exact finite laws, controlled synthetic failure evidence, controlled repair evidence, and one CPU NumPy semi-learned object-centric artifact. Unsupported claims include real-robot validation, broad benchmark superiority, and universal object learning.

See `results/claims_status.md` and `docs/final_audit.md` after running the scripts.
