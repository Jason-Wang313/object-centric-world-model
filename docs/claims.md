# Claims

## Supported

- Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.
- Controlled object-centric scenes show selected-tail binding failure where object score can increase while real utility stagnates or falls; the good-scene negative control avoids the same collapse, and dense-object OOD variants reproduce the corrupted-setting failure.
- Identity consistency, hidden-property calibration, targeted probing, observable-only repair, pilot-label calibration, and combined repair improve selected utility in the controlled synthetic setting, including paired seed gains, bootstrap confidence checks, repair ablations, seed-block robustness, score-noise sensitivity, dense-object OOD stress, held-out domain-randomized stress, counterfactual target-swap stress, toy model-family proxy diagnostics, and a high-N stress panel.
- A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories, with feature ablations showing object information matters.

## Unsupported

- Real-robot validation.
- Broad benchmark superiority over graph physics, latent dynamics, or diffusion world models; the toy proxy panel is only a controlled diagnostic comparison.
- Universal object learning or general robotic reliability.

The executable audit writes the machine-readable version to `results/claims_status.json`. Core claims are `strongly_supported` only if numeric thresholds in the generated claim-strength record pass, required artifacts verify, artifact hashes are written, and paper text avoids supported overclaims.
