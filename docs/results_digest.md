# Results Digest

This digest is generated from the current result artifacts.

## Headline Numbers
- Exact-law mean absolute error: 0.0004629118998613018
- Raw selected-tail score gain: 0.5759192453426587
- Raw selected-tail utility drop: 0.36397088780796794
- Combined repair raw Nmax gain: 0.8803086375224858
- Combined repair raw ablation dominance: 0.27709535654229556
- Stress combined mean selected utility: 0.8494153088926296
- Seed-block robustness pass rate: 1.0
- Top raw-score calibration gap: 1.095467459764002
- Combined repair low-noise minimum utility: 0.8374916544433992
- Combined-vs-raw low-noise sensitivity margin: 0.8534731753001509
- Good-control raw high-N utility: 0.6554531451148605
- Good-minus-corrupted raw high-N utility: 0.608449149414918
- Learned full-minus-no-mass property accuracy: 0.1229166666666666
- Learned full-minus-kinematic-pair identity accuracy: 0.0760416666666666
- OOD combined mean selected utility: 0.876852645124664
- OOD combined-vs-raw gain: 0.8411636120707556
- Toy proxy combined-vs-best-proxy gain: 0.5504614056934154
- Bootstrap audit minimum CI margin: 0.08294730684862575

## Learned Model
- Property accuracy: 0.9958333333333333 versus baseline 0.75
- Identity alignment accuracy: 0.9875 versus baseline 0.5
- Transition MSE ratio: 0.007032264292782088
- Reward correlation: 0.953061460933608

## Claim Status
- C1: strongly_supported - Exact finite tie-aware Best-of-N laws predict selected utility on finite object-candidate populations.
- C2: strongly_supported - In controlled object-centric scenes, high-N selection can increase object score while real utility stagnates or falls due to binding failures.
- C3: strongly_supported - Identity, hidden-property, and targeted-probe repairs improve selected utility in the controlled synthetic setting.
- C4: strongly_supported - A CPU NumPy semi-learned object-centric model improves property, identity-alignment, and transition prediction over simple baselines on generated trajectories.
- C5: unsupported - The method is validated on real robot systems.
- C6: unsupported - The method establishes broad benchmark superiority over graph physics, latent, or diffusion world models.

## Boundaries
Real-robot validation and broad benchmark superiority remain unsupported and are not claimed as supported paper results.
