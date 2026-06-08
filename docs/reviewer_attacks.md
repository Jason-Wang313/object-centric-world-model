# Reviewer Attacks

## Attack: The law is generic, not object-centric.

Response: Correct. The finite law is generic. The object-centric contribution is defining the score/utility pairs around slots, target identity, occlusion, hidden properties, and repair signals, then showing controlled selected-tail failures.

## Attack: The experiments are synthetic.

Response: Correct. Claims are limited to controlled synthetic and semi-learned CPU evidence. Real-robot and broad benchmark claims are explicitly unsupported.

## Attack: Repairs may use generator diagnostics.

Response: Correct. The repair stack uses controlled diagnostic signals such as identity instability, merge evidence, property entropy, and targeted probe observations. This is appropriate for a v1 mechanism study, not a deployment claim.

## Attack: The repair result could be a seed fluke.

Response: The upgraded run reports paired per-seed gains in `paired_effects.csv` and a separate high-N stress panel in `stress_metrics.csv`. The claim audit requires positive paired raw-to-repair gain, high win rate, targeted hidden-property gain, and stress utility before marking the repair claim strongly supported.

## Attack: The learned model is too simple.

Response: The learned artifact is intentionally CPU NumPy and semi-learned. It now includes transition, hidden-property, identity-alignment, reward, and learning-curve evidence. It does not establish modern benchmark performance.

## Attack: Oracle rows make repairs look weak or strong.

Response: Oracle rows are upper bounds for interpreting regret and oracle gap. Repair claims are based on controlled improvements over raw and random selectors, not equality to oracle in every setting.
