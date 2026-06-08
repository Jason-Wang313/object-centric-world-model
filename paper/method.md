# Method

The method has four pieces.

First, we compute exact Best-of-N selected utility for a finite scored population with tie-aware selection. The law is independent of the object model.

Second, we generate controlled object-centric candidates. Each candidate contains predicted slots, object identities, hidden-mass estimates, trajectories, an object score, and measured real utility.

Third, we train a small NumPy object-centric model on generated slot trajectories. It learns linear transition, hidden-property, and reward predictors from slot features. This is a semi-learned artifact, not a broad benchmark model.

Fourth, we compare selectors: raw score, identity-consistent score, property-calibrated score, targeted-probe repair, combined repair, random selection, and oracle selection.
