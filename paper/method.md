# Method

The method has four pieces.

First, we compute exact Best-of-N selected utility for a finite scored population with tie-aware selection. The law is independent of the object model.

Second, we generate controlled object-centric candidates. Each candidate contains predicted slots, object identities, hidden-mass estimates, trajectories, an object score, and measured real utility.

Third, we train a small NumPy object-centric model on generated slot trajectories. It learns linear transition, hidden-property, identity-alignment, and reward predictors from slot and slot-pair features, then is checked on held-out dense, occluded, crossing, and mixed object-count synthetic variants. A separate learned repair policy fits a ridge utility predictor from observable candidate diagnostics plus those learned heads, then selects with a conservative blend of the ridge utility estimate, learned identity-reward score, and normalized observable repair score on held-out benchmark-style synthetic variants. The learned repair-policy audit reports mean gain over learned identity, strict win rate, non-loss rate, and worst seed-level loss against the learned identity+reward selector. This is a semi-learned artifact, not a broad benchmark model.

Fourth, we compare selectors: raw score, identity-consistent score, property-calibrated score, targeted-probe repair, observable-only repair, combined repair, random selection, and oracle selection. Probe-cost sensitivity subtracts a fixed diagnostic-action cost from probe-using selectors before reporting net selected utility.

Fifth, we separate repair evidence into tiers. Deployable no-leak rows may use generated futures, model scores, generated uncertainty features, and pilot labels, but not evaluation real utility or hidden truth features. Support-covered rows may use simulator-visible diagnostic/probe proxies. Oracle upper-bound rows include all-candidate labeled selectors or hidden-truth features and are not deployment evidence.

Sixth, we aggregate paired seed gains, high-N stress rows, dense OOD rows, extreme 10/12-object rows, and deployment-gate policy rows. The nested repair panel splits whole conditions, not candidates from the same condition, into pilot train, pilot calibration, dev, and final test. A small fixed grid is selected only on dev, and final-test robustness is reported across five split seeds. The claim audit requires numeric margins before marking core claims strongly supported.

For pilot-label calibration, we also sweep the number of labeled candidate futures used to fit the utility calibrator. The held-out scenes are fixed across budgets, so the budget curve reflects calibration data size rather than a changing evaluation set.
