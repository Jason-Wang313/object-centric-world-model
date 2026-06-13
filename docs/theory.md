# Theory

## Object-Centric Setup

Each imagined future is a finite object-centric candidate with slots, predicted object identities, attributes, trajectories, hidden-property beliefs, an object score, and a real utility measured against the true target object. A score-tail selector samples N candidates and selects the candidate with maximal score.

The object-specific concern is that the score may reward plausible-looking slots while real utility depends on the true target identity, hidden mass/friction, and whether the chosen trajectory preserves object binding through occlusion or crossing.

## Finite Tie-Aware Law

Let the finite candidate population be indexed by `i in {1,...,m}`. Candidate `i` has score `s_i` and real utility `u_i`. Draw N candidates independently and uniformly with replacement. Select a draw with maximal score; if several sampled positions tie for maximal score, break the tie uniformly across tied positions.

Partition candidates into score tie groups `G_1,...,G_K` sorted from high score to low score. For group `G_k`, let `b_k` be the number of candidates in lower-score groups. The probability that group `G_k` is selected is:

```text
P(G_k selected) = ((|G_k| + b_k) / m)^N - (b_k / m)^N
```

Conditional on selecting group `G_k`, the selected expected utility is the mean utility of candidates in that group. Therefore:

```text
E[u_selected] = sum_k P(G_k selected) mean_{i in G_k} u_i
```

This handles real utility, binary success, ties, constant utilities, oracle scores, and anti-aligned scores. The implementation is in `src/object_binding_tail_audit/theory.py`, with Monte Carlo validation in `results/tables/exact_law_validation.csv`.

## Why Objects Matter

The law itself is not object-specific. The object-centric contribution is the choice of score and utility variables: a future can score well because the slots look coherent while the selected real utility falls because the target identity swapped, a hidden property was misread, or two objects were merged into one slot. The law predicts the selected tail once these object-level score/utility pairs are defined.

## Differentiation From WAM

This repository reuses only the abstract finite score-tail law pattern. It changes the experimental object to object-centric world models and studies slots, identities, occlusion, hidden properties, binding failures, and object-specific repair. It does not claim WAM-style dynamics evidence or reuse WAM experiments.
