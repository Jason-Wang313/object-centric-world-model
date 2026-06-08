import math

import numpy as np

from object_centric_best_of_n.theory import (
    anti_aligned_score_check,
    exact_best_of_n_expected_utility,
    exact_binary_expected_success,
    monte_carlo_best_of_n,
    oracle_score_check,
    selected_tie_groups,
)


def test_exact_law_real_utility_with_ties():
    utilities = [0.1, 0.5, 0.9]
    scores = [1.0, 2.0, 2.0]
    groups = selected_tie_groups(utilities, scores, n=2)
    expected = (1.0 - (1 / 3) ** 2) * 0.7 + (1 / 3) ** 2 * 0.1
    assert math.isclose(sum(group.probability for group in groups), 1.0)
    assert math.isclose(exact_best_of_n_expected_utility(utilities, scores, n=2), expected)


def test_binary_constant_oracle_anti_aligned_and_monte_carlo():
    utilities = np.array([0.0, 0.25, 0.75, 1.0])
    assert exact_binary_expected_success([0, 1, 1], [0.2, 0.3, 0.3], n=4) > 0.6
    assert math.isclose(exact_best_of_n_expected_utility([0.42, 0.42, 0.42], [3, 2, 1], n=16), 0.42)
    assert oracle_score_check(utilities, n=8)["passes"]
    assert anti_aligned_score_check(utilities, n=8)["passes"]
    exact = exact_best_of_n_expected_utility(utilities, [0.0, 0.2, 0.9, 1.0], n=4)
    mc = monte_carlo_best_of_n(utilities, [0.0, 0.2, 0.9, 1.0], n=4, trials=25_000, seed=3)
    assert abs(exact - mc) < 0.025
