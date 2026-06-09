from object_centric_best_of_n.envs import make_scene
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator
from object_centric_best_of_n.repair import (
    GATE_ACTIONS,
    brute_force_slot_alignment,
    conservative_selected_tail_stop_rule,
    fit_pilot_calibrator,
    pilot_calibrated_score,
    pilot_calibration_features,
    property_posterior_update,
    targeted_diagnostic_probe,
)
from object_centric_best_of_n.selection import SELECTORS


def _candidates():
    scene = make_scene(seed=11, occlusion=True, hidden_property=True, crossing=True)
    generator = ObjectCentricFutureGenerator(seed=5)
    return scene, generator.generate_candidates(scene, n=16, scenario="raw", seed=12)


def test_selectors_return_candidates_and_repairs_are_available():
    scene, candidates = _candidates()
    raw = SELECTORS["raw"](candidates, scene, seed=0)
    oracle = SELECTORS["oracle"](candidates, scene, seed=0)
    combined = SELECTORS["combined_repair"](candidates, scene, seed=0)
    observable = SELECTORS["observable_repair"](candidates, scene, seed=0)
    assert raw.candidate_id in range(len(candidates))
    assert oracle.real_utility >= min(c.real_utility for c in candidates)
    assert combined.diagnostics["selector_score_label"] == "combined_repair"
    assert observable.diagnostics["selector_score_label"] == "observable_repair"


def test_property_probe_update_and_slot_alignment():
    scene, candidates = _candidates()
    before = property_posterior_update(0.5, "heavy", reliability=0.85)
    after = targeted_diagnostic_probe(candidates[0], scene, seed=1)
    assert before > 0.5
    assert "property_posterior_heavy" in after.diagnostics
    mapping, cost = brute_force_slot_alignment(candidates[0].slots, candidates[1].slots)
    assert isinstance(mapping, dict)
    assert cost >= 0.0


def test_deployment_gate_vocabulary():
    action = conservative_selected_tail_stop_rule(
        {"N": 32, "identity_error": 0.5, "object_real_gap": 0.4, "property_entropy": 0.2}
    )
    assert action in GATE_ACTIONS
    assert action == "block_high_n"


def test_pilot_calibrator_scores_candidates_from_named_features():
    _, candidates = _candidates()
    calibrator = fit_pilot_calibrator(candidates, ridge=1e-3)
    score = pilot_calibrated_score(candidates[0], calibrator)
    assert len(calibrator["feature_names"]) == len(pilot_calibration_features(candidates[0]))
    assert 0.0 <= score <= 1.0
    assert calibrator["n_train_candidates"] == len(candidates)
