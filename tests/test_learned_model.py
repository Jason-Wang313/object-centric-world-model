import json

from object_binding_tail_audit.envs import make_scene
from object_binding_tail_audit.learned_model import learned_candidate_scores, train_and_evaluate
from object_binding_tail_audit.object_model import ObjectCentricFutureGenerator


def test_learned_model_improves_over_simple_baselines(tmp_path):
    metrics, model = train_and_evaluate(tmp_path, seed=4, n_train_scenes=120, n_test_scenes=60)
    assert metrics.property_accuracy > metrics.random_property_accuracy
    assert metrics.identity_alignment_accuracy > metrics.random_identity_alignment_accuracy
    assert metrics.transition_mse < metrics.constant_transition_mse
    assert metrics.reward_correlation > 0.25
    slots = model.encode_slots(seed=5, n_scenes=2)
    assert slots.shape[0] > 0
    summary = json.loads((tmp_path / "learned_object_model_summary.json").read_text(encoding="utf-8"))
    assert summary["passes_minimum_learned_artifact_checks"] is True
    assert summary["domain_shift_rows"] >= 4
    assert summary["domain_shift_min_property_margin"] >= 0.12
    assert summary["domain_shift_min_identity_margin"] >= 0.15
    assert (tmp_path / "tables" / "learned_learning_curve.csv").exists()
    assert (tmp_path / "tables" / "learned_ablation.csv").exists()
    assert (tmp_path / "tables" / "learned_domain_shift.csv").exists()
    scene = make_scene(seed=42, occlusion=True, hidden_property=True, crossing=True)
    candidates = ObjectCentricFutureGenerator(seed=9).generate_candidates(scene, n=5, scenario="raw", seed=44)
    scores = learned_candidate_scores(model, candidates, scene)
    assert scores["learned_reward"].shape == (5,)
    assert scores["learned_identity_reward"].shape == (5,)
    assert scores["learned_identity_reward"].min() >= 0.0
    assert scores["learned_identity_reward"].max() <= 1.0
