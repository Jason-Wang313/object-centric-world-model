import numpy as np

from object_centric_best_of_n.envs import make_scene, observe_slots, retarget_scene, trajectory
from object_centric_best_of_n.object_model import ObjectCentricFutureGenerator


def test_scene_exposes_identity_occlusion_hidden_property_and_crossing():
    scene = make_scene(seed=7, occlusion=True, hidden_property=True, crossing=True)
    assert scene.target_id == 0
    assert scene.target().hidden_mass >= 0.70
    assert scene.occlusion_band is not None
    assert scene.crossing
    traj = trajectory(scene, steps=5)
    assert set(traj) == {obj.obj_id for obj in scene.objects}
    assert traj[0][0][0] < traj[1][0][0]


def test_slot_dropout_merge_split_bookkeeping():
    scene = make_scene(seed=9, occlusion=True)
    slots, bookkeeping = observe_slots(
        scene,
        rng=np.random.default_rng(0),
        slot_dropout=1.0,
        merge_split=1.0,
        identity_swap=1.0,
    )
    assert bookkeeping["merged"] == 1
    assert bookkeeping["swapped"] == 1
    assert any(slot.merged_ids for slot in slots)
    assert any(slot.dropped for slot in slots)


def test_retarget_scene_updates_true_target_flags():
    scene = make_scene(seed=12, occlusion=True, hidden_property=True, crossing=True)
    retargeted = retarget_scene(scene, target_id=1)
    assert retargeted.target_id == 1
    assert retargeted.target().obj_id == 1
    assert {obj.obj_id for obj in retargeted.objects if obj.is_target} == {1}
    assert "counterfactual target id 1" in retargeted.notes


def test_generator_identity_error_uses_scene_target_not_hardcoded_object_one():
    scene = retarget_scene(make_scene(seed=13, occlusion=True, hidden_property=True, crossing=True), target_id=1)
    generator = ObjectCentricFutureGenerator(seed=3)
    candidates = generator.generate_candidates(scene, n=128, scenario="raw", seed=91)
    assert all(candidate.target_id == 1 for candidate in candidates)
    identity_errors = [candidate for candidate in candidates if candidate.identity_error]
    assert identity_errors
    assert all(candidate.predicted_target_id != 1 for candidate in identity_errors if candidate.predicted_target_id is not None)
