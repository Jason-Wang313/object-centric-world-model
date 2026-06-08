import numpy as np

from object_centric_best_of_n.envs import make_scene, observe_slots, trajectory


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
