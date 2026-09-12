"""Regressions for poses crossing the GraspGenX / Dexonomy boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from icra.grasp_frames import aligned_pose, filtered_pose, pose_of


@pytest.fixture
def alignment():
    path = Path(__file__).resolve().parents[2] / "scripts/teacher/align_grasps.py"
    spec = importlib.util.spec_from_file_location("alignment_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(position=(0, 0, 0), axis=(0, 0, 1), angle=0.0):
    quat, rotation = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=float), angle)
    mujoco.mju_quat2Mat(rotation, quat)
    pose = np.eye(4)
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = position
    return pose


def filtered_entry(**changes):
    return dict(
        rotation=np.eye(3).tolist(), position=[1.0, 2.0, 3.0],
        roll_deg=90.0, depth_mm=-15.0, grip_offset=[0.10, 0.02, 0.03], **changes,
    )


def test_filter_corrections_have_known_rotation_and_translation():
    expected = transform([0.9796, 2.0796, 3.015], angle=np.pi / 2)
    np.testing.assert_allclose(filtered_pose(filtered_entry()), expected, atol=1e-12)
    # The explicit filter API and serialized-record API must denote the same pose.
    np.testing.assert_allclose(
        pose_of(filtered_entry(), np.pi / 2, [.1, .02, .03], -.015), expected, atol=1e-12,
    )


def test_seed_roundtrip_preserves_distinct_filtered_variants(alignment):
    entries = {"a": filtered_entry(), "b": dict(filtered_entry(), depth_mm=30, roll_deg=180)}
    world_object = transform([.7, -.2, .8], axis=(1, 0, 0), angle=.9)
    rotations, translations = alignment.poses_in_object_frame(entries, world_object)
    assert rotations.shape == (2, 3, 3)
    for i, entry in enumerate(entries.values()):
        local = np.eye(4)
        local[:3, :3], local[:3, 3] = rotations[i], translations[i]
        np.testing.assert_allclose(world_object @ local, filtered_pose(entry), atol=2e-7)
    assert not np.allclose(translations[0], translations[1])
    assert not np.allclose(rotations[0], rotations[1])


def test_default_does_not_add_unchecked_rolls(alignment):
    assert alignment.parse_args([]).in_plane == 1
    with pytest.raises(ValueError, match="full 4x4"):
        alignment.poses_in_object_frame({"a": filtered_entry()}, np.zeros(3))
    with pytest.raises(ValueError, match="nonnegative"):
        alignment.poses_in_object_frame({"a": filtered_entry()}, np.eye(4), -1)


@pytest.mark.parametrize("scale", [[1, 1, 1], [.07, .07, .07], [.08, .12, .05]])
def test_matcher_contact_seed_reconstructs_hand_for_each_sampled_template(alignment, scale):
    torch = pytest.importorskip("torch")
    hands = np.stack([
        transform([.02, -.03, .1], axis=(0, 1, 0), angle=.7),
        transform([-.02, .04, .08], axis=(1, 0, 0), angle=-.4),
    ])
    contacts = np.stack([
        transform([.03, .01, -.02], axis=(0, 0, 1), angle=1.2),
        transform([-.04, .02, .01], axis=(1, 0, 0), angle=-.8),
    ])
    tmpl = {
        "rot_c2h": torch.tensor(contacts[None, :, :3, :3]),
        "trans_c2h": torch.tensor(contacts[None, :, :3, 3]),
    }
    sample = {"obj_scale": torch.tensor([scale]), "sentinel": object()}
    seeded = alignment.contact_seed_sample(tmpl, sample, hands[:, :3, :3], hands[:, :3, 3])
    r_oc = seeded["rot_o2c_init"][0].numpy()
    t_oc = seeded["trans_o2c_init"][0].numpy() * scale
    # Reconstruct using the matcher's actual output convention: O_C @ C_H.
    for i in range(2):
        contact_in_object = np.eye(4)
        contact_in_object[:3, :3], contact_in_object[:3, 3] = r_oc[i], t_oc[i]
        np.testing.assert_allclose(contact_in_object @ contacts[i], hands[i], atol=1e-12)
    assert "rot_o2c_init" not in sample
    assert seeded["sentinel"] is sample["sentinel"]


def test_saved_hand_root_uses_both_object_poses_without_second_urdf_offset():
    world_object = transform([.7, .2, .9], axis=(1, 0, 0), angle=.8)
    dex_object = transform([-.1, .3, .05], angle=-1.2)
    local_hand = transform([.03, -.04, .12], axis=(0, 1, 0), angle=.4)
    result = aligned_pose(world_object, dex_object, dex_object @ local_hand)
    np.testing.assert_allclose(result, world_object @ local_hand, atol=1e-12)


def test_object_frame_is_per_body_and_includes_rotation(alignment):
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <body name="object_a" pos=".5 -.18 .8" euler="90 0 0"/>
          <body name="object_b" pos=".5 .18 .8" euler="0 0 90"/>
        </worldbody></mujoco>
    ''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    a = alignment.body_pose(data, model.body("object_a").id)
    b = alignment.body_pose(data, model.body("object_b").id)
    np.testing.assert_allclose(a, transform([.5, -.18, .8], (1, 0, 0), np.pi / 2))
    np.testing.assert_allclose(b, transform([.5, .18, .8], angle=np.pi / 2))
    with pytest.raises(ValueError, match="absent"):
        alignment.body_pose(data, -1)


def test_patches_and_working_directory_restore_after_failure(alignment, tmp_path):
    torch = pytest.importorskip("torch")
    sampler = lambda *args: None  # noqa: E731
    loader = SimpleNamespace(sample_init_pose=sampler)
    received = []

    class Matcher:
        def forward(self, tmpl, sample):
            received.append(sample)
            raise RuntimeError("matcher failed")

    original = Matcher.forward
    previous = Path.cwd()
    tmpl = {"rot_c2h": torch.eye(3)[None, None], "trans_c2h": torch.ones(1, 1, 3)}
    sample = {"obj_scale": torch.ones(1, 3)}
    with pytest.raises(RuntimeError, match="matcher failed"):
        with alignment.working_directory(tmp_path), alignment.matcher_seeds(
            loader, Matcher, np.eye(3)[None], np.zeros((1, 3))
        ):
            assert Path.cwd() == tmp_path
            assert loader.sample_init_pose(None, 10, 10)[0].shape == (1, 3, 3)
            Matcher().forward(tmpl, sample)
    assert Matcher.forward is original
    assert loader.sample_init_pose is sampler
    assert Path.cwd() == previous
    np.testing.assert_allclose(received[0]["trans_o2c_init"].numpy(), -np.ones((1, 1, 3)))


def test_invalid_object_scale_is_rejected(alignment):
    torch = pytest.importorskip("torch")
    tmpl = {"rot_c2h": torch.eye(3)[None, None], "trans_c2h": torch.zeros(1, 1, 3)}
    with pytest.raises(ValueError, match="positive"):
        alignment.contact_seed_sample(
            tmpl, {"obj_scale": torch.tensor([[1., 0., 1.]])},
            np.eye(3)[None], np.zeros((1, 3)),
        )


def test_swallowed_matcher_error_still_fails_alignment_stage(alignment):
    torch = pytest.importorskip("torch")
    loader = SimpleNamespace(sample_init_pose=lambda *a: None)

    class Matcher:
        def forward(self, template, sample):
            raise RuntimeError("upstream failure")

    original = Matcher.forward
    template = {"rot_c2h": torch.eye(3)[None, None], "trans_c2h": torch.zeros(1, 1, 3)}
    with pytest.raises(RuntimeError, match="failed during seeded alignment"):
        with alignment.matcher_seeds(loader, Matcher, np.eye(3)[None], np.zeros((1, 3))):
            # Simulate operate_init's exception swallowing.
            try:
                Matcher().forward(template, {"obj_scale": torch.ones(1, 3)})
            except RuntimeError:
                pass
    assert Matcher.forward is original


def test_real_dexonomy_matcher_preserves_seed_without_optimization(alignment, monkeypatch):
    """Exercise upstream pre/post-processing; forbid backward and optimizer steps."""
    torch = pytest.importorskip("torch")
    backend = pytest.importorskip("dexonomy.op.gen_init")

    def forbidden(*args, **kwargs):
        raise AssertionError("this verification must not optimize or train")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: SimpleNamespace(
        zero_grad=lambda: None, step=forbidden,
    ))
    desired = np.stack([
        transform([.03, -.02, .12], axis=(0, 1, 0), angle=.7),
        transform([-.02, .01, .08], axis=(1, 0, 0), angle=-.4),
    ])
    contact = np.stack([
        transform([.01, -.04, -.02], angle=.9),
        transform([-.03, .01, .02], axis=(1, 0, 0), angle=-.6),
    ])
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    template = {
        "rot_c2h": tensor(contact[None, :, :3, :3]),
        "trans_c2h": tensor(contact[None, :, :3, 3]),
        "hand_cpn_c": torch.zeros(1, 2, 2, 6),
        "hand_skt_h": torch.zeros(1, 2, 1, 3),
        "grasp_qpos": torch.zeros(1, 2, 27),
        "n_evo": torch.zeros(1, 2),
    }
    dex_pose7 = [.1, -.2, .04, np.cos(.3), 0, 0, np.sin(.3)]
    sample = {
        "obj_scale": tensor([[.07, .07, .07]]),
        "pose_w2so": tensor([dex_pose7]),
        "ext_center": torch.zeros(1, 3),
        "ext_wrench": torch.zeros(1, 6),
    }
    # Only correspondence lookup is stubbed: real pose conversion and the
    # matcher's one-pass forward are executed with no gradient update.
    collision = SimpleNamespace(query_point=lambda p: (p.clone(), torch.zeros_like(p)))
    matcher = backend.HandObjMatcher(SimpleNamespace(opt_step=1), "cpu", collision)
    loader = SimpleNamespace(sample_init_pose=lambda *a: None)
    with alignment.matcher_seeds(loader, backend.HandObjMatcher,
                                desired[:, :3, :3], desired[:, :3, 3]):
        result = matcher.forward(template, sample)
    dex_object = alignment.pose7(dex_pose7)
    for i, pose in enumerate(result["grasp_pose"].detach().numpy()):
        np.testing.assert_allclose(alignment.pose7(pose), dex_object @ desired[i], atol=2e-6)


def scaled_asset(root, name="core_bottle_x", scale="011"):
    """A DGN-shaped set: the object under processed_data, its configs beside it."""
    directory = root / "processed_data" / name
    directory.mkdir(parents=True)
    config = root / "scene_cfg" / name / "floating" / f"scale{scale}.npy"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"")
    return directory, config


def test_scene_config_is_found_for_both_object_layouts(alignment, tmp_path):
    flat = tmp_path / "icra" / "Mustard"
    flat.mkdir(parents=True)
    (flat / "scene_cfg.npy").write_bytes(b"")
    assert alignment.scene_config(flat.parent, "Mustard", 1.0) == flat / "scene_cfg.npy"

    directory, config = scaled_asset(tmp_path / "DGN")
    assert alignment.scene_config(directory.parent, directory.name, 0.11) == config


def test_scene_config_follows_a_single_object_directory_of_links(alignment, tmp_path):
    """A one-object scene is staged as links into the set, which is where the
    configurations live -- the object directory alone does not lead to them."""
    directory, config = scaled_asset(tmp_path / "DGN")
    staged = tmp_path / "one"
    staged.mkdir()
    (staged / directory.name).symlink_to(directory)
    assert alignment.scene_config(staged, directory.name, 0.11) == config


def test_a_missing_scene_config_is_raised_not_passed_on(alignment, tmp_path):
    """Dexonomy reports a missing configuration as an empty result, which reads
    exactly like a hard alignment problem: 0 of 18 seeds, and no reason given."""
    directory, _ = scaled_asset(tmp_path / "DGN", scale="011")
    with pytest.raises(FileNotFoundError, match="scale014"):
        alignment.scene_config(directory.parent, directory.name, 0.14)
    with pytest.raises(FileNotFoundError, match="no Dexonomy scene configuration"):
        alignment.scene_config(tmp_path / "nowhere", "absent", 0.11)


def test_an_explicit_scene_config_directory_wins_over_searching(alignment, tmp_path):
    directory, _ = scaled_asset(tmp_path / "DGN")
    given = tmp_path / "elsewhere"
    (given / directory.name / "floating").mkdir(parents=True)
    (given / directory.name / "floating" / "scale011.npy").write_bytes(b"")
    assert alignment.scene_config(directory.parent, directory.name, 0.11, given) \
        == given / directory.name / "floating" / "scale011.npy"


def test_a_tabletop_layout_is_found_in_a_given_directory(alignment, tmp_path):
    """A table-aware set is laid out like DGN's but under ``tabletop``. Without
    this the refiner silently falls back to the floating scene and optimises
    against a world the executor does not have."""
    directory, _ = scaled_asset(tmp_path / "DGN")
    given = tmp_path / "with-table"
    (given / directory.name / "tabletop").mkdir(parents=True)
    (given / directory.name / "tabletop" / "scale011.npy").write_bytes(b"")
    assert alignment.scene_config(directory.parent, directory.name, 0.11, given) \
        == given / directory.name / "tabletop" / "scale011.npy"


def test_floating_still_wins_when_a_given_directory_holds_both(alignment, tmp_path):
    """Pointing this at DGN's own directory must keep getting DGN's own file."""
    directory, _ = scaled_asset(tmp_path / "DGN")
    given = tmp_path / "both"
    for layout in ("floating", "tabletop"):
        (given / directory.name / layout).mkdir(parents=True)
        (given / directory.name / layout / "scale011.npy").write_bytes(b"")
    assert alignment.scene_config(directory.parent, directory.name, 0.11, given) \
        == given / directory.name / "floating" / "scale011.npy"


def test_the_scene_config_path_is_absolute(alignment, tmp_path, monkeypatch):
    """The matcher runs from Dexonomy's directory and globs this path, so a
    relative one finds nothing and is reported as zero objects, not an error."""
    directory, config = scaled_asset(tmp_path / "DGN")
    monkeypatch.chdir(tmp_path)
    relative = directory.parent.relative_to(tmp_path)
    found = alignment.scene_config(relative, directory.name, 0.11)
    assert found.is_absolute() and found == config.resolve()

    given = tmp_path / "with-table"
    (given / directory.name / "tabletop").mkdir(parents=True)
    (given / directory.name / "tabletop" / "scale011.npy").write_bytes(b"")
    found = alignment.scene_config(relative, directory.name, 0.11,
                                   given.relative_to(tmp_path))
    assert found.is_absolute()


def test_initialisations_are_found_under_a_nested_scene_id(alignment, tmp_path):
    """DGN scene ids carry pose and scale, so op=init files one level deeper."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "0_0_grasp.npy").write_bytes(b"")
    assert alignment.initialisations(flat) == [flat / "0_0_grasp.npy"]

    nested = tmp_path / "core_bottle_x"
    deep = nested / "floating" / "scale011"
    deep.mkdir(parents=True)
    for index in range(3):
        (deep / f"0_{index}_grasp.npy").write_bytes(b"")
    assert len(alignment.initialisations(nested)) == 3
    assert alignment.initialisations(tmp_path / "empty") == []
