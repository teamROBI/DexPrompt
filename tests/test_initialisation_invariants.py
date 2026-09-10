"""Guard the saved initialisation and the within-experiment comparison."""

import importlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
perturb = importlib.import_module('perturb_initialisation')
report = importlib.import_module('report_initialisation_sensitivity')


class TranslatedLink:
    """One translating link with a known world-to-link transform."""

    def __init__(self):
        self._model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".01"/>'
            '</body></worldbody></mujoco>')
        self._data = mujoco.MjData(self._model)

    def reset_qpos(self, qpos):
        self._data.qpos[:] = np.asarray(qpos)[:7]

    def transform_cpn_w2b(self, bodies, points):
        result = np.asarray(points, dtype=float).copy()
        result[:, :3] -= self._data.qpos[:3]
        return result

    def transform_cpn_b2w(self, bodies, points):
        result = np.asarray(points, dtype=float).copy()
        result[:, :3] += self._data.qpos[:3]
        return result


def saved(path, *, corruption=0):
    qpos = np.zeros((1, 27), dtype=np.float32)
    qpos[0, :4] = [.1, .2, .3, 1]
    reference = np.array([[.01, .02, .03, 1, 0, 0]], dtype=np.float32)
    points = reference.copy()
    points[:, :3] += qpos[0, :3]
    points[0, 0] += corruption
    data = dict(grasp_qpos=qpos, hand_cpn_w=points, hand_cbody=['tip'],
                obj_cpn_w=reference.copy(), required_cbody=[['tip']], tmpl_name='test',
                obj_pose=np.eye(4), obj_scale=1., ext_wrench=np.zeros(6),
                ext_center=np.zeros(3), scene_path='fixture')
    np.save(path, data)
    return reference


def test_check_reads_the_saved_world_targets(tmp_path):
    path = tmp_path / 'good.npy'
    reference = saved(path)
    _, drift = perturb.check_saved_targets(TranslatedLink(), path, reference)
    assert drift < perturb.TOLERANCE


def test_corrupt_saved_targets_fail_even_if_transform_roundtrip_is_exact(tmp_path):
    path = tmp_path / 'bad.npy'
    reference = saved(path, corruption=.005)
    with pytest.raises(ValueError, match='contact targets moved'):
        perturb.check_saved_targets(TranslatedLink(), path, reference)


def test_original_variant_preserves_file_bytes(tmp_path, monkeypatch):
    source, dest = tmp_path / 'source.npy', tmp_path / 'original.npy'
    saved(source)
    monkeypatch.setattr(perturb, 'environment', lambda _: TranslatedLink())
    monkeypatch.setattr(perturb, 'collision_state', lambda _: {})
    row = perturb.write_variant({'init_path': str(source)}, 'original', 'none', 0, dest)
    assert source.read_bytes() == dest.read_bytes()
    assert all(row['fields_unchanged'].values())
    with pytest.raises(FileExistsError):
        perturb.write_variant({'init_path': str(source)}, 'original', 'none', 0, dest)


def test_lost_contacts_use_current_original_not_older_run():
    rows = [dict(object='bowl', sha256='candidate', variant='original', missing_digits=[5]),
            dict(object='bowl', sha256='candidate', variant='tilt_plus', missing_digits=[3, 5],
                 digits_lost_since_previous_400_step_run=[])]
    report.compare_current_original(rows)
    assert rows[1]['digits_lost_since_current_original'] == [3]
    assert rows[1]['digits_lost_since_previous_400_step_run'] == []


def test_missing_original_cannot_be_counted_as_no_contact_loss():
    with pytest.raises(ValueError, match='missing current original'):
        report.compare_current_original([
            dict(object='bowl', sha256='candidate', variant='tilt_plus', missing_digits=[5])])


# --- digit-5 flexion ---------------------------------------------------------

# Radians declared explicitly: MJCF reads joint ranges in degrees by default, so
# without this "0 2" compiles to [0, 0.035] and every offset is (correctly)
# refused as out of range.
FINGER_XML = """
<mujoco><compiler angle="radian"/><worldbody>
  <body name="hand-base"><freejoint/><geom size=".01"/>
    <body name="hand-finger_r_link17" pos=".02 0 0">
      <joint name="hand-finger_r_joint17" axis="1 0 0" range="-.6 .6"/>
      <geom size=".005"/>
      <body name="hand-finger_r_link18" pos="0 0 .02">
        <joint name="hand-finger_r_joint18" axis="0 1 0" range="0 2"/>
        <geom size=".005"/>
        <body name="hand-finger_r_link19" pos="0 0 .03">
          <joint name="hand-finger_r_joint19" axis="0 1 0" range="0 1.5"/>
          <geom size=".005"/>
        </body>
      </body>
    </body>
  </body>
</worldbody></mujoco>"""


class Finger:
    """A hand root and one finger, with named joints and the hand prefix."""

    def __init__(self):
        self._model = mujoco.MjModel.from_xml_string(FINGER_XML)
        self._data = mujoco.MjData(self._model)
        self._cfg = type("Cfg", (), {"hand_prefix": "hand-"})()

    def reset_qpos(self, qpos):
        self._data.qpos[:] = np.asarray(qpos)[: self._model.nq]

    def transform_cpn_w2b(self, bodies, points):
        return np.asarray(points, dtype=float).copy()

    def transform_cpn_b2w(self, bodies, points):
        return np.asarray(points, dtype=float).copy()


def test_flexion_changes_exactly_the_named_joint():
    env = Finger()
    qpos = np.zeros(27)
    qpos[3] = 1.0
    index = int(env._model.joint("hand-finger_r_joint18").qposadr[0])
    moved, where, reason = perturb.flexed(env, qpos, "finger_r_joint18", 0.3)
    assert reason is None and where == index
    assert moved[index] == pytest.approx(0.3)
    others = [i for i in range(27) if i != index]
    assert np.array_equal(moved[others], qpos[others])


def test_an_offset_past_the_range_is_refused_not_clipped():
    """1.9 + 0.3 leaves [0, 2]: the answer is a reason, never 2.0."""
    env = Finger()
    qpos = np.zeros(27)
    qpos[int(env._model.joint("hand-finger_r_joint18").qposadr[0])] = 1.9
    moved, _, reason = perturb.flexed(env, qpos, "finger_r_joint18", 0.3)
    assert moved is None
    assert reason.startswith("out_of_range")


def test_moved_bodies_are_the_joint_link_and_everything_below_it():
    moved = perturb.moved_bodies(Finger(), "finger_r_joint18")
    assert moved == {"finger_r_link18", "finger_r_link19"}


def test_an_unrunnable_variant_writes_nothing(tmp_path, monkeypatch):
    env = Finger()
    source = tmp_path / "source.npy"
    saved(source)
    data = np.load(source, allow_pickle=True).item()
    data["grasp_qpos"][0, int(env._model.joint("hand-finger_r_joint18").qposadr[0])] = 1.9
    np.save(source, data)
    monkeypatch.setattr(perturb, "environment", lambda _scene: env)
    destination = tmp_path / "variant.npy"
    row = perturb.write_variant({"init_path": str(source)}, "pip_plus_030", "flex",
                                0.3, destination)
    assert row["not_run"].startswith("out_of_range")
    assert row["path"] is None
    assert not destination.exists()


def test_targets_on_links_the_joint_does_not_move_keep_their_bytes(tmp_path, monkeypatch):
    env = Finger()
    source = tmp_path / "source.npy"
    saved(source)
    data = np.load(source, allow_pickle=True).item()
    data["hand_cbody"] = ["base", "finger_r_link19"]
    data["hand_cpn_w"] = np.array([[.1234567, .2, .3, 1, 0, 0],
                                   [.4, .5, .6, 0, 1, 0]], dtype=np.float32)
    data["obj_cpn_w"] = data["hand_cpn_w"].copy()
    np.save(source, data)
    monkeypatch.setattr(perturb, "environment", lambda _scene: env)
    row = perturb.write_variant({"init_path": str(source)}, "pip_plus_015", "flex",
                                0.15, tmp_path / "variant.npy")
    written = np.load(row["path"], allow_pickle=True).item()
    assert written["hand_cpn_w"][0].tobytes() == data["hand_cpn_w"][0].tobytes()
    assert row["other_joints_unchanged"] is True
    assert row["changed_joint"] == "finger_r_joint18"
