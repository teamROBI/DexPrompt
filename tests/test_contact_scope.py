"""What the refiner's contact test counts, asked of the refiner's own function.

The rejection this diagnosis turns on is "the required contact bodies are not
touching the object". That verdict only means what it says if the set it reads
holds hand-object contacts and nothing else: a hand resting on the table, or one
finger pressing another, must not be able to satisfy a required group.

So this calls ``MuJoCo_OptEnv.get_contacts`` itself rather than restating the
rule it applies -- a copy of the classification cannot notice the original
changing. It also asserts first that the pose really does produce a contact with
the table and a contact between two fingers, because a test that only checks
those are absent from the result passes just as well on an empty scene.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "dexonomy"))

CONTROLS = ROOT / "out" / "contact-diagnosis" / "controls.json"
HAND = ROOT / "third_party" / "dexonomy" / "assets" / "hand" / "hx5_d20" / "right.xml"


def a_candidate_with_a_table():
    """A recorded candidate whose scene carries the table plane."""
    if not (CONTROLS.is_file() and HAND.is_file()):
        return None
    controls = json.loads(CONTROLS.read_text())
    for record in controls["negative"]:
        path = Path(record["init_path"])
        if path.is_file():
            data = np.load(path, allow_pickle=True).item()
            if Path(str(data["scene_path"])).is_file():
                return data
    return None


@pytest.fixture(scope="module")
def env_and_pose():
    pytest.importorskip("mujoco")
    candidate = a_candidate_with_a_table()
    if candidate is None:
        pytest.skip("needs the recorded controls and the hx5_d20 assets")
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    env = MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=str(HAND), freejoint=True),
        scene_cfg=load_scene_cfg(str(candidate["scene_path"])),
        sim_cfg=MuJoCo_OptCfg(obj_margin=0.0),
    )
    return env, np.asarray(candidate["grasp_qpos"])[0]


def raw_pairs(env):
    """Every contact MuJoCo reports, as body-name pairs."""
    model = env._model
    return [(model.body(model.geom(c.geom1).bodyid[0]).name,
             model.body(model.geom(c.geom2).bodyid[0]).name)
            for c in env._data.contact]


def press_into_the_table(env, qpos):
    """Lower the whole hand until it is well inside the table plane."""
    import mujoco

    for drop in np.arange(0.0, 0.40, 0.005):
        moved = np.asarray(qpos, dtype=float).copy()
        moved[2] -= drop
        env.reset_qpos(moved)
        mujoco.mj_forward(env._model, env._data)
        if any("world" in pair for pair in raw_pairs(env)):
            return True
    return False


def curl_the_fingers(env, qpos):
    """Close every finger joint hard, so fingers meet each other."""
    import mujoco

    model = env._model
    for scale in np.arange(0.2, 2.01, 0.1):
        moved = np.asarray(qpos, dtype=float).copy()
        for index in range(7, len(moved)):
            joint = model.jnt_range[min(index, model.njnt - 1)]
            moved[index] = float(np.clip(moved[index] * 0 + scale * joint[1],
                                         joint[0], joint[1]))
        env.reset_qpos(moved)
        mujoco.mj_forward(env._model, env._data)
        hand_hand = [p for p in raw_pairs(env)
                     if "finger_r" in p[0] and "finger_r" in p[1]]
        if hand_hand:
            return True
    return False


def test_the_table_contact_exists_and_is_excluded(env_and_pose):
    """First that the pose touches the table at all, then that it is left out."""
    env, qpos = env_and_pose
    assert press_into_the_table(env, qpos), "could not make the hand touch the table"
    world_pairs = [p for p in raw_pairs(env) if "world" in p]
    assert world_pairs, "the scene must actually contain a table contact"

    hand_object, hand_hand = env.get_contacts(0.002)
    for reported in (hand_object, hand_hand):
        assert "world" not in set(reported["bn1"]) | set(reported["bn2"])
    # The table is body zero, so it cannot be the object side either.
    assert all("world" not in name for name in hand_object["bn2"])


def test_finger_on_finger_contact_exists_and_is_not_hand_object(env_and_pose):
    """A self-touch must never be able to satisfy a required contact group."""
    env, qpos = env_and_pose
    if not curl_the_fingers(env, qpos):
        pytest.skip("this hand's closed pose produces no finger-finger contact")
    raw = [p for p in raw_pairs(env) if "finger_r" in p[0] and "finger_r" in p[1]]
    assert raw, "the scene must actually contain a finger-finger contact"

    hand_object, hand_hand = env.get_contacts(0.002)
    assert len(hand_hand["dist"]) > 0, "the self-touch must land in the hand-hand set"
    for first, second in zip(hand_object["bn1"], hand_object["bn2"], strict=True):
        assert not (str(first).startswith("finger_r")
                    and str(second).startswith("finger_r"))


def test_a_real_object_contact_is_counted(env_and_pose):
    """The complement: the function does report hand-object contact when there is one."""
    import mujoco

    env, qpos = env_and_pose
    env.reset_qpos(qpos)
    mujoco.mj_forward(env._model, env._data)
    hand_object, _ = env.get_contacts(0.005)
    assert len(hand_object["dist"]) > 0
    assert all(str(name).startswith("finger_r") or str(name).startswith("hx5")
               for name in hand_object["bn1"])
