"""The hand's structure and what limits it, as tests.

These load the real model -- it is 47 MB and committed, so there is nothing to skip for.
The one that earns its place is the contact comparison: it pins the finding that the
fingers are stopped by collision geometry rather than by actuator authority, which is the
difference between "this hand needs tuning" and "this hand cannot close".
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from icra.hand import FINGER_NAMES, Hand, finger_groups, load_model

SCENE = "model/scene_ffw_sh5.xml"


@pytest.fixture(scope="module")
def model():
    return load_model(SCENE)


def curl(hand: Hand) -> dict[str, float]:
    """Every flexion joint to its far limit, spread axes at zero."""
    targets = {}
    for joint in hand.joint_names:
        low, high = hand.joint_range(joint)
        targets[joint] = 0.0 if low < 0 < high else high
    return targets


class TestStructure:
    def test_five_fingers_of_four_joints(self, model):
        groups = finger_groups(model, "r")
        assert list(groups) == list(FINGER_NAMES)
        assert [len(j) for j in groups.values()] == [4, 4, 4, 4, 4]

    def test_grouping_comes_from_the_chain_not_the_names(self, model):
        """Every joint in a finger must sit on a body that descends from the one before."""
        for joints in finger_groups(model, "r").values():
            bodies = [
                int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in joints
            ]
            for parent, child in zip(bodies, bodies[1:], strict=False):
                assert int(model.body_parentid[child]) == parent

    def test_every_joint_has_its_own_actuator(self):
        """No mimic linkage, unlike the SG2 gripper's four-into-one."""
        hand = Hand.create("r", SCENE)
        assert len({hand.actuator_id(j) for j in hand.joint_names}) == 20

    def test_both_hands_are_present(self, model):
        assert len(finger_groups(model, "l")) == 5

    def test_an_unknown_side_is_refused(self, model):
        with pytest.raises(ValueError, match="no joints named"):
            finger_groups(model, "x")


class TestWhatStopsAFist:
    """The finding: collision geometry, not actuator authority."""

    def test_without_contacts_every_joint_reaches_its_target(self):
        hand = Hand.create("r", SCENE)
        hand.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        targets = curl(hand)
        hand.command(targets)
        hand.settle(4000)
        residuals = [abs(hand.read(j) - targets[j]) for j in hand.joint_names]
        assert max(residuals) < 0.01, "the actuators and limits are not the constraint"

    def test_with_contacts_a_fist_falls_well_short(self):
        hand = Hand.create("r", SCENE)
        targets = curl(hand)
        hand.command(targets)
        hand.settle(4000)
        residuals = [abs(hand.read(j) - targets[j]) for j in hand.joint_names]
        assert np.mean(residuals) > 0.1, "if this passes, the palm collision was fixed"

    def test_removing_only_palm_collision_reduces_the_closure_obstruction(self):
        """Causal ablation; contact manifold counts differ across MuJoCo versions.

        The old oracle used a percentage of all robot contacts, including the
        other hand and floor. Removing palm collision should improve tracking
        while finger collisions remain active; it need not remove every obstacle.
        """
        baseline = Hand.create("r", SCENE)
        without_palm = Hand.create("r", SCENE)
        palm = without_palm.model.body("hx5_r_base").id
        palm_geoms = np.flatnonzero(without_palm.model.geom_bodyid == palm)
        assert np.any(without_palm.model.geom_contype[palm_geoms])
        without_palm.model.geom_contype[palm_geoms] = 0
        without_palm.model.geom_conaffinity[palm_geoms] = 0
        remaining = np.flatnonzero(without_palm.model.geom_bodyid != palm)
        np.testing.assert_array_equal(without_palm.model.geom_contype[remaining],
                                      baseline.model.geom_contype[remaining])
        np.testing.assert_array_equal(without_palm.model.geom_conaffinity[remaining],
                                      baseline.model.geom_conaffinity[remaining])
        np.testing.assert_array_equal(without_palm.model.actuator_gainprm,
                                      baseline.model.actuator_gainprm)
        np.testing.assert_array_equal(without_palm.model.actuator_forcerange,
                                      baseline.model.actuator_forcerange)

        residuals = []
        for hand in (baseline, without_palm):
            targets = curl(hand)
            hand.command(targets)
            hand.settle(4000)
            residuals.append(np.mean([abs(hand.read(j)-targets[j]) for j in hand.joint_names]))
        # Require a meaningful causal improvement, beyond integration noise.
        assert residuals[0]-residuals[1] > 0.05
        finger_contacts = 0
        for contact in without_palm.data.contact[:without_palm.data.ncon]:
            names = [without_palm.model.body(without_palm.model.geom_bodyid[g]).name
                     for g in (contact.geom1, contact.geom2)]
            finger_contacts += int(all(name.startswith("finger_r_") for name in names))
        assert finger_contacts > 0, "finger-finger collision must remain physically active"

    def test_gravity_is_not_the_cause(self):
        def residual(gravity: bool) -> float:
            hand = Hand.create("r", SCENE)
            if not gravity:
                hand.model.opt.gravity[:] = 0.0
            targets = curl(hand)
            hand.command(targets)
            hand.settle(4000)
            return float(
                np.mean([abs(hand.read(j) - targets[j]) for j in hand.joint_names])
            )

        assert residual(True) == pytest.approx(residual(False), abs=0.01)


class TestFingersAreIndependent:
    def test_closing_one_finger_leaves_the_others_alone(self):
        """Compared against an uncommanded baseline, not against zero.

        Against zero this looked like 3-4 joints of cross-talk per finger, which was an
        artefact: uncommanded joints do not sit at zero.
        """
        baseline = Hand.create("r", SCENE)
        baseline.settle(4000)
        rest = {j: baseline.read(j) for j in baseline.joint_names}

        hand = Hand.create("r", SCENE)
        index = hand.fingers["index"]
        targets = {j: 0.0 for j in hand.joint_names}
        for joint in index:
            low, high = hand.joint_range(joint)
            targets[joint] = high if low >= 0 else 0.0
        hand.command(targets)
        hand.settle(4000)

        for joint in hand.joint_names:
            if joint in index:
                continue
            assert abs(hand.read(joint) - rest[joint]) < 0.05, joint


class TestCommandingIsSafe:
    def test_targets_are_clipped_to_the_joint_range(self):
        hand = Hand.create("r", SCENE)
        joint = hand.fingers["index"][1]
        low, high = hand.joint_range(joint)
        hand.command({joint: high + 5.0})
        assert hand.data.ctrl[hand.actuator_id(joint)] == pytest.approx(high)

    def test_a_joint_with_no_actuator_is_an_error(self):
        hand = Hand.create("r", SCENE)
        with pytest.raises(KeyError, match="no actuator"):
            hand.actuator_id("not_a_joint")

    def test_a_missing_scene_names_the_path(self):
        with pytest.raises(FileNotFoundError, match="no MuJoCo scene at"):
            load_model("model/does_not_exist.xml")
