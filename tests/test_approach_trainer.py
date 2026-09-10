"""What --check has to notice before anyone starts a training run.

A check that records numbers and exits zero regardless is a file, not a check.
These pin the judgement: the shapes the trainer will feed the policy, the
finiteness of everything the update consumes, and the two controls that say the
environment still behaves -- the recorded approach completing, and a still arm
failing inside its budget.
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_approach_rl.py"
    spec = importlib.util.spec_from_file_location("approach_trainer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rollout(**changes) -> dict:
    return {
        "rollout": 0, "observation_size": 41, "action_size": 7,
        "reward_finite": True, "value_finite": True, "advantage_finite": True,
        "return_finite": True, "losses_finite": True, **changes,
    }


def controls(**changes) -> dict:
    return {
        "scripted_approach": {"success": True, "executor_status": "completed",
                              "handover_tick": 451, "status": "completed"},
        "held_still": {"status": "approach_timeout", "terminated": True,
                       "truncated": False, "success": False},
        **changes,
    }


def test_a_healthy_run_has_nothing_to_report(trainer):
    assert trainer.check_problems([rollout()], controls(), 41) == []


def test_the_shapes_the_policy_is_given_are_checked(trainer):
    wide = trainer.check_problems([rollout(observation_size=40)], controls(), 41)
    assert any("observations are 40 wide" in problem for problem in wide)
    hand = trainer.check_problems([rollout(action_size=27)], controls(), 41)
    assert any("seven joints" in problem for problem in hand)


@pytest.mark.parametrize("field", ["reward_finite", "value_finite", "advantage_finite",
                                   "return_finite", "losses_finite"])
def test_anything_the_update_consumes_must_be_finite(trainer, field):
    problems = trainer.check_problems([rollout(**{field: False})], controls(), 41)
    assert any(field.replace("_", " ") in problem for problem in problems)


def test_the_recorded_success_must_still_succeed(trainer):
    """If it does not, the environment differs from the run the candidate came
    from, and nothing trained in it can be compared against that run."""
    for broken in ({"success": False, "executor_status": "lift_timeout",
                    "handover_tick": 451, "status": "lift_timeout"},
                   {"success": False, "executor_status": None,
                    "handover_tick": None, "status": "approach_timeout"}):
        problems = trainer.check_problems(
            [rollout()], controls(scripted_approach=broken), 41
        )
        assert problems


def test_a_still_arm_must_fail_inside_its_budget(trainer):
    """Terminated, not truncated: arriving within the budget is the task."""
    bootstrapped = {"status": "approach_timeout", "terminated": False,
                    "truncated": True, "success": False}
    problems = trainer.check_problems([rollout()], controls(held_still=bootstrapped), 41)
    assert any("terminated approach_timeout" in problem for problem in problems)

    scored = {"status": "approach_timeout", "terminated": True, "truncated": False,
              "success": True}
    problems = trainer.check_problems([rollout()], controls(held_still=scored), 41)
    assert any("scored as a success" in problem for problem in problems)


def test_every_rollout_is_judged_not_only_the_last(trainer):
    problems = trainer.check_problems(
        [rollout(rollout=0), rollout(rollout=1, advantage_finite=False)], controls(), 41
    )
    assert len(problems) == 1 and "rollout 1" in problems[0]


def test_the_loss_is_the_one_training_uses(trainer):
    """The check computes the update's own terms, so a shape or dtype error in
    the loss shows up before a training run rather than during one."""
    torch = pytest.importorskip("torch")

    policy = trainer.Policy(6, 7, torch.device("cpu"))
    observations = torch.zeros((5, 6))
    actions = torch.zeros((5, 7))
    old = torch.zeros(5)
    terms = trainer.ppo_losses(policy, torch, observations, actions, old,
                               torch.ones(5), torch.zeros(5))
    assert set(terms) == {"loss", "policy_loss", "value_loss", "entropy", "ratio",
                          "log_probability", "value"}
    assert torch.isfinite(terms["loss"])
    assert terms["ratio"].shape == (5,) and terms["value"].shape == (5,)
    # Built with a graph, so training can take a gradient from the same call.
    assert terms["loss"].requires_grad
