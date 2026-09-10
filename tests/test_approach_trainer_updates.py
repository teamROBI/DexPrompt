"""The update itself: separate rates, a frozen actor during warm-up, one extra term.

The overnight comparison rests on three mechanical claims -- that the actor can
be slowed without slowing the critic, that nothing touches the actor while the
value function is warming up, and that the demonstration term is a term on the
actor and on nothing else. Each is cheap to state and easy to get wrong, so each
is pinned here rather than inferred from a training curve.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_approach_rl.py"
    spec = importlib.util.spec_from_file_location("trainer_updates_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(trainer):
    return trainer.Policy(6, 7, torch.device("cpu"))


def rollout(size=32):
    return (torch.zeros((size, 6)), torch.zeros((size, 7)), torch.zeros(size),
            torch.ones(size), torch.zeros(size))


def test_the_actor_and_the_critic_can_be_given_different_rates(trainer, policy):
    rates = trainer.separate_rates(policy, torch, 3e-5, 3e-4)
    assert rates == {"policy_lr": 3e-5, "critic_lr": 3e-4}
    groups = policy.optimiser.param_groups
    assert [group["lr"] for group in groups] == [3e-5, 3e-4]
    actor = {id(p) for p in policy.actor.parameters()} | {id(policy.log_std)}
    assert {id(p) for p in groups[0]["params"]} == actor
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in policy.critic.parameters()}


def test_the_value_only_loss_leaves_the_actor_alone(trainer, policy):
    """What warm-up does: the critic learns, the policy and its exploration do not."""
    trainer.separate_rates(policy, torch, 3e-4, 3e-4)
    before = {"actor": [p.detach().clone() for p in policy.actor.parameters()],
              "log_std": policy.log_std.detach().clone(),
              "critic": [p.detach().clone() for p in policy.critic.parameters()]}
    terms = trainer.ppo_losses(policy, torch, *rollout())
    loss = trainer.VALUE_WEIGHT * terms["value_loss"]
    policy.optimiser.zero_grad()
    loss.backward()
    policy.optimiser.step()
    for parameter, original in zip(policy.actor.parameters(), before["actor"],
                                   strict=True):
        assert torch.equal(parameter, original)
    assert torch.equal(policy.log_std, before["log_std"])
    assert any(not torch.equal(p, o) for p, o in
               zip(policy.critic.parameters(), before["critic"], strict=True))


def test_the_demonstration_term_moves_the_actor_and_nothing_else(trainer, policy):
    trainer.separate_rates(policy, torch, 3e-4, 3e-4)
    batch = (torch.zeros((8, 6)), torch.ones((8, 7)))
    before = {"critic": [p.detach().clone() for p in policy.critic.parameters()],
              "log_std": policy.log_std.detach().clone(),
              "actor": [p.detach().clone() for p in policy.actor.parameters()]}
    loss = trainer.retention_loss(policy, batch, torch)
    assert float(loss) > 0
    policy.optimiser.zero_grad()
    loss.backward()
    policy.optimiser.step()
    assert any(not torch.equal(p, o) for p, o in
               zip(policy.actor.parameters(), before["actor"], strict=True))
    for parameter, original in zip(policy.critic.parameters(), before["critic"],
                                   strict=True):
        assert torch.equal(parameter, original)
    assert torch.equal(policy.log_std, before["log_std"])


def test_the_retention_term_is_measured_in_exploration_widths(trainer, policy):
    """Scaled by the 0.05 the policy was cloned with, so beta is comparable across
    runs rather than being in radians."""
    policy.actor[-1].weight.data.zero_()
    policy.actor[-1].bias.data.zero_()
    batch = (torch.zeros((4, 6)), torch.full((4, 7), 0.05))
    assert float(trainer.retention_loss(policy, batch, torch)) == pytest.approx(1.0)
    batch = (torch.zeros((4, 6)), torch.full((4, 7), 0.10))
    assert float(trainer.retention_loss(policy, batch, torch)) == pytest.approx(4.0)


def test_no_demonstrations_means_no_term(trainer, policy):
    """beta 0 has to leave the PPO loss exactly as it was."""
    assert float(trainer.retention_loss(policy, None, torch)) == 0.0
    generator = np.random.default_rng(0)
    assert trainer.demonstration_batch(None, generator, 8, torch, torch.device("cpu")) is None


def test_the_demonstration_batch_is_drawn_reproducibly(trainer):
    demos = {"observations": np.arange(60, dtype=np.float32).reshape(10, 6),
             "actions": np.arange(70, dtype=np.float32).reshape(10, 7)}
    device = torch.device("cpu")
    first = trainer.demonstration_batch(demos, np.random.default_rng(3), 4, torch, device)
    again = trainer.demonstration_batch(demos, np.random.default_rng(3), 4, torch, device)
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert first[0].shape == (4, 6) and first[1].shape == (4, 7)


def test_a_checkpoint_appears_only_once_it_is_complete(trainer, policy, tmp_path):
    """Evaluated while it is being written, a half-saved checkpoint is a mystery
    rather than a result."""
    trainer.separate_rates(policy, torch, 3e-4, 3e-4)
    path = tmp_path / "policy_step_00040960.pt"
    trainer.save_checkpoint(policy, torch, path, steps=40960, updates=40,
                            actor_updates=0, observation_size=6, manifest={"kind": "test"})
    assert path.is_file() and not path.with_suffix(".partial").exists()
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["steps"] == 40960 and state["actor_updates"] == 0
    assert state["observation_size"] == 6 and state["action_size"] == 7


def test_the_kl_is_measured_on_the_finished_policy(trainer, policy):
    """Measured after the update, on the states the update was computed from:
    the running average collected during it is read before each optimizer step
    and describes the drift so far, not where the update ended up."""
    observations = torch.zeros((16, 6))
    before = trainer.gaussian_parameters(policy, torch, observations)
    assert trainer.gaussian_kl(
        before, trainer.gaussian_parameters(policy, torch, observations), torch
    ) == pytest.approx(0.0, abs=1e-9)

    with torch.no_grad():
        policy.actor[-1].bias.add_(0.1)
    moved = trainer.gaussian_kl(
        before, trainer.gaussian_parameters(policy, torch, observations), torch)
    assert moved > 0


def test_an_update_can_be_undone_including_the_optimizer(trainer, policy):
    """Restoring the weights alone leaves Adam carrying the momentum of the step
    that was rejected, and the next update applies it anyway."""
    trainer.separate_rates(policy, torch, 3e-4, 3e-4)
    keep = trainer.policy_state(policy, torch)

    terms = trainer.ppo_losses(policy, torch, *rollout())
    policy.optimiser.zero_grad()
    (terms["loss"] + policy.log_std.sum()).backward()
    policy.optimiser.step()
    assert policy.optimiser.state_dict()["state"], "the optimizer has moments now"
    assert not torch.equal(policy.log_std, keep["log_std"])

    trainer.restore_policy(policy, torch, keep)
    assert torch.equal(policy.log_std, keep["log_std"])
    for parameter, original in zip(policy.actor.parameters(), 
                                   [torch.as_tensor(v) for v in keep["actor"].values()],
                                   strict=False):
        assert parameter.shape == original.shape
    restored = policy.optimiser.state_dict()["state"]
    assert not restored, "the moments from the rejected step are gone too"


def test_restoring_leaves_the_next_update_where_it_would_have_been(trainer, policy):
    """Two updates from the same state must agree: undo has to be complete."""
    trainer.separate_rates(policy, torch, 3e-4, 3e-4)
    keep = trainer.policy_state(policy, torch)

    def one_update():
        terms = trainer.ppo_losses(policy, torch, *rollout())
        policy.optimiser.zero_grad()
        terms["loss"].backward()
        policy.optimiser.step()
        return [p.detach().clone() for p in policy.critic.parameters()]

    first = one_update()
    trainer.restore_policy(policy, torch, keep)
    second = one_update()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_the_kl_is_the_divergence_not_a_signed_estimate(trainer):
    """mean(old_logp - new_logp) is signed: on a policy whose spread shrank it
    comes out negative while the divergence is positive, and a positive limit
    then passes an update that moved a long way."""
    zero = torch.zeros((4, 7))
    wide = {"mean": zero, "log_std": torch.zeros(7)}
    narrow = {"mean": zero, "log_std": torch.full((7,), -1.0)}

    # One dimension, means one standard deviation apart: KL is 0.5.
    apart = {"mean": torch.ones((1, 1)), "log_std": torch.zeros(1)}
    same = {"mean": torch.zeros((1, 1)), "log_std": torch.zeros(1)}
    assert trainer.gaussian_kl(same, apart, torch) == pytest.approx(0.5)
    assert trainer.gaussian_kl(apart, same, torch) == pytest.approx(0.5)

    assert trainer.gaussian_kl(wide, wide, torch) == pytest.approx(0.0, abs=1e-9)
    # Any real move is positive, whichever direction the spread went.
    assert trainer.gaussian_kl(wide, narrow, torch) > 0
    assert trainer.gaussian_kl(narrow, wide, torch) > 0


def test_the_kl_sums_the_action_dimensions_and_averages_the_states(trainer):
    states, dimensions = 5, 7
    before = {"mean": torch.zeros((states, dimensions)), "log_std": torch.zeros(dimensions)}
    after = {"mean": torch.ones((states, dimensions)), "log_std": torch.zeros(dimensions)}
    assert trainer.gaussian_kl(before, after, torch) == pytest.approx(0.5 * dimensions)


def test_the_distribution_is_captured_from_the_policy(trainer, policy):
    observations = torch.zeros((3, 6))
    captured = trainer.gaussian_parameters(policy, torch, observations)
    assert captured["mean"].shape == (3, 7)
    assert torch.equal(captured["log_std"], policy.log_std.detach())
    with torch.no_grad():
        policy.log_std.add_(1.0)
    # A copy, not a view: the record of "before" must not follow the policy.
    assert not torch.equal(captured["log_std"], policy.log_std.detach())
    assert trainer.gaussian_kl(
        captured, trainer.gaussian_parameters(policy, torch, observations), torch) > 0
