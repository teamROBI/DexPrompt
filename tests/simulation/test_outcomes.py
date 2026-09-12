"""One rule for both reports.

The two reports disagreed about the same attempts because they asked their
questions in different orders. These pin the order and the distinctions.
"""

import numpy as np

from icra import outcomes


def groups(pattern):
    return np.array(pattern, dtype=bool)


def test_completed_wins_over_everything():
    assert outcomes.classify(True, "completed", groups([[1, 1]]), "hold") \
        == outcomes.COMPLETED


def test_never_taken_is_about_the_hand_not_the_controller():
    """A run that never closed all groups is that, whatever the status says."""
    never = groups([[1, 0], [1, 0]])
    assert outcomes.classify(False, "no_quality_feasible_command", never, None) \
        == outcomes.NEVER_TAKEN
    assert outcomes.classify(False, "closure_timeout", never, "close") \
        == outcomes.NEVER_TAKEN


def test_still_holding_is_not_a_loss():
    held = groups([[1, 1], [1, 1]])
    assert outcomes.classify(False, "lift_timeout", held, "hold") \
        == outcomes.STILL_HOLDING


def test_contacts_outside_hold_do_not_imply_either_loss_or_stable_hold():
    held = groups([[1, 1], [1, 1]])
    assert outcomes.classify(False, "closure_timeout", held, "close") \
        == outcomes.CONTACTS_AT_END


def test_quality_termination_with_contacts_is_not_a_budget_timeout():
    held = groups([[1, 1], [1, 1]])
    assert outcomes.classify(False, "quality_penetration", held, "hold") \
        == outcomes.CONTACTS_AT_END


def test_taken_then_lost():
    lost = groups([[1, 1], [1, 0]])
    assert outcomes.classify(False, "lift_timeout", lost, "hold") \
        == outcomes.TAKEN_THEN_LOST


def test_an_attempt_that_recorded_nothing_keeps_its_status():
    assert outcomes.classify(False, "no_validated_home_path", None, None) \
        == "no_validated_home_path"
    assert outcomes.classify(False, "", None, None) == outcomes.NO_PATH


def test_trailing_hold_counts_back_from_the_end():
    assert outcomes.trailing_hold([True, False, True, True]) == 2
    assert outcomes.trailing_hold([True, True, False]) == 0
    assert outcomes.trailing_hold([]) == 0


def test_quality_failures_are_counted_alongside_not_inside():
    rows = [{"status": "no_quality_feasible_command"},
            {"status": "lift_timeout"},
            {"status": "no_quality_feasible_command"}]
    assert outcomes.quality_command_failures(rows) == 2
