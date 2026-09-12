"""One name for what happened to an execution attempt.

Two reports counted the same attempts with rules that differed in their order --
one asked whether the closure ever found a quality-passing command before asking
what the hand achieved, the other after -- and the same run came out under two
names. The distinctions worth keeping are physical and independent of that
order, so they are made once, here:

* the arm never found a validated path and nothing ran;
* the hand never closed every required contact group at the same moment;
* it closed them and had lost some by the end;
* it still held them all in the hold phase when the tick budget ran out;
* every group was present at termination, but the hold-timeout description does not apply;
* it met the criterion.

The last two are different facts and reporting them as one would be wrong in
either direction: a run that is still holding has not met the criterion and is
not a demonstration, and it has also not dropped the object.
"""

from __future__ import annotations

import numpy as np

NO_PATH = "no_validated_home_path"
NEVER_TAKEN = "contacts_never_all_taken"
TAKEN_THEN_LOST = "contacts_taken_then_lost"
STILL_HOLDING = "still_holding_when_the_budget_ran_out"
CONTACTS_AT_END = "all_contacts_present_at_termination"
COMPLETED = "completed"


def trailing_hold(held) -> int:
    """Consecutive ticks with every group held, counted back from the end."""
    count = 0
    for value in np.asarray(held)[::-1]:
        if not value:
            break
        count += 1
    return int(count)


def classify(accepted: bool, status: str, groups, phase: str | None) -> str:
    """What became of one attempt. ``groups`` is ticks x required, or None."""
    if accepted:
        return COMPLETED
    if groups is None:
        return status or NO_PATH
    groups = np.asarray(groups, dtype=bool)
    if groups.size == 0:
        return status or NO_PATH
    required = groups.shape[1]
    if int(groups.sum(axis=1).max()) < required:
        return NEVER_TAKEN
    if int(groups[-1].sum()) == required:
        if phase == "hold" and status == "lift_timeout":
            return STILL_HOLDING
        return CONTACTS_AT_END
    return TAKEN_THEN_LOST


def quality_command_failures(rows) -> int:
    """How many attempts ended because no command passed the quality gate.

    Reported alongside the classes above rather than inside them: it says what
    the closure controller did, not what the hand achieved, and an attempt can
    be described by both.
    """
    return sum(1 for row in rows
               if row.get("status") == "no_quality_feasible_command")
