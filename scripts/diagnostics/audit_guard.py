"""``collect_teacher_dataset.py`` with the executor's command guard recorded -- nothing else.

The executor previews each proposed command (``TeacherExecutor.guarded``): the full
step, then half, then a quarter, then holding or opening, each simulated for
``preview_ticks`` and dropped at the first quality violation. Only the fraction it
settled on survives in the archive (``guard_scaled_ticks``). This wraps ``tick`` and
``guarded`` for this process only to write, for every guarded control tick, which
choices were tried, why each rejected one was rejected and which robot contacts
were penetrating at that moment, which choice was applied, and -- on the committed
tick -- the target contact groups and each digit's surface distance to the target.

Nothing that decides the motion is touched: the original methods run unchanged,
exceptions are re-raised, the simulator state is only read. Whether the run is
still the original is checked afterwards against the archived episode, field by
field. ``ICRA_GUARD_LOG`` names the JSONL output; ``ICRA_CLUTTER_PLACEMENT``, if set,
places distractors as ``collect_clutter.py`` does. Arguments are the collector's.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# The order TeacherExecutor.guarded builds its choices in (execution.py).
ACQUIRE = ("full", "half", "quarter", "hold_previous", "open_fingers")
CARRY = ("full", "half", "quarter",
         "arm_full_fingers_previous", "arm_full_fingers_open",
         "arm_half_fingers_previous", "arm_half_fingers_open",
         "arm_quarter_fingers_previous", "arm_quarter_fingers_open",
         "fingers_only", "open_fingers", "hold_previous")


def penetrating_contacts(executor, limit=8):
    """Robot contacts with negative distance, deepest first; read-only."""
    r = executor.r
    m, d = r.model, r.data
    meter = executor.meter
    rows = []
    for c in d.contact[:d.ncon]:
        if c.dist >= 0:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in meter.robot_bodies) == (b2 in meter.robot_bodies):
            continue
        robot, other, other_geom = (b1, b2, c.geom2) if b1 in meter.robot_bodies \
            else (b2, b1, c.geom1)
        kind = ("target" if other in meter.targets else
                "obstacle" if other in meter.obstacles else
                "table" if other_geom in meter.table_geoms else "other")
        rows.append({"robot_body": m.body(robot).name, "other_body": m.body(other).name,
                     "kind": kind, "depth_mm": -float(c.dist) * 1000})
    rows.sort(key=lambda x: -x["depth_mm"])
    return rows[:limit]


def digit_distances(executor):
    """Each digit's smallest geom-to-geom surface distance to the target (m)."""
    import mujoco
    import numpy as np

    r = executor.r
    m, d = r.model, r.data
    cache = getattr(executor, "_audit_geoms", None)
    if cache is None:
        def collidable(g):
            return m.geom_contype[g] or m.geom_conaffinity[g]
        target = [g for g in range(m.ngeom)
                  if int(m.geom_bodyid[g]) in executor.meter.targets and collidable(g)]
        digits = []
        for digit in range(5):
            names = {f"finger_r_link{4 * digit + k}" for k in range(1, 5)}
            digits.append([g for g in range(m.ngeom)
                           if m.body(int(m.geom_bodyid[g])).name in names and collidable(g)])
        cache = executor._audit_geoms = (target, digits)
    target, digits = cache
    # mj_geomDistance writes the segment into a writeable float64 array.
    fromto = np.zeros(6)
    out = []
    for geoms in digits:
        best = 0.2
        for a in geoms:
            for b in target:
                best = min(best, float(mujoco.mj_geomDistance(m, d, a, b, 0.2, fromto)))
        out.append(best)
    return out


def install(log_path: str) -> None:
    import numpy as np

    from icra import execution as ex

    stream = open(log_path, "a")  # noqa: SIM115 -- kept open for the process
    original_tick = ex.TeacherExecutor.tick
    original_guarded = ex.TeacherExecutor.guarded

    def tick(self, phase, action, *, record):
        audit = getattr(self, "_audit", None)
        if audit is not None and not record and phase == "feedback":
            if audit["last_action"] is not action:
                audit["choices"].append({"index": len(audit["choices"]), "preview_ticks": 0,
                                         "rejected": None})
                audit["last_action"] = action
            choice = audit["choices"][-1]
            choice["preview_ticks"] += 1
            try:
                return original_tick(self, phase, action, record=record)
            except RuntimeError as error:
                choice["rejected"] = str(error)
                choice["contacts"] = penetrating_contacts(self)
                raise
        result = original_tick(self, phase, action, record=record)
        pending = getattr(self, "_audit_pending", None)
        if record and pending is not None:
            self._audit_pending = None
            self._audit_step = getattr(self, "_audit_step", -1) + 1
            pending.update(
                event="committed", step=self._audit_step, phase=phase,
                contact_groups=np.asarray(result["contact_groups"]).tolist(),
                group_forces_min=np.asarray(result["group_forces_min"]).tolist(),
                digit_force_max=np.asarray(result["digit_force_max"]).tolist(),
                robot_obstacle_penetration_mm=float(result["robot_obstacle_penetration_max"])
                * 1000,
                robot_object_penetration_mm=float(result["robot_object_penetration_max"]) * 1000,
                digit_target_distance_m=digit_distances(self),
                committed_contacts=penetrating_contacts(self))
            stream.write(json.dumps(pending) + "\n")
            stream.flush()
        return result

    def guarded(self, proposed, *, carrying=False):
        labels = CARRY if carrying else ACQUIRE
        before = np.asarray(self.r.targets(), float).copy()
        self._audit = {"choices": [], "last_action": None}
        record = {"carrying": bool(carrying),
                  "proposed_arm_delta": float(np.linalg.norm(proposed[:7] - before[:7])),
                  "proposed_finger_delta": float(np.linalg.norm(proposed[7:] - before[7:]))}
        try:
            action = original_guarded(self, proposed, carrying=carrying)
        except RuntimeError as error:
            record.update(event="guard_failed", failed=str(error),
                          choices=[{**c, "label": labels[c["index"]]}
                                   for c in self._audit["choices"]])
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            raise
        finally:
            choices = self._audit["choices"]
            self._audit = None
        accepted = len(choices) - 1  # the last choice tried is the one kept
        record.update(choices=[{**c, "label": labels[c["index"]]} for c in choices],
                      accepted_index=accepted, accepted_label=labels[accepted],
                      applied_arm_delta=float(np.linalg.norm(action[:7] - before[:7])),
                      applied_finger_delta=float(np.linalg.norm(action[7:] - before[7:])),
                      guard_fraction=float(self.guard_fractions[-1]),
                      guard_finger_fraction=float(self.guard_finger_fractions[-1]))
        self._audit_pending = record
        return action

    ex.TeacherExecutor.tick = tick
    ex.TeacherExecutor.guarded = guarded


def main() -> None:
    install(os.environ["ICRA_GUARD_LOG"])
    placement = os.environ.get("ICRA_CLUTTER_PLACEMENT")
    if placement:
        from collect_clutter import with_offsets

        from icra import scene as scene_lib

        offsets = json.loads(Path(placement).read_text())["offsets"]
        scene_lib.build = with_offsets(scene_lib.build, offsets)
    import collect_teacher_dataset

    sys.argv = ["collect_teacher_dataset.py", *sys.argv[1:]]
    collect_teacher_dataset.main()


if __name__ == "__main__":
    main()
