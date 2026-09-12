"""Milk under multiccd B: the actual archived start states, examined -- three runs only.

1. Replay the accepted Power Disk episode's recorded commands from its archived
   compiled model and archived initial state (``restore_recorder``: no rebuild, no
   re-brace, B flag as archived) and compare every recorded physics signal. If it
   does not reproduce, the observations below are recorded but not interpreted.
2. From two archived initial states -- the accepted Power Disk episode's, and one
   failed Large Diameter candidate's (fixed rule: the first failed LD in execution
   order whose archived state differs from the success's) -- hold the recorded
   actuator command in that state and watch the object for 4 s.

Time base, stated because it differs from the original check: an archived initial
state is taken *after* the executor's own 600-step (1.2 s) settle, with the
candidate's calibrated open hand. So t = 0 here is already 1.2 s after reset, and
"t = 1.2 s" here is 2.4 s after reset. This is extra waiting at the start of the
real run, not the original gate, which started from the pre-settle state. The
existing ``initial_gate_pass`` / ``sustained_stability_pass`` are not touched.

    third_party/dexonomy/.venv/bin/python scripts/teacher/milk_state_diagnosis.py \\
        --out /data1/yspark/sim/icra/hope-milk-state-diagnosis-v1
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_hx5_student import restore_recorder  # noqa: E402

from icra.dataset import load_episode, load_manifest, validate_dataset  # noqa: E402
from icra.execution import ContactMeter, aggregate, subtree  # noqa: E402
from icra.simulation import forward, restore  # noqa: E402

SESSION = Path("/data1/yspark/sim/icra/hope-milk-multiccd-v1")
SUCCESS = SESSION / "runs/08-new-Milk-centre-23a2d8fd"
FAILED_LD = SESSION / "runs/01-new-Milk-centre-1879dac1"
EPISODE = "teacher_0000_yaw_000"
TICKS, FIRST, GATE, TAIL, FORWARD_EVERY = 2000, 250, 600, 250, 50


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def replay(collection: Path) -> dict:
    """Run 1: the recorded commands, stepped exactly as the executor stepped them."""
    import mujoco

    validate_dataset(collection)
    manifest = load_manifest(collection, require_train=False)
    entry = next(e for e in manifest["episodes"] + manifest["rejected"]
                 if e["episode_id"] == EPISODE)
    recorder, initial, _ = restore_recorder(collection.resolve(), manifest, entry, render=False)
    archived = load_episode(collection.resolve(), entry)
    r = recorder
    meta = entry["provenance"]
    body = r.object_body(meta["object_id"])
    restore(r.model, r.data, initial)
    meter = ContactMeter(r, body, meta["required_contact_groups"], [], None)
    rows = []
    for tick, command in enumerate(np.asarray(archived["action"], dtype=float)):
        row = {"right_state": r.measured().copy(), "timestamp": float(r.data.time)}
        ratio = round(float(archived["control_dt"][tick]) / r.model.opt.timestep)
        r.command(command)
        samples = []
        for _ in range(ratio):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            samples.append(meter.read())
        rows.append({**row, **aggregate(samples)})
    measured = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    errors = {k: float(np.max(np.abs(measured[k].astype(float) - archived[k])))
              for k in measured if measured[k].dtype != np.bool_}
    mismatches = {k: int(np.count_nonzero(measured[k] != archived[k]))
                  for k in measured if measured[k].dtype == np.bool_}
    execution = meta["execution"]
    return {"collection": str(collection), "episode_id": EPISODE,
            "episode_sha256": entry["sha256"],
            "model": execution["model"], "model_sha256": sha(collection / execution["model"]),
            "initial_state_sha256": sha(collection / execution["initial_state"]),
            "archived_model_multiccd": bool(r.model.opt.enableflags
                                            & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
            "ticks": len(rows), "max_absolute_errors": errors,
            "boolean_mismatches": mismatches,
            "reproduces": bool(all(v == 0.0 for v in errors.values())
                               and all(v == 0 for v in mismatches.values()))}


def window(speeds: np.ndarray) -> dict:
    return {"linear_max_m_s": float(speeds[:, 0].max()),
            "linear_rms_m_s": float(np.sqrt((speeds[:, 0] ** 2).mean())),
            "angular_max_rad_s": float(speeds[:, 1].max()),
            "angular_rms_rad_s": float(np.sqrt((speeds[:, 1] ** 2).mean()))}


def observe(collection: Path, label: str) -> dict:
    """Runs 2 and 3: hold the archived start state's own command for 4 s."""
    import mujoco

    attempts = json.loads((collection / "attempts.json").read_text())
    attempt = attempts[0]
    provenance = attempt.get("provenance")
    if provenance is None:  # a candidate whose path failed keeps its record here
        provenance = json.loads((collection / "diagnostics" / f"{EPISODE}.json").read_text())
    execution = provenance["execution"]
    model_path = collection / execution["model"]
    state_path = collection / execution["initial_state"]
    model = mujoco.MjModel.from_binary_path(str(model_path))  # archived: no rebuild, no brace
    data = mujoco.MjData(model)
    initial = np.load(state_path, allow_pickle=False)
    restore(model, data, initial)
    forward(model, data)
    held_command = data.ctrl.copy()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{provenance['object_id']}")
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    table = {g for g in range(model.ngeom) if model.geom(g).name == "table"}
    robot = subtree(model, model.body("base_link").id)

    def frame():
        f = np.eye(4)
        f[:3, :3] = data.xmat[body].reshape(3, 3)
        f[:3, 3] = data.xpos[body]
        return f

    start = frame()
    speeds = np.zeros((TICKS, 2))
    table_contacts = np.zeros(TICKS, int)
    robot_contacts = np.zeros(TICKS, int)
    robot_depth = np.zeros(TICKS)
    frames = {}
    for tick in range(TICKS):
        mujoco.mj_step(model, data)
        if (tick + 1) % FORWARD_EVERY == 0:
            forward(model, data)
        speeds[tick] = [np.linalg.norm(data.qvel[dof:dof + 3]),
                        np.linalg.norm(data.qvel[dof + 3:dof + 6])]
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 == body and c.geom2 in table) or (b2 == body and c.geom1 in table):
                table_contacts[tick] += 1
            if (b1 == body and b2 in robot) or (b2 == body and b1 in robot):
                robot_contacts[tick] += 1
                robot_depth[tick] = max(robot_depth[tick], -float(c.dist) * 1000)
        if tick + 1 in (FIRST, GATE, TICKS):
            forward(model, data)
            frames[tick + 1] = frame()
    if not np.array_equal(data.ctrl, held_command):
        raise RuntimeError("the actuator command changed during the observation")

    def moved(at):
        return {"translation_mm": float(np.linalg.norm(frames[at][:3, 3] - start[:3, 3]) * 1000),
                "rotation_degrees": angle_degrees(start[:3, :3], frames[at][:3, :3])}

    def contacts(sl):
        return {"object_table_mean": float(table_contacts[sl].mean()),
                "object_table_min": int(table_contacts[sl].min()),
                "object_table_max": int(table_contacts[sl].max()),
                "robot_object_ticks": int((robot_contacts[sl] > 0).sum()),
                "robot_object_max_depth_mm": float(robot_depth[sl].max())}

    np.savez_compressed(Path(ARGS.out) / f"{label}-observation.npz", speeds=speeds,
                        table_contacts=table_contacts, robot_contacts=robot_contacts,
                        robot_depth_mm=robot_depth)
    return {
        "label": label, "collection": str(collection),
        "candidate_id": provenance.get("candidate_id"), "outcome": attempt.get("status"),
        "model_sha256": sha(model_path), "initial_state_sha256": sha(state_path),
        "archived_model_multiccd": bool(model.opt.enableflags
                                        & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
        "held_command_sha256": hashlib.sha256(held_command.tobytes()).hexdigest(),
        "time_base": "t=0 is the archived initial state, already 1.2 s after reset "
                     "(executor settle with the candidate's open hand); extra waiting",
        "first_0_5s": {**window(speeds[:FIRST]), "moved_at_0_5s": moved(FIRST),
                       **contacts(slice(0, FIRST))},
        "at_1_2s": {"linear_m_s": float(speeds[GATE - 1, 0]),
                    "angular_rad_s": float(speeds[GATE - 1, 1]),
                    "moved": moved(GATE), "object_table_contacts": int(table_contacts[GATE - 1]),
                    "robot_object_contacts": int(robot_contacts[GATE - 1])},
        "last_0_5s": {**window(speeds[-TAIL:]), "moved_at_4s": moved(TICKS),
                      **contacts(slice(TICKS - TAIL, TICKS))},
        "whole_4s": {**window(speeds), **contacts(slice(0, TICKS))},
        "same_thresholds_descriptive_only": {
            "at_1_2s_lin_le_0_005_and_ang_le_0_05": bool(
                speeds[GATE - 1, 0] <= .005 and speeds[GATE - 1, 1] <= .05),
            "last_0_5s_max_lin_le_0_005_and_ang_le_0_05": bool(
                speeds[-TAIL:, 0].max() <= .005 and speeds[-TAIL:, 1].max() <= .05)},
    }


def main() -> int:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().isoformat(timespec="seconds")
    result = {"started": started, "physics_runs_budget": 3, "physics_runs": []}
    rep = replay(SUCCESS)
    result["replay"] = rep
    result["physics_runs"].append("replay: accepted Power Disk episode")
    observations = []
    for collection, label in ((SUCCESS, "powerdisk-success"), (FAILED_LD, "largediameter-failed")):
        observations.append(observe(collection, label))
        result["physics_runs"].append(f"observation: {label}")
    result["observations"] = observations
    result["interpretable"] = rep["reproduces"]
    result["note"] = ("existing HANDOFF values initial_gate_pass=true / "
                      "sustained_stability_pass=false are unchanged; this is a separate diagnostic "
                      "with a different time base")
    result["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "state-diagnosis.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"replay_reproduces": rep["reproduces"],
                      "observations": [{o["label"]: {k: o[k] for k in ("first_0_5s", "at_1_2s",
                                                                       "last_0_5s")}}
                                       for o in observations]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    ARGS = parser.parse_args()
    raise SystemExit(main())
