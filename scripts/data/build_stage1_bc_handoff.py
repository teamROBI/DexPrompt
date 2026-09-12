# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Stage-1 BC data contract: every accepted teacher episode, linked, checked, split -- read only.

Finds every accepted Dexonomy teacher episode under ``out/`` and ``/data1/yspark/sim/icra``,
collapses reproductions (same episode sha256), links each to its mesh, real scale, object
pose, joint states, executed actions, Dexonomy type and candidate, and checks the archived
arrays against the collection code:

* ``right_state[0]`` equals the 27 driven joints of the archived initial state (float32);
* ``timestamp[0]`` equals the initial state's time and steps by ``control_dt``;
* ``action[0, 7:]`` equals the candidate's calibrated open hand (first transit tick);
* the recorded world hand-root target equals initial object frame @ teacher relative root.

The model file is only loaded and the archived state only set (``mj_setState``); nothing is
stepped, rendered or trained. No source file or dataset is changed. Output: one new folder.
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from icra.paths import source_value

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SEARCH = (ROOT / "out", Path("/data1/yspark/sim/icra"))
TRAIN_LEDGER = Path("/data1/yspark/sim/icra/selector-compare-v1/train-ledger/executions.jsonl")
B_COLLECTIONS = ("hope-milk-multiccd-v1/", "hope-multiccd-ab-v1/grasp/runs/")
CURRENT_CONTROLLER = "fe2434f4e03e346513de372c0da1153a037e2f1f4239014be36df93d12b3af94"
PHASES = ("transit", "descend", "approach", "arrive", "close", "lift", "hold")
# Placement holdouts, fixed by magnitude before looking at any learning result.
INNER = {"x_minus10mm", "x_plus10mm", "y_minus10mm", "y_plus10mm", "yaw_minus7_5",
         "yaw_plus7_5"}
OUTER = {"x_minus20mm", "x_plus20mm", "y_minus20mm", "y_plus20mm", "yaw_minus15",
         "yaw_plus15"}
_HASH: dict = {}
COMPARE = Path("/data1/yspark/sim/icra/selector-compare-v1")
LEDGERS = {"selector-compare-v1": COMPARE / "ledger.json",
           "teacher-followup-5h": Path("/data1/yspark/sim/icra/"
                                       "teacher-followup-5h-20260912-v1/ledger.json"),
           "target-switch-v1": Path("/data1/yspark/sim/icra/target-switch-v1/ledger.json")}
# (experiment, role) pairs whose runs were themselves an evaluation of a condition.
EVALUATION_ROLES = {
    ("selector-compare-v1", "evaluate"): "selector-compare evaluation pool",
    ("teacher-followup-5h", "pool"): "selector evaluation pool completion (follow-up)",
    ("teacher-followup-5h", "placement"): "follow-up placement grid",
    ("first_run", "expand"): "hope-pilot placement transfer",
    ("candidate_diversity", "placement"): "candidate-diversity placement transfer",
    ("target-switch-v1", "pair"): "target-switch pair",
    ("clutter", "clutter"): "clutter",
    ("multiccd_ab_grasp", "arm_B"): "multiccd B arm"}
TRAINING_ROLES = {("selector-compare-v1", "label"): "selector label (selector training data)",
                  ("selector_labels", "label"): "selector label (selector training data)"}
AFTER_ACTION = ("contact_groups", "group_forces_min", "group_forces_max", "digit_force_max",
                "target_height_min", "object_xy_displacement_max",
                "non_target_displacement_max", "robot_object_penetration_max",
                "robot_table_penetration_max", "robot_obstacle_penetration_max")
FIELDS = {
    "before_episode": {
        "mesh, mesh_sha256, scale, collision_parts": "the executed object asset",
        "initial_object_poses": "world pose of every object in the archived initial state "
                                "(after settling), m and quaternion wxyz",
        "initial_joint_qpos": "27 driven joints at the initial state (rad)",
        "scene": "table, robot base, lift, head, arm start pose, object offset/yaw",
        "dexonomy_target": "Dexonomy grasp/squeeze/pregrasp qpos, required and hand contact "
                           "bodies, contact points; object frame. Input for a goal-conditioned "
                           "executor; label for a shape-to-target generator",
        "executed_target": "world hand-root target, open and closure finger targets, "
                           "approach axis/sign/standoff actually used"},
    "per_tick_observation_before_action_i": {
        "right_state": "27 driven joint positions (arm_r_joint1..7, finger_r_joint1..20), rad, "
                       "float32, measured at timestamp[i] before action[i]",
        "rgbm": "head RGB at timestamp[i]; alpha = simulator target mask (privileged, "
                "stage 2 only)",
        "right_cam_img": "wrist RGB at timestamp[i]",
        "timestamp": "simulation time t_i, s", "control_dt": "0.02 s"},
    "per_tick_action_i": {
        "action": "27 actuator commands held over [t_i, t_i + control_dt] (10 physics "
                  "substeps); fingers = position targets (rad); arm = joint position target "
                  "+ gravity bias qfrc_bias/kp at t_i (the bias itself is not stored)"},
    "teacher_internal_not_an_observation": {
        "phase": "the teacher controller's stage label for tick i"},
    "per_tick_after_action_i_never_input_for_action_i": {
        k: "aggregated over the 10 substeps after action[i]" for k in AFTER_ACTION},
    "episode_outcome_never_input": ["accepted", "quality", "status", "reproduction verdict",
                                    "past_evaluation_use", "future states"],
    "not_stored": {
        "object pose per tick": "only the initial pose (initial_state) and per-tick "
                                "target_height_min / object_xy_displacement_max",
        "joint velocities": "right_state holds positions only",
        "arm gravity-bias term per tick": "so the pure arm joint target is not separable "
                                          "from action without replay",
        "hand root pose per tick": "not stored; derivable by forward kinematics of "
                                   "right_state with the fixed base/lift (not done here)"}}


def sha(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _HASH:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                digest.update(block)
        _HASH[key] = digest.hexdigest()
    return _HASH[key]


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def placement_name(offset, yaw) -> str:
    dx, dy = (round(float(v) * 1000) for v in offset)
    yaw = float(yaw)
    if dx == dy == 0 and yaw == 0:
        return "centre"
    if dy == 0 and yaw == 0:
        return f"x_{'plus' if dx > 0 else 'minus'}{abs(dx)}mm"
    if dx == 0 and yaw == 0:
        return f"y_{'plus' if dy > 0 else 'minus'}{abs(dy)}mm"
    if dx == dy == 0:
        text = f"{abs(yaw):g}".replace(".", "_")
        return f"yaw_{'plus' if yaw > 0 else 'minus'}{text}"
    return f"offset_{dx}_{dy}mm_yaw_{yaw:g}"


def scan() -> tuple[list[dict], dict]:
    """Accepted episodes, once each, with every directory that holds the same file."""
    seen_dirs, episodes, copies = set(), {}, defaultdict(list)
    for root in SEARCH:
        for directory, _, files in os.walk(root, followlinks=True):
            if "attempts.json" not in files or "quarantine" in directory:
                continue
            real = os.path.realpath(directory)
            if real in seen_dirs:
                continue
            seen_dirs.add(real)
            for attempt in json.loads(Path(real, "attempts.json").read_text()):
                if not attempt.get("accepted") or not attempt.get("path"):
                    continue
                digest = attempt["sha256"]
                copies[digest].append(real)
                episodes.setdefault(digest, (real, attempt))
    return [{"dir": d, "attempt": a} for d, a in episodes.values()], {
        "collections_scanned": len(seen_dirs), "copies": copies}


def physics_labels() -> dict[str, str]:
    labels = {}
    for line in TRAIN_LEDGER.read_text().splitlines():
        row = json.loads(line)
        if row["source"].get("episode_sha256"):
            labels[row["source"]["episode_sha256"]] = row["cohort"]["physics"]
    return labels


def experiment_roles() -> dict[str, list[dict]]:
    """Every ledger row that produced a given accepted episode (originals and repeats)."""
    roles: dict[str, list[dict]] = defaultdict(list)
    for line in TRAIN_LEDGER.read_text().splitlines():
        r = json.loads(line)
        digest = r["source"].get("episode_sha256")
        if digest and r["labels"].get("accepted"):
            roles[digest].append({"experiment": r["source"]["name"], "role": r["flags"]["role"],
                                  "attempt_uuid": r["source"]["attempt_uuid"],
                                  "replicate_of": r.get("replicate_of")})
    for name, path in LEDGERS.items():
        for r in json.loads(path.read_text()) if path.is_file() else []:
            if r.get("episode_sha256") and r.get("accepted"):
                roles[r["episode_sha256"]].append({
                    "experiment": name, "role": r.get("role"),
                    "attempt_uuid": r.get("attempt_uuid"),
                    "reproduction_of": r.get("reproduction_of")})
    for digest, rows in roles.items():
        unique = {row["attempt_uuid"]: row for row in rows}
        roles[digest] = list(unique.values())
    return roles


def hand_joint_order() -> dict:
    """Dexonomy's HX5 finger joints, in file order, against the executor's HAND_JOINTS."""
    import xml.etree.ElementTree as ET

    from icra import paths, scene

    xml = paths.DEXONOMY_HAND / "right.xml"
    names = [j.get("name") for j in ET.parse(xml).getroot().iter("joint")
             if j.get("name") and j.get("type") != "free"]
    fingers = [n for n in names if n.startswith("finger_r_joint")]
    return {"xml": str(xml), "xml_sha256": sha(xml), "finger_joints_in_file_order": fingers,
            "equals_executor_order": fingers == list(scene.HAND_JOINTS)}


def link(item: dict, labels: dict, issues: list) -> dict:
    """One episode: mesh, scale, object pose, joints, actions, type, candidate; checked."""
    import mujoco
    from collect_episodes import in_object_frame
    from collect_teacher_dataset import checked_candidate

    from icra import objects, scene

    directory, attempt = Path(item["dir"]), item["attempt"]
    provenance = attempt["provenance"]
    execution = provenance["execution"]
    collection = json.loads((directory / "collection.json").read_text())
    manifest_path = Path(collection["candidate_manifest"])
    manifest = json.loads(manifest_path.read_text())
    row = next(r for r in manifest["candidates"] if r["source_sha256"] == attempt["candidate_id"])
    asset = objects.load(Path(row["directory"]))
    sources = execution.get("source_hashes") or {}
    digest = attempt["sha256"]
    real = str(directory)
    physics = labels.get(digest) or ("B-multiccd" if any(p in real for p in B_COLLECTIONS)
                                     else "A")
    config = collection["scene"]
    distractors = execution.get("distractor_ids") or []
    record = {
        "episode_sha256": digest, "episode": str(directory / attempt["path"]),
        "episode_id": attempt["episode_id"], "collection": real,
        "object_id": attempt["object_id"], "candidate_id": attempt["candidate_id"],
        "grasp_type": provenance.get("grasp_type"),
        "physics": physics, "physics_source": "selector ledger cohort" if digest in labels
        else "collection path",
        "controller_sha256": source_value(sources, "src/icra/teacher/execution.py"),
        "collector_sha256": source_value(sources, "scripts/teacher/collect_teacher_dataset.py"),
        "visual_profile": execution.get("visual_profile"),
        "distractors": distractors,
        "placement": placement_name(config.get("object_offset", [0, 0]),
                                    config.get("object_yaw_degrees", 0)),
        "episode_yaw_deg": round(float(np.degrees(execution.get("yaw_rad") or 0.0)), 6),
        "grasp_root_offset_m": [float(v) for v in (execution.get("teacher_root_offset_m")
                                                   or [0.0, 0.0, 0.0])],
        "scene": {k: config.get(k) for k in ("object_offset", "object_yaw_degrees", "table",
                                              "robot_base_pose", "lift", "arm_init_pose",
                                              "head", "object_appearance")},
        "mesh": str(asset.mesh), "mesh_sha256": sha(asset.mesh),
        "collision_parts": len(asset.parts), "asset": str(row["directory"]),
        "scale": float(execution["object_scale"]), "asset_scale": float(asset.scale),
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "teacher_source": row["source"], "teacher_source_sha256": sha(row["source"]),
        "teacher_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode())
        .hexdigest(),
        "model": str(directory / execution["model"]),
        "model_sha256": next(iter(execution["geometry_hashes"].values())),
        "initial_state": str(directory / execution["initial_state"]),
        "initial_state_sha256": execution["initial_state_sha256"],
        "initial_state_spec": execution.get("initial_state_spec"),
        "yaw_rad": execution.get("yaw_rad"),
        "teacher_root_offset_m": execution.get("teacher_root_offset_m"),
        "original_target": execution["original_target"],
        "quality": {k: attempt["quality"].get(k) for k in (
            "max_robot_object_penetration", "max_digit_force", "limits",
            "longest_contact_preserving_hold_seconds", "max_contact_groups")}}

    def problem(what, **detail):
        issues.append({"episode_sha256": digest, "episode": record["episode"],
                       "problem": what, **detail})

    for field, recorded in (("model", record["model_sha256"]),
                            ("initial_state", record["initial_state_sha256"]),
                            ("episode", digest)):
        now = sha(record[field])
        if now != recorded:
            problem(f"{field} hash differs or file missing", recorded=recorded, file=now)
    if abs(record["scale"] - record["asset_scale"]) > 1e-9:
        problem("executed scale differs from the asset scale")
    if record["teacher_source_sha256"] is None:
        problem("Dexonomy source file missing")

    with np.load(record["episode"], allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    frames = len(arrays["action"])
    record["arrays"] = {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()}
    model = mujoco.MjModel.from_binary_path(record["model"])
    data = mujoco.MjData(model)
    state = np.load(record["initial_state"])
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    names = list(scene.ARM_JOINTS) + list(scene.HAND_JOINTS)
    joints = [model.joint(n).id for n in names]
    actuators = [model.actuator(n).id for n in names]
    initial_q = data.qpos[model.jnt_qposadr[joints]]
    bodies = {}
    for name in [record["object_id"], *[Path(d).name for d in distractors]]:
        body = model.body(f"object_{name}")
        address = model.jnt_qposadr[model.body_jntadr[body.id]]
        bodies[name] = {"position_m": data.qpos[address:address + 3].tolist(),
                        "quaternion_wxyz": data.qpos[address + 3:address + 7].tolist()}
    record["initial_object_poses"] = bodies
    record["initial_joint_qpos"] = initial_q.tolist()

    pose = np.eye(4)
    obj = bodies[record["object_id"]]
    mujoco.mju_quat2Mat(rotation := np.zeros(9), np.asarray(obj["quaternion_wxyz"]))
    pose[:3, :3], pose[:3, 3] = rotation.reshape(3, 3), obj["position_m"]
    _, raw, _ = checked_candidate(dict(row))
    passed = next((p for p in attempt.get("path_attempts") or []
                   if p.get("status") == "passed"), None)
    controller = execution.get("controller") or {}
    pregrasp = np.asarray(raw.get("pregrasp_qpos", np.zeros((0, 27))), float).reshape(-1, 27)
    record["dexonomy_target"] = {
        "source": row["source"], "source_sha256": record["teacher_source_sha256"],
        "template": str(raw.get("tmpl_name")), "hand": str(raw.get("hand_name")),
        "frame": "Dexonomy object frame = the object asset's own frame (mesh placed with its "
                 "lowest point at z = 0), scaled by obj_scale; source obj_pose "
                 f"{np.asarray(raw.get('obj_pose', [])).round(6).tolist()}",
        "units": "qpos[0:3] root position m; qpos[3:7] root quaternion w,x,y,z; qpos[7:27] "
                 "20 finger joints rad, used by the executor in finger_r_joint1..20 order",
        "grasp_qpos": raw["grasp_qpos"].reshape(27).tolist(),
        "squeeze_qpos": raw["squeeze_qpos"].reshape(27).tolist(),
        "pregrasp_qpos": pregrasp.tolist(), "pregrasp_stages": len(pregrasp),
        "required_contact_groups": np.asarray(raw["required_cbody"]).tolist(),
        "hand_contact_bodies": np.asarray(raw.get("hand_cbody", [])).tolist(),
        "contact_points_normals_shape": {
            "object": list(np.asarray(raw.get("obj_cpn_w", [])).shape),
            "hand": list(np.asarray(raw.get("hand_cpn_w", [])).shape),
            "meaning": "obj_cpn_w / hand_cpn_w rows = point xyz (m) + normal xyz, Dexonomy "
                       "object frame; kept in the source file"},
        "obj_scale": np.asarray(raw.get("obj_scale", [])).reshape(-1).tolist()}
    record["executed_target"] = {
        "world_root": execution["original_target"]["world_root"],
        "frame": "MuJoCo world, m; = initial object frame @ Dexonomy root @ "
                 "translate(teacher_root_offset_m)",
        "teacher_root_offset_m": record["teacher_root_offset_m"],
        "finger_open_command": [float(v) for v in row["open_qpos"]],
        "finger_closure_target": raw["squeeze_qpos"].reshape(27)[7:].tolist(),
        "approach": {"axis": (passed or {}).get("axis"), "sign": (passed or {}).get("sign"),
                     "standoff_m": controller.get("standoff"),
                     "overhead_m": controller.get("overhead"),
                     "path_attempts": [p.get("status") for p in
                                       attempt.get("path_attempts") or []],
                     "note": "the executor's own pre-grasp: overhead -> retreat (target - "
                             "sign*axis*standoff) -> target with the open hand; Dexonomy "
                             "pregrasp_qpos is recorded but was not executed"},
        "adapted_final": execution.get("adapted_target")}
    relative = in_object_frame(raw, offset=np.asarray(record["teacher_root_offset_m"]))
    root_error = float(np.max(np.abs(pose @ relative
                                     - np.asarray(record["original_target"]["world_root"]))))
    stamps = np.asarray(arrays["timestamp"], float)
    dt = np.asarray(arrays["control_dt"], float)
    phases = [str(p) for p in arrays["phase"]]
    order = [PHASES.index(p) if p in PHASES else -1 for p in phases]
    low = model.actuator_ctrlrange[actuators, 0]
    high = model.actuator_ctrlrange[actuators, 1]
    checks = {
        "frames": frames, "attempt_ticks": attempt.get("ticks"),
        "frames_equal_ticks": frames == attempt.get("ticks"),
        "state_first_equals_initial_float32": bool(np.array_equal(
            arrays["right_state"][0], initial_q.astype(np.float32))),
        "timestamp_first_equals_initial_time": bool(abs(stamps[0] - data.time) < 1e-9),
        "timestamp_step_equals_control_dt": bool(np.allclose(np.diff(stamps), dt[:-1],
                                                             rtol=0, atol=1e-9)),
        "control_dt": float(dt[0]),
        "first_hand_command_equals_open_pose": bool(np.array_equal(
            arrays["action"][0, 7:], np.clip(np.asarray(row["open_qpos"], float),
                                             low[7:], high[7:]))),
        "actions_within_ctrlrange": bool(np.all(arrays["action"] >= low - 1e-12)
                                         and np.all(arrays["action"] <= high + 1e-12)),
        "phases_in_order": bool(all(o >= 0 for o in order)
                                and all(b >= a for a, b in zip(order, order[1:], strict=False))),
        "phases_present": sorted(set(phases), key=lambda p: PHASES.index(p)
                                 if p in PHASES else 99),
        "world_root_reconstruction_max_abs": root_error,
        "world_root_reconstructed": root_error < 1e-6,
        "state_action_shapes": [list(arrays["right_state"].shape),
                                list(arrays["action"].shape)]}
    record["checks"] = checks
    for key in ("frames_equal_ticks", "state_first_equals_initial_float32",
                "timestamp_first_equals_initial_time", "timestamp_step_equals_control_dt",
                "first_hand_command_equals_open_pose", "actions_within_ctrlrange",
                "phases_in_order", "world_root_reconstructed"):
        if not checks[key]:
            problem(f"check failed: {key}", value=checks.get(key))
    return record


def pool_of(r: dict) -> tuple[str, str | None]:
    """Default training pool membership, or why an episode is kept apart."""
    if r["physics"] != "A":
        return "separate", "B-multiccd physics"
    if r["distractors"]:
        return "separate", "multi-object scene"
    if r["controller_sha256"] != CURRENT_CONTROLLER:
        return "separate", "earlier controller (different teacher behaviour)"
    if r["visual_profile"]:
        return "separate", "visual profile (different image size and head pose)"
    return "default", None


def splits(records: list[dict]) -> dict:
    """Proposals only. New-placement and new-shape evaluation are built separately."""
    pool = [r for r in records if r["pool"] == "default"]
    by_id = {r["episode_sha256"]: r for r in pool}
    hope = [r for r in pool if r["object_id"] in ("AlphabetSoup", "Peaches", "Raisins")]
    seen_train = [r["episode_sha256"] for r in pool]
    result = {
        "status": "PROPOSAL ONLY -- not final; to be decided with the stage-1 model plan",
        "unit": "whole episode (episode sha256); reproductions are collapsed into the one "
                "episode they reproduce (listed in produced_by); a trajectory is never split",
        "pool": "physics A, single object, current controller, no visual profile",
        "pool_size": len(pool),
        "relations": {
            "candidate_group": "placement variants of one (object, candidate) share it; the "
                               "placement proposals keep a group's other placements in train "
                               "on purpose, the shape proposals hold whole objects out",
            "past_evaluation_use": "per episode; conditions already used as an evaluation in "
                                   "an earlier experiment are marked, not removed"},
        "candidate_groups": {g: sorted(r["episode_sha256"] for r in pool
                                       if r["candidate_group"] == g)
                             for g in sorted({r["candidate_group"] for r in pool})}}
    result["A_trained_condition_rollout"] = {
        "train": seen_train,
        "evaluate": "roll out in the (object, candidate, placement) conditions of the "
                    "training episodes; no held-out episode; checks fitting, not transfer",
        "conditions": sorted({(r["object_id"], r["candidate_id"][:10], r["placement"])
                              for r in pool})}
    placement, shape = {}, {}
    for name, held in (("inner_10mm_7_5deg", INNER), ("outer_20mm_15deg", OUTER)):
        test = [r["episode_sha256"] for r in hope if r["placement"] in held]
        placement[name] = {
            "held_out_placements": sorted(held),
            "train": [h for h in seen_train if h not in set(test)], "test": test,
            "note": "new placement only: every test object and candidate also appears in "
                    "training at other placements (same candidate_group)"}
    for obj in ("AlphabetSoup", "Peaches", "Raisins"):
        test = [r["episode_sha256"] for r in pool if r["object_id"] == obj]
        shape[f"leave_out_{obj}"] = {
            "train": [h for h in seen_train if h not in set(test)], "test": test,
            "note": "new shape: every episode of the object (all placements and candidates) "
                    "held out" + ("; Raisins is the only Large Diameter object, so this fold "
                                  "also holds out the grasp type" if obj == "Raisins" else "")}
    for split in (*placement.values(), *shape.values()):
        assert not set(split["train"]) & set(split["test"])
        split["counts"] = {
            "train": len(split["train"]), "test": len(split["test"]),
            "test_objects": dict(Counter(by_id[h]["object_id"] for h in split["test"])),
            "test_previously_evaluated": sum(1 for h in split["test"]
                                             if by_id[h]["past_evaluation_use"]),
            "test_candidate_groups_also_in_train": len(
                {by_id[h]["candidate_group"] for h in split["test"]}
                & {by_id[h]["candidate_group"] for h in split["train"]})}
    result["B_new_placement_evaluation"] = placement
    result["C_new_shape_evaluation"] = shape
    result["not_in_any_proposal"] = {
        "core_can (2, teacher-expansion/tabletop, scale 0.08)": "could serve as an extra shape "
                                                                "but comes from an older scene "
                                                                "and scale family",
        "kept apart": "B physics, multi-object, earlier controller, visual profile"}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/2026-09-12/stage1-bc-data-contract")
    args = parser.parse_args(argv)
    items, meta = scan()
    labels = physics_labels()
    roles = experiment_roles()
    evaluation_pool = set(json.loads((COMPARE / "evaluation-candidates.json").read_text()))
    issues: list[dict] = []
    records = []
    for item in items:
        try:
            record = link(item, labels, issues)
        except Exception as error:  # noqa: BLE001 -- recorded, never silently dropped
            issues.append({"episode_sha256": item["attempt"].get("sha256"),
                           "episode": str(Path(item["dir"]) / item["attempt"]["path"]),
                           "problem": f"could not link: {type(error).__name__}: {error}"})
            continue
        record["copies"] = meta["copies"][record["episode_sha256"]]
        record["pool"], record["kept_apart_because"] = pool_of(record)
        produced = roles.get(record["episode_sha256"], [])
        record["produced_by"] = produced
        record["repeats_of_this_episode"] = sum(
            1 for p in produced if p.get("replicate_of") or p.get("reproduction_of")
            or "reproduce" in str(p.get("role")) or p.get("role") == "control")
        record["past_evaluation_use"] = sorted({EVALUATION_ROLES[(p["experiment"], p["role"])]
                                                for p in produced
                                                if (p["experiment"], p["role"])
                                                in EVALUATION_ROLES})
        record["past_training_use"] = sorted({TRAINING_ROLES[(p["experiment"], p["role"])]
                                              for p in produced
                                              if (p["experiment"], p["role"]) in TRAINING_ROLES})
        record["candidate_in_selector_evaluation_pool"] = record["candidate_id"] \
            in evaluation_pool
        record["candidate_group"] = f"{record['object_id']}/{record['candidate_id'][:12]}"
        record["condition_key"] = [record["object_id"], record["candidate_id"],
                                   record["placement"], record["episode_yaw_deg"],
                                   record["grasp_root_offset_m"], record["physics"],
                                   record["distractors"], record["controller_sha256"]]
        records.append(record)
    records.sort(key=lambda r: (r["pool"], r["object_id"], r["candidate_id"], r["placement"]))
    default = [r for r in records if r["pool"] == "default"]
    conditions = Counter(json.dumps(r["condition_key"]) for r in records)
    summary = {
        "collections_scanned": meta["collections_scanned"],
        "unique_accepted_episodes": len(records), "could_not_link": sum(
            1 for i in issues if i["problem"].startswith("could not link")),
        "copies_collapsed": sum(len(r["copies"]) - 1 for r in records),
        "same_condition_distinct_episodes": [json.loads(k) for k, v in conditions.items()
                                             if v > 1],
        "hand_joint_order": hand_joint_order(),
        "dexonomy_pregrasp_available": sum(1 for r in records
                                           if r["dexonomy_target"]["pregrasp_stages"]),
        "default_pool_previously_evaluated": sum(1 for r in records if r["pool"] == "default"
                                                 and r["past_evaluation_use"]),
        "default_pool": {"episodes": len(default),
                         "objects": dict(Counter(r["object_id"] for r in default)),
                         "candidates": len({(r["object_id"], r["candidate_id"])
                                            for r in default}),
                         "candidates_by_object": dict(Counter(
                             o for o, _ in {(r["object_id"], r["candidate_id"])
                                            for r in default})),
                         "grasp_types": dict(Counter(r["grasp_type"] for r in default)),
                         "placements": dict(Counter(r["placement"] for r in default)),
                         "frames": int(sum(r["checks"]["frames"] for r in default))},
        "kept_apart": dict(Counter(r["kept_apart_because"] for r in records
                                   if r["pool"] != "default")),
        "checks_passed": {k: sum(bool(r["checks"][k]) for r in records) for k in (
            "frames_equal_ticks", "state_first_equals_initial_float32",
            "timestamp_first_equals_initial_time", "timestamp_step_equals_control_dt",
            "first_hand_command_equals_open_pose", "actions_within_ctrlrange",
            "phases_in_order", "world_root_reconstructed")},
        "issues": len(issues), "student_ready": False}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    dump(out / "split-proposals.json", splits(records))
    dump(out / "fields.json", FIELDS)
    dump(out / "issues.json", issues)
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
