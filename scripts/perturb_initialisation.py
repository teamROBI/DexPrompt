"""Vary only the hand pose the refiner starts from, and prove nothing else moved.

The ADMM starts from ``grasp_qpos`` and pulls the hand's contact points onto the
object's. Perturbing that pose alone is not as simple as writing a different
number into it: the objective's hand-side targets are stored in the world frame
and converted to each link's own frame **at the starting pose**, so moving the
hand and leaving the stored points alone quietly moves the targets with it, and
the comparison stops being about initialisation.

So the link-frame targets are held fixed and the stored world-frame points are
recomputed at the new pose. Everything else in the file -- the template name,
the required bodies, the contact indices, the object-frame targets, the scene,
the scale and the external wrench -- is copied through untouched, and the result
is checked against the original rather than assumed.

Perturbations are fixed here, before any of them is run, and applied to the
hand's root: two tilts about the root's own x axis and two slides along its own
z. No random draw is involved, so there is no seed to choose.

    third_party/dexonomy/.venv/bin/python scripts/perturb_initialisation.py \\
        --controls out/contact-diagnosis/controls.json \\
        --out third_party/dexonomy/output --prefix init4 \\
        --report out/init-sensitivity/variants.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "dexonomy"))

import numpy as np  # noqa: E402

HAND = ROOT / "third_party" / "dexonomy" / "assets" / "hand" / "hx5_d20" / "right.xml"

# Fixed before running. Angles in degrees about the root's own x axis, applied
# about the root position; slides in millimetres along the root's own z.
PERTURBATIONS = (
    ("tilt_plus", "rotate", +6.0),
    ("tilt_minus", "rotate", -6.0),
    ("slide_plus", "translate", +5.0),
    ("slide_minus", "translate", -5.0),
)
# Digit 5's proximal flexion joint, and offsets from where each candidate starts.
# Fixed before running. Positive rotation curls the finger (measured: the tip
# moves toward the palm). The PIP joint is used alone because it has the most
# travel and moves both distal links the required contacts sit on; the hammer's
# DIP starts within 0.1 rad of its limit, so offsetting every flexion joint
# together would have made "curl more" unrunnable for half the negatives. An
# offset that leaves the joint's range is recorded as not run, never clipped.
FINGER_JOINT = "finger_r_joint18"
FINGER_PERTURBATIONS = (
    ("pip_minus_030", "flex", -0.30),
    ("pip_minus_015", "flex", -0.15),
    ("pip_plus_015", "flex", +0.15),
    ("pip_plus_030", "flex", +0.30),
)
SETS = {"root": PERTURBATIONS, "finger5": FINGER_PERTURBATIONS}

# The initialisation files store contacts as float32, so a target written and
# read back moves by about 1e-8 m however exactly it was computed. A micrometre
# is far below anything the hand could act on and far above that round-off, so
# a drift under it means the targets were carried across, not that the check is
# slack: a perturbation that actually moved them shifts them by millimetres.
TOLERANCE = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Dexonomy output root; new experiments are made here.")
    parser.add_argument("--prefix", default="init4")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--set", choices=sorted(SETS), default="root",
                        help="Which fixed perturbation set to apply.")
    return parser.parse_args(argv)


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(scene_path: str):
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    return MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=str(HAND), freejoint=True),
        scene_cfg=load_scene_cfg(scene_path),
        sim_cfg=MuJoCo_OptCfg(obj_margin=0.0),
    )


def moved_root(qpos: np.ndarray, kind: str, amount: float) -> np.ndarray:
    """The same hand, its root tilted or slid in its own frame."""
    import mujoco

    out = np.asarray(qpos, dtype=float).copy()
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, out[3:7])
    frame = rotation.reshape(3, 3)
    if kind == "rotate":
        half = np.radians(amount) / 2.0
        axis = frame @ np.array([1.0, 0.0, 0.0])
        turn = np.concatenate([[np.cos(half)], np.sin(half) * axis])
        composed = np.zeros(4)
        mujoco.mju_mulQuat(composed, turn, out[3:7])
        out[3:7] = composed
    elif kind == "translate":
        out[:3] = out[:3] + frame @ np.array([0.0, 0.0, amount / 1000.0])
    else:
        raise ValueError(kind)
    return out


def flexed(env, qpos: np.ndarray, joint: str, amount: float):
    """The same hand with one finger joint offset, or the reason it cannot be.

    Returns ``(qpos, index, None)`` or ``(None, index, reason)``. The index is
    the joint's position in the hand's own configuration vector, resolved from
    the model by name rather than assumed.
    """
    model = env._model
    handle = model.joint(env._cfg.hand_prefix + joint)
    index = int(handle.qposadr[0])
    if index >= len(qpos):
        raise ValueError(f"{joint} at {index} is outside a {len(qpos)}-long hand pose")
    low, high = (float(v) for v in model.jnt_range[handle.id])
    value = float(qpos[index]) + amount
    if value < low or value > high:
        return None, index, (f"out_of_range: {joint} {float(qpos[index]):.4f}"
                             f"{amount:+.2f} = {value:.4f} outside [{low:.4f}, {high:.4f}]")
    out = np.asarray(qpos, dtype=float).copy()
    out[index] = value
    return out, index, None


def moved_bodies(env, joint: str) -> set[str]:
    """Hand bodies a joint moves: the one it sits on and everything below it."""
    model = env._model
    prefix = env._cfg.hand_prefix
    first = int(model.jnt_bodyid[model.joint(prefix + joint).id])
    moved = set()
    for body in range(model.nbody):
        walk = body
        while walk > 0 and walk != first:
            walk = int(model.body_parentid[walk])
        if walk == first:
            moved.add(model.body(body).name.removeprefix(prefix))
    return moved


def joint_state(env) -> dict:
    """Whether any finger joint sits outside its range at this pose."""
    model, data = env._model, env._data
    outside, margins = [], []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] == 3:
            value = float(data.qpos[model.jnt_qposadr[j]])
            low, high = model.jnt_range[j]
            margins.append(min(value - low, high - value))
            if value < low - 1e-9 or value > high + 1e-9:
                outside.append(model.joint(j).name)
    return {"joints_outside_range": outside,
            "tightest_margin_rad": float(min(margins)) if margins else None}


def collision_state(env) -> dict:
    """How far the hand is into the object and into the table at this pose."""
    model, data = env._model, env._data
    prefix = env._cfg.hand_prefix
    into_object, into_world, into_hand = [], [], []
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        first = model.body(model.geom(contact.geom1).bodyid[0]).name
        second = model.body(model.geom(contact.geom2).bodyid[0]).name
        depth = -float(contact.dist)
        hands = [name.startswith(prefix) for name in (first, second)]
        if "world" in (first, second):
            into_world.append(depth)
        elif all(hands):
            into_hand.append(depth)
        elif any(hands):
            into_object.append(depth)
    return {"into_object_mm": float(max(into_object) * 1000) if into_object else 0.0,
            "into_world_mm": float(max(into_world) * 1000) if into_world else 0.0,
            "into_hand_mm": float(max(into_hand) * 1000) if into_hand else 0.0}


def check_saved_targets(env, destination, reference):
    """Check the serialized targets at the serialized pose, not a synthetic round trip."""
    import mujoco

    saved = np.load(destination, allow_pickle=True).item()
    env.reset_qpos(np.asarray(saved["grasp_qpos"])[0])
    mujoco.mj_forward(env._model, env._data)
    actual = np.asarray(env.transform_cpn_w2b(
        list(saved["hand_cbody"]), np.asarray(saved["hand_cpn_w"])), dtype=float)
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("saved contact targets have invalid shape or values")
    drift = float(np.abs(actual - reference).max())
    if drift > TOLERANCE:
        raise ValueError(f"saved link-frame contact targets moved: {drift}")
    return saved, drift


def write_variant(record: dict, name: str, kind: str, amount: float,
                  destination: Path) -> dict:
    """One perturbed initialisation, with the invariants checked, not assumed."""
    import mujoco

    source = Path(record["init_path"])
    if destination.exists():
        raise FileExistsError(f"variant output already exists: {destination}")
    data = np.load(source, allow_pickle=True).item()
    env = environment(str(data["scene_path"]))
    bodies = list(data["hand_cbody"])
    original_qpos = np.asarray(data["grasp_qpos"])[0]

    env.reset_qpos(original_qpos)
    mujoco.mj_forward(env._model, env._data)
    targets_in_link_frame = np.asarray(
        env.transform_cpn_w2b(bodies, np.asarray(data["hand_cpn_w"])), dtype=float)

    changed_index = None
    if kind == "flex":
        qpos, changed_index, reason = flexed(env, original_qpos, FINGER_JOINT, amount)
        if reason is not None:
            # Not run, and not replaced: the row stays in the denominator.
            return {"variant": name, "kind": kind, "amount": amount,
                    "joint": FINGER_JOINT, "path": None, "not_run": reason}
    elif kind == "none":
        qpos = original_qpos
    else:
        qpos = moved_root(original_qpos, kind, amount)
    env.reset_qpos(qpos)
    mujoco.mj_forward(env._model, env._data)
    recomputed_world = np.asarray(
        env.transform_cpn_b2w(bodies, targets_in_link_frame), dtype=float)

    written = dict(data)
    written["grasp_qpos"] = np.asarray(qpos, dtype=np.float32)[None]
    written["hand_cpn_w"] = np.asarray(recomputed_world, dtype=np.float32)
    if kind == "flex":
        # Only the rows on links the joint moves are recomputed; the rest keep
        # their stored bytes, so an untouched target cannot drift by rounding.
        moving = moved_bodies(env, FINGER_JOINT)
        stored = np.asarray(data["hand_cpn_w"]).copy()
        for row, body in enumerate(bodies):
            if body in moving:
                stored[row] = np.asarray(recomputed_world[row], dtype=stored.dtype)
        written["hand_cpn_w"] = stored
        written["grasp_qpos"] = np.asarray(data["grasp_qpos"]).copy()
        written["grasp_qpos"][0, changed_index] = np.float32(qpos[changed_index])

    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "none":
        # A zero perturbation must not rewrite/round the baseline contact targets.
        shutil.copyfile(source, destination)
    else:
        np.save(destination, written, allow_pickle=True)
    written, drift = check_saved_targets(env, destination, targets_in_link_frame)

    unchanged = {
        key: bool(np.array_equal(np.asarray(data[key]), np.asarray(written[key])))
        for key in ("obj_cpn_w", "hand_cbody", "required_cbody", "tmpl_name",
                    "obj_pose", "obj_scale", "ext_wrench", "ext_center",
                    "scene_path")
    }
    if not all(unchanged.values()):
        raise ValueError(f"variant changed fixed fields: {unchanged}")
    other_joints_unchanged = None
    if kind == "flex":
        before = np.asarray(data["grasp_qpos"])[0]
        after = np.asarray(written["grasp_qpos"])[0]
        others = [i for i in range(len(before)) if i != changed_index]
        other_joints_unchanged = bool(np.array_equal(before[others], after[others]))
        if not other_joints_unchanged or after[changed_index] == before[changed_index]:
            raise ValueError("a finger variant must change exactly its one joint")
        moving = moved_bodies(env, FINGER_JOINT)
        still = [r for r, b in enumerate(bodies) if b not in moving]
        if not np.array_equal(np.asarray(data["hand_cpn_w"])[still],
                              np.asarray(written["hand_cpn_w"])[still]):
            raise ValueError("targets on links the joint does not move were rewritten")
    return {
        "variant": name, "kind": kind, "amount": amount,
        "path": str(destination),
        "link_frame_target_drift": drift,
        "link_frame_targets_preserved": drift <= TOLERANCE,
        "fields_unchanged": unchanged,
        "root_moved_mm": float(np.linalg.norm(
            np.asarray(written["grasp_qpos"])[0][:3] - original_qpos[:3]) * 1000),
        "finger_joints_changed": bool(not np.array_equal(
            np.asarray(written["grasp_qpos"])[0][7:], original_qpos[7:])),
        "changed_joint": FINGER_JOINT if kind == "flex" else None,
        "changed_index": changed_index,
        "joint_value_before": float(original_qpos[changed_index])
            if changed_index is not None else None,
        "joint_value_after": float(np.asarray(written["grasp_qpos"])[0][changed_index])
            if changed_index is not None else None,
        "other_joints_unchanged": other_joints_unchanged,
        "initial_state": {**joint_state(env), **collision_state(env)},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controls = json.loads(args.controls.read_text())
    chosen = SETS[args.set]
    units = {"rotate": "degrees about the root's own x",
             "translate": "millimetres along the root's own z",
             "flex": f"radians added to {FINGER_JOINT}"}
    report = {"set": args.set, "perturbations": [
        {"name": n, "kind": k, "amount": a, "units": units[k]}
        for n, k, a in chosen],
        "note": "Fixed before running; deterministic, so no seed is involved.",
        "candidates": []}

    for group in ("positive", "negative"):
        for record in controls.get(group, []):
            source = Path(record["init_path"])
            parts = list(source.parts)
            index = parts.index("output")
            experiment = parts[index + 1]
            tail = Path(*parts[index + 2:])
            rows = []
            for name, kind, amount in (("original", "none", 0.0),) + chosen:
                destination = (args.out / f"{args.prefix}_{name}_{experiment}"
                               / tail.parent / source.name)
                rows.append(write_variant(record, name, kind, amount, destination))
            report["candidates"].append({
                "control": group, "object": record["object"],
                "grasp_type": record["grasp_type"],
                "sha256": record["sha256"],
                "source_init": str(source),
                "source_init_sha256": digest(source),
                "experiment": experiment,
                "variants": rows,
            })
            ran = [r for r in rows if not r.get("not_run")]
            worst = max(r["link_frame_target_drift"] for r in ran)
            unchanged = all(all(r["fields_unchanged"].values()) for r in ran)
            print(f"{group:<9}{record['object']:<8}{record['sha256'][:8]}  "
                  f"{len(ran)}/{len(rows)} written  target drift {worst:.2e}  "
                  f"other fields unchanged: {unchanged}")
            for row in rows:
                if row.get("not_run"):
                    print(f"          not run  {row['variant']:<14}{row['not_run']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
