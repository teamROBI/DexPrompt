"""Is the corrected grip offset worth anything? One object, one prediction, two arms.

The offset moves the hand from where GraspGenX assumes the object sits in the
gripper to where it actually sits. It was being computed from a reconstruction of
the scene rather than from the render, which put it 180 mm out on any object not
standing in the nominal slot and skipped it entirely on the DGN layout. That is
fixed; this measures what the fix buys, and what having no offset at all costs.

The comparison is only honest if the two arms differ in **one** thing:

* **One prediction.** GraspGenX is sampled once. The zero-offset arm is a copy of
  those candidates with the offset field zeroed -- not a second prediction run,
  which would draw different poses from the same model and confound the two.
  Every candidate id, pose and confidence is checked equal before anything runs.
* **One scene.** Each stage builds its own MuJoCo model from a scene file and a
  directory of objects. Given the whole DGN folder the filter stands five
  thousand objects on the table, which is not the scene the camera saw. Both the
  scene file and the object set are checked against what the observation
  recorded -- object pose, mesh scale, and the robot's own joints.
* **One set of settings.** Rolls, depths, solver iterations, damping and the
  thumb threshold are passed identically to both filter runs, and the solver
  starts from the scene's rest pose every time, so the initial guess is fixed.

What comes out is a stage-by-stage tally per arm: how many pose variants were out
of the arm's reach, failed inverse kinematics, hit something (and what), arrived
with the thumb too far away, and passed.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/compare_grip_offset.py \\
        --template 33_Inferior_Pincer --objects out/offset-compare/objects \\
        --observation out/offset-compare/observation --predicted out/offset-compare/predicted \\
        --out out/offset-compare
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
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402
from icra import scene as scene_lib  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: How far the scene a later stage builds may differ from the observed one.
POSITION_TOLERANCE = 1e-9
SCALE_TOLERANCE = 1e-12
JOINT_TOLERANCE = 1e-9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--objects", type=Path, required=True,
                        help="The object directory every stage uses; one object.")
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--predicted", type=Path, required=True,
                        help="Prediction to compare, generated once beforehand.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-object", type=int, default=10)
    parser.add_argument("--roll-steps", type=int, default=12)
    parser.add_argument("--depth-span", type=float, default=0.03)
    parser.add_argument("--depth-steps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--thumb-reach", type=float, default=0.05)
    return parser.parse_args(argv)


def scene_matches(meta: dict, scene_path: Path, objects: Path) -> list[str]:
    """Every way the scene a filter run would build differs from the observed one.

    Built here exactly as ``filter_grasps`` builds it -- the same scene file, the
    same directory of objects, ``apply_init`` and nothing else -- and compared
    against what the observation wrote down.
    """
    import mujoco

    problems = []
    digest = meta.get("scene") or {}
    if not digest:
        return ["the observation does not record which scene file it used"]
    actual = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if digest.get("sha256") != actual:
        problems.append(
            f"scene file differs: observed {digest.get('sha256', '?')[:12]}, "
            f"this run {actual[:12]} ({scene_path})"
        )

    scene = scene_lib.load(scene_path)
    directories = sorted(d for d in objects.iterdir() if d.is_dir())
    model = scene_lib.build(scene, directories)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)
    mujoco.mj_forward(model, data)

    placement = meta.get("placement") or {}
    here = {d.name for d in directories}
    if here != set(placement):
        problems.append(
            f"object set differs: observed {sorted(placement)}, this run {sorted(here)}"
        )
    for name in sorted(here & set(placement)):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}")
        observed = np.asarray(placement[name]["world_transform"], dtype=float)
        if not np.allclose(observed[:3, 3], data.xpos[body], atol=POSITION_TOLERANCE):
            problems.append(
                f"{name}: observed at {observed[:3, 3].round(6).tolist()}, "
                f"this run at {np.asarray(data.xpos[body]).round(6).tolist()}"
            )
        if not np.allclose(observed[:3, :3],
                           np.asarray(data.xmat[body]).reshape(3, 3), atol=1e-9):
            problems.append(f"{name}: resting rotation differs from the observation")
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_visual")
        scale = model.mesh_scale[model.geom_dataid[geom]]
        if not np.allclose(placement[name]["scale"], scale, atol=SCALE_TOLERANCE):
            problems.append(
                f"{name}: observed at scale {placement[name]['scale']}, "
                f"this run at {[float(v) for v in scale]}"
            )

    observed_joints = meta.get("robot") or {}
    if not observed_joints:
        problems.append("the observation does not record the robot's joints")
    for name, value in observed_joints.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            problems.append(f"joint {name} is not in this run's model")
            continue
        here_value = float(data.qpos[model.jnt_qposadr[joint]])
        if abs(here_value - value) > JOINT_TOLERANCE:
            problems.append(
                f"joint {name}: observed {value:.6f}, this run {here_value:.6f}"
            )
    return problems


def candidate_digest(grasps: dict) -> str:
    """A fingerprint of the candidates themselves -- ids, poses, confidences.

    Recorded for both arms so the claim that they ran on identical candidates is
    something the output shows rather than something this script asserts.
    """
    return hashlib.sha256(
        json.dumps(grasps, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def clone_without_offset(source: Path, destination: Path) -> dict[str, str]:
    """The same candidates, with the offset taken out. Raises if anything else moves."""
    import yaml

    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in sorted(source.glob("*.yml")):
        original = yaml.safe_load(path.read_text())
        zeroed = dict(original)
        zeroed["grip_offset"] = [0.0, 0.0, 0.0]
        zeroed["grip_offset_applied"] = False
        zeroed["derived_from"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) \
            else str(path)
        (destination / path.name).write_text(yaml.safe_dump(zeroed, sort_keys=False))
        # Read back rather than trust the write: the whole comparison rests on
        # these two files holding the same candidates.
        written = yaml.safe_load((destination / path.name).read_text())
        if written["grasps"] != original["grasps"]:
            raise ValueError(f"{path.name}: candidates changed while zeroing the offset")
        if written["object"] != original["object"]:
            raise ValueError(f"{path.name}: object changed while zeroing the offset")
        copied[path.name] = candidate_digest(original["grasps"])
    if not copied:
        raise ValueError(f"no predictions under {source}")
    return copied


def clear_record(directory: Path) -> None:
    """Remove a previous run's completion record, so its presence means this run.

    Timestamps are not enough: a filter run finishes inside one second, and file
    systems that keep mtime to the second would make a fresh record look stale.
    """
    record = directory / "run.json"
    if record.is_file():
        record.unlink()


def completion_problems(directory: Path, expected: list[str],
                        candidates: int) -> list[str]:
    """Whether a filter run actually finished, or only left a directory behind.

    A run that keeps nothing exits nonzero and is a result; a run that dies
    halfway exits nonzero too, having already written some of its files. The
    output directory alone cannot tell those apart, so this asks for the record
    the filter writes last -- cleared beforehand by :func:`clear_record` -- plus
    one stage file per object with the expected number of candidates in it.
    """
    problems = []
    record = directory / "run.json"
    if not record.is_file():
        return [f"{record} is missing: the filter did not reach the end of its run"]
    finished = json.loads(record.read_text())
    if not finished.get("completed"):
        problems.append(f"{record} does not record a completed run")
    if sorted(finished.get("objects") or []) != sorted(expected):
        problems.append(
            f"filtered {sorted(finished.get('objects') or [])}, expected {sorted(expected)}"
        )
    for name in expected:
        stages = directory / f"{name}.stages.json"
        if not stages.is_file():
            problems.append(f"{stages} is missing")
            continue
        tried = json.loads(stages.read_text()).get("candidates_tried")
        if tried != candidates:
            problems.append(f"{name}: {tried} candidates tried, expected {candidates}")
    return problems


def filter_run(args, predicted: Path, out: Path, expected: list[str],
               candidates: int) -> list[str]:
    """One filter run. Returns what is wrong with it, empty when it finished."""
    clear_record(out / args.template)
    command = [
        str(paths.DEXONOMY_PYTHON), str(ROOT / "scripts/teacher/filter_grasps.py"),
        "--template", args.template, "--scene", str(args.scene),
        "--objects", str(args.objects), "--predicted", str(predicted),
        "--out", str(out), "--max-per-object", str(args.max_per_object),
        "--roll-steps", str(args.roll_steps), "--depth-span", str(args.depth_span),
        "--depth-steps", str(args.depth_steps), "--iterations", str(args.iterations),
        "--damping", str(args.damping), "--thumb-reach", str(args.thumb_reach),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    print("    " + "\n    ".join((result.stdout or "").strip().splitlines()[-8:]))
    # 0 kept exits 1, so the code alone says nothing; the tail of stderr is
    # printed because that is where a crash says what happened.
    problems = completion_problems(out / args.template, expected, candidates)
    if problems and result.stderr:
        print("    " + "\n    ".join(result.stderr.strip().splitlines()[-5:]))
    return problems


def stages_of(directory: Path) -> dict:
    return {
        path.name.removesuffix(".stages.json"): json.loads(path.read_text())
        for path in sorted(directory.glob("*.stages.json"))
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    observation = args.observation / args.camera
    predicted = args.predicted / args.template
    problems = paths.check(args.scene, args.objects, predicted,
                           observation / "meta_data.json")
    if problems:
        print("\n".join(problems))
        return 1

    meta = json.loads((observation / "meta_data.json").read_text())
    print("checking that every stage sees one scene")
    mismatches = scene_matches(meta, args.scene, args.objects)
    if mismatches:
        print("\n".join(f"  {problem}" for problem in mismatches))
        print("\nthe stages would not be looking at the same scene; nothing was run")
        return 1
    names = sorted(meta.get("placement") or {})
    print(f"  {len(names)} object(s): {', '.join(names)}"
          + "".join(f"\n    {n} at scale {meta['placement'][n]['scale']}" for n in names))

    zeroed = args.out / "predicted-zero" / args.template
    copied = clone_without_offset(predicted, zeroed)
    record = predicted / "grip_offset.json"
    offsets = json.loads(record.read_text()) if record.is_file() else {}
    print(f"\ncloned {len(copied)} prediction file(s) with the offset zeroed")
    for name, digest in copied.items():
        print(f"  {name}: candidates {digest[:16]} (identical in both arms)")

    arms = {"offset": predicted, "zero": zeroed}
    results = {}
    for arm, source in arms.items():
        print(f"\nfiltering with the {arm} candidates")
        out = args.out / f"reachable-{arm}"
        broken = filter_run(args, source.parent, out, names, args.max_per_object)
        if broken:
            print("\n".join(f"  {problem}" for problem in broken))
            print(f"\nthe {arm} filter run did not complete; nothing is comparable")
            return 1
        results[arm] = stages_of(out / args.template)

    summary = {
        "template": args.template, "camera": args.camera,
        "objects": [str(args.objects)], "candidates": args.max_per_object,
        "settings": {k: getattr(args, k) for k in
                     ("roll_steps", "depth_span", "depth_steps", "iterations",
                      "damping", "thumb_reach")},
        "prediction_offsets": offsets,
        "candidate_digest": copied,
        "arms": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(summary, indent=4))

    stages = ("out_of_reach", "ik_failure", "collision", "thumb_far", "passed")
    print(f"\n{'object':28s} {'arm':7s} " + " ".join(f"{s:>12s}" for s in stages) + "   kept")
    for name in sorted(set().union(*(set(r) for r in results.values()))):
        for arm in arms:
            record = results[arm].get(name)
            if record is None:
                print(f"{name[:28]:28s} {arm:7s} (no output)")
                continue
            counts = record["stages"]
            print(f"{name[:28]:28s} {arm:7s} "
                  + " ".join(f"{counts[s]:12d}" for s in stages)
                  + f"   {record['kept']}")
    print(f"\n{args.out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
