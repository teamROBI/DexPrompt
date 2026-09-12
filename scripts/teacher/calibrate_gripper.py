# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Give each grasp type its own gripper, because they are not the same gripper.

GraspGenX places a **tool centre point** on the object: it predicts poses such
that ``config["fingertip"]`` lands where the grip happens. That point, the volume
the fingers sweep, and the shape the hand approaches in are all properties of the
gripper -- and on a 20-DoF hand they are not one gripper. Wrapping a bottle and
pinching a card put the contact at depths 53 mm apart along the approach axis and
hold the hand in shapes that differ by nearly a hundred millimetres of reach.

    palmar_pinch        contact at z = 0.100 m
    small_diameter                    0.100
    parallel_extension                0.121
    large_diameter                    0.133
    power_sphere                      0.153

One averaged value serves neither end. So this writes **one gripper per grasp
template**, each with that template's own numbers, and GraspGenX is then run once
per type against the type it is calibrated for.

Everything here is measured from the templates rather than annotated. That
matters because the alternative -- the onboarding wizard -- put the tool centre
at the fingertips, 102 mm past where this hand actually grips, and left the
knuckle joints at zero in both the open and closed poses it recorded. GraspGenX
renders the gripper at those poses and conditions its model on the result, so it
was being shown a hand that curls only its fingertips and told the grip happens
at their tips. Neither is true of this hand.

Each generated gripper gets:

* ``fingertip`` -- the centroid of that template's contact points;
* ``sweep_volume`` -- the box those points occupy, reaching back to the palm,
  since an object can sit anywhere between the fingers and the palm;
* ``close`` -- the template's own pose;
* ``open`` -- that pose with the flexion backed off but not undone. A hand with
  straight fingers reaches 236 mm past the palm on this robot and drives four
  fingertips through the table on any top-down approach;
* ``bbox`` and a collision mesh, both measured on the hand at that open pose,
  because that is the shape that has to fit around the object.

Run with Dexonomy's interpreter, which has the mujoco this reads the hand with::

    third_party/dexonomy/.venv/bin/python scripts/teacher/calibrate_gripper.py
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

#: Room left around the contacts when sizing the sweep volume, so an object a
#: little larger than the template was fitted to is still proposed.
MARGIN = 0.015

#: How much of the closed pose's flexion the open pose keeps.
OPEN_FLEXION = 0.4

#: Joints that flex a finger, by position in each four-joint chain. The first of
#: each chain abducts or opposes and holds its value between open and close.
FLEXION_OFFSETS = (1, 2, 3)

#: The gripper the wizard produced, which supplies the URDF, the meshes and the
#: fields that are the same whatever the grasp type.
BASE_GRIPPER = "hx5_d20_right"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grippers", type=Path,
        default=paths.GRASPGENX / "ext/gripper_descriptions/gripper_descriptions"
                                  "/assets/x_grippers",
    )
    parser.add_argument("--base", default=BASE_GRIPPER)
    parser.add_argument("--templates", type=Path, default=paths.DEXONOMY_HAND / "init_tmpl")
    parser.add_argument("--hand-xml", type=Path, default=paths.DEXONOMY_HAND / "right.xml")
    parser.add_argument("--template", dest="only", action="append")
    return parser.parse_args(argv)


def joint_limits(hand_xml: Path) -> dict[str, tuple[float, float]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(hand_xml))
    limits = {}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name:
            limits[name] = tuple(float(v) for v in model.jnt_range[index])
    return limits


def open_pose(close: dict[str, float], limits: dict) -> dict[str, float]:
    """The closed pose with its flexion backed off, not undone."""
    opened = dict(close)
    for chain in range(5):
        for offset in FLEXION_OFFSETS:
            name = f"finger_r_joint{4 * chain + 1 + offset}"
            if name in opened:
                opened[name] = float(
                    np.clip(close[name] * OPEN_FLEXION, *limits[name])
                )
    return opened


def hand_mesh(hand_xml: Path, pose: dict[str, float]):
    """The whole hand as one mesh, at ``pose``, in the hand's own frame."""
    import mujoco
    import trimesh

    model = mujoco.MjModel.from_xml_path(str(hand_xml))
    data = mujoco.MjData(model)
    for name, value in pose.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = float(value)
    mujoco.mj_forward(model, data)

    pieces = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        faces = np.asarray(model.mesh_face[face_start : face_start + face_count])
        rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        pieces.append(
            trimesh.Trimesh(
                vertices=vertices @ rotation.T + data.geom_xpos[geom],
                faces=faces, process=False,
            )
        )
    return trimesh.util.concatenate(pieces)


def write_gripper(
    destination: Path, base_dir: Path, base_config: dict,
    contacts: np.ndarray, close: dict[str, float], opened: dict[str, float],
    mesh,
) -> dict:
    """One gripper directory: shared geometry, this template's numbers."""
    destination.mkdir(parents=True, exist_ok=True)

    # The URDF and its meshes are the same hand whatever shape it is in, so they
    # are linked rather than copied -- the meshes alone run to tens of megabytes.
    for name in ("gripper.urdf", "meshes"):
        link = destination / name
        if not link.exists():
            link.symlink_to(Path("..") / base_dir.name / name)

    # The collision mesh is *not* shared: it is the hand at this template's open
    # pose, which is the shape GraspGenX checks against the scene.
    mesh.export(destination / "coll_mesh.obj")
    mesh.export(destination / "vis_mesh.obj")

    low = contacts.min(axis=0) - MARGIN
    high = contacts.max(axis=0) + MARGIN
    # An object can sit anywhere between the fingers and the palm, so the near
    # face stays where the base gripper put it however shallow this grip is.
    sweep = base_config["sweep_volume"]
    low[2] = min(low[2], sweep["offset"][2] - sweep["extents"][2] / 2)

    bounds = mesh.bounds
    config = dict(base_config)
    config["fingertip"] = [round(float(v), 5) for v in contacts.mean(axis=0)]
    config["sweep_volume"] = {
        "offset": [round(float(v), 5) for v in (low + high) / 2],
        "extents": [round(float(v), 5) for v in (high - low)],
        "offset2": [round(float(v), 5) for v in (low + high) / 2],
        "extents2": [round(float(v) * 0.6, 5) for v in (high - low)],
    }
    config["open"] = {k: round(v, 4) for k, v in opened.items()}
    config["close"] = {k: round(v, 4) for k, v in close.items()}
    config["bbox"] = [
        [round(float(v), 5) for v in bounds[0]],
        [round(float(v), 5) for v in bounds[1]],
    ]
    (destination / "config.json").write_text(json.dumps(config, indent=4))
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.grippers / args.base
    problems = paths.check(base_dir / "config.json", args.templates, args.hand_xml)
    if problems:
        print("\n".join(problems))
        return 1

    # The wizard's file is kept as the source of the fields that do not depend on
    # the grasp type, and never used for the ones that do.
    original = base_dir / "config.wizard.json"
    base_config = json.loads(
        (original if original.exists() else base_dir / "config.json").read_text()
    )
    limits = joint_limits(args.hand_xml)
    joints = [f"finger_r_joint{i}" for i in range(1, 21)]

    names = sorted(p.stem for p in args.templates.glob("*.npy"))
    if args.only:
        names = [n for n in names if n in set(args.only)]
    if not names:
        print(f"no templates under {args.templates}")
        return 1

    print(f"{'gripper':34s} {'tool centre z':>14} {'reach at open':>14} {'contacts':>9}")
    for name in names:
        template = np.load(args.templates / f"{name}.npy", allow_pickle=True).item()
        contacts = np.asarray(template["hand_cpn_w"], dtype=float)[:, :3]
        close = {
            joint: float(value)
            for joint, value in zip(joints, np.asarray(template["grasp_qpos"])[0, 7:])
        }
        opened = open_pose(close, limits)
        mesh = hand_mesh(args.hand_xml, opened)

        gripper = f"{args.base}__{name}"
        config = write_gripper(
            args.grippers / gripper, base_dir, base_config,
            contacts, close, opened, mesh,
        )
        print(f"{gripper:34s} {config['fingertip'][2]:13.4f}m "
              f"{mesh.bounds[1][2]:13.4f}m {len(contacts):9d}")

    print(f"\n{len(names)} gripper(s) under {args.grippers}")
    print("each is GraspGenX's view of this hand doing one thing; "
          "run predict_grasps.py once per type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
