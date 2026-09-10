"""Build a self-contained URDF of the whole FFW-SH5 (AI-Worker), for GraspGenX.

Same treatment as :mod:`export_hand_urdf` but for the full robot, because the
hand alone was only ever a stand-in: the UR10e it was bolted to is not our arm.
ROBOTIS's ``ffw_sh5_follower.urdf`` already carries the arms **and** the hands in
one file, so unlike the UR10e case there is nothing to merge -- the whole robot
comes in as one URDF and the right arm's chain runs straight into the right
hand's twenty joints.

What has to change before Newton and cuRobo can load it
-------------------------------------------------------
* ``package://ffw_description/...`` resolves only inside a built ROS 2
  workspace. Rewritten to ``meshes/<subdir>/<basename>``. The subdirectory has to
  survive: the left and right HX5-D20 hands are mirrored copies that reuse the
  same nine filenames, so flattening to basenames alone would silently give one
  hand the other's geometry.
* ``${swerve_meshes_dir}`` is an unexpanded xacro variable that survived into the
  released URDF. It only ever names the three wheel-drive meshes, and it gets the
  same rewrite.
* ``file:///root/ros2_ws/install/realsense2_description/...`` points at a path
  inside ROBOTIS's own container for the two D405 camera meshes. Those meshes are
  not redistributed, so the two ``camera_*_link`` visuals are dropped rather than
  left dangling -- they are cosmetic and carry no collision role for planning.
* All 138 meshes carry ``scale="0.001"`` because the STLs are millimetres.
  GraspGenX's renderer drops that attribute (it writes only a mesh path and a
  transform into ``trajectory.json``), so the scale is **baked into the STLs**
  and the attribute removed, exactly as ``export_hand_urdf`` does. Without this
  the robot renders 1000x oversized while the physics stays correct.
* ``<gazebo>`` and ``<ros2_control>`` blocks are stripped. They are simulation
  and hardware-interface metadata for Gazebo/ros2_control; yourdfpy and Newton
  ignore them, but they carry plugin references that make the file noisy to diff.

Scope: right arm only
---------------------
The robot has 63 actuated joints -- lift 1, head 2, left arm 7, right arm 7,
left hand 20, right hand 20, wheels 6. Only ``arm_r_joint1..7`` and
``finger_r_joint_a..t`` (27) are driven; see :data:`RIGHT_ARM_JOINTS` and
:data:`RIGHT_HAND_JOINTS`. **Geometry for the rest is kept** -- the left arm and
the torso are exactly what the right arm must not collide with -- so the other
36 joints are locked in the cuRobo config rather than deleted here.

The base is fixed. cuRobo plans for a fixed base, and the six wheel joints are
not something a grasp trajectory should be steering.

The lift is fixed too, at :data:`LIFT_HEIGHT`. ``lift_joint`` is prismatic along
Z over [-0.5, 0], and **0 is the top of its travel** -- the pose the robot ships
in. Measured over 3000 random right-arm configurations, only 21 put the palm
anywhere useful above a 0.70 m table at that height; at -0.4 it is several times
that, because the whole arm base drops with it. Baking the value in rather than
locking it in the cuRobo config keeps the planner and the Newton replay in
agreement by construction: a locked joint is invisible to the URDF that Newton
imports, so it would start the simulation at 0 while cuRobo planned at -0.4.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402

#: Mesh reference forms that appear in the released file.
PACKAGE_PREFIX = "package://ffw_description/meshes/common/"
XACRO_VAR = "${swerve_meshes_dir}/"

#: The two camera meshes ROBOTIS references by absolute container path.
UNRESOLVABLE = "file:///root/ros2_ws/"

#: Where the lift is parked, or ``None`` to leave ``lift_joint`` prismatic.
#: Leave it free while choosing a value in ``scene_setup_gui.py``; park it once
#: the value is settled, so the planner and the Newton replay cannot disagree.
LIFT_HEIGHT = None

MESH_SOURCE = Path("model/assets")
MM_TO_M = 0.001
MESH_SCALE_RE = re.compile(r'\s*scale="[^"]*"')

#: The 27 joints this project drives. Everything else is locked, not removed.
RIGHT_ARM_JOINTS = tuple(f"arm_r_joint{i}" for i in range(1, 8))

#: The full robot names the hand's joints ``finger_r_joint1..20`` while ROBOTIS's
#: standalone ``hx5_d20_right.urdf`` -- the one GraspGenX was onboarded with, and
#: the one the wizard's ``config.json`` refers to -- names the same twenty joints
#: ``finger_r_joint_a..t``. Verified identical: same parent/child links in the
#: same order, same axes, same limits. So the two map by **chain position**,
#: never by name, and anything carrying a pose from the gripper config to the
#: full robot has to go through :data:`HAND_JOINT_ALIAS`.
RIGHT_HAND_JOINTS = tuple(f"finger_r_joint{i}" for i in range(1, 21))

#: ``finger_r_joint_a`` -> ``finger_r_joint1``, and so on.
HAND_JOINT_ALIAS = {
    f"finger_r_joint_{c}": f"finger_r_joint{i}"
    for i, c in enumerate("abcdefghijklmnopqrst", start=1)
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=paths.ROBOT_URDF_SOURCE)
    parser.add_argument("--meshes", type=Path, default=MESH_SOURCE)
    parser.add_argument("--out", type=Path, default=paths.ROBOT_URDF.parent)
    parser.add_argument(
        "--lift", type=float, default=LIFT_HEIGHT,
        help="Park lift_joint at this height (metres, range [-0.5, 0]) and make it "
             "fixed. Omit to leave it prismatic so a GUI can drive it.",
    )
    return parser.parse_args(argv)


def _resolve(ref: str, mesh_root: Path) -> tuple[str, Path]:
    """Map one URDF mesh reference to ``(output subpath, local source file)``.

    The left/right hands share filenames, so a reference under ``hx5_d20_left``
    or ``hx5_d20_right`` is resolved inside that directory and keeps it in the
    output path. Everything else is unique by basename and lands in ``ffw_s/``.
    """
    name = ref.rsplit("/", 1)[-1]
    for side in ("hx5_d20_left", "hx5_d20_right"):
        if side in ref:
            return f"{side}/{name}", mesh_root / "hx5_d20" / side / name
    return f"ffw_s/{name}", mesh_root / "ffw_s" / name


def _park_lift(root: ET.Element, height: float) -> None:
    """Fix ``lift_joint`` at ``height`` by folding the travel into its origin.

    Raises:
        ValueError: when the joint is missing or the height is outside its range,
            rather than silently parking the arm somewhere it cannot go.
    """
    for joint in root.findall("joint"):
        if joint.get("name") != "lift_joint":
            continue
        limit = joint.find("limit")
        low, high = float(limit.get("lower")), float(limit.get("upper"))
        if not low <= height <= high:
            raise ValueError(f"lift height {height} outside the joint's range [{low}, {high}]")
        origin = joint.find("origin")
        x, y, z = (float(v) for v in origin.get("xyz").split())
        origin.set("xyz", f"{x} {y} {z + height}")
        joint.set("type", "fixed")
        joint.remove(limit)
        for tag in ("axis", "dynamics"):
            element = joint.find(tag)
            if element is not None:
                joint.remove(element)
        return
    raise ValueError("no lift_joint in the URDF to park")


def _drop_unresolvable_visuals(root: ET.Element) -> list[str]:
    """Remove visual/collision elements whose mesh cannot be resolved."""
    dropped = []
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                mesh = element.find("geometry/mesh")
                if mesh is not None and UNRESOLVABLE in (mesh.get("filename") or ""):
                    link.remove(element)
                    dropped.append(f"{link.get('name')}/{tag}")
    return dropped


def rewrite(source: Path, out_dir: Path, mesh_root: Path,
            lift: float | None = None) -> tuple[Path, list[str], list[str]]:
    """Write ``ffw_sh5.urdf`` and ``meshes/`` into ``out_dir``."""
    text = source.read_text()
    refs = sorted({m for m in re.findall(r'filename="([^"]+\.stl)"', text)
                   if UNRESOLVABLE not in m})
    resolved = {ref: _resolve(ref, mesh_root) for ref in refs}
    missing = [ref for ref, (_, src) in resolved.items() if not src.is_file()]
    if missing:
        raise FileNotFoundError(f"{mesh_root} is missing {len(missing)} mesh(es): {missing}")

    mesh_dir = out_dir / "meshes"
    wanted = []
    for ref, (sub, src) in resolved.items():
        dest = mesh_dir / sub
        dest.parent.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(src, force="mesh")
        mesh.apply_scale(MM_TO_M)
        dest.write_bytes(trimesh.exchange.stl.export_stl(mesh))
        text = text.replace(f'filename="{ref}"', f'filename="meshes/{sub}"')
        wanted.append(sub)

    # Drop the scales now baked into the files themselves.
    text = MESH_SCALE_RE.sub("", text)

    root = ET.fromstring(text)
    if lift is not None:
        _park_lift(root, lift)
    dropped = _drop_unresolvable_visuals(root)
    for tag in ("gazebo", "ros2_control"):
        for element in root.findall(tag):
            root.remove(element)

    urdf_path = out_dir / "ffw_sh5.urdf"
    urdf_path.write_text(ET.tostring(root, encoding="unicode"))
    return urdf_path, wanted, dropped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source.exists():
        raise FileNotFoundError(
            f"no vendored URDF at {args.source.resolve()}. It comes from "
            f"ROBOTIS-GIT/ai_worker, "
            f"ffw_description/urdf/ffw_sh5_rev1_follower/ffw_sh5_follower.urdf"
        )
    urdf_path, meshes, dropped = rewrite(args.source, args.out, args.meshes, args.lift)

    root = ET.parse(urdf_path).getroot()
    actuated = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    driven = [n for n in actuated if n in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS]
    print(f"wrote {urdf_path}")
    print(f"  links {len(root.findall('link'))}  actuated joints {len(actuated)}")
    print(f"  driven (right arm + right hand): {len(driven)}   to lock: {len(actuated) - len(driven)}")
    print(f"  lift: {'parked at %.2f m (fixed)' % args.lift if args.lift is not None else 'left prismatic'}")
    print(f"  meshes converted to metres: {len(meshes)}")
    if dropped:
        print(f"  dropped unresolvable visuals: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
