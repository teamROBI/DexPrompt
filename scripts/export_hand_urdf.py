"""Build a self-contained URDF of the right HX5-D20 hand, for GraspGenX.

GraspGenX onboards a new gripper from a **URDF**, and this project carries MJCF.
Rather than convert the MJCF -- which loses the joint limits and inertials, and
invents collision geometry -- this takes ROBOTIS's own URDF for the same hand.
`ffw_description/urdf/common/hx5_d20/hx5_d20_right.urdf` is already expanded (not
xacro), is standalone, and references exactly the nine STLs this repo already has
under ``model/assets/hx5_d20/hx5_d20_right``. It is vendored as
``model/hx5_d20_right.urdf.orig``, following the same convention as
``ffw_sh5.xml.orig``. Both it and the menagerie MJCF are Apache-2.0.

Two things have to change before the wizard can load it:

* ``package://ffw_description/meshes/common/hx5_d20/hx5_d20_right/X.stl`` resolves
  only inside a built ROS 2 workspace. Rewritten to ``meshes/X.stl``.
* The wizard copies a ``meshes/`` directory sitting next to the URDF, so the STLs
  are placed there rather than referenced across the tree.

The URDF's **zero pose is the open pose** -- every flexion joint has 0 at one end
of its range -- so the wizard's step 2 ("confirm open configuration") needs no
slider work. The ``scale="0.001"`` on each mesh is kept as-is: the STLs are in
millimetres, exactly as the MJCF's own ``scale="0.001 0.001 0.001"`` assumes.

Note the joint names differ between the two models for the same twenty joints:
the MJCF calls them ``finger_r_joint1..20`` and the URDF ``finger_r_joint_a..t``.
Anything that carries a pose from one to the other has to map them by chain
position, not by name.
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

#: What the URDF asks for, and where this repo actually keeps it.
PACKAGE_PREFIX = "package://ffw_description/meshes/common/hx5_d20/hx5_d20_right/"
MESH_SOURCE = Path("model/assets/hx5_d20/hx5_d20_right")

#: Every curated GraspGenX gripper URDF starts with a bare ``world`` link joined
#: to the hand by a fixed ``world_joint``. Two things depend on it:
#: ``build_ur10e_gripper.py`` finds the link to bolt onto the arm's ``tool0`` by
#: looking for the joint whose parent is ``world``, and the joint's origin is the
#: offset that also goes into the robot YAML's ``grasp_to_tool_transform``
#: (``surge_hand`` uses the same numbers in both places).
#:
#: So ``world`` *is* the canonical grasp frame, and this origin is where the
#: hand's own base sits inside it. The aperture between the opposed thumb and
#: the four fingertips centres on x=+0.0204, y=-0.0004 in the URDF's frame --
#: measured off the FK at the open pose -- so putting the base at the negative of
#: that centres the aperture on the approach axis, which is how every curated
#: gripper is authored. Carry these numbers into
#: ``grasp_to_tool_transform`` when building the arm+hand robot config.
WORLD_JOINT_ORIGIN = (-0.0204, 0.0004, 0.0)
BASE_LINK = "hx5_d20_right_base"

#: ROBOTIS's STLs are in millimetres. See :func:`_write_metres`.
MM_TO_M = 0.001

#: ROBOTIS ships the same twenty joints under two names: the standalone
#: ``hx5_d20_right.urdf`` calls them ``finger_r_joint_a..t``, the full
#: ``ffw_sh5_follower.urdf`` calls them ``finger_r_joint1..20``. Same parent and
#: child links in the same order, same axes, same limits -- only the labels
#: differ. The full robot is what this project actually drives, so its names win
#: and the standalone hand is renamed to match. Keeping both conventions alive
#: is how a pose silently lands on the wrong joint.
JOINT_RENAME = {
    f"finger_r_joint_{c}": f"finger_r_joint{i}"
    for i, c in enumerate("abcdefghijklmnopqrst", start=1)
}
MESH_SCALE_RE = re.compile(r'\s*scale="[^"]*"')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=paths.HAND_URDF_SOURCE)
    parser.add_argument("--meshes", type=Path, default=MESH_SOURCE)
    parser.add_argument("--out", type=Path, default=paths.HAND_URDF.parent)
    return parser.parse_args(argv)


def rewrite(source: Path, out_dir: Path, mesh_source: Path) -> tuple[Path, list[str]]:
    """Write ``gripper.urdf`` and ``meshes/`` into ``out_dir``.

    Returns the URDF path and the mesh basenames copied.

    Raises:
        FileNotFoundError: when a mesh the URDF names is not in ``mesh_source``.
            Copying only what happens to be there would produce a URDF that loads
            with links silently missing their geometry.
    """
    text = source.read_text()
    wanted = sorted(set(re.findall(re.escape(PACKAGE_PREFIX) + r"([\w.]+\.stl)", text)))
    if not wanted:
        raise ValueError(f"{source} references no meshes under {PACKAGE_PREFIX!r}")

    missing = [name for name in wanted if not (mesh_source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{mesh_source} is missing {len(missing)} mesh(es): {missing}")

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        _write_metres(mesh_source / name, mesh_dir / name)

    text = text.replace(PACKAGE_PREFIX, "meshes/")
    text = MESH_SCALE_RE.sub("", text)
    # Longest names first: renaming _a before _t is fine, but a bare prefix
    # replace would turn "finger_r_joint_a" into "finger_r_joint1" *inside*
    # an already-renamed string if the order let it.
    for old in sorted(JOINT_RENAME, key=len, reverse=True):
        text = text.replace(f'"{old}"', f'"{JOINT_RENAME[old]}"')
    urdf_path = out_dir / "gripper.urdf"
    urdf_path.write_text(_add_world_link(text))
    return urdf_path, wanted


def _write_metres(source: Path, dest: Path) -> None:
    """Copy an STL, converting millimetres to metres.

    ROBOTIS ships these meshes in millimetres and carries the conversion in the
    URDF as ``scale="0.001 0.001 0.001"``. That is valid URDF and MuJoCo, cuRobo
    and Newton all honour it -- but GraspGenX's **renderer** does not. Its
    exporter writes only ``mesh_rel`` and ``transform`` into ``trajectory.json``
    (see ``export_trajectory`` in ``e2e_grasp_demo.py``), and
    ``render_trajectory_mp4.py`` then ``trimesh.load``s that path and applies the
    transform alone. The scale is dropped, so the hand renders 1000x oversized --
    a giant ring of teeth swallowing the scene -- while the physics stays correct.
    Every curated gripper ships meshes already in metres, which is why the bug has
    never surfaced upstream.

    Baking the conversion in and dropping the attribute makes the URDF, the
    physics and the renderer agree, and matches how the curated grippers are
    packaged.
    """
    mesh = trimesh.load(source, force="mesh")
    mesh.apply_scale(MM_TO_M)
    dest.write_bytes(trimesh.exchange.stl.export_stl(mesh))


def _add_world_link(text: str) -> str:
    """Prepend the ``world`` link and ``world_joint`` described above.

    Done as a string insert rather than an ElementTree rewrite so the rest of
    ROBOTIS's file -- comments, formatting, attribute order -- survives
    untouched and a diff against ``.orig`` stays readable.

    Raises:
        ValueError: when the anchor is not found, rather than returning the
            input unchanged and letting the arm merge fail later with a much
            less obvious message.
    """
    anchor = f'<link name="{BASE_LINK}">'
    if anchor not in text:
        raise ValueError(f"cannot find {anchor!r} to insert the world link before")
    x, y, z = WORLD_JOINT_ORIGIN
    block = (
        '<link name="world"/>\n'
        '   <joint name="world_joint" type="fixed">\n'
        '      <parent link="world"/>\n'
        f'      <child link="{BASE_LINK}"/>\n'
        f'      <origin xyz="{x} {y} {z}" rpy="0 0 0"/>\n'
        '   </joint>\n'
        f'   {anchor}'
    )
    return text.replace(anchor, block, 1)


def summarise(urdf_path: Path) -> tuple[int, int]:
    """Actuated joints and links, read back from what was written."""
    root = ET.parse(urdf_path).getroot()
    actuated = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    return len(actuated), len(root.findall("link"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source.exists():
        raise FileNotFoundError(
            f"no vendored URDF at {args.source.resolve()}. It comes from "
            f"ROBOTIS-GIT/ai_worker, ffw_description/urdf/common/hx5_d20/hx5_d20_right.urdf"
        )

    urdf_path, meshes = rewrite(args.source, args.out, args.meshes)
    joints, links = summarise(urdf_path)
    print(f"wrote {urdf_path}  ({joints} actuated joints, {links} links)")
    print(f"copied {len(meshes)} meshes into {args.out / 'meshes'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
