"""Turn a mesh into the assets Dexonomy loads for one object.

Dexonomy does not take a bare mesh. Its object loader wants three things beside
each other:

* a **MuJoCo XML** of *convex* pieces, because MuJoCo only collides convex
  geometry -- ``_add_rigid_object`` calls ``MjSpec.from_file(xml_path)`` and
  attaches its worldbody, so a concave mesh dropped in whole would collide as
  its own hull and grasps would be scored against a shape the object does not
  have. CoACD does the decomposition.
* the **original mesh** at ``file_path``, which the loader reads with trimesh
  for the centre of mass and for the surface samples that ``op=init`` uses as
  candidate contact points.
* a **scene config** naming both, plus the task, the object's pose, and the
  **table** it stands on;
* an **info JSON** holding ``mass``, ``density`` and ``scale``. ``op=eval`` reads
  it only to recover the mesh's volume -- ``mass / (density * scale ** 3)`` -- and
  then rewrites the density so the object weighs whatever the run's ``obj_mass``
  says. Writing a consistent triple is all that is asked of it.

This was previously buried inside ``dexonomy_from_graspgen.py``, which existed to
replace ``op=init`` with GraspGenX's pose predictions. That replacement is gone;
the asset half is not GraspGenX-specific and is what ``op=init`` reads.

The table
---------
An object floating in space can be grasped from underneath, and roughly half of
the approach directions ``op=init`` samples do exactly that. Nothing downstream
wants those: the robot works at a table, and a grasp that reaches through it is
not a grasp. So the object is stood on a plane at *z = 0* and the plane goes into
the scene, where ``op=init``'s ``check_plane`` filter keeps the hand skeleton
above it and MuJoCo collides against it in the later stages.

The plane is at the origin rather than at the scene's table height. Dexonomy
works in the object's own frame and the grasps it produces are poses relative to
the object, so where the table sits in the robot's world is a separate transform
applied when the grasp is placed in the scene -- and keeping it out of here means
the same grasps survive moving the table.

Run with Dexonomy's interpreter, which is the one with coacd::

    third_party/dexonomy/.venv/bin/python scripts/build_object_assets.py \
        --mesh path/to/object.obj --scale 0.001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import paths  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, nargs="+")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Mesh units to metres. 0.001 for a mesh authored in millimetres.",
    )
    parser.add_argument(
        "--upright", action="store_true", default=True,
        help="Turn the mesh a quarter turn about X before standing it up, which "
             "is what HOPE-style meshes need to be the way up they are used.",
    )
    parser.add_argument("--no-upright", dest="upright", action="store_false")
    parser.add_argument(
        "--floating", action="store_true",
        help="Leave the object in space with no table, the way Dexonomy's own "
             "force-closure scenes are set up.",
    )
    parser.add_argument(
        "--out", type=Path, default=paths.DEXONOMY / "assets/object/icra",
        help="One subdirectory per object is written here.",
    )
    return parser.parse_args(argv)


def build_object(
    mesh_path: Path, scale: float, out_dir: Path, name: str,
    upright: bool = True, on_table: bool = True,
) -> Path:
    """Write the CoACD-decomposed MJCF, the source mesh and the scene config.

    Returns the scene config path, which is what ``op=init`` globs for.
    """
    import coacd
    import trimesh
    import trimesh.transformations as tra

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meshes").mkdir(exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale)
    if upright:
        # HOPE meshes lie on their side in their canonical frame; the same
        # quarter turn scene_setup_gui.py applies stands them up.
        mesh.apply_transform(tra.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if on_table:
        # Resting on the plane, not floating above or sunk into it.
        mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])
    # Everything downstream expresses grasps about this mesh's origin, so the
    # placement is baked once, here, and never applied twice.
    mesh.export(out_dir / "mesh.obj")

    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces))
    assets, geoms = [], []
    for index, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        piece.export(out_dir / "meshes" / f"part_{index}.obj")
        assets.append(f'    <mesh name="part_{index}" file="part_{index}.obj"/>')
        # Named, and this is not cosmetic. ``set_obj_margin`` only touches geoms
        # whose name starts with the environment's object prefix, and attaching
        # prefixes existing names -- an unnamed geom stays unnamed and is skipped.
        # Its margin then keeps whatever the environment was constructed with,
        # which for op=grasp is the *pregrasp* distance of 45 mm: MuJoCo reports
        # contact while the hand is 45 mm away, every filter passes, and the
        # grasp holds nothing when a fresh environment evaluates it.
        geoms.append(
            f'      <geom name="part_{index}" type="mesh" mesh="part_{index}" group="3"/>'
        )

    (out_dir / "coacd.xml").write_text(
        f'<mujoco model="{name}">\n'
        '  <compiler meshdir="meshes"/>\n'
        "  <asset>\n" + "\n".join(assets) + "\n  </asset>\n"
        "  <worldbody>\n" + "\n".join(geoms) + "\n  </worldbody>\n"
        "</mujoco>\n"
    )

    # op=eval divides mass by density and by scale cubed to recover the volume
    # it was measured at, so any consistent triple works; this states the mesh's
    # own volume at unit scale.
    density = 1000.0
    (out_dir / "info.json").write_text(
        json.dumps(
            {"mass": density * float(mesh.volume), "density": density, "scale": 1.0},
            indent=4,
        )
    )

    scene_cfg = {
        "scene_id": name,
        # Force closure: the grasp is judged on the hand's grip alone, which is
        # what op=grasp refines toward. The table is a constraint on how the hand
        # may get there, not part of what holds the object.
        "task": {"type": "force_closure", "obj_name": name},
        "scene": {
            name: {
                "type": "rigid_object",
                "xml_path": "coacd.xml",
                "file_path": "mesh.obj",
                "info_path": "info.json",
                # Per-axis, not a scalar: op=init reshapes it to (batch, 1, 1, 3)
                # and a scalar fails there rather than broadcasting.
                "scale": [1.0, 1.0, 1.0],
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
        },
    }
    if on_table:
        scene_cfg["scene"]["table"] = {
            "type": "plane",
            "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "size": [0.0, 0.0, 1.0],
        }
    scene_path = out_dir / "scene_cfg.npy"
    np.save(scene_path, scene_cfg, allow_pickle=True)
    return scene_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for mesh_path in args.mesh:
        name = mesh_path.stem
        scene = build_object(
            mesh_path, args.scale, args.out / name, name,
            upright=args.upright, on_table=not args.floating,
        )
        print(f"  {name}  ->  {scene}")
    where = "floating" if args.floating else "standing on a table plane at z = 0"
    print(f"\n{len(args.mesh)} object(s) {where}, under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
