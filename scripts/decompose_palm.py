"""Split the palm mesh into convex parts, so MuJoCo can collide against its real shape.

MuJoCo's narrow phase requires convex geometry, so a ``<geom type="mesh">`` is always
replaced by that mesh's convex hull -- for this palm, 583 hull faces standing in for
41,332 mesh faces, and a hull of **475.9 cm3** standing in for a part whose real volume is
**172.3 cm3**. The hull fills the hollow that fingers are meant to curl into, so the
fingers collide with geometry the real hand does not have. Measured evidence: 12 palm
contacts with the hand fully *extended*, and a fist that stops 0.22 rad short with ten of
twenty joints saturated at their 2 Nm limit.

There is no setting that fixes this. MuJoCo 3.12 ships ``libsdf_plugin.so``, whose
signed-distance collisions would handle concave shapes, but the plugin it registers are
analytic solids -- ``nut``, ``bolt``, ``torus``, ``gear`` -- and ``mujoco.sdf.sdflib``,
the one that would take an arbitrary mesh, is not in this build.

So the palm is decomposed into convex pieces whose union approximates the concave shape.
Each piece is exactly representable, which is the closest thing to "collide against the
mesh faces" that the engine allows.

Only the **collision** geom is replaced. The visual geom keeps the original mesh, because
nothing is wrong with how the palm looks.

    .venv/bin/python scripts/decompose_palm.py --threshold 0.03
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coacd
import numpy as np
import trimesh


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/ffw_sh5.xml"))
    parser.add_argument(
        "--side", default="r", choices=["r", "l", "both"], help="which palm to decompose"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="CoACD concavity threshold. Lower means more parts and a closer fit; the "
        "cost is collision pairs, so this is the accuracy/speed dial.",
    )
    parser.add_argument("--max-parts", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def hull_volume(mesh: trimesh.Trimesh) -> float:
    return float(mesh.convex_hull.volume)


def decompose(stl: Path, out_dir: Path, stem: str, threshold: float, max_parts: int):
    """CoACD the mesh, write one STL per part, return (parts, stats)."""
    mesh = trimesh.load_mesh(stl, process=False)
    original_volume = float(abs(mesh.volume))
    original_hull = hull_volume(mesh)

    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=threshold,
        max_convex_hull=max_parts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    names, union_volume = [], 0.0
    for index, (vertices, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        # Each piece is already convex, so its hull volume *is* its volume; summing them
        # overcounts wherever pieces overlap, which is why this is reported as an upper
        # bound rather than as "the volume".
        union_volume += float(abs(piece.volume))
        name = f"{stem}_part{index:02d}"
        piece.export(out_dir / f"{name}.stl")
        names.append(name)

    # The STL is in millimetres -- the MJCF applies scale="0.001" -- so volumes convert
    # with 1e-3, not the 1e6 that a metre-unit mesh would need. Reported in cm3 for
    # comparison with the palm's real 172.3 cm3.
    to_cm3 = 1e-3
    return names, {
        "mesh_volume_cm3": round(original_volume * to_cm3, 1),
        "single_hull_cm3": round(original_hull * to_cm3, 1),
        "parts": len(names),
        # Summed, so overlaps are counted twice: an upper bound on the union, not the
        # union. The measurement that decides whether this helped is the hand's own
        # behaviour, not this ratio.
        "parts_volume_sum_cm3": round(union_volume * to_cm3, 1),
        "hull_inflation": round(original_hull / original_volume, 2),
        "parts_inflation_upper_bound": round(union_volume / original_volume, 2),
    }


def patch_model(model_path: Path, side: str, part_names: list[str], mesh_dir: str) -> None:
    """Swap the palm's collision geom for the parts, and declare their meshes.

    Edits the XML as text on purpose: the file is ROBOTIS's, and a targeted two-line
    substitution is reviewable against their original in a way a rewritten tree is not.
    """
    text = model_path.read_text()
    palm = f"hx5_{side}_base"

    asset_line = f'    <mesh name="{palm}" '
    if asset_line not in text:
        raise ValueError(f"no mesh asset line for {palm} in {model_path}")
    declarations = "".join(
        f'    <mesh name="{name}" file="{mesh_dir}/{name}.stl" scale="0.001 0.001 0.001"/>\n'
        for name in part_names
    )
    marker = "<!-- palm collision parts -->"
    if marker not in text:
        index = text.index(asset_line)
        text = text[:index] + f"    {marker}\n" + declarations + text[index:]

    collision = f'                          <geom type="mesh" mesh="{palm}" class="collision"/>'
    if collision not in text:
        raise ValueError(
            f"could not find the palm collision geom for {palm}. It may already be "
            f"replaced, or the file's indentation changed."
        )
    replacement = "\n".join(
        f'                          <geom type="mesh" mesh="{name}" class="collision"/>'
        for name in part_names
    )
    text = text.replace(collision, replacement)
    model_path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sides = ["r", "l"] if args.side == "both" else [args.side]

    for side in sides:
        stl = Path(
            f"model/assets/hx5_d20/hx5_d20_{'right' if side == 'r' else 'left'}/"
            f"hx5_d20_base_unit.stl"
        )
        if not stl.exists():
            raise FileNotFoundError(stl)
        out_dir = stl.parent / "convex"
        stem = f"hx5_{side}_base"
        names, stats = decompose(stl, out_dir, stem, args.threshold, args.max_parts)
        print(f"{side} palm: {stats}")
        print(f"  wrote {len(names)} parts to {out_dir}")

        if args.dry_run:
            continue
        backup = args.model.with_suffix(".xml.orig")
        if not backup.exists():
            shutil.copy2(args.model, backup)
            print(f"  kept the original at {backup}")
        rel = f"hx5_d20/hx5_d20_{'right' if side == 'r' else 'left'}/convex"
        patch_model(args.model, side, names, rel)
        print(f"  patched {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
