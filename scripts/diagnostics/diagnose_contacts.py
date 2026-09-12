"""Why a required contact is not made: mapping, or the optimisation.

Dexonomy rejects a refined grasp when the grasp type's required contact bodies
are not all touching the object. That verdict is the same whether the required
bodies were named wrongly, the contact points were transformed wrongly, the
contact test counted the wrong thing, or the hand simply never got there. This
separates those, on a fixed set of candidates, by checking the mapping first and
only then measuring residuals.

The mapping checks, per candidate:

* every required body name resolves to a body in the hand, and that body has
  collision geometry -- a name that resolves to nothing can never be reported as
  touching anything;
* the required set the initialisation carries is the one the template it was
  drawn from carries, and the one any refined output carries;
* the contact test counts only the target object -- proved by rebuilding the
  scene and confirming that hand/table contacts are absent from the hand-object
  set the refiner reads;
* the object-frame contact targets lie on the *scaled* object surface, which
  they cannot if the scale is applied twice or not at all.

Then, per required contact, at the pose as initialised and (where one exists) as
refined: how far the target contact point is from the link's own surface, the
angle between the target normal and the surface normal there, which bodies are
actually touching, how far the hand is into the object and the table, and how
close each joint is to its limit.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/diagnose_contacts.py \\
        --controls out/contact-diagnosis/controls.json \\
        --out out/contact-diagnosis/diagnosis.json
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
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "dexonomy"))

import numpy as np  # noqa: E402

FINGERS = {1: "digit 1", 2: "digit 2", 3: "digit 3", 4: "digit 4", 5: "digit 5"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True,
                        help="The recorded control groups.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.005,
                        help="Contact distance the refiner reads with, metres.")
    return parser.parse_args(argv)


def finger_of(name: str) -> int | None:
    """Which digit a link belongs to: four links each, numbered from one."""
    tail = name.removeprefix("finger_r_link")
    return (int(tail) - 1) // 4 + 1 if tail.isdigit() else None


def environment(scene_path: str):
    """The refiner's own environment for this candidate, built the same way."""
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    scene_cfg = load_scene_cfg(scene_path)
    # The same file config/hand/hx5_d20.yaml names, so the hand examined here
    # is the hand the refiner refined.
    hand_xml = str(ROOT / "third_party" / "dexonomy" / "assets" / "hand"
                   / "hx5_d20" / "right.xml")
    return MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=hand_xml, freejoint=True),
        scene_cfg=scene_cfg,
        sim_cfg=MuJoCo_OptCfg(obj_margin=0.0),
    ), scene_cfg


def body_report(env, names: list[str]) -> dict:
    """Does each required name name a body that could ever report a contact?"""
    model = env._model
    prefix = env._cfg.hand_prefix
    report = {}
    for name in names:
        try:
            body = model.body(prefix + name)
        except KeyError:
            report[name] = {"resolves": False}
            continue
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id]
        collidable = [g for g in geoms
                      if model.geom_contype[g] or model.geom_conaffinity[g]]
        report[name] = {
            "resolves": True, "body_id": int(body.id),
            "geoms": len(geoms), "collidable_geoms": len(collidable),
            "digit": finger_of(name),
        }
    return report


def surface_residuals(env, names, points) -> dict:
    """How far each contact target sits from its own link's collision surface.

    Measured with the refiner's own projection, so the number here is the one
    the optimisation was working against rather than a second opinion computed
    a different way.
    """
    import trimesh
    from dexonomy.op.gen_grasp_admm import create_hand_trimesh, project_to_hand

    meshes = create_hand_trimesh(env)
    proxies = [trimesh.proximity.ProximityQuery(m) if m is not None else None
               for m in meshes]
    points = np.asarray(points, dtype=float)
    projected, _ = project_to_hand(points.copy(), list(names), env, meshes,
                                   proxies, "collision")
    out = {}
    for index, name in enumerate(names):
        out[name] = {
            "distance_to_link_surface_mm":
                float(np.linalg.norm(projected[index] - points[index]) * 1000),
            "digit": finger_of(name),
        }
    return out


def object_surface_residual(scene_cfg, points) -> dict:
    """The object-frame targets against the scaled mesh: scale applied once?"""
    import trimesh

    scene = scene_cfg["scene"]
    name, entry = next((n, e) for n, e in scene.items()
                       if e["type"] == "rigid_object")
    mesh = trimesh.load_mesh(entry["file_path"])
    scaled = mesh.copy()
    scaled.apply_scale(entry["scale"])
    unscaled = mesh
    twice = mesh.copy()
    twice.apply_scale(np.asarray(entry["scale"]) ** 2)
    points = np.asarray(points)[:, :3]
    out = {"object": name, "scale": np.asarray(entry["scale"]).tolist()}
    for label, candidate in (("scaled_once", scaled), ("unscaled", unscaled),
                             ("scaled_twice", twice)):
        _, distance, _ = trimesh.proximity.closest_point(candidate, points)
        out[label + "_mm"] = {
            "median": float(np.median(distance) * 1000),
            "max": float(np.max(distance) * 1000),
        }
    return out


def contacts_now(env, threshold: float) -> dict:
    """What the refiner would read at the current pose, and what it excludes."""
    ho, hh = env.get_contacts(threshold)
    model = env._model
    everything = []
    for contact in env._data.contact:
        first = model.body(model.geom(contact.geom1).bodyid).name
        second = model.body(model.geom(contact.geom2).bodyid).name
        everything.append((first, second, float(contact.dist)))
    return {
        "hand_object_bodies": sorted(set(ho["bn1"])),
        "hand_object_pairs": len(ho["dist"]),
        "hand_hand_pairs": len(hh["dist"]),
        "min_hand_object_distance_mm":
            float(min(ho["dist"]) * 1000) if len(ho["dist"]) else None,
        "all_contact_pairs": len(everything),
        "pairs_touching_the_world":
            sorted({f"{a}|{b}" for a, b, _ in everything
                    if a == "world" or b == "world"}),
    }


def joint_margins(env) -> dict:
    """How much travel each hand joint has left before its limit."""
    model, data = env._model, env._data
    margins = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] == 3:  # hinge
            address = model.jnt_qposadr[j]
            low, high = model.jnt_range[j]
            value = float(data.qpos[address])
            margins.append((model.joint(j).name, min(value - low, high - value)))
    if not margins:
        return {}
    margins.sort(key=lambda row: row[1])
    return {
        "tightest": [{"joint": n, "margin_rad": float(m)} for n, m in margins[:5]],
        "at_limit_within_1e-3": sum(1 for _, m in margins if m < 1e-3),
        "joints": len(margins),
    }


def penetration(env) -> dict:
    """How far the hand is inside the object and inside the table."""
    model = env._model
    into_object, into_world = [], []
    for contact in env._data.contact:
        first = model.body(model.geom(contact.geom1).bodyid).name
        second = model.body(model.geom(contact.geom2).bodyid).name
        if contact.dist >= 0:
            continue
        if first == "world" or second == "world":
            into_world.append(-float(contact.dist))
        elif "finger_r" in first or "finger_r" in second:
            into_object.append(-float(contact.dist))
    return {
        "into_object_max_mm": float(max(into_object) * 1000) if into_object else 0.0,
        "into_world_max_mm": float(max(into_world) * 1000) if into_world else 0.0,
    }


def examine(record: dict, threshold: float) -> dict:
    """One candidate, at the pose it was initialised with and any refined pose."""
    initialisation = np.load(record["init_path"], allow_pickle=True).item()
    required = [list(group) for group in initialisation["required_cbody"]]
    hand_bodies = list(initialisation["hand_cbody"])
    env, scene_cfg = environment(initialisation["scene_path"])

    out = {
        "candidate": record,
        "template": str(initialisation["tmpl_name"]),
        "required_groups": required,
        "required_digits": sorted({finger_of(n) for g in required for n in g}),
        "hand_bodies_with_targets": hand_bodies,
        "mapping": body_report(env, sorted({n for g in required for n in g})),
        "object_targets_on_surface": object_surface_residual(
            scene_cfg, initialisation["obj_cpn_w"]),
    }

    poses = {"initialised": np.asarray(initialisation["grasp_qpos"])[0]}
    if record.get("refined_path"):
        refined = np.load(record["refined_path"], allow_pickle=True).item()
        poses["refined"] = np.asarray(refined["grasp_qpos"])[0]
        out["required_preserved_in_refined"] = (
            [list(g) for g in refined["required_cbody"]] == required)

    out["states"] = {}
    for label, qpos in poses.items():
        env.reset_qpos(qpos)
        import mujoco
        mujoco.mj_forward(env._model, env._data)
        state = {
            "contacts": contacts_now(env, threshold),
            "penetration": penetration(env),
            "joints": joint_margins(env),
        }
        state["contact_target_residuals"] = surface_residuals(
            env, hand_bodies, np.asarray(initialisation["hand_cpn_w"])[:, :3])
        touching = set(state["contacts"]["hand_object_bodies"])
        state["groups_satisfied"] = [bool(set(g) & touching) for g in required]
        state["groups_satisfied_count"] = int(sum(state["groups_satisfied"]))
        out["states"][label] = state
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controls = json.loads(args.controls.read_text())
    report = {"contact_threshold_m": args.contact_threshold, "candidates": []}
    for group in ("positive", "negative"):
        for record in controls[group]:
            row = examine(dict(record, control=group), args.contact_threshold)
            report["candidates"].append(row)
            state = row["states"].get("refined") or row["states"]["initialised"]
            unresolved = [n for n, m in row["mapping"].items() if not m["resolves"]]
            print(f"{group:<9}{record['object']:<10}{row['template']:<20}"
                  f"{Path(record['init_path']).name:<18}"
                  f"groups {state['groups_satisfied_count']}/{len(row['required_groups'])}"
                  f"  unresolved names {len(unresolved)}"
                  f"  targets on scaled surface (median mm) "
                  f"{row['object_targets_on_surface']['scaled_once_mm']['median']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
