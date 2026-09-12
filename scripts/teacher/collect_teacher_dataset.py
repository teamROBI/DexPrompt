# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Prepare teacher candidates or collect audited home-to-hold demonstrations.

No training is performed. Only schema-3 episodes passing measured trajectory
quality gates enter episodes/. Failures remain in diagnostics/ and attempts.json.
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
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from collect_episodes import Recorder, in_object_frame  # noqa: E402
from collect_episodes import parse_args as recorder_args  # noqa: E402

from icra import objects, paths, scene  # noqa: E402
from icra.contact_control import right_contact_digits  # noqa: E402
from icra.dataset import QualityLimits, file_sha256, finalize_dataset, write_episode  # noqa: E402
from icra.execution import ExecutionConfig, TeacherExecutor  # noqa: E402

SOURCES = (
    "scripts/teacher/collect_teacher_dataset.py",
    "scripts/teacher/collect_episodes.py",
    "src/icra/teacher/execution.py",
    "src/icra/sim/simulation.py",
    "src/icra/teacher/contact_control.py",
    "src/icra/data/dataset.py",
    "src/icra/vision/conditioning.py",
    "src/icra/sim/scene.py",
    "src/icra/robots/arm.py",
    "src/icra/sim/objects.py",
    "src/icra/vision/visual_observation.py",
)
DIRECTIONS = {"x": (0, 1), "-x": (0, -1), "y": (1, 1), "-y": (1, -1), "z": (2, 1), "-z": (2, -1)}


def dump(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def serialize(candidate):
    return {
        key: value.tolist()
        if isinstance(value, np.ndarray)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in candidate.__dict__.items()
    }


def select(args):
    from icra.grasp_env import reachable_yaws, read_candidates

    if args.out.exists():
        raise FileExistsError(args.out)
    verdicts: dict[str, str] = {}
    candidates = read_candidates(args.grasps, args.objects, limit=args.limit,
                                 report=verdicts)
    if args.screen_yaws:
        candidates = reachable_yaws(candidates, scene=scene.load(args.scene))
    rows = [serialize(candidate) for candidate in candidates]
    for row in rows:
        row["source_sha256"] = file_sha256(row["source"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(
        args.out,
        {
            "schema_version": 1,
            "teacher": "Dexonomy",
            "candidates": rows,
            "scope": "static candidate filters; full dynamic paths remain unvalidated",
        },
    )
    # One verdict per grasp file, beside the manifest: a candidate that never
    # reaches execution should say where it was dropped rather than vanish.
    dump(args.out.with_suffix(".verdicts.json"), verdicts)
    print(json.dumps({"selected": len(rows), "manifest": str(args.out),
                      "verdicts": str(args.out.with_suffix(".verdicts.json"))}), flush=True)


def teacher_scene_object(path: Path) -> str | None:
    """The object a Dexonomy scene configuration was built for, read off its path.

    DGN keeps one configuration per scale under the set root, as
    ``<set>/scene_cfg/<object>/<floating|tabletop>/scaleNNN.npy``; the flat sets
    (HOPE, the earlier ``icra`` assets) keep a single ``scene_cfg.npy`` beside the
    mesh, as ``<object>/scene_cfg.npy``. Reading only the first layout refused
    every flat-set teacher as belonging to an object called ``objects``.
    """
    if path.name == "scene_cfg.npy":
        return path.parent.name or None
    return path.parents[1].name if len(path.parents) >= 2 else None


def checked_candidate(row):
    """Bind original teacher labels and pose to the actual imported source/scale."""
    if not isinstance(row, dict):
        raise ValueError("each candidate must be a metadata dictionary")
    row = dict(row)
    for key in ("source", "directory", "object_name", "template"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"candidate {key} must be a nonempty string")
    source, directory = Path(row["source"]), Path(row["directory"])
    if not source.is_file() or not directory.is_dir():
        raise ValueError("teacher source or object assets are unavailable")
    digest = file_sha256(source)
    if row.get("source_sha256", digest) != digest:
        raise ValueError("teacher source hash differs from candidate selection")
    raw = np.load(source, allow_pickle=True).item()  # trusted local Dexonomy exports
    if not isinstance(raw, dict):
        raise ValueError("teacher export must contain one metadata dictionary")
    if raw.get("tmpl_name") != row["template"] or raw.get("hand_name") != "hx5_d20":
        raise ValueError("teacher grasp type and HX5 hand identity must match the original source")
    teacher_scene = Path(str(raw.get("scene_path", "")))
    if teacher_scene_object(teacher_scene) != row["object_name"]:
        raise ValueError("teacher source scene identifies a different target object")
    for key in ("grasp_qpos", "squeeze_qpos"):
        values = np.asarray(raw[key], float)
        if values.shape not in ((27,), (1, 27)):
            raise ValueError(f"teacher {key} must contain exactly one 27D configuration")
        actual = values.reshape(27)
        declared = np.asarray(row[key], float)
        if declared.shape != (27,) or not np.isfinite(actual).all():
            raise ValueError(f"invalid original teacher {key}")
        if not np.allclose(declared, actual, rtol=0, atol=1e-7):
            raise ValueError(f"manifest {key} has drifted from original teacher")
        if abs(np.linalg.norm(actual[3:7]) - 1.0) > 1e-4:
            raise ValueError("teacher root quaternion must be normalized wxyz")
        raw[key] = actual.reshape(1, 27)
    grasp_root, squeeze_root = (raw[key][0, :7] for key in ("grasp_qpos", "squeeze_qpos"))
    grasp_quat = grasp_root[3:] / np.linalg.norm(grasp_root[3:])
    squeeze_quat = squeeze_root[3:] / np.linalg.norm(squeeze_root[3:])
    if np.dot(grasp_quat, squeeze_quat) < 0:
        squeeze_quat = -squeeze_quat
    if (not np.allclose(grasp_root[:3], squeeze_root[:3], rtol=0, atol=1e-7)
            or not np.allclose(grasp_quat, squeeze_quat, rtol=0, atol=1e-7)):
        raise ValueError("executor does not support a changing grasp-to-squeeze root pose")
    right_contact_digits(row["required"])
    if row["required"] != [list(group) for group in raw["required_cbody"]]:
        raise ValueError("teacher contact annotations were changed in candidate manifest")
    asset = objects.load(directory)
    if asset is None:
        raise ValueError("object assets could not be loaded")
    stored = np.asarray(raw.get("obj_scale", [])).reshape(-1)
    if (
        stored.size not in (1, 3)
        or not np.isfinite(stored).all()
        or not np.allclose(stored, asset.scale, rtol=0, atol=1e-6)
    ):
        raise ValueError("teacher and executed object scales must match explicitly")
    if row["object_name"] != directory.name:
        raise ValueError("target object identity does not match geometry directory")
    opened = np.asarray(row["open_qpos"], float)
    if opened.shape != (20,) or not np.isfinite(opened).all():
        raise ValueError("calibrated 20-joint opening is required")
    row.update(
        source=str(source.resolve()), directory=str(directory.resolve()), source_sha256=digest
    )
    return row, raw, asset


def compiled_profile(r, body):
    m = r.model
    return {
        "mujoco_version": r.mj.__version__,
        "timestep": float(m.opt.timestep),
        "solver": int(m.opt.solver),
        "integrator": int(m.opt.integrator),
        "iterations": int(m.opt.iterations),
        "tolerance": float(m.opt.tolerance),
        "target_mass_kg": float(m.body_mass[body]),
        "actuator_kp": m.actuator_gainprm[r.actuators, 0].tolist(),
        "actuator_kv": (-m.actuator_biasprm[r.actuators, 2]).tolist(),
        "actuator_force_limits": m.actuator_forcerange[r.actuators].tolist(),
        "contact_solref": np.unique(m.geom_solref, axis=0).tolist(),
        "contact_solimp": np.unique(m.geom_solimp, axis=0).tolist(),
        "geom_friction": np.unique(m.geom_friction, axis=0).tolist(),
        "geom_condim": np.unique(m.geom_condim).tolist(),
        "force_semantics": "sum of nonnegative contact-normal forces per whole digit, N",
        "note": (
            "object anti-spin friction also mixes into finger-object contacts; defaults unchanged"
        ),
    }


def collect(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version", 1) != 1 or manifest.get("teacher", "Dexonomy") != "Dexonomy":
        raise ValueError("unsupported candidate selection schema or teacher")
    if not isinstance(manifest.get("candidates"), list):
        raise ValueError("candidate selection must contain a candidates list")
    rows = manifest["candidates"][: args.max_candidates or None]
    if not rows:
        raise ValueError("candidate manifest is empty")
    directions = [DIRECTIONS[name] for name in args.directions.split(",")]
    config = ExecutionConfig(
        **(json.loads(args.execution_config.read_text()) if args.execution_config else {})
    )
    limits = QualityLimits()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "simulation").mkdir()
    source_hashes = {}
    for name in SOURCES:
        dest = args.out / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        source_hashes[name] = file_sha256(dest)
    dump(
        args.out / "collection.json",
        {
            "candidate_manifest": str(args.manifest.resolve()),
            "candidate_manifest_sha256": file_sha256(args.manifest),
            "execution": asdict(config),
            "quality_limits": asdict(limits),
            "source_hashes": source_hashes,
            "training_started": False,
            "scene": scene.load(args.scene),
            "scope": (
                "bounded dynamic path and contact feedback; no general motion-planning guarantee"
            ),
        },
    )
    entries, attempts = [], []
    names = list(scene.ARM_JOINTS + scene.HAND_JOINTS)
    for index, candidate in enumerate(rows):
        try:
            row, raw, asset = checked_candidate(candidate)
        except (ValueError, TypeError, KeyError, OSError, EOFError) as error:
            attempts.append({"candidate_index": index, "accepted": False, "status": str(error)})
            dump(args.out / "attempts.json", attempts)
            continue
        r = None
        try:
            recording_args = recorder_args(
                ["--width", str(args.width), "--height", str(args.height)])
            recording_args.visual_profile = getattr(args, "visual_profile", None)
            r = Recorder(
                recording_args,
                scene.load(args.scene),
                [Path(row["directory"]), *args.distractor],
            )
            opened = np.asarray(row["open_qpos"])
            bounds = r.model.actuator_ctrlrange[r.actuators[7:]]
            if np.any(opened < bounds[:, 0]) or np.any(opened > bounds[:, 1]):
                raise ValueError("calibrated open command exceeds executed hand joint limits")
            part = None
            if args.part_annotation:
                import trimesh

                from icra.conditioning import PartRegion

                part = PartRegion.from_file(args.part_annotation, row["object_name"])
                geometry_bounds = trimesh.load(asset.mesh, force="mesh").bounds * asset.scale
                part.validate_geometry(geometry_bounds, asset.scale)
            model_path = args.out / "simulation" / f"model_{index:04d}.mjb"
            r.mj.mj_saveModel(r.model, str(model_path))
            model_hash = file_sha256(model_path)
            distractors = [r.object_body(directory.name) for directory in args.distractor]
            if len(set(distractors)) != len(distractors) or any(b < 0 for b in distractors):
                raise ValueError("distractors must identify distinct existing scene objects")
            yaws = args.yaw if args.yaw is not None else row.get("yaws") or [0.0]
            if not np.isfinite(yaws).all():
                raise ValueError("candidate yaws must be finite radians")
            for yaw_index, yaw in enumerate(yaws[: args.max_yaws or None]):
                identifier = f"teacher_{index:04d}_yaw_{yaw_index:03d}"
                print(
                    json.dumps(
                        {
                            "episode_id": identifier,
                            "status": "starting",
                            "object_id": row["object_name"],
                            "yaw_rad": float(yaw),
                        }
                    ),
                    flush=True,
                )
                executor = TeacherExecutor(
                    r, row, config=config, limits=limits, part=part, distractors=distractors
                )
                executor.initialize(in_object_frame(raw, offset=r.offset), float(yaw))
                initial_path = args.out / "simulation" / f"{identifier}_initial.npy"
                np.save(initial_path, executor.initial)
                executor.run(directions)
                evidence = executor.evidence()
                if r.visual_profile:
                    evidence["visual_profile"] = r.visual_profile
                evidence.update(
                    {
                        "source_hashes": source_hashes,
                        "geometry_hashes": {str(model_path.relative_to(args.out)): model_hash},
                        "compiled_profile": compiled_profile(r, executor.body),
                        "distractor_ids": [d.name for d in args.distractor],
                        "object_scale": asset.scale,
                        "teacher_root_offset_m": r.offset.tolist(),
                        "teacher_hand_xml_sha256": file_sha256(paths.DEXONOMY_HAND / "right.xml"),
                        "yaw_rad": float(yaw),
                        "initial_state": str(initial_path.relative_to(args.out)),
                        "initial_state_sha256": file_sha256(initial_path),
                        "model": str(model_path.relative_to(args.out)),
                        "initial_state_spec": "mjSTATE_INTEGRATION",
                    }
                )
                metadata = {
                    "object_id": row["object_name"],
                    "candidate_id": row["source_sha256"],
                    "grasp_type": row["template"],
                    "teacher": {
                        "name": "Dexonomy",
                        "source_path": row["source"],
                        "source_sha256": row["source_sha256"],
                    },
                    "required_contact_groups": row["required"],
                    "condition": part.condition
                    if part
                    else {"kind": "object", "object_id": row["object_name"]},
                    "execution": evidence,
                }
                arrays = executor.arrays()
                if arrays is None:
                    entry = {
                        "episode_id": identifier,
                        "accepted": False,
                        "reasons": [executor.status],
                        "path": None,
                    }
                    (args.out / "diagnostics").mkdir(exist_ok=True)
                    dump(args.out / "diagnostics" / f"{identifier}.json", metadata)
                else:
                    entry = write_episode(args.out, identifier, arrays, metadata, limits=limits)
                    entries.append(entry)
                attempt = {
                    **entry,
                    "status": executor.status,
                    "ticks": len(executor.frames),
                    "object_id": row["object_name"],
                    "yaw_rad": float(yaw),
                    "path_attempts": executor.path_attempts,
                }
                attempts.append(attempt)
                dump(args.out / "attempts.json", attempts)
                print(
                    json.dumps(
                        {
                            key: attempt[key]
                            for key in ("episode_id", "accepted", "status", "ticks", "path")
                        }
                    ),
                    flush=True,
                )
                # Preserve recovery entries without sealing the immutable final
                # manifest until collection has ended.
                dump(args.out / "episode_entries.json", entries)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            attempts.append(
                {
                    "candidate_index": index,
                    "object_id": row["object_name"],
                    "accepted": False,
                    "status": str(error),
                }
            )
            dump(args.out / "attempts.json", attempts)
            print(json.dumps(attempts[-1]), flush=True)
        finally:
            if r is not None:
                r.renderer.close()
    result = finalize_dataset(args.out, entries, names, split_by=args.split_by, seed=args.seed)
    summary = {
        "attempts": len(attempts),
        "accepted": sum(a["accepted"] for a in attempts),
        "manifest": str(args.out / "manifest.json"),
        "training_started": False,
        "training_ready": result["training_ready"],
        "split_by": args.split_by,
    }
    dump(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    s = subs.add_parser("select", help="filter existing Dexonomy exports; does not train teacher")
    s.add_argument("--grasps", type=Path, required=True)
    s.add_argument("--objects", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--limit", type=int, default=32)
    s.add_argument("--screen-yaws", action="store_true")
    s.add_argument("--scene", type=Path, default=paths.SCENE)
    c = subs.add_parser(
        "collect", help="home approach, contact closure, lift/hold and quality gate"
    )
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--scene", type=Path, default=paths.SCENE)
    c.add_argument("--max-candidates", type=int, default=0)
    c.add_argument("--max-yaws", type=int, default=1)
    c.add_argument("--yaw", type=float, action="append", help="explicit object yaw in radians")
    c.add_argument("--directions", default="x,-x,y,-y,z,-z")
    c.add_argument("--width", type=int, default=320)
    c.add_argument("--height", type=int, default=240)
    c.add_argument("--visual-profile", type=Path,
                   help="native resolution and head pose; overrides --width/--height")
    c.add_argument("--execution-config", type=Path)
    c.add_argument("--part-annotation", type=Path)
    c.add_argument("--distractor", type=Path, action="append", default=[])
    c.add_argument("--split-by", choices=("object", "candidate"), default="object")
    c.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "collect":
        if any(name not in DIRECTIONS for name in args.directions.split(",")):
            p.error("directions must be comma-separated signed axes x,-x,y,-y,z,-z")
        if min(args.width, args.height) < 16 or args.max_candidates < 0 or args.max_yaws < 0:
            p.error("image sizes must be >=16 and candidate/yaw limits nonnegative")
        if args.yaw and not np.isfinite(args.yaw).all():
            p.error("yaw values must be finite")
    elif args.limit < 1:
        p.error("candidate selection limit must be positive")
    {"select": select, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
