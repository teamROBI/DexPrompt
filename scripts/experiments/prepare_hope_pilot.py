# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Build the six preselected HOPE pilot assets serially, without grasp generation."""

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
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("AlphabetSoup", "Peaches", "Mustard", "BBQSauce", "ChocolatePudding", "Raisins")


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "objects").mkdir()
    (args.out / "work").mkdir()
    inventory = json.loads((ROOT / "results/2026-09-10/hope-data-readiness/inventory.json")
                           .read_text())
    indexed = {row["object"]: row for row in inventory["objects"]}
    summary = {"training_started": False, "grasp_generation_started": False, "objects": []}
    for name in NAMES:
        source = ROOT / "out/hope-audit/source/full" / f"{name}.obj"
        row = {"object": name, "unit_to_m": 0.001, "upright_x_degrees": 90,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "status": "running", "timeout": False, "exit_code": None}
        if row["source_sha256"] != indexed[name]["full_mesh_sha256"]:
            raise ValueError(f"{name}: source changed since unit audit")
        summary["objects"].append(row)
        write(args.out / "build.json", summary)
        stage = args.out / "work" / name
        existing = ROOT / "third_party/dexonomy/assets/object/icra" / name
        start = time.monotonic()
        if existing.is_dir():
            row["method"] = "copy_existing"
            shutil.copytree(existing, stage)
            row["exit_code"] = 0
        else:
            row["method"] = "coacd_default_seed_0"
            command = [sys.executable, str(ROOT / "scripts/assets/build_object_assets.py"),
                       "--mesh", str(source), "--scale", "0.001", "--out",
                       str(args.out / "work")]
            row["command"] = command
            with (args.out / f"{name}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    row["exit_code"] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    row["timeout"] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    row["exit_code"] = process.returncode
        row["elapsed_seconds"] = time.monotonic() - start
        if row["exit_code"] == 0 and not row["timeout"]:
            import numpy as np
            import trimesh

            mesh = trimesh.load(stage / "mesh.obj", force="mesh")
            delta = float(np.max(np.abs(mesh.extents * 1000 - np.array(
                indexed[name]["upright_width_depth_height_mm"])) ))
            parts = sorted((stage / "meshes").glob("part_*.obj"))
            if delta > 0.001 or not parts or not (stage / "scene_cfg.npy").is_file():
                raise ValueError(f"{name}: incomplete or incorrectly scaled build")
            row.update(status="geometry_built", extent_error_mm=delta, parts=len(parts),
                       mesh_sha256=hashlib.sha256((stage / "mesh.obj").read_bytes()).hexdigest())
            stage.rename(args.out / "objects" / name)
        else:
            row["status"] = "timeout" if row["timeout"] else "build_failed"
        write(args.out / "build.json", summary)
        print(json.dumps(row), flush=True)
    summary["completed"] = True
    summary["all_geometry_built"] = all(row["status"] == "geometry_built"
                                        for row in summary["objects"])
    write(args.out / "build.json", summary)
    return 0 if summary["all_geometry_built"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
