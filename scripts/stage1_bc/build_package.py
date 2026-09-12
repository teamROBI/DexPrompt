"""Stage-1 goal-conditioned BC package: the exact files one training run needs, read only.

Takes the HOPE episodes of the stage-1 data contract's default pool (physics A, single
object, current controller), and writes a staging folder of

``models/<sha>.mjb``         symlink to the archived compiled model (deduplicated by hash)
``initial/<sha>.npy``        symlink to the archived initial state
``arrays/<episode>.npz``     the episode arrays without images (state, action, time, phase,
                             per-tick signals), written fresh; the original npz is untouched
``episodes.jsonl``           per episode: identity, targets, initial poses, hashes, limits
``MANIFEST.sha256``          every file, for the transfer check on the other machine

The staging folder is copied with ``rsync -aL`` (symlinks followed). Nothing is simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "records/experiments/2026-09-12/stage1-bc-data-contract/episodes.jsonl"
HOPE = ("AlphabetSoup", "Peaches", "Raisins")
KEEP = ("right_state", "action", "timestamp", "control_dt", "phase", "contact_groups",
        "group_forces_min", "group_forces_max", "digit_force_max", "target_height_min",
        "object_xy_displacement_max", "non_target_displacement_max",
        "robot_object_penetration_max", "robot_table_penetration_max",
        "robot_obstacle_penetration_max")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def link(target: Path, source: str) -> None:
    """Named by content hash: an existing link to any copy of the same file is kept.

    Reproductions and hard-linked copies hold the same model under different paths; the
    content itself is checked against the name when MANIFEST.sha256 is written.
    """
    if target.is_symlink() or target.exists():
        return
    if not os.path.isfile(source):
        raise SystemExit(f"missing source {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out
    records = [json.loads(line) for line in CONTRACT.read_text().splitlines()]
    chosen = [r for r in records if r["pool"] == "default" and r["object_id"] in HOPE]
    rows = []
    for r in chosen:
        name = (f"{r['object_id']}-{r['candidate_id'][:10]}-{r['placement']}-"
                f"{r['episode_sha256'][:8]}")
        link(out / "models" / f"{r['model_sha256']}.mjb", r["model"])
        link(out / "initial" / f"{r['initial_state_sha256']}.npy", r["initial_state"])
        with np.load(r["episode"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in KEEP}
        compact = out / "arrays" / f"{name}.npz"
        compact.parent.mkdir(parents=True, exist_ok=True)
        if not compact.exists():
            np.savez_compressed(compact, **arrays)
        rows.append({
            "name": name, "episode_sha256": r["episode_sha256"], "object_id": r["object_id"],
            "candidate_id": r["candidate_id"], "grasp_type": r["grasp_type"],
            "placement": r["placement"], "candidate_group": r["candidate_group"],
            "model": f"models/{r['model_sha256']}.mjb", "model_sha256": r["model_sha256"],
            "initial_state": f"initial/{r['initial_state_sha256']}.npy",
            "initial_state_sha256": r["initial_state_sha256"],
            "arrays": f"arrays/{name}.npz", "arrays_sha256": sha(compact),
            "frames": r["checks"]["frames"],
            "required_contact_groups": r["dexonomy_target"]["required_contact_groups"],
            "distractors": r["distractors"],
            "initial_object_poses": r["initial_object_poses"],
            "initial_joint_qpos": r["initial_joint_qpos"],
            "scale": r["scale"], "mesh_sha256": r["mesh_sha256"],
            "dexonomy_target": {k: r["dexonomy_target"][k] for k in (
                "grasp_qpos", "squeeze_qpos", "pregrasp_qpos", "required_contact_groups",
                "template", "frame", "units", "source_sha256")},
            "executed_target": {k: r["executed_target"][k] for k in (
                "world_root", "teacher_root_offset_m", "finger_open_command",
                "finger_closure_target", "approach")},
            "quality_limits": r["quality"]["limits"],
            "past_evaluation_use": r["past_evaluation_use"],
            "initial_execution": None})
    (out / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        digest = sha(path)
        if path.parent.name in ("models", "initial") and path.stem != digest:
            raise SystemExit(f"{path} content hash {digest} differs from its name")
        lines.append(f"{digest}  {path.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps({"episodes": len(rows), "models": len(list((out / 'models').iterdir())),
                      "files": len(lines)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
