"""Replay stored commands and watch the object move inside the hand.

The archive keeps forces, contact groups and the object's height, which say that
a grasp came apart but not whether the object slid out of the fingers or the
fingers let go of it. This replays the same commands, from the same archived
model and initial state, and adds observation only: where the object sits and
how it is turned **in the hand's frame**, and for each digit its distance to the
object and the contact normal. Nothing about the controller, the contacts or the
criteria is changed, and the replayed forces and contact groups are compared
with the archive so the added signals belong to a faithful run.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/probe_grasp_slip.py \\
        --data out/bowl-scale/s008o4/executed --episode teacher_0000_yaw_000 \\
        --out out/bowl-drop/slip_bowl0000.npz
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_hx5_student import restore_recorder  # noqa: E402

from icra.dataset import load_episode, load_manifest, validate_dataset  # noqa: E402
from icra.execution import ContactMeter, aggregate  # noqa: E402
from icra.simulation import forward, restore  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args(argv)


def pose_in_hand(model, data, hand_body: int, object_body: int):
    """The object's position and orientation expressed in the hand's own frame."""
    hand_rotation = np.array(data.xmat[hand_body], dtype=float).reshape(3, 3)
    relative = hand_rotation.T @ (np.array(data.xpos[object_body], dtype=float)
                                  - np.array(data.xpos[hand_body], dtype=float))
    object_rotation = np.array(data.xmat[object_body], dtype=float).reshape(3, 3)
    return relative, hand_rotation.T @ object_rotation


def digit_geometry(model, data, object_body: int):
    """Per digit: nearest distance to the object and the mean contact normal."""

    distance = np.full(5, np.inf)
    normal = np.zeros((5, 3))
    counts = np.zeros(5)
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = [int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])]
        names = [model.body(b).name for b in bodies]
        for side in (0, 1):
            if not names[side].startswith("finger_r_link") or bodies[1 - side] != object_body:
                continue
            digit = (int(names[side].removeprefix("finger_r_link")) - 1) // 4
            if not 0 <= digit < 5:
                continue
            distance[digit] = min(distance[digit], float(contact.dist))
            sign = 1.0 if side == 0 else -1.0
            normal[digit] += sign * np.array(contact.frame[0:3], dtype=float)
            counts[digit] += 1
    for digit in range(5):
        if counts[digit]:
            length = np.linalg.norm(normal[digit])
            if length:
                normal[digit] /= length
    distance[~np.isfinite(distance)] = np.nan
    return distance, normal, counts


def main(argv: list[str] | None = None) -> int:

    args = parse_args(argv)
    validate_dataset(args.data)
    manifest = load_manifest(args.data, require_train=False)
    root = (args.data if args.data.is_dir() else args.data.parent).resolve()
    matches = [e for e in manifest["episodes"] + manifest["rejected"]
               if e["episode_id"] == args.episode]
    if len(matches) != 1:
        raise SystemExit("episode ID must select one recorded trajectory")
    entry = matches[0]
    recorder, initial, _ = restore_recorder(root, manifest, entry, render=False)
    archived = load_episode(root, entry)

    metadata = entry["provenance"]
    r = recorder
    body = r.object_body(metadata["object_id"])
    hand = r.model.body("hx5_r_base").id
    restore(r.model, r.data, initial)
    meter = ContactMeter(r, body, metadata["required_contact_groups"], [], None)

    series: dict[str, list] = {k: [] for k in
                               ("relative_position", "relative_rotation_deg",
                                "relative_rotation", "digit_distance", "digit_normal",
                                "digit_contacts", "contact_groups", "digit_force_max")}
    reference = None
    for tick, action in enumerate(archived["action"]):
        ratio = round(float(archived["control_dt"][tick]) / r.model.opt.timestep)
        r.command(action)
        samples = []
        for _ in range(ratio):
            r.mj.mj_step(r.model, r.data)
            forward(r.model, r.data)
            samples.append(meter.read())
        signals = aggregate(samples)
        position, rotation = pose_in_hand(r.model, r.data, hand, body)
        if reference is None:
            reference = (position.copy(), rotation.copy())
        turned = reference[1].T @ rotation
        angle = float(np.degrees(np.arccos(np.clip((np.trace(turned) - 1) / 2, -1, 1))))
        distance, normal, counts = digit_geometry(r.model, r.data, body)
        series["relative_position"].append((position - reference[0]) * 1000)
        series["relative_rotation_deg"].append(angle)
        # The orientation itself, so a rotation between any two ticks can be
        # composed properly: subtracting two angles-to-a-reference is not the
        # angle between them.
        series["relative_rotation"].append(rotation)
        series["digit_distance"].append(distance * 1000)
        series["digit_normal"].append(normal)
        series["digit_contacts"].append(counts)
        series["contact_groups"].append(signals["contact_groups"])
        series["digit_force_max"].append(signals["digit_force_max"])

    out = {k: np.asarray(v) for k, v in series.items()}
    fidelity = {
        "contact_group_mismatches": int(np.count_nonzero(
            out["contact_groups"] != archived["contact_groups"])),
        "max_digit_force_error_N": float(np.max(np.abs(
            out["digit_force_max"] - archived["digit_force_max"]))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"{args.episode}: {len(out['relative_rotation_deg'])} ticks  "
          f"contact-group mismatches {fidelity['contact_group_mismatches']}  "
          f"max digit-force error {fidelity['max_digit_force_error_N']:.2e} N")
    if args.summary:
        args.summary.write_text(json.dumps(
            {"episode": args.episode, "fidelity": fidelity,
             "output": str(args.out)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
