"""Fix the digit-5 flexion experiment's conditions, with hashes, before it runs.

Reads the variant report the perturbation step wrote and adds what the run must
be judged against: the joint and the evidence for its direction, the offsets,
which conditions could not be run and why, the hashes of everything held fixed,
the budget, and the rule for when to stop.

    third_party/dexonomy/.venv/bin/python scripts/record_finger5_plan.py \\
        --variants out/finger5/variants.json --out out/finger5/plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "third_party" / "dexonomy" / "assets" / "hand" / "hx5_d20" / "right.xml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path) -> str:
    return sha(Path(path).read_bytes())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = json.loads(args.variants.read_text())
    plan = {
        "recorded_at": subprocess.run(["date", "-Is"], capture_output=True,
                                      text=True).stdout.strip(),
        "question": "With the template, the required contacts, the contact targets, "
                    "the scene, the objective and the budget held fixed, does "
                    "digit 5's initial flexion change whether every required "
                    "contact is made?",
        "primary_measure": "every required contact group satisfied at once AND the "
                           "refiner's full filter sequence passed (a refined file)",
        "joint": {
            "name": "finger_r_joint18", "role": "digit 5 PIP flexion",
            "index_in_grasp_qpos": 24, "range_rad": [0.0, 2.0071],
            "positive_direction": "flexion; measured on bowl 498809d2: link20 to "
                                  "palm 169.37 / 162.29 / 153.80 mm at -0.2 / 0 / +0.2",
            "coupling": "none: the model's only equality is the hand-root weld, and "
                        "it has no tendons",
            "why_this_joint_alone": "PIP has the most travel and moves both distal "
                                    "links the required contacts sit on. The hammer "
                                    "candidates start with DIP at 1.47-1.48 of 1.5708, "
                                    "so a common offset on every flexion joint would "
                                    "leave 'curl more' unrunnable for half the "
                                    "negatives.",
            "held": ["finger_r_joint17 (abduction)", "finger_r_joint19", "finger_r_joint20",
                     "every other finger", "the root pose"],
        },
        "offsets_rad": [p["amount"] for p in variants["perturbations"]],
        "offset_names": [p["name"] for p in variants["perturbations"]],
        "seed": "none; the offsets are fixed",
        "budget": {"op.grasp.step": 400, "n_worker": 2, "threads_per_worker": 1,
                   "nice": 15},
        "baseline_protocol": "the unperturbed initialisation is a byte-for-byte copy "
                             "of the original file, run first and on its own. It is "
                             "compared with the earlier iter400 run (fed byte-exact "
                             "copies, so it should match exactly) and with the "
                             "init4_original run (fed float32-rewritten files, so a "
                             "difference there has an explanation to test). Any "
                             "unexplained difference stops the experiment before "
                             "the offsets are run.",
        "stop_rule": "if no negative passes the full filter sequence, stop: no further "
                     "change to the budget, the offset range or the contact threshold",
        "not_a_proof": "a failure here is not evidence that the contact is physically "
                       "unreachable",
        "hand_xml_sha256": file_sha(HAND),
        "candidates": [],
    }
    not_run = []
    for candidate in variants["candidates"]:
        data = np.load(candidate["source_init"], allow_pickle=True).item()
        entry = {
            "control": candidate["control"], "object": candidate["object"],
            "grasp_type": candidate["grasp_type"],
            "template_name": str(data["tmpl_name"]),
            "source_init": candidate["source_init"],
            "source_init_sha256": candidate["source_init_sha256"],
            "scene_cfg_sha256": file_sha(str(data["scene_path"])),
            "required_cbody_sha256": sha(json.dumps(
                [list(g) for g in data["required_cbody"]], sort_keys=True).encode()),
            "hand_cbody_sha256": sha(json.dumps(list(data["hand_cbody"])).encode()),
            "obj_cpn_w_sha256": sha(np.ascontiguousarray(data["obj_cpn_w"]).tobytes()),
            "obj_pose_sha256": sha(np.ascontiguousarray(data["obj_pose"]).tobytes()),
            "variants": [],
        }
        for row in candidate["variants"]:
            item = {"variant": row["variant"], "amount": row.get("amount")}
            if row.get("not_run"):
                item["not_run"] = row["not_run"]
                not_run.append((candidate["object"], candidate["source_init_sha256"][:8],
                                row["variant"], row["not_run"]))
            else:
                item.update({
                    "path": row["path"], "sha256": file_sha(row["path"]),
                    "joint_value_before": row.get("joint_value_before"),
                    "joint_value_after": row.get("joint_value_after"),
                    "other_joints_unchanged": row.get("other_joints_unchanged"),
                    "link_frame_target_drift": row["link_frame_target_drift"],
                    "initial_state": row["initial_state"],
                })
                if row["variant"] == "original":
                    item["byte_copy_of_source"] = (file_sha(row["path"])
                                                   == candidate["source_init_sha256"])
            entry["variants"].append(item)
        plan["candidates"].append(entry)
    plan["not_run"] = [{"object": o, "candidate": c, "variant": v, "reason": r}
                       for o, c, v, r in not_run]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1, default=float) + "\n")
    total = sum(len(c["variants"]) for c in plan["candidates"])
    print(f"{len(plan['candidates'])} candidates, {total} conditions, "
          f"{len(not_run)} not run")
    for o, c, v, r in not_run:
        print(f"  not run  {o:<7}{c}  {v:<14}{r}")
    copies = [v.get("byte_copy_of_source") for c in plan["candidates"]
              for v in c["variants"] if v["variant"] == "original"]
    print(f"baselines that are byte copies of the source: {sum(bool(x) for x in copies)}"
          f"/{len(copies)}")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
