"""Did the unperturbed baseline reproduce, and against which reference?

A comparison of perturbations is only as good as the row it is compared to.
Two earlier 400-step runs exist for these candidates, and they did not agree
with each other on one contact. One (``iter400_*``) was fed byte-for-byte copies
of the original initialisations; the other (``init4_original_*``) was fed files
that had been recomputed and re-saved as float32. A fresh baseline made by
copying the originals again should match the first exactly if the refiner is
deterministic, and any difference from the second is then explained by its
inputs. This checks both, candidate by candidate, before anything is perturbed.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/compare_baseline_reproduction.py \\
        --variants out/finger5/variants.json --end-state out/finger5/end-state \\
        --out out/finger5/baseline-reproduction.json
"""

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
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

REFERENCES = (
    ("iter400", "iter400_", ROOT / "out" / "contact-diagnosis" / "end-state-iter400"),
    ("init4_original", "init4_original_", ROOT / "out" / "init-sensitivity" / "end-state"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--end-state", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def key_for(path) -> str:
    parts = Path(path).parts
    experiment = parts[parts.index("output") + 1]
    return "__".join((experiment,) + parts[-4:-1] + (Path(path).stem,))


def renamed(path: str, prefix: str) -> Path:
    """The same candidate's file in another experiment of the same source."""
    parts = list(Path(path).parts)
    index = parts.index("output")
    parts[index + 1] = prefix + parts[index + 1]
    return Path(*parts)


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(record, required) -> dict:
    bodies = sorted(str(b) for b in record["contact_bodies"])
    satisfied = [bool(set(g) & set(bodies)) for g in required]
    return {
        "hand_qpos": np.asarray(record["hand_qpos"], dtype=float),
        "bodies": bodies,
        "groups_satisfied": int(sum(satisfied)),
        "passed": bool(record["passed"]) if "passed" in record else None,
        "filter_status": ({str(k): str(v) for k, v in record["filter_status"]}
                          if "filter_status" in record else None),
        "threshold_m": float(record["contact_threshold_m"])
            if "contact_threshold_m" in record else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.variants.read_text())
    rows = []
    for candidate in plan["candidates"]:
        original = next(v for v in candidate["variants"] if v["variant"] == "original")
        source = Path(candidate["source_init"])
        fresh_path = args.end_state / f"{key_for(original['path'])}.npz"
        if not fresh_path.is_file():
            raise SystemExit(f"no fresh baseline record: {fresh_path.name}")
        required = [list(g) for g in np.load(source, allow_pickle=True).item()["required_cbody"]]
        fresh = summary(np.load(fresh_path, allow_pickle=True), required)
        row = {"control": candidate["control"], "object": candidate["object"],
               "sha256": candidate["source_init_sha256"],
               "fresh_input_bytes_equal_source": digest(original["path"]) == digest(source),
               "fresh": {k: v for k, v in fresh.items() if k != "hand_qpos"},
               "references": {}}
        for label, prefix, directory in REFERENCES:
            copy = renamed(str(source), prefix)
            record_path = directory / f"{key_for(copy)}.npz"
            if not record_path.is_file():
                row["references"][label] = {"record": None}
                continue
            other = summary(np.load(record_path, allow_pickle=True), required)
            row["references"][label] = {
                "input_bytes_equal_source": copy.is_file() and digest(copy) == digest(source),
                "max_hand_qpos_difference": float(np.abs(other["hand_qpos"]
                                                         - fresh["hand_qpos"]).max()),
                "same_contact_bodies": other["bodies"] == fresh["bodies"],
                "groups_satisfied": other["groups_satisfied"],
                "same_verdict": other["passed"] == fresh["passed"],
                "filter_status": other["filter_status"],
                "threshold_m": other["threshold_m"],
                "bodies_only_in_reference": sorted(set(other["bodies"]) - set(fresh["bodies"])),
                "bodies_only_in_fresh": sorted(set(fresh["bodies"]) - set(other["bodies"])),
            }
        rows.append(row)
        cells = []
        for label, _, _ in REFERENCES:
            ref = row["references"][label]
            if ref.get("record", True) is None:
                cells.append(f"{label}: no record")
                continue
            cells.append(f"{label}: dq {ref['max_hand_qpos_difference']:.1e} "
                         f"bodies {'=' if ref['same_contact_bodies'] else 'DIFF'} "
                         f"groups {ref['groups_satisfied']}")
        print(f"{row['control']:<9}{row['object']:<7}{row['sha256'][:8]}  "
              f"fresh groups {fresh['groups_satisfied']}  passed {fresh['passed']}  | "
              + " | ".join(cells))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
