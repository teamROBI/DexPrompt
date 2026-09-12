"""A selection-learning ledger from the recorded HOPE teacher executions.

Reads the sources named in a JSON file (ledgers, their collections, candidate
pools, the geometry audit) and writes, under ``--out`` only:

``executions.jsonl``  One row per physical execution of a candidate in a scene:
                      where it came from (paths and hashes, original attempt ID),
                      what was run (object, mesh, geometry group, scale, placement,
                      distractors, robot initial state, candidate, type, contact
                      groups, physics cohort, controller version), what happened
                      (outcome, per-stage reached/succeeded with observation
                      masks), what it cost, and how it relates to other rows
                      (replicate of, retry of, infrastructure failure).
``candidates.jsonl``  Every candidate in the pools, with the features available
                      *before* execution and whether it was ever executed. A
                      candidate never executed has no label -- it is not a failure.
``costs.jsonl``       Every physics start in the sources, including runs that lost
                      their result and non-grasp diagnostics, so cost is not lost
                      when labels are.
``meta.json``         Counts reconciled against every input ledger, cohorts,
                      geometry groups, label availability, and the input/label split.

Inputs (``row["inputs"]``) and labels (``row["labels"]``) are kept apart: nothing
read from an outcome -- contact forces, the final pose, status, file names of
episodes -- is placed among the inputs.

    third_party/dexonomy/.venv/bin/python scripts/teacher/build_selector_ledger.py \\
        --sources <sources.json> --out <dir>
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from icra.paths import source_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Statuses that end an execution before the recorded approach: nothing after
# the approach was observed.
BEFORE_APPROACH = {"initial_collision", "no_validated_home_path", "ik_failure",
                   "not_started"}


def sha(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def phases_of(collection: Path, attempt: dict) -> list[str] | None:
    import numpy as np

    rel = attempt.get("path")
    if not rel or not (collection / rel).is_file():
        return None
    with np.load(collection / rel, allow_pickle=True) as z:
        phases = [str(p) for p in z["phase"]]
        groups = np.asarray(z["contact_groups"], bool)
    return phases, bool(groups.all(axis=1).any()), int(groups.sum(axis=1).max())


def stage_labels(status: str | None, accepted: bool, phases) -> dict:
    """Reached/succeeded per stage, with an explicit observed mask.

    A stage that was never reached has neither a success nor a failure label.
    """
    seen = set(phases[0]) if phases else set()
    all_groups = phases[1] if phases else False
    close_reached = "close" in seen
    lift_reached = "lift" in seen or "hold" in seen
    hold_reached = "hold" in seen
    # The approach succeeded exactly when the hand reached the grasp pose and the
    # close began; a quality violation during the approach is an approach failure.
    labels = {
        "approach": {"observed": status is not None,
                     "succeeded": close_reached if status is not None else None},
        # The executor starts lifting only after the required groups have held
        # for its ready ticks: that is its own closure success.
        "close": {"observed": close_reached,
                  "succeeded": lift_reached if close_reached else None,
                  "all_groups_ever": all_groups if close_reached else None},
        "carry": {"observed": lift_reached,
                  "succeeded": bool(accepted) if lift_reached else None},
        "full": {"observed": status is not None, "succeeded": bool(accepted)
                 if status is not None else None},
    }
    labels["end_phase"] = phases[0][-1] if phases and phases[0] else None
    labels["hold_reached"] = hold_reached
    return labels


def attempt_for(collection: Path, episode_id: str | None, index: int | None) -> dict | None:
    path = collection / "attempts.json"
    if not path.is_file():
        return None
    attempts = json.loads(path.read_text())
    for a in attempts:
        if episode_id and a.get("episode_id") == episode_id:
            return a
    if index is not None:
        for a in attempts:
            if a.get("candidate_index") == index:
                return a
    return attempts[0] if len(attempts) == 1 else None


def attempt_by_candidate(collection: Path, candidate_sha: str) -> dict | None:
    """The attempt of a candidate in a multi-candidate collection whose attempt
    entries carry no candidate ID (the first run's early schema): the collection's
    own manifest order gives the episode ID."""
    collection_json = collection / "collection.json"
    attempts_json = collection / "attempts.json"
    if not collection_json.is_file() or not attempts_json.is_file():
        return None
    manifest = resolve(json.loads(collection_json.read_text())["candidate_manifest"])
    if not manifest.is_file():
        return None
    rows = json.loads(manifest.read_text())["candidates"]
    index = next((i for i, r in enumerate(rows) if r["source_sha256"] == candidate_sha), None)
    if index is None:
        return None
    episode = f"teacher_{index:04d}_yaw_000"
    return next((a for a in json.loads(attempts_json.read_text())
                 if a.get("candidate_id") == candidate_sha or a.get("episode_id") == episode),
                None)


def collection_hashes(collection: Path | None) -> dict:
    """Code hashes from any attempt or diagnostics file of the same collection run."""
    if not collection:
        return {}
    attempts_json = collection / "attempts.json"
    if attempts_json.is_file():
        for a in json.loads(attempts_json.read_text()):
            hashes = ((a.get("provenance") or {}).get("execution") or {}).get("source_hashes")
            if hashes:
                return hashes
    for path in sorted((collection / "diagnostics").glob("*.json")) \
            if (collection / "diagnostics").is_dir() else []:
        hashes = (json.loads(path.read_text()).get("execution") or {}).get("source_hashes")
        if hashes:
            return hashes
    return {}


def candidate_row(collection: Path, attempt: dict | None) -> dict:
    """The candidate row the collection actually ran, from its own manifest."""
    collection_json = collection / "collection.json"
    if not collection_json.is_file():
        return {}
    manifest = Path(json.loads(collection_json.read_text())["candidate_manifest"])
    manifest = resolve(manifest)
    if not manifest.is_file():
        return {}
    rows = json.loads(manifest.read_text())["candidates"]
    if attempt and attempt.get("candidate_id"):
        for row in rows:
            if row["source_sha256"] == attempt["candidate_id"]:
                return row
    if attempt and attempt.get("episode_id"):
        index = int(attempt["episode_id"].split("_")[1])
        return rows[index] if index < len(rows) else {}
    return rows[0] if len(rows) == 1 else {}


def execution_of(collection: Path | None, attempt: dict | None) -> dict:
    """The execution provenance, from the attempt or, for a run that recorded no
    episode (an approach failure), from its diagnostics file."""
    execution = ((attempt or {}).get("provenance") or {}).get("execution")
    if execution:
        return execution
    if collection and attempt and attempt.get("episode_id"):
        path = collection / "diagnostics" / f"{attempt['episode_id']}.json"
        if path.is_file():
            return (json.loads(path.read_text()).get("execution") or {})
    return {}


def pool_index(sources: dict) -> dict:
    """(physics, object, candidate) -> (type, required groups), from the pools."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_expansion_manifest import pooled

    index = {}
    for entry in sources["pools"]:
        rows, _ = pooled(resolve(entry["manifests"]), entry["object"])
        for r in rows:
            index[(entry["physics"], entry["object"], r["source_sha256"])] = (
                r["template"], r["required"])
    return index


def geometry_groups(path: Path) -> dict:
    data = json.loads(path.read_text())
    groups = {}
    for group in data.get("connected_groups") or []:
        members = group if isinstance(group, list) else group.get("objects", [])
        for name in members:
            groups[name] = "+".join(sorted(members))
    for entry in data.get("objects", []):
        groups.setdefault(entry["object"], entry["object"])
    return groups


def scene_info(scene_path: Path | None) -> dict:
    if not scene_path or not scene_path.is_file():
        return {}
    import yaml

    scene = yaml.safe_load(scene_path.read_text())
    return {"scene_sha256": sha(scene_path),
            "object_yaw_degrees": scene.get("object_yaw_degrees", 0),
            "object_offset": scene.get("object_offset"),
            "appearance": scene.get("object_appearance"),
            "arm_init_pose_sha256": hashlib.sha256(json.dumps(
                scene.get("arm_init_pose"), sort_keys=True).encode()).hexdigest()}


def normalise(source: dict, row: dict) -> dict:
    """Map one source ledger row to the common fields it carries."""
    kind = source["kind"]
    if kind == "clutter":
        # Its "placement" field is the distractor placement file; the target stands
        # at its own centre placement in every condition.
        placement, condition = "centre", row.get("condition") or "single"
    else:
        placement, condition = row.get("placement") or "centre", "single"
    obj = row.get("object") or row.get("target")
    return {
        "attempt_uuid": row.get("attempt_uuid"),
        "object": obj, "placement": placement, "condition": condition,
        "role": row.get("role") or row.get("stage") or (f"arm_{row['arm']}" if "arm" in row
                                                        else kind),
        "candidate_sha256": row.get("candidate_sha256"),
        "collection": row.get("collection"),
        "episode_sha256": row.get("episode_sha256"),
        "model_sha256": row.get("model_sha256"),
        "initial_state_sha256": row.get("initial_state_sha256"),
        "simulation_started": bool(row.get("simulation_started")),
        "input_refusal": bool(row.get("input_refusal")),
        "infrastructure_error": row.get("infrastructure_error"),
        "quarantined": bool(row.get("quarantined")),
        "retry_of": row.get("retry_of"),
        "status": row.get("outcome"), "accepted": bool(row.get("accepted")),
        "distractors": row.get("distractors") or [],
        "placement_file": row.get("placement") if kind == "clutter" else None,
        "placement_file_sha256": row.get("placement_sha256"),
        "scene": row.get("scene"),
        "seconds": row.get("seconds"),
        "not_run": row.get("not_run"),
    }


def build(sources: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    groups = geometry_groups(resolve(sources["geometry"]))
    inventory = {o["object"]: o for o in
                 json.loads(resolve(sources["inventory"]).read_text())["objects"]}
    executions, costs, reconcile = [], [], {}
    types = pool_index(sources)
    for source in sources["ledgers"]:
        path = resolve(source["path"])
        rows = load_rows(path)
        reconcile[source["name"]] = {"path": str(path), "sha256": sha(path),
                                     "input_rows": len(rows), "physics_started": 0,
                                     "input_refusals": 0, "not_run": 0, "rows_written": 0}
        for index, raw in enumerate(rows):
            r = normalise(source, raw)
            tally = reconcile[source["name"]]
            if r["not_run"]:
                tally["not_run"] += 1
                continue
            if r["simulation_started"]:
                tally["physics_started"] += 1
            if r["input_refusal"]:
                tally["input_refusals"] += 1
            collection = resolve(r["collection"]) if r["collection"] else None
            attempt = attempt_for(collection, None, None) if collection else None
            if collection and r["episode_sha256"] and (collection / "attempts.json").is_file():
                attempt = next((a for a in json.loads((collection / "attempts.json").read_text())
                                if a.get("sha256") == r["episode_sha256"]), attempt)
            if collection and attempt is None and r["candidate_sha256"]:
                attempt = attempt_by_candidate(collection, r["candidate_sha256"])
            candidate = candidate_row(collection, attempt) if collection else {}
            execution = execution_of(collection, attempt)
            hashes = execution.get("source_hashes") or {}
            controller_source = "attempt" if hashes else None
            if not hashes and r["simulation_started"]:
                hashes = collection_hashes(collection)
                controller_source = "same_collection_run" if hashes else None
            # A refusal before simulation is not an infrastructure failure; a started
            # run that lost its result (error or quarantine without an episode) is.
            infra = bool(not r["input_refusal"] and (
                r["infrastructure_error"] or (r["quarantined"] and r["episode_sha256"] is None)))
            physics = (source.get("physics_by_arm") or {}).get(raw.get("arm"), source["physics"])
            phases = phases_of(collection, attempt) if (collection and attempt) else None
            scene_path = resolve(r["scene"]) if r["scene"] else None
            if scene_path is None and source.get("scene_template"):
                scene_path = resolve(source["scene_template"].format(object=r["object"],
                                                                     placement=r["placement"]))
            status = (attempt or {}).get("status") or r["status"]
            observed = r["simulation_started"] and not infra and not r["input_refusal"]
            labels = stage_labels(status if observed else None,
                                  r["accepted"] if observed else False, phases)
            mesh = None
            for template in (source.get("mesh_template"), sources.get("mesh_template")):
                if template and resolve(template.format(object=r["object"])).is_file():
                    mesh = template.format(object=r["object"])
                    break
            candidate_sha = r["candidate_sha256"] or candidate.get("source_sha256")
            pooled_type, pooled_required = types.get(
                (physics, r["object"], candidate_sha), (None, None))
            cost = {"source": source["name"], "attempt_uuid": r["attempt_uuid"],
                    "physics_started": r["simulation_started"],
                    "ticks": (attempt or {}).get("ticks"),
                    "seconds": r["seconds"], "infrastructure_error": infra,
                    "kind": "grasp_execution"}
            costs.append(cost)
            tally["rows_written"] += 1
            executions.append(
                {
                    "row_id": f"{source['name']}:{index}",
                    "source": {
                        "name": source["name"],
                        "ledger": str(path),
                        "ledger_sha256": sha(path),
                        "ledger_index": index,
                        "attempt_uuid": r["attempt_uuid"],
                        "collection": str(collection) if collection else None,
                        "attempts_sha256": sha(collection / "attempts.json")
                        if collection
                        else None,
                        "episode_sha256": r["episode_sha256"],
                        "model_sha256": r["model_sha256"]
                        or next(iter((execution.get("geometry_hashes") or {}).values()), None),
                        "initial_state_sha256": r["initial_state_sha256"]
                        or execution.get("initial_state_sha256"),
                    },
                    "cohort": {
                        "physics": physics,
                        "controller_sha256": source_value(hashes, "src/icra/teacher/execution.py"),
                        "contact_control_sha256": source_value(
                            hashes, "src/icra/teacher/contact_control.py"
                        ),
                        "controller_hash_source": controller_source,
                        "scene_kind": "clutter" if r["distractors"] else "single",
                    },
                    "inputs": {
                        "object": r["object"],
                        "geometry_group": groups.get(r["object"], r["object"]),
                        "mesh_sha256": sha(resolve(mesh)) if mesh else None,
                        "object_dims_mm": (inventory.get(r["object"]) or {}).get(
                            "upright_width_depth_height_mm"
                        ),
                        "scale": execution.get("object_scale", 1.0),
                        "placement": r["placement"],
                        "condition": r["condition"],
                        "mesh_path": str(resolve(mesh)) if mesh else None,
                        **scene_info(scene_path),
                        "distractors": r["distractors"],
                        "placement_file": r["placement_file"],
                        "placement_file_sha256": r["placement_file_sha256"],
                        "robot_initial_state_sha256": r["initial_state_sha256"]
                        or execution.get("initial_state_sha256"),
                        "candidate_sha256": candidate_sha,
                        "grasp_type": candidate.get("template") or pooled_type,
                        "required_contact_groups": candidate.get("required") or pooled_required,
                    },
                    "labels": {
                        **labels,
                        "status": status if observed else None,
                        "accepted": r["accepted"] if observed else None,
                        "reasons": ((attempt or {}).get("quality") or {}).get("reasons"),
                    },
                    "flags": {
                        "simulation_started": r["simulation_started"],
                        "input_refusal": r["input_refusal"],
                        "infrastructure_error": infra,
                        "quarantined": r["quarantined"],
                        "retry_of": r["retry_of"],
                        "role": r["role"],
                    },
                    "cost": cost,
                }
            )
    # Replicates: the same candidate, scene, distractor placement, physics and
    # controller run again. The first in ledger order is the primary.
    primary = {}
    for row in executions:
        key = (row["cohort"]["physics"], row["cohort"]["controller_sha256"],
               row["inputs"]["candidate_sha256"], row["inputs"].get("scene_sha256"),
               row["inputs"]["placement_file_sha256"], tuple(row["inputs"]["distractors"]),
               row["inputs"]["placement"])
        if row["flags"]["infrastructure_error"] or not row["flags"]["simulation_started"]:
            row["replicate_of"] = None
            continue
        if key in primary:
            first = primary[key]
            row["replicate_of"] = first["row_id"]
            row["identical_to_primary"] = bool(
                row["source"]["episode_sha256"]
                and row["source"]["episode_sha256"] == first["source"]["episode_sha256"])
        else:
            primary[key] = row
            row["replicate_of"] = None
    candidates = pool(sources, executions)
    write_jsonl(out / "executions.jsonl", executions)
    write_jsonl(out / "candidates.jsonl", candidates)
    for extra in sources.get("non_grasp_physics", []):
        costs.append({"source": extra["name"], "physics_started": True, "runs": extra["runs"],
                      "kind": "non_grasp_diagnostic"})
    write_jsonl(out / "costs.jsonl", costs)
    meta = summarise(executions, candidates, costs, reconcile, sources)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def pool(sources: dict, executions: list[dict]) -> list[dict]:
    """Every pooled candidate with its pre-execution features and observation state."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_expansion_manifest import pooled

    executed = collections.defaultdict(list)
    for row in executions:
        if row["flags"]["simulation_started"] and not row["flags"]["infrastructure_error"]:
            executed[(row["cohort"]["physics"], row["inputs"]["object"],
                      row["inputs"]["candidate_sha256"])].append(row)
    out = []
    for entry in sources["pools"]:
        rows, parents = pooled(resolve(entry["manifests"]), entry["object"])
        for r in rows:
            key = (entry["physics"], entry["object"], r["source_sha256"])
            runs = executed.get(key, [])
            primaries = [x for x in runs if x.get("replicate_of") is None]
            out.append({
                "physics": entry["physics"], "object": entry["object"],
                "candidate_sha256": r["source_sha256"], "grasp_type": r["template"],
                "parents": parents[r["source_sha256"]],
                "features": {"grasp_qpos": r["grasp_qpos"], "squeeze_qpos": r["squeeze_qpos"],
                             "open_qpos": r["open_qpos"], "required": r["required"]},
                "source_sha256_verified": sha(r["source"]) == r["source_sha256"]
                if Path(r["source"]).is_file() else None,
                "executed_rows": [x["row_id"] for x in runs],
                "independent_executions": len(primaries),
                "label_observed": bool(primaries),
                "any_accepted": any(x["labels"]["accepted"] for x in runs) if runs else None,
            })
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def summarise(executions, candidates, costs, reconcile, sources) -> dict:
    by = collections.Counter
    labelled = [r for r in executions if r["labels"]["full"]["observed"]]
    primaries = [r for r in labelled if r.get("replicate_of") is None]
    stage = {s: {"observed": sum(r["labels"][s]["observed"] for r in primaries),
                 "succeeded": sum(bool(r["labels"][s]["succeeded"]) for r in primaries
                                  if r["labels"][s]["observed"])}
             for s in ("approach", "close", "carry", "full")}
    return {
        "rows": len(executions), "labelled_rows": len(labelled),
        "independent_labelled_rows": len(primaries),
        "replicates": sum(r.get("replicate_of") is not None for r in executions),
        "infrastructure_rows": sum(r["flags"]["infrastructure_error"] for r in executions),
        "input_refusals": sum(r["flags"]["input_refusal"] for r in executions),
        "physics_starts_grasp": sum(c["physics_started"] for c in costs
                                    if c["kind"] == "grasp_execution"),
        "physics_starts_non_grasp": sum(c.get("runs", 0) for c in costs
                                        if c["kind"] == "non_grasp_diagnostic"),
        "cohorts": dict(by(f"{r['cohort']['physics']}|{r['cohort']['scene_kind']}"
                           for r in executions)),
        "controllers": dict(by(str(r["cohort"]["controller_sha256"])[:12] for r in executions)),
        "objects": dict(by(r["inputs"]["object"] for r in primaries)),
        "geometry_groups": dict(by(r["inputs"]["geometry_group"] for r in primaries)),
        "grasp_types": dict(by(str(r["inputs"]["grasp_type"]) for r in primaries)),
        "distinct_candidates_executed": len({r["inputs"]["candidate_sha256"] for r in primaries}),
        "stage_counts_independent": stage,
        "full_success_independent": {"accepted": sum(bool(r["labels"]["accepted"])
                                                     for r in primaries),
                                     "failed": sum(r["labels"]["accepted"] is False
                                                   for r in primaries)},
        "end_phase_independent": dict(by(str(r["labels"]["end_phase"]) for r in primaries)),
        "candidate_pool": {"total": len(candidates),
                           "executed": sum(c["label_observed"] for c in candidates),
                           "unobserved": sum(not c["label_observed"] for c in candidates),
                           "by_object": dict(by(f"{c['physics']}|{c['object']}"
                                                for c in candidates))},
        "reconciliation": reconcile,
        "input_fields": ["object", "geometry_group", "mesh_sha256", "object_dims_mm", "scale",
                         "placement", "scene_sha256", "object_yaw_degrees", "object_offset",
                         "distractors", "placement_file_sha256", "robot_initial_state_sha256",
                         "candidate_sha256", "grasp_type", "required_contact_groups",
                         "candidate features (grasp/squeeze/open qpos) in candidates.jsonl"],
        "label_fields": ["approach/close/carry/full observed+succeeded", "end_phase", "status",
                         "accepted", "reasons"],
        "leakage_rule": "no outcome-derived value (status, forces, contacts, final pose, "
                        "episode file names) is an input",
        "sources": sources,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    meta = build(json.loads(args.sources.read_text()), args.out)
    print(json.dumps({k: meta[k] for k in ("rows", "labelled_rows", "independent_labelled_rows",
                                           "replicates", "infrastructure_rows", "input_refusals",
                                           "cohorts", "stage_counts_independent",
                                           "full_success_independent", "candidate_pool")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
