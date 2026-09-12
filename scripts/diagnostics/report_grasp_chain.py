"""What became of each predicted grasp, from the camera to the wrench test.

Counts are reported against the **predictions**, not against the intermediate
files, because that is the number the pipeline is judged on: of the poses
GraspGenX proposed for this object, how many produced a grasp the robot actually
executed. One prediction fans out into roll/depth variants, each of those into
an alignment per seed, and identical refined grasps recur across seeds -- so a
count of files says more about the branching factor than about the predictions.

Three numbers, and they answer different questions:

* **completed** -- predictions with at least one execution that reached the end
  of approach, closure, lift and hold. Failures are kept in the denominator,
  including the executor's own failure to validate an approach path: that is a
  failure of this pipeline to produce a usable grasp, whatever its cause.
* **stages** -- what went in and what came out at each step, plus how many
  alignments, refinements and executions were actually run.
* **held every direction** -- of those, the ones whose grasp also resists an
  external wrench from each of six directions in Dexonomy's evaluator. That is
  a simulated test at one friction setting, not a proof of force closure.

Run with any interpreter; it only reads the records the pipeline wrote::

    python3 scripts/diagnostics/report_grasp_chain.py --template 33_Inferior_Pincer ...
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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

#: The executor's own outcome names, grouped by where the attempt stopped.
STOPPED_AT = {
    "no_validated_home_path": "approach",
    "closure_timeout": "closure",
    "no_quality_feasible_command": "closure",
    "lift_timeout": "lift",
    "completed": "completed",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--alignment", type=Path, default=paths.OUT / "alignment")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument("--exp", dest="exps", action="append", required=True,
                        help="Alignment experiment name; repeatable, one per seed.")
    parser.add_argument("--selection", dest="selections", type=Path, action="append",
                        default=[], help="select manifest; its .verdicts.json is read too.")
    parser.add_argument("--executed", type=Path, help="collect output directory")
    parser.add_argument("--eval-audit", type=Path,
                        help="Directory of audit_grasp_eval.py records.")
    parser.add_argument("--predictions", type=Path,
                        help="predict_grasps YAML, to count predictions never filtered.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def blank() -> dict:
    """What is tracked per prediction: where it reached, and what happened there."""
    return {"variants": set(), "refined": set(), "executed": set(), "outcomes": []}


def chain(args) -> dict:
    """Join every record the pipeline wrote, keyed by refined-grasp identity."""
    predictions: dict[str, dict] = {}
    if args.predictions and args.predictions.is_file():
        import yaml

        for name in (yaml.safe_load(args.predictions.read_text()).get("grasps") or {}):
            predictions[name] = blank()

    grasps: dict[str, dict] = {}
    alignments = []
    for exp in args.exps:
        record = (args.alignment / f"{exp}_{args.hand}" / args.template
                  / f"{args.object}.provenance.json")
        if not record.is_file():
            raise FileNotFoundError(f"{record} is missing; run align_grasps first")
        provenance = read(record)
        alignments.append({
            "experiment": exp, "seed": provenance["seed"],
            "variants": len(provenance["variants"]),
            "initialisations": len(provenance["init_files"]),
            "filters": {stage["filter"]: len(stage["seeds_passed"])
                        for stage in provenance["filters"]},
        })
        for variant in provenance["variants"]:
            predictions.setdefault(
                variant["prediction_id"], blank()
            )["variants"].add(variant["variant_id"])
        # The refined file keeps its initialisation's name, so the two join by
        # name within one experiment.
        refined = paths.DEXONOMY / "output" / f"{exp}_{args.hand}" / "grasp_data"
        for path in sorted(refined.rglob("*.npy")) if refined.is_dir() else []:
            entry = provenance["init_files"].get(path.name)
            if entry is None:
                continue
            # Resolved, because the same file is reached through a symlinked
            # checkout in one record and through its real location in another.
            key = str(path.resolve())
            grasps.setdefault(key, {"parents": [], "experiment": exp})
            grasps[key]["parents"].append({
                "prediction_id": entry["prediction_id"],
                "variant_id": entry["variant_id"],
                "experiment": exp, "seed": provenance["seed"],
                "template_sha256": entry["template"]["sha256"],
                "contact_index": entry["template"]["contact_index"],
            })
            # Credited here, where refinement happened, not where execution did:
            # a grasp the refiner produced and the selector then dropped still
            # says its prediction reached refinement.
            predictions.setdefault(
                entry["prediction_id"], blank(),
            )["refined"].add(key)

    verdicts: dict[str, str] = {}
    selected = 0
    for manifest in args.selections:
        if manifest.is_file():
            selected += len(read(manifest)["candidates"])
        record = manifest.with_suffix(".verdicts.json")
        if record.is_file():
            verdicts.update(read(record))
    verdicts = {str(Path(path).resolve()): value for path, value in verdicts.items()}
    for path, entry in grasps.items():
        entry["select_verdict"] = verdicts.get(path, "not_selected")

    # Identical refined grasps recur across seeds; each is executed once and
    # every parent it came from keeps the result. Attempts are joined to the
    # manifest by position, because an attempt that fails before an episode
    # exists carries no candidate identity of its own -- only its outcome.
    outcomes: list[dict] = []
    if args.executed and (args.executed / "attempts.json").is_file():
        used = read(args.executed / "collection.json")["candidate_manifest"]
        manifest = read(used)
        rows = manifest["candidates"]
        for index, attempt in enumerate(read(args.executed / "attempts.json")):
            if index >= len(rows):
                raise ValueError(f"{used} has fewer candidates than attempts")
            row = rows[index]
            recorded = ((attempt.get("provenance") or {}).get("teacher") or {})
            if recorded and Path(recorded.get("source_path", "")).resolve() \
                    != Path(row["source"]).resolve():
                raise ValueError(
                    f"attempt {attempt['episode_id']} ran {recorded['source_path']}, "
                    f"manifest position {index} is {row['source']}"
                )
            digest = row["source_sha256"]
            parents = manifest.get("parents", {}).get(digest) or \
                grasps.get(str(Path(row["source"]).resolve()), {}).get("parents", [])
            held = None
            if args.eval_audit:
                audit = args.eval_audit / f"{digest[:12]}.json"
                if audit.is_file():
                    held = all(d["held"] for d in read(audit)["directions"])
            status = attempt.get("status") or "unknown"
            tried = attempt.get("path_attempts") or (
                (attempt.get("provenance") or {}).get("execution") or {}
            ).get("path_attempts") or []
            outcomes.append({
                "episode_id": attempt["episode_id"], "source": row["source"],
                "candidate_id": digest, "status": status,
                "accepted": bool(attempt["accepted"]),
                "stopped_at": STOPPED_AT.get(status, "other"),
                "approach_attempts": [entry["status"] for entry in tried],
                "held_every_direction": held,
                "parents": parents,
            })

    for outcome in outcomes:
        for parent in outcome["parents"]:
            entry = predictions.setdefault(parent["prediction_id"], blank())
            entry["refined"].add(str(Path(outcome["source"]).resolve()))
            entry["executed"].add(str(Path(outcome["source"]).resolve()))
            entry["outcomes"].append(outcome)
    return {"alignments": alignments, "grasps": grasps, "outcomes": outcomes,
            "selected": selected, "predictions": predictions}


def summarise(joined: dict) -> dict:
    predictions = joined["predictions"]
    completed = sorted(
        name for name, entry in predictions.items()
        if any(o["accepted"] for o in entry["outcomes"])
    )
    held = sorted(
        name for name, entry in predictions.items()
        if any(o["accepted"] and o["held_every_direction"] for o in entry["outcomes"])
    )
    refined = sorted(name for name, entry in predictions.items() if entry["refined"])
    executed = sorted(name for name, entry in predictions.items() if entry["executed"])
    verdicts = Counter(
        entry.get("select_verdict", "not_selected") for entry in joined["grasps"].values()
    )
    filtered = sorted(name for name, entry in predictions.items() if entry["variants"])
    stopped = Counter(o["stopped_at"] for o in joined["outcomes"])
    approach = Counter(
        reason for o in joined["outcomes"] if o["stopped_at"] == "approach"
        for reason in o["approach_attempts"]
    )
    return {
        "predictions_total": len(predictions),
        "predictions_filtered": len(filtered),
        "predictions_refined": len(refined),
        "predictions_executed": len(executed),
        "select_verdicts": dict(verdicts),
        "predictions_completed": len(completed),
        "predictions_held_every_direction": len(held),
        "completed": completed, "held_every_direction": held,
        "variants": sum(len(e["variants"]) for e in predictions.values()),
        "alignment_runs": len(joined["alignments"]),
        "initialisations": sum(a["initialisations"] for a in joined["alignments"]),
        "refined_files": len(joined["grasps"]),
        "selected": joined["selected"],
        "executions": len(joined["outcomes"]),
        "stopped_at": dict(stopped),
        "approach_attempt_reasons": dict(approach),
        "filters": [a["filters"] for a in joined["alignments"]],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    joined = chain(args)
    report = summarise(joined)

    print(f"predictions        {report['predictions_total']}")
    print(f"  filtered         {report['predictions_filtered']}"
          f"   ({report['variants']} roll/depth variants)")
    print(f"  refined          {report['predictions_refined']}"
          f"   ({report['refined_files']} refined files)")
    print(f"  executed         {report['predictions_executed']}"
          f"   ({report['selected']} selected, {report['executions']} executed;"
          f" select verdicts "
          + ", ".join(f"{name} {count}" for name, count in
                      sorted(report["select_verdicts"].items())) + ")")
    print(f"  completed        {report['predictions_completed']}"
          f"   {report['completed']}")
    print(f"  held 6 wrenches  {report['predictions_held_every_direction']}"
          f"   {report['held_every_direction']}")
    print("\nexecutions stopped at: "
          + ", ".join(f"{where} {count}" for where, count in
                      sorted(report["stopped_at"].items())))
    if report["approach_attempt_reasons"]:
        print("approach attempts:     "
              + ", ".join(f"{reason} {count}" for reason, count in
                          sorted(report["approach_attempt_reasons"].items())))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "summary": report,
            "alignments": joined["alignments"],
            "outcomes": joined["outcomes"],
            "predictions": {
                name: {"variants": sorted(entry["variants"]),
                       "refined": sorted(entry["refined"]),
                       "executed": sorted(entry["executed"]),
                       "statuses": [o["status"] for o in entry["outcomes"]],
                       "held_every_direction": [
                           o["held_every_direction"] for o in entry["outcomes"]
                       ]}
                for name, entry in sorted(joined["predictions"].items())
            },
        }, indent=2) + "\n")
        print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
