"""Read-only intake release with explicit, narrowly scoped preflight recovery."""

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
from pathlib import Path

from report_hope_student_readiness import digest, read, verified_episode
from watch_visual_intake import write


def recovered_preflight(original, recovery):
    allowed = {
        "vision.source: local DINOv2 checkout with hubconf.py",
        "vision.head_weights: local pretrained DINOv2 weights",
        "vision.wrist_weights: local pretrained DINOv2 weights",
    }
    if not original["missing"] or not set(original["missing"]).issubset(allowed):
        raise ValueError("recovery is only allowed for missing DINO asset paths")
    if original["cache_sha256"] != recovery["cache_sha256"]:
        raise ValueError("recovery changed the data cache")
    if recovery.get("training_ready") is not True or recovery.get("missing"):
        raise ValueError("recovery did not pass")
    if recovery.get("optimizer_steps") != 0:
        raise ValueError("recovery performed learning")
    return recovery


def report(root, recoveries=()):
    raw = read(root / "results.json")
    if raw.get("completed") is not True:
        raise ValueError("intake is still running")
    plan = read(root / "plan.json")
    indexed = {r["teacher_episode_sha256"]: r for r in raw["episodes"]}
    recovery_map = {p.parent.parent.resolve(): p for p in recoveries}
    result = {
        "completed": True,
        "training_started": False,
        "episodes": [],
        "raw_results_sha256": digest(root / "results.json"),
        "plan_sha256": digest(root / "plan.json"),
        "scope": "diagnostic data only; fixed7 unchanged; no new DINO forward",
    }
    for source in plan["episodes"]:
        row = indexed[source["episode_sha256"]]
        work = Path(row["work"])
        item = {
            **row,
            "student_intake_passed": False,
            "candidate_sha256": source["candidate_sha256"],
        }
        item["raw_intake_status"] = row["status"]
        if "error" in item:
            item["raw_intake_error"] = item.pop("error")
        result["episodes"].append(item)
        try:
            manifest = Path(source["source_manifest"])
            if digest(manifest) != source["source_manifest_sha256"]:
                raise ValueError("source manifest changed")
            verified_episode(manifest.parent, read(manifest)["episodes"][0])
            visual = read(work / "visual/manifest.json")["episodes"]
            if len(visual) != 1:
                raise ValueError("expected one visual episode")
            verified_episode(work / "visual", visual[0])
            if (
                visual[0]["provenance"]["visual_derivation"]["source_episode_sha256"]
                != source["episode_sha256"]
            ):
                raise ValueError("visual derivation changed source episode")
            render = read(work / "visual/render-report.json")
            audit = read(work / "audit/audit.json")
            controls = read(work / "controls/summary.json")
            preflight = read(work / "preflight/preflight.json")
            recovery = recovery_map.get(work.resolve())
            if recovery:
                preflight = recovered_preflight(preflight, read(recovery))
                item["preflight_recovery"] = {"path": str(recovery), "sha256": digest(recovery)}
            if not all(
                (
                    render.get("completed"),
                    audit.get("completed"),
                    audit.get("all_home_references_available"),
                    controls.get("completed"),
                    controls.get("checks_passed"),
                    preflight.get("training_ready"),
                )
            ):
                raise ValueError("one or more evidence stages did not pass")
            if audit["dataset_sha256"] != digest(work / "visual/manifest.json"):
                raise ValueError("audit manifest mismatch")
            item.update(
                student_intake_passed=True,
                status="data_checks_passed",
                data=str(work / "visual"),
                cache=str(work / "cache"),
                frames=render["episodes"][0]["frames"],
                rendering=render["episodes"][0],
                controls=controls,
                audit=audit,
                preflight=preflight,
                visual_manifest_sha256=digest(work / "visual/manifest.json"),
            )
        except (ValueError, KeyError, OSError) as error:
            item.update(status="evidence_failed", release_error=repr(error))
    result["ready_count"] = sum(r["student_intake_passed"] for r in result["episodes"])
    result["ready_frames"] = sum(
        r.get("frames", 0) for r in result["episodes"] if r["student_intake_passed"]
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--preflight-recovery", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write(args.out, report(args.root, args.preflight_recovery))
