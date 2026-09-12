# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""Freeze reproduced follow-up episodes for diagnostic visual intake, without training."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()


import argparse
from collections import defaultdict
from pathlib import Path

from report_hope_student_readiness import digest, read, verified_episode
from watch_visual_intake import source_guard, write


def balanced_order(rows):
    """Visit every candidate before its second placement; no evaluation split implied."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["object"], row["candidate_sha256"])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (-r["quality_margin"]["margin"], r["placement"]))
    return [group[i] for i in range(max(map(len, groups.values()), default=0))
            for _, group in sorted(groups.items()) if i < len(group)]


def check_pair(row, ledger, hash_file=digest):
    repeat = row["reproduction"]
    if repeat.get("identical") is not True or repeat.get("accepted") is not True:
        raise ValueError("reproduction is not accepted and identical")
    manifests = []
    for attempt in (row["attempt_uuid"], repeat["attempt_uuid"]):
        record = ledger[attempt]
        if (record.get("status") != "completed" or record.get("exit_code") != 0
                or record.get("accepted") is not True
                or record.get("infrastructure_error") or record.get("timeout_reason")
                or record.get("ended") != "exited"):
            raise ValueError("source/reproduction execution did not complete cleanly")
        if (record["object"], record["condition"], record["candidate_sha256"]) != (
                row["object"], row["placement"], row["candidate_sha256"]):
            raise ValueError("source/reproduction identity mismatch")
        root = Path(record["collection"])
        manifest = root / "manifest.json"
        entries = read(manifest)["episodes"]
        if len(entries) != 1:
            raise ValueError("expected one source episode")
        entry = entries[0]
        if verified_episode(root, entry) != row["episode_sha256"]:
            raise ValueError("episode hash mismatch")
        if entry["candidate_id"] != row["candidate_sha256"]:
            raise ValueError("manifest candidate mismatch")
        execution = entry["provenance"]["execution"]
        paths = {"episode": root / entry["path"],
                 "model": root / execution["model"],
                 "initial_state": root / execution["initial_state"]}
        for key, path in paths.items():
            expected = row[key + "_sha256"]
            if record[key + "_sha256"] != expected or hash_file(path) != expected:
                raise ValueError(f"{key} hash mismatch")
        manifests.append(manifest)
    if repeat["episode_sha256"] != row["episode_sha256"]:
        raise ValueError("reproduction summary hash mismatch")
    return manifests[0]


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    ledger_rows = read(source / "ledger.json")
    ledger = {r["attempt_uuid"]: r for r in ledger_rows}
    if len(ledger) != len(ledger_rows):
        raise ValueError("duplicate attempt UUID")
    handoff = read(source / "HANDOFF.json")
    rows = handoff["demonstrations"]
    if len({r["episode_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate episode within follow-up handoff")
    audit, eligible = [], []
    cache = {}

    def cached_hash(path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = digest(path)
        return cache[key]

    for row in rows:
        record = {"object": row["object"], "placement": row["placement"],
                  "candidate_sha256": row["candidate_sha256"],
                  "episode_sha256": row["episode_sha256"], "student_ready": False}
        try:
            manifest = check_pair(row, ledger, cached_hash)
            entry = read(manifest)["episodes"][0]
            violation = source_guard(manifest.parent, entry)
            record.update(status="guard_rejected" if violation else "eligible_for_visual_intake",
                          first_violation=violation, source_manifest=str(manifest))
            if violation is None:
                eligible.append({**row, "source_manifest": str(manifest),
                                 "source_manifest_sha256": digest(manifest)})
        except Exception as error:
            record.update(status="evidence_error", error=repr(error))
        audit.append(record)
        write(out / "guard-audit.json", {"completed": False, "episodes": audit})
    write(out / "guard-audit.json", {"completed": True, "episodes": audit,
                                    "training_started": False})
    plan = {"training_started": False, "scope": "follow-up diagnostic intake; A12 unchanged",
            "handoff_sha256": digest(source / "HANDOFF.json"),
            "ledger_sha256": digest(source / "ledger.json"),
            "source": str(source), "episodes": balanced_order(eligible),
            "cross_dataset_deduplication": "pending Claude audit; do not merge",
            "split": "diagnostic only; not an independent evaluation set"}
    write(out / "intake-plan.json", plan)
    print(f"guard passed {len(eligible)}/{len(rows)}; training not started", flush=True)
    if any(r["status"] == "evidence_error" for r in audit):
        raise ValueError("evidence errors recorded; batch not approved for automatic intake")
    if not eligible:
        raise ValueError("no episodes eligible for visual intake")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)
