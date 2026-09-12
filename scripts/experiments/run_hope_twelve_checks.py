"""Assemble exactly the released 7+5 HOPE demonstrations; no training or new physics."""

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
import time
import zipfile
from pathlib import Path

import run_single_student_lecun as job

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_sources(fixed, added):
    if not fixed["completed"] or not added["completed"]:
        raise ValueError("release incomplete")
    bundle = fixed["bundle"]
    if bundle["episodes"] != 7 or not fixed["remote_validation"]["completed"]:
        raise ValueError("fixed seven release invalid")
    if not fixed["remote_validation"]["controls"]["checks_passed"]:
        raise ValueError("fixed seven controls failed")
    if len(added["episodes"]) != 5 or added["ready_count"] != 5:
        raise ValueError("expected exactly five newly approved episodes")
    output = [
        dict(
            data=bundle["remote_path"],
            manifest_sha256=bundle["manifest_sha256"],
            frames=bundle["frames"],
            episodes=7,
            placements={e["bundle_episode_id"]: e["placement"] for e in bundle["mapping"]},
        )
    ]
    for row in added["episodes"]:
        if not (
            row["student_intake_passed"]
            and row["status"] == "data_checks_passed"
            and row["audit"]["completed"]
            and row["controls"]["checks_passed"]
            and row["preflight"]["training_ready"]
        ):
            raise ValueError("new episode lacks final intake approval")
        if row["controls"]["dataset_sha256"] != row["visual_manifest_sha256"]:
            raise ValueError("approval belongs to a different dataset")
        output.append(
            dict(
                data=row["data"],
                manifest_sha256=row["visual_manifest_sha256"],
                frames=row["frames"],
                episodes=1,
                candidate=row["candidate_sha256"],
                teacher_episode_sha256=row["teacher_episode_sha256"],
                placement=row["placement"],
            )
        )
    if sum(s["frames"] for s in output) != 12477 or len({s["data"] for s in output}) != 6:
        raise ValueError("release count or source mismatch")
    return output


def members(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate NPZ members")
        for name in names:
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            result[name] = h.hexdigest()
    return result


def verify_import(sources, data):
    from icra.dataset import load_manifest

    result = load_manifest(data, require_train=True)
    expected, mapping = {}, []
    for index, source in enumerate(sources):
        root = Path(source["data"])
        if digest(root / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("source manifest changed")
        manifest = load_manifest(root, require_train=True)
        if len(manifest["episodes"]) != source["episodes"]:
            raise ValueError("source count changed")
        for row in manifest["episodes"]:
            expected[f"source{index}_{row['episode_id']}"] = (source, root, row)
    if set(expected) != {e["episode_id"] for e in result["episodes"]}:
        raise ValueError("merge lost or added episodes")
    for entry in result["episodes"]:
        source, root, old = expected[entry["episode_id"]]
        if entry["split"] != "train" or entry["quality"] != old["quality"]:
            raise ValueError("split or quality changed")
        # Exact uncompressed NPY bytes, including every physical signal, not only image inputs.
        if members(root / old["path"]) != members(data / entry["path"]):
            raise ValueError("episode content changed")
        teacher = entry["provenance"]["teacher"]["source_sha256"]
        if "candidate" in source and teacher != source["candidate"]:
            raise ValueError("unexpected teacher candidate")
        derivation = entry["provenance"].get("visual_derivation", {})
        if (
            "teacher_episode_sha256" in source
            and derivation["source_episode_sha256"] != source["teacher_episode_sha256"]
        ):
            raise ValueError("unexpected teacher episode")
        mapping.append(
            dict(
                episode_id=entry["episode_id"],
                object=entry["object_id"],
                placement=source.get("placement") or source["placements"][old["episode_id"]],
                teacher_candidate=teacher,
                source=str(root),
                source_episode_id=old["episode_id"],
                source_sha256=old["sha256"],
                imported_sha256=entry["sha256"],
                all_arrays_byte_identical=True,
            )
        )
    if len({r["teacher_candidate"] for r in mapping}) != 5:
        raise ValueError("expected five distinct teacher candidates")
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed-release", type=Path, required=True)
    p.add_argument("--new-release", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    if args.data.exists() or args.cache.exists() or args.out.exists():
        raise FileExistsError("all output directories must be fresh")
    sources = approved_sources(read(args.fixed_release), read(args.new_release))
    for source in sources:
        if digest(Path(source["data"]) / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("source differs from approved release")
    args.out.mkdir(parents=True)
    job.ROOT, job.JOB = ROOT, args.out
    os.nice(15)
    os.environ.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        MUJOCO_GL="egl",
        CUDA_VISIBLE_DEVICES=args.gpu,
    )
    job.write(
        args.out / "plan.json",
        dict(
            sources=sources,
            training_started=False,
            fixed_release_sha256=digest(args.fixed_release),
            new_release_sha256=digest(args.new_release),
            expected_episodes=12,
            expected_frames=12477,
            expected_candidates=5,
            scope="diagnostic train-only; no held-out generalization; physics A only",
        ),
    )
    try:
        command = ["scripts/data/prepare_student_diagnostic.py", "--out", args.data]
        for source in sources:
            command.extend(["--input", source["data"]])
        job.stage("assemble", command, 3600)
        mapping = verify_import(sources, args.data)
        job.write(args.out / "mapping.json", dict(episodes=mapping, completed=True))
        trainer = "scripts/student/train_hx5_student.py"
        job.stage(
            "cache", [trainer, "prepare-cache", "--data", args.data, "--out", args.cache], 1800
        )
        common = ["--data", args.data, "--cache", args.cache, "--config", args.config]
        job.stage(
            "audit",
            ["scripts/diagnostics/audit_visual_intake.py", *common, "--out", args.out / "audit"],
            1800,
        )
        job.stage(
            "preflight",
            [
                trainer,
                "preflight",
                *common,
                "--allow-train-only-diagnostic",
                "--out",
                args.out / "preflight",
            ],
            1800,
        )
        preflight = read(args.out / "preflight/preflight.json")
        if not preflight["training_ready"] or preflight["episode_counts"] != dict(
            train=12, val=0, test=0
        ):
            raise ValueError("train-only preflight failed")
        # Select tick 0 and 500 from one episode of each new candidate, not the first batch twice.
        from icra.dataset import load_manifest
        from icra.student_cache import StudentCache
        from icra.student_data import EpisodeChunks

        config = read(args.config)
        manifest = load_manifest(args.data, require_train=True)
        cache = StudentCache(args.cache, args.data, manifest)
        chunks = EpisodeChunks(
            args.data,
            manifest,
            "train",
            config["action_horizon"],
            config["image_size"],
            cache=cache,
            observation_version=2,
            goal_image_size=config["goal_image_size"],
            goal_reference_policy=config["goal_reference_policy"],
        )
        if any(chunks.starts):
            raise ValueError("unexpected prefix removed; recalculate sample mapping")
        new_ids = {s["candidate"] for s in sources if "candidate" in s}
        chosen, offset = {}, 0
        for e in chunks.entries:
            candidate = e["provenance"]["teacher"]["source_sha256"]
            length = cache.length(e["episode_id"])
            if candidate in new_ids and candidate not in chosen:
                chosen[candidate] = dict(
                    episode_id=e["episode_id"], sample_indices=[offset, offset + 500]
                )
                if length <= 500:
                    raise ValueError("representative episode too short")
            offset += length
        if len(chosen) != 2 or offset != 12477:
            raise ValueError("sample mapping incorrect")
        job.write(args.out / "full-model-samples.json", chosen)
        code = (
            "import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction(0.85,0); "
            "sys.argv=['scripts/student/train_hx5_student.py',*sys.argv[1:]]; "
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        for index, row in enumerate(chosen.values()):
            job.GPU = args.gpu
            job.gpu_free()
            name = f"full-model-{index}"
            command = [
                "-c",
                code,
                "dry-run",
                *common,
                "--full-model",
                "--device",
                "cuda:0",
                "--out",
                args.out / name,
            ]
            for value in row["sample_indices"]:
                command.extend(["--sample-index", value])
            job.stage(name, command, 1800)
            check = read(args.out / name / "dry_run.json")
            if not (
                check["loss_finite"]
                and check["model_unchanged"]
                and check["full_training_architecture_checked"]
                and check["prediction_shape"] == [2, 64, 27]
                and check["optimizer_steps"] == check["backward_calls"] == 0
            ):
                raise ValueError("DINO check failed")
        for source in sources:
            if digest(Path(source["data"]) / "manifest.json") != source["manifest_sha256"]:
                raise ValueError("source mutated during checks")
        job.write(
            args.out / "release.json",
            dict(
                completed=True,
                training_started=False,
                data=str(args.data),
                cache=str(args.cache),
                manifest_sha256=digest(args.data / "manifest.json"),
                cache_sha256=digest(args.cache / "manifest.json"),
                episodes=12,
                frames=12477,
                teacher_candidates=5,
                object_placement_conditions=len({(r["object"], r["placement"]) for r in mapping}),
                objects=sorted({r["object"] for r in mapping}),
                mapping=mapping,
                controls="source-bound existing replay/stationary controls; not rerun",
                all_episode_arrays_preserved=True,
                full_model_batches=2,
                optimizer_steps=0,
                scope="diagnostic train-only; no independent validation/test",
            ),
        )
        job.write(
            args.out / "status.json", dict(phase="completed", running=False, training_started=False)
        )
    except BaseException as exc:
        job.write(
            args.out / "status.json",
            dict(
                phase="failed",
                running=False,
                training_started=False,
                error=repr(exc),
                updated_at=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
