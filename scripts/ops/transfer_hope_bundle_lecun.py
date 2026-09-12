"""Resume-safe transfer and verification of the sealed HOPE bundle to lecun."""

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
import subprocess
import time
from pathlib import Path

HOST = "yspark@147.47.200.171"
PORT = "19423"
REMOTE_CODE = Path("/home/yspark/projects/DexPrompt/out/student-x0-code-v1")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--remote-final", type=Path, required=True)
    parser.add_argument("--remote-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("transfer record directory must be fresh")
    args.out.mkdir(parents=True)
    remote_partial = args.remote_final.parent / f".{args.remote_final.name}.partial"
    ssh = [
        "ssh",
        "-i",
        str(args.key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        PORT,
        HOST,
    ]
    scp = [
        "scp",
        "-i",
        str(args.key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-P",
        PORT,
    ]
    report = {
        "completed": False,
        "training_started": False,
        "source": str(args.source),
        "manifest": str(args.manifest),
        "remote_partial": str(remote_partial),
        "remote_final": str(args.remote_final),
        "remote_record": str(args.remote_record),
        "stages": [],
    }
    write_json(args.out / "status.json", report)

    def run(name: str, command: list[str]) -> None:
        started = time.monotonic()
        with (args.out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        report["stages"].append(
            {
                "name": name,
                "exit_code": result.returncode,
                "seconds": time.monotonic() - started,
            }
        )
        write_json(args.out / "status.json", report)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    try:
        prepare = (
            f"test ! -e {args.remote_final} && "
            f"mkdir -p {remote_partial} {args.remote_record.parent}"
        )
        run("prepare-remote", [*ssh, prepare])
        rsync_shell = (
            f"ssh -i {args.key} -o IdentitiesOnly=yes -o BatchMode=yes "
            f"-o ConnectTimeout=15 -p {PORT}"
        )
        run(
            "rsync",
            [
                "rsync",
                "-a",
                "--partial",
                "--info=progress2",
                "-e",
                rsync_shell,
                f"{args.source}/",
                f"{HOST}:{remote_partial}/",
            ],
        )
        remote_manifest = args.remote_record.parent / "transfer-manifest.json"
        run("copy-manifest", [*scp, str(args.manifest), f"{HOST}:{remote_manifest}.partial"])
        run(
            "publish-manifest",
            [*ssh, f"mv {remote_manifest}.partial {remote_manifest}"],
        )
        helpers = (
            "scripts/diagnostics/audit_visual_intake.py",
            "scripts/ops/check_hope_bundle_lecun.py",
            "scripts/ops/verify_tree_transfer.py",
        )
        helper_hashes = {}
        for relative in helpers:
            source = Path(__file__).resolve().parents[2] / relative
            destination = REMOTE_CODE / relative
            helper_hashes[relative] = digest(source)
            check = (
                f"if test -e {destination}; then "
                f"test \"$(sha256sum {destination} | cut -d' ' -f1)\" = {helper_hashes[relative]}; "
                "fi"
            )
            run(f"guard-{source.name}", [*ssh, check])
            run(f"copy-{source.name}", [*scp, str(source), f"{HOST}:{destination}.g2-partial"])
            publish = (
                f"test \"$(sha256sum {destination}.g2-partial | cut -d' ' -f1)\" = "
                f"{helper_hashes[relative]} && mv {destination}.g2-partial {destination}"
            )
            run(f"publish-{source.name}", [*ssh, publish])
        verifier = REMOTE_CODE / "scripts/ops/verify_tree_transfer.py"
        run(
            "verify-partial",
            [
                *ssh,
                f"/home/yspark/projects/DexPrompt/.venv-student/bin/python {verifier} "
                f"--root {remote_partial} --manifest {remote_manifest} --verify",
            ],
        )
        publish_data = (
            f"test ! -e {args.remote_final} && mv {remote_partial} {args.remote_final}"
        )
        run("publish-data", [*ssh, publish_data])
        run(
            "verify-final",
            [
                *ssh,
                f"/home/yspark/projects/DexPrompt/.venv-student/bin/python {verifier} "
                f"--root {args.remote_final} --manifest {remote_manifest} --verify",
            ],
        )
        manifest = json.loads(args.manifest.read_text())
        receipt = {
            "completed": True,
            "training_started": False,
            "data": str(args.remote_final),
            "transfer_manifest": str(remote_manifest),
            "transfer_manifest_sha256": digest(args.manifest),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "helper_sha256": helper_hashes,
        }
        local_receipt = args.out / "remote-receipt.json"
        write_json(local_receipt, receipt)
        run("copy-receipt", [*scp, str(local_receipt), f"{HOST}:{args.remote_record}.partial"])
        run(
            "publish-receipt",
            [*ssh, f"mv {args.remote_record}.partial {args.remote_record}"],
        )
        report.update(
            completed=True,
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
            helper_sha256=helper_hashes,
        )
        write_json(args.out / "status.json", report)
        return 0
    except BaseException as error:
        report["error"] = repr(error)
        write_json(args.out / "status.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
