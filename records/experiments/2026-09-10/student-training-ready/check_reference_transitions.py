"""Check first accepted reference transitions using actual sealed demonstration data."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from icra.dataset import load_manifest  # noqa: E402
from icra.goal_memory import GoalMemory  # noqa: E402
from icra.student_cache import StudentCache  # noqa: E402
from icra.student_data import EpisodeChunks, observation_tensor  # noqa: E402


def main():
    torch.set_num_threads(1)
    cfg = json.loads((ROOT / "configs/hx5_student.bottle-can.local.json").read_text())
    root = ROOT / "out/student-bottle-can-v1"
    manifest = load_manifest(root)
    cache = StudentCache(ROOT / "out/student-bottle-can-v1-cache", root, manifest)
    data = EpisodeChunks(
        root,
        manifest,
        "train",
        64,
        518,
        cache=cache,
        observation_version=2,
        goal_reference_policy=cfg["goal_reference_policy"],
    )
    report, offset = [], 0
    for i, entry in enumerate(data.entries):
        episode = data.episode(i)
        first = data.reference_indices[i]
        pixels = episode["target_pixels"]
        checks = {0, len(pixels) - 1}
        for tick in first:
            if tick is not None:
                checks.update(t for t in [tick - 1, tick, tick + 1] if 0 <= t < len(pixels))
        memory = GoalMemory(cfg["goal_reference_policy"])
        for tick in range(len(pixels)):
            obs = {key: value[tick] for key, value in episode.items()}
            memory.update(obs, entry["condition"])
            if tick in checks:
                online = observation_tensor(
                    obs,
                    entry["condition"],
                    518,
                    observation_version=2,
                    goal_references=memory.references,
                )
                offline = data[offset + tick]["obs"]
                assert online.keys() == offline.keys()
                assert all(torch.equal(value, offline[key]) for key, value in online.items())
        report.append(
            {
                "episode": entry["episode_id"],
                "checked_ticks": sorted(checks),
                "reference_ticks": first,
                "reference_pixels": [
                    int(pixels[t, c]) if t is not None else None
                    for t, c in zip(first, [0, 2], strict=True)
                ],
            }
        )
        offset += len(pixels)
    output = {
        "passed": True,
        "training_started": False,
        "episodes": report,
        "observations_compared": sum(len(r["checked_ticks"]) for r in report),
    }
    (Path(__file__).parent / "reference-transitions.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
