"""Bowl scale x matcher steps: what survived each stage, and why it stopped.

Four cells, one object and one grasp type. The stages are kept apart because
they fail for different reasons and a single "0" hides which: the initialisation
filters inside op=init (including the loss filter, whose verdict the padding
then overrides), the reach recheck after alignment, the ADMM refinement, the
static screen, and the dynamic run. A cell that produced no candidate is
reported as that, not as an error -- the two now carry different exit codes.

    third_party/dexonomy/.venv/bin/python scripts/diagnostics/report_bowl_scale_matcher.py \\
        --out out/bowl-scale/summary.json
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
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = (("0.05", 25, "s005"), ("0.05", 4, "s005o4"),
         ("0.08", 25, "s008"), ("0.08", 4, "s008o4"))
TYPE = "10_Power_Disk"
SEEDS = (0, 1, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "out" / "bowl-scale")
    parser.add_argument("--alignment", type=Path, default=ROOT / "out" / "alignment")
    parser.add_argument("--dexonomy", type=Path,
                        default=ROOT / "third_party" / "dexonomy" / "output")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def filter_counts(run: Path, tag: str) -> dict:
    """Prediction and reach/collision filtering, from the run logs."""
    base = tag[:4] if tag.endswith("o4") else tag
    out = {}
    for log in sorted(run.glob("run*.log")):
        text = log.read_text()
        for line in text.splitlines():
            found = re.match(r"(\d+) grasps -> \S+/" + base + r"/predicted/", line)
            if found:
                out["predicted"] = int(found[1])
            found = re.match(r"variants: .*passed (\d+)", line)
            if found and "variants_passed" not in out:
                pass
    # the per-tag lines appear in order; read them from the tag's own section
    for log in sorted(run.glob("run*.log")):
        section, keep = None, {}
        for line in log.read_text().splitlines():
            if line.startswith("##### scale"):
                section = line.split()[2]
            found = re.match(r"variants: out_of_reach \d+, ik_failure (\d+), "
                             r"collision (\d+), thumb_far (\d+), passed (\d+)", line)
            if found and section:
                keep.setdefault(section, {}).update(
                    ik_failure=int(found[1]), collision=int(found[2]),
                    thumb_far=int(found[3]), variants_passed=int(found[4]))
            found = re.match(r"(\d+) reachable, collision-free", line)
            if found and section:
                keep.setdefault(section, {})["reachable"] = int(found[1])
            found = re.match(r"(\d+) grasps -> ", line)
            if found and section:
                keep.setdefault(section, {})["predicted"] = int(found[1])
        scale = "0.05" if tag.startswith("s005") else "0.08"
        if scale in keep and keep[scale]:
            out.update(keep[scale])
    return out


def init_filters(alignment: Path, tag: str) -> dict:
    """op=init's own filters, before valid_index_padding refills them."""
    stages: dict = {}
    for seed in SEEDS:
        directory = alignment / f"bowl{tag}_{TYPE}_s{seed}_hx5_d20" / TYPE
        found = sorted(directory.glob("*.provenance.json")) if directory.is_dir() else []
        if not found:
            continue
        for stage in json.loads(found[0].read_text()).get("filters", []):
            row = stages.setdefault(stage["filter"], {"passed": 0, "in": 0, "padded_back": 0})
            row["passed"] += len(stage["seeds_passed"])
            row["in"] += len(stage["seeds_in"])
            row["padded_back"] += stage["padded_back"]
    return stages


def alignment_counts(alignment: Path, tag: str) -> dict:
    produced, kept, reasons = [], [], Counter()
    for seed in SEEDS:
        directory = alignment / f"bowl{tag}_{TYPE}_s{seed}_hx5_d20" / TYPE
        files = [p for p in directory.glob("*.json")
                 if not p.name.endswith(".provenance.json")] if directory.is_dir() else []
        entries = json.loads(files[0].read_text())["aligned"] if files else []
        produced.append(len(entries))
        kept.append(sum(e["kept"] for e in entries))
        for entry in entries:
            if entry["kept"]:
                reasons["kept"] += 1
            elif not entry["ik_passed"]:
                reasons["ik_failure"] += 1
            elif entry["collisions"]:
                reasons["collision " + " | ".join(map(str, entry["collisions"][0][:2]))] += 1
    return {"aligned": produced, "kept_by_reach_recheck": kept,
            "reasons": dict(reasons.most_common())}


def refinement(dexonomy: Path, tag: str) -> dict:
    counts, rejects = [], Counter()
    for seed in SEEDS:
        experiment = dexonomy / f"bowl{tag}_{TYPE}_s{seed}_hx5_d20"
        counts.append(len(list(experiment.rglob("grasp_data/**/*.npy")))
                      if experiment.is_dir() else 0)
        log = experiment / "log" / "grasp_admm_0" / "main.log"
        if log.is_file():
            text = log.read_text()
            for name, pattern in (("required contact", r"required:"),
                                  ("collision ho", r"collision ho"),
                                  ("collision hh", r"collision hh"),
                                  ("no contact", r"no contact"),
                                  ("qp error", r"bad QP error"),
                                  ("qpos limit", r"qpos limit")):
                n = len(re.findall(pattern, text))
                if n:
                    rejects[name] += n
    return {"refined": counts, "rejections": dict(rejects)}


def screen(run: Path, tag: str) -> dict:
    verdicts: Counter = Counter()
    for seed in SEEDS:
        path = run / "manifests" / f"{tag}-{TYPE}-s{seed}.verdicts.json"
        if path.is_file():
            verdicts.update(json.loads(path.read_text()).values())
    return dict(verdicts)


def dynamic(run: Path, tag: str) -> dict:
    path = run / tag / "executed" / "attempts.json"
    if not path.is_file():
        return {"attempts": 0, "completed": 0, "statuses": {}}
    rows = json.loads(path.read_text())
    return {"attempts": len(rows),
            "completed": sum(bool(r["accepted"]) for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows))}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {}
    for scale, matcher, tag in CELLS:
        cell = {
            "scale": scale, "matcher_steps": matcher, "tag": tag,
            "filter": filter_counts(args.run, tag),
            "init_filters": init_filters(args.alignment, tag),
            "alignment": alignment_counts(args.alignment, tag),
            "refinement": refinement(args.dexonomy, tag),
            "static_screen": screen(args.run, tag),
            "dynamic": dynamic(args.run, tag),
        }
        refined = sum(cell["refinement"]["refined"])
        if cell["dynamic"]["attempts"]:
            cell["outcome"] = ("completed" if cell["dynamic"]["completed"]
                               else "executed, none completed")
        elif refined == 0:
            cell["outcome"] = "no candidates: nothing survived to refinement"
        else:
            cell["outcome"] = "refined but not executed"
        report[f"scale {scale}, matcher {matcher}"] = cell

    print(f"{'cell':<26}{'reach':>7}{'kept':>16}{'refined':>14}{'screen':>10}"
          f"{'run':>6}{'done':>6}   outcome")
    for name, cell in report.items():
        print(f"{name:<26}{cell['filter'].get('reachable', 0):>7}"
              f"{str(cell['alignment']['kept_by_reach_recheck']):>16}"
              f"{str(cell['refinement']['refined']):>14}"
              f"{sum(cell['static_screen'].values()):>10}"
              f"{cell['dynamic']['attempts']:>6}{cell['dynamic']['completed']:>6}"
              f"   {cell['outcome']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
