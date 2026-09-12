"""Read-only sidecar: compare preselected final policies; never count missing runs as failures."""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
import math
import statistics
import time
from pathlib import Path

EPOCHS = (1, 3, 6, 9, 13)
SEEDS = (1000, 1001, 1002)
FRAMES = (0, 8, 16, 32, 64, 200, 450, 700)
LEGAL = {
    None,
    "penetration",
    "table_collision",
    "obstacle_collision",
    "digit_force",
    "distractor_displacement",
    "approach_displacement",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evaluation_rows(result):
    output = {}
    for item in result["evaluations"]:
        epoch, summary = item["epoch"], item["summary"]
        if epoch not in EPOCHS or epoch in output:
            raise ValueError("unexpected or duplicate epoch")
        if item["preselected_final"] != (epoch == 13):
            raise ValueError("final epoch changed")
        if not summary["completed"] or summary["attempts"] != 3:
            raise ValueError("incomplete evaluation")
        if len(summary["episodes"]) != 1:
            raise ValueError("expected one fixed task")
        outcome = summary["episodes"][0]["outcome"]
        rows = outcome["replays"]
        if len(rows) != 3 or sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError("evaluation seeds differ")
        if any(r["failure_reason"] not in LEGAL for r in rows):
            raise ValueError("infrastructure error is not a task failure")
        if any(type(r["teacher_compliant_lift"]) is not bool for r in rows):
            raise ValueError("success must be explicit")
        successes = sum(r["teacher_compliant_lift"] for r in rows)
        if successes != summary["compliant_successes"]:
            raise ValueError("success accounting mismatch")
        output[epoch] = dict(
            successes=successes,
            attempts=3,
            rows=rows,
            dataset_sha256=summary["dataset_sha256"],
            sources=summary["evaluation_sources"],
            model_sha256=outcome["model_sha256"],
            initial_state_sha256=outcome["initial_state_sha256"],
            checkpoint_sha256=outcome["checkpoint_sha256"],
        )
    return output


def compare_evaluations(a, b):
    left, right = evaluation_rows(a), evaluation_rows(b)
    if set(left) != set(EPOCHS):
        raise ValueError("baseline is incomplete")
    table = []
    for epoch in EPOCHS:
        baseline_row, r = left[epoch], right.get(epoch)
        if r is not None:
            for key in ("dataset_sha256", "sources", "model_sha256", "initial_state_sha256"):
                if baseline_row[key] != r[key]:
                    raise ValueError(f"A/B contract differs: {key}")
        table.append(dict(epoch=epoch, preselected_final=epoch == 13, A=baseline_row, B=r))
    final = None
    if 13 in right:
        la = {r["seed"]: r["teacher_compliant_lift"] for r in left[13]["rows"]}
        rb = {r["seed"]: r["teacher_compliant_lift"] for r in right[13]["rows"]}
        final = dict(
            gained_seeds=[s for s in SEEDS if rb[s] and not la[s]],
            lost_seeds=[s for s in SEEDS if la[s] and not rb[s]],
        )
    return dict(evaluations_complete=set(right) == set(EPOCHS), epochs=table, final=final)


def compare_diagnostics(a, b):
    def keyed(d):
        if not d["completed"] or d["new_action_chunks"] != 24:
            raise ValueError("teacher observation diagnostic incomplete")
        rows = {}
        if len(d["cases"]) != len(FRAMES):
            raise ValueError("teacher frames differ")
        for case in d["cases"]:
            if not case["preprocessing_equal"]:
                raise ValueError("preprocessing mismatch")
            for sample in case["samples"]:
                key = (case["frame"], sample["seed"])
                if key in rows:
                    raise ValueError("duplicate diagnostic")
                value = sample["errors"]["8"]["arm_rmse_rad"]
                if not math.isfinite(value):
                    raise ValueError("nonfinite diagnostic")
                rows[key] = (case["observation_sha256"], value)
        if set(rows) != {(f, s) for f in FRAMES for s in SEEDS}:
            raise ValueError("teacher frame/seed mismatch")
        return rows

    left, right = keyed(a), keyed(b)
    rows = []
    for (frame, seed), (observation, value) in left.items():
        other_observation, other_value = right[(frame, seed)]
        if observation != other_observation:
            raise ValueError("teacher input changed")
        rows.append(
            dict(
                frame=frame,
                seed=seed,
                A_arm_rmse_rad=value,
                B_arm_rmse_rad=other_value,
                delta_B_minus_A=other_value - value,
            )
        )
    return dict(
        rows=rows,
        improved=sum(r["delta_B_minus_A"] < 0 for r in rows),
        median_delta=statistics.median(r["delta_B_minus_A"] for r in rows),
    )


def render(report):
    lines = [
        "# Low-noise 단일 요인 비교",
        "",
        f"상태: {report['status']}",
        "",
        "최종 정책은 사전 고정 epoch 13이다. 누락된 평가는 미완료로 표시한다.",
        "같은 병 한 과제·같은 seed 3개이며 일반화 또는 학습 seed 재현성 실험이 아니다.",
        "가중 학습 손실과 기존 손실의 크기를 직접 비교하지 않는다.",
        "",
        "| epoch | A 완주 | B 완주 |",
        "| --- | --- | --- |",
    ]
    for row in report.get("comparison", {}).get("epochs", []):
        b = "미완료" if row["B"] is None else f"{row['B']['successes']}/3"
        lines.append(f"| {row['epoch']} | {row['A']['successes']}/3 | {b} |")
    diagnostic = report.get("teacher_observations")
    if diagnostic:
        lines += [
            "",
            f"첫 8행동 팔 RMSE가 감소한 teacher 관측: {diagnostic['improved']}/24.",
            f"B−A 중앙값: {diagnostic['median_delta']:.6f} rad.",
            "Teacher 관측에서의 행동 오차 개선과 폐루프 파지 성공은 별도 판정한다.",
        ]
    if report.get("error"):
        lines += ["", f"검사 오류: {report['error']}"]
    return "\n".join(lines) + "\n"


def snapshot(args):
    status = read(args.job / "status.json")
    baseline = read(args.baseline)
    current = (
        read(args.job / "results.json")
        if (args.job / "results.json").exists()
        else {"evaluations": []}
    )
    report = dict(
        status="pending",
        runner_status=status,
        completed=False,
        comparison=compare_evaluations(baseline, current),
    )
    path = args.job / "teacher-observations/diagnosis.json"
    if path.exists():
        report["teacher_observations"] = compare_diagnostics(
            read(args.baseline_diagnostic), read(path)
        )
    if status["phase"] == "failed":
        report.update(status="runner_failed", error=status.get("error"))
    elif status["phase"] == "completed":
        if not report["comparison"]["evaluations_complete"] or "teacher_observations" not in report:
            raise ValueError("runner completed but required results missing")
        report.update(status="completed", completed=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-diagnostic", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--wait-seconds", type=int, default=21600)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            report = snapshot(args)
        except Exception as exc:
            report = dict(status="report_failed", completed=False, error=repr(exc))
        if report["status"] == "pending" and time.monotonic() >= deadline:
            report["status"] = "wait_timeout"
        write(args.out / "comparison.json", report)
        (args.out / "README.md").write_text(render(report))
        if report["status"] != "pending" or not args.watch:
            return 0 if report["completed"] else 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
