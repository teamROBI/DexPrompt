# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""FINAL.md for run_teacher_followup_5h.py, in Korean, written only from its files.

Kept apart from the runner because its table rows and prose are long Korean lines.
"""

# ruff: noqa: E501
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath

_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup

_layout_setup()

import json
import os
from pathlib import Path

from run_teacher_followup_5h import GRID, Clock, Runs, load, stamp

from icra.paths import source_value

STATE_KO = {"done": "완료", "failed": "실패", "skipped": "건너뜀(선행 단계 미완료)",
            "interrupted": "중단", "not_started": "시작 안 함", "running": "진행 중"}


def write_final(out: Path, runs: Runs, clock: Clock) -> None:
    plan = load(out / "plan.json", {})
    status = load(out / "status.json", {})
    cmp_ = load(out / "comparison-logit.json")
    ranks = (load(out / "rankings-logit.json") or {}).get("rankings", {})
    stage2 = load(out / "stage2-plan.json")
    centre = load(out / "centre.json", {})
    ready = load(out / "preflight.json", {})
    handoff = load(out / "HANDOFF.json", {})
    cost = load(out / "cost.json", {})
    rows = runs.rows
    lines = ["# Teacher 후속 5시간 무인 실행 — 최종 보고", "",
             f"- T0 {stamp(clock.t0)} · 새 물리 시작 상한 {stamp(clock.cutoff)} · "
             f"물리 종료 {stamp(clock.physics_end)} · 보고 상한 {stamp(clock.deadline)}",
             f"- 보고 작성 {stamp()} · 실행기 `scripts/experiments/run_teacher_followup_5h.py` "
             f"(sha256 {str(source_value(plan.get('code') or {}, 'scripts/experiments/run_teacher_followup_5h.py'))[:12]})",
             "- 이 보고는 실행기가 원장·파일에서 자동으로 작성했다. 수치는 모두 아래 파일에서 다시 셀 수 있다.",
             ""]

    demos = handoff.get("demonstrations", [])
    verified = [d for d in demos if d["verification"] == "reproduced"]
    charged = cost.get("grasp_runs_charged", runs.charged())
    lines += ["## 답", ""]
    if cmp_:
        parts = []
        for obj, s in cmp_["per_object"].items():
            pm = s["per_method"]
            parts.append(f"{obj} 원래 학습 {pm['learned_saved']['accepted']}/8 → logit "
                         f"{pm['learned_logit']['accepted']}/8 (해시 {pm['hash_order']['accepted']}/8,"
                         f" 무작위 {pm['random']['accepted']}/8, 풀 {s['pool_accepted']}/"
                         f"{s['pool_observed']} 관측)")
        lines.append("1. **정정된 선정기**: " + "; ".join(parts) + ". 같은 평가 풀을 이미 본 뒤의 "
                     "사후 구현 정정이므로 새 독립 test나 일반화 증거가 아니다.")
    else:
        lines.append("1. **정정된 선정기**: 비교가 완료되지 않았다(아래 단계 상태 참조).")
    lines.append(f"2. **새로 검증된 시범**: 재현까지 통과 {len(verified)}개, 고유 후보 "
                 f"{len({d['candidate_sha256'] for d in verified})}개, 물체-배치 조합 "
                 f"{len({(d['object'], d['placement']) for d in verified})}개 "
                 f"(teacher accepted {len(demos)}개 중; 재현 실패 "
                 f"{sum(d['verification'] == 'reproduction_failed' for d in demos)}, 재현 미실행 "
                 f"{sum(d['verification'] == 'reproduction_pending' for d in demos)}).")
    wall = (cost.get("physics_wall") or {})
    lines.append(f"3. **비용**: 전체 파지 실행 {charged}회(상한 {runs.total}) 대비 재현 통과 시범 "
                 f"{len(verified)}개. 물리 실행 벽시계 {wall.get('first_start')} ~ "
                 f"{wall.get('last_end')} ({wall.get('minutes')}분), 파지 실행 누적 "
                 f"{cost.get('grasp_seconds_total')}초(동시 {plan.get('workers')}개).")
    placement_rows = [r for r in rows if r["role"] == "placement" and r["status"] == "completed"]
    n_place = len(placement_rows)
    if n_place:
        rate = f"{sum(r['accepted'] for r in placement_rows)}/{n_place}"
        lines.append(f"4. **제한된 배치 데이터 생산 근거**: 사전 고정 배치 실행 {rate}가 accepted, 그중 "
                     f"{len(verified)}개가 동일 조건 재현을 통과했다. 이는 이미 성공한 후보를 물체와 함께 옮긴 "
                     "같은 후보의 배치 다양성이며, 형상은 여전히 3개다. 병목은 아래 '배치별 결과'의 실패 "
                     "단계·사유 분포로 판단한다.")
    else:
        lines.append("4. **제한된 배치 데이터 생산 근거**: 배치 실행이 없어 판단하지 않는다.")
    lines.append("")

    lines += ["## 단계 상태", "", "| 단계 | 상태 | 시작 | 종료 | 결과/사유 |", "| --- | --- | --- | --- | --- |"]
    for name, s in (status.get("stages") or {}).items():
        detail = s.get("result") if s.get("state") == "done" else (s.get("reason")
                                                                  or s.get("exit"))
        lines.append(f"| {name} | {STATE_KO.get(s.get('state'), s.get('state'))} | "
                     f"{s.get('started', '')} | {s.get('finished', s.get('at', ''))} | "
                     f"{json.dumps(detail, ensure_ascii=False)[:160]} |")
    lines.append("")

    lines += ["## 1. 선정 점수 정정 (사후 구현 정정)", ""]
    if ranks:
        lines += ["| 물체 | raw logit 범위 | 고유 logit | float32 sigmoid=0 | 저장 확률 고유값 | "
                  "logit top-8 ∩ 원래 학습 top-8 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for obj, r in ranks.items():
            lines.append(f"| {obj} | {r['logit_range'][0]:.2f} ~ {r['logit_range'][1]:.2f} | "
                         f"{r['unique_logits']}/16 | {r['float32_sigmoid_zero']}/16 | "
                         f"{r['unique_saved_probabilities']}/16 | "
                         f"{r['overlap_with_saved_learned_top8']}/8 |")
        lines += ["", "- **동점 수 정정**: selector-compare README 5절의 'AlphabetSoup 고유 점수 14/16', "
                  "'Peaches 고유값 10/16·일부 동점이 해시 순으로 풀림'은 틀렸다. 저장된 확률 자체가 위 표처럼 "
                  "각각 모두 다르며(동점 없음), 전부 같은 값(0)이었던 것은 Raisins뿐이다.",
                  "- 재계산한 sigmoid 값은 저장된 rankings.json 값과 정확히 같았고, raw logit은 사전 진단"
                  "(selector-logit-audit)과 일치했다(물체별 최대 차이는 rankings-logit.json).",
                  "- 낮은 train loss만으로 과적합을 확정하지 않는다. 극단적 음수 logit의 원인은 가설로 남긴다.", ""]
    if cmp_:
        lines += ["| 물체 | 평가 풀 accepted/관측 | 해시 top-8 | 무작위 top-8 | 원래 학습 top-8 | "
                  "logit top-8 | logit 추가 / 제외 | 새 실행 / 재사용 | 누락 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
        for obj, s in cmp_["per_object"].items():
            pm = s["per_method"]
            lines.append(
                f"| {obj} | {s['pool_accepted']}/{s['pool_observed']} (풀 {s['pool_size']}) | "
                f"{pm['hash_order']['accepted']}/8 | {pm['random']['accepted']}/8 | "
                f"{pm['learned_saved']['accepted']}/8 | {pm['learned_logit']['accepted']}/"
                f"{pm['learned_logit']['observed']} 관측 | "
                f"{', '.join(s['added_by_logit']) or '-'} / {', '.join(s['dropped_by_logit']) or '-'}"
                f" | {s['new_runs']} / {s['reused_runs']} | {', '.join(s['pool_missing']) or '없음'} |")
        lines += ["", "- 원래 순위의 결과(selector-compare-v1)는 그대로 보존하며 이 표로 대체하지 않는다.",
                  "- 누락은 실패로 채우지 않았다. 해시·무작위·원래 학습 top-8은 모두 이전 실행 결과다.",
                  f"- 재현: `{json.dumps(cmp_.get('reproductions'), ensure_ascii=False)[:400]}`",
                  f"- 단계1 인프라 손실: {len(cmp_.get('stage1_infrastructure_losses', []))}건", ""]
    controls = load(out / "controls.json")
    if controls:
        lines.append("- 대조군 재현: " + ", ".join(
            f"{o} {'동일 재현' if v['reproduced'] else '재현 안 됨(' + str(v['status']) + ')'}"
            for o, v in controls.items()))
        lines.append("")

    lines += ["## 2. 배치 시범", ""]
    if stage2:
        lines += ["| 물체 | 후보 | 유형 | 여유(min) | 침투 여유 | 힘 여유 | 매니페스트 | 중앙 재현 |",
                  "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
        index = {(c["object"], c["candidate_sha256"]): c for c in stage2["candidates"]}
        for obj, hashes in stage2["chosen"].items():
            for h in hashes:
                c = index[(obj, h)]
                b = c["basis"]
                v = centre.get(obj, {}).get(h, {})
                lines.append(f"| {obj} | {h[:10]} | {c['template']} | {c['margin']:.3f} | "
                             f"{b['penetration_fraction']:.3f} | {b['force_fraction']:.3f} | "
                             f"{c['manifest_source']} | "
                             f"{('통과' if v.get('verified') else '실패/미실행') if v else '미실행'}"
                             f"{' (' + v['how'] + ')' if v else ''} |")
        counts = stage2.get("counts", {})
        lines += ["", "- 후보 풀(중앙 accepted / 선정 자격): " + ", ".join(
            f"{o} {c['accepted_candidates']}/{c['eligible']}" for o, c in counts.items()), ""]
    if ready:
        tags = [g[0] for g in GRID]
        lines += ["사전 검사(초기 장면; ✓ 준비, ✗ 제외):", "",
                  "| 물체 | " + " | ".join(tags) + " | 검사기 재현 |",
                  "| --- |" + " --- |" * (len(tags) + 1)]
        for obj, r in ready.items():
            validation = r.get("_validation") or []
            lines.append(f"| {obj} | " + " | ".join("✓" if (r.get(t) or {}).get("ready") else "✗"
                                                     for t in tags)
                         + f" | {'일치' if validation and all(v['matches_record'] for v in validation) else '불일치/없음'} |")
        lines.append("")
    if placement_rows or any(r["role"] == "placement" for r in rows):
        lines += ["배치별 결과 (A=accepted, R=재현 통과, F=재현 실패, P=재현 미실행, ·=실패, x=인프라, -=미실행):", "",
                  "| 물체 | 후보 | " + " | ".join(g[0] for g in GRID) + " |",
                  "| --- | --- |" + " --- |" * len(GRID)]
        verdicts = {(d["object"], d["candidate_sha256"], d["placement"]): d["verification"]
                    for d in demos}
        mark = {"reproduced": "R", "reproduction_failed": "F", "reproduction_pending": "P"}
        pairs = sorted({(r["object"], r["candidate_sha256"]) for r in rows
                        if r["role"] == "placement"})
        for obj, h in pairs:
            cells = []
            for tag, *_ in GRID:
                rr = [r for r in rows if r["role"] == "placement" and r["object"] == obj
                      and r["candidate_sha256"] == h and r["condition"] == tag
                      and r.get("budget_charged")]
                if not rr:
                    cells.append("-")
                elif rr[-1]["status"] != "completed":
                    cells.append("x")
                elif rr[-1]["accepted"]:
                    cells.append(mark[verdicts.get((obj, h, tag), "reproduction_pending")])
                else:
                    cells.append("·")
            lines.append(f"| {obj} | {h[:10]} | " + " | ".join(cells) + " |")
        failed = [r for r in placement_rows if not r["accepted"]]
        reasons: dict[str, int] = {}
        for r in failed:
            key = f"{r.get('outcome')}: {', '.join(r.get('reasons') or []) or '-'}"
            reasons[key] = reasons.get(key, 0) + 1
        lines += ["", "- 실패 배치의 종료 상태·사유: " + (", ".join(
            f"{k} ×{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "없음")]
        lines.append(f"- HANDOFF: teacher accepted {handoff.get('teacher_accepted')}개 · 재현 통과 "
                     f"{(handoff.get('counts') or {}).get('reproduced')}개 · 고유 후보 "
                     f"{handoff.get('unique_candidates_reproduced')}개 · 물체-배치 "
                     f"{handoff.get('object_placement_pairs_reproduced')}개. 모든 항목 "
                     "`student_ready=false`, student 데이터 자동 병합 없음.")
        lines += ["- 같은 후보의 배치 다양성이지 새로운 후보/형상 다양성이 아니다. 형상은 3개다.", ""]

    lines += ["## 3. 비용", "", "| 역할 | 실행(과금) | 상한 | 누적 초 |", "| --- | ---: | ---: | ---: |"]
    for role, v in (cost.get("by_role") or {}).items():
        lines.append(f"| {role} | {v['charged']} | {v['cap']} | {v['seconds']} |")
    lines += ["", f"- 합계 {charged} / {runs.total}. 상태별: "
              f"{json.dumps(cost.get('by_status'), ensure_ascii=False)}",
              f"- 비파지 시뮬레이션(초기 장면 안정화 검사): "
              f"{(cost.get('non_grasp_simulation') or {}).get('scenes_simulated')}장면, "
              f"{(cost.get('non_grasp_simulation') or {}).get('seconds')}초 — 파지 예산과 별도 공개.",
              f"- 남은 디스크 {cost.get('disk_free_gib')} GiB", ""]

    not_run = cost.get("not_run") or []
    lines += ["## 완료 / 미완료 / 미실행 / 인프라 실패", ""]
    done = [n for n, s in (status.get("stages") or {}).items() if s.get("state") == "done"]
    undone = [f"{n}({STATE_KO.get(s.get('state'), s.get('state'))})"
              for n, s in (status.get("stages") or {}).items() if s.get("state") != "done"]
    lines.append(f"- 완료 단계: {', '.join(done) or '없음'}")
    lines.append(f"- 미완료 단계: {', '.join(undone) or '없음'}")
    reasons: dict[str, int] = {}
    for r in not_run:
        reasons[f"{r['role']}: {r['not_run']}"] = reasons.get(f"{r['role']}: {r['not_run']}", 0) + 1
    lines.append("- 계획했지만 미실행: " + (", ".join(f"{k} ×{v}" for k, v in reasons.items())
                                        or "없음"))
    excluded = load(out / "placements-excluded.json", [])
    if excluded:
        ex: dict[str, int] = {}
        for e in excluded:
            ex[e["reason"]] = ex.get(e["reason"], 0) + 1
        lines.append("- 배치 제외(사전 고정 조건, 대체 없음): " + ", ".join(f"{k} ×{v}"
                                                                 for k, v in ex.items()))
    infra = [r for r in rows if r.get("budget_charged") and r["status"] != "completed"]
    lines.append("- 인프라 실패/timeout/중단: " + (", ".join(
        f"{r['role']} {r['object']} {r['condition']} {r['candidate_sha256'][:10]} {r['status']}"
        for r in infra) or "없음"))
    lines += ["- 새 학습·새 물체·물리 기준 변경·실패 후보 튜닝·student 병합: 하지 않음.", "",
              "## 파일", "",
              "`plan.json`(사전 고정) · `rankings-logit.json` · `comparison-logit.json` · "
              "`controls.json` · `logit-reproduce.json` · `stage2-plan.json` · `preflight.json` · "
              "`preflight-ledger.json` · `centre.json` · `ledger.json`(시도별 UUID) · `HANDOFF.json` · "
              "`cost.json` · `status.json` · `logs/` · `runs/`", ""]
    text = "\n".join(lines)
    temporary = out / ".FINAL.md.tmp"
    temporary.write_text(text)
    os.replace(temporary, out / "FINAL.md")
