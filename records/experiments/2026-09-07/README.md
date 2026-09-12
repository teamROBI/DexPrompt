<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Historical measurements, 2026-09-07

These are retained raw records, not the current system's performance report.
The arm execution measurements predate corrective commit **55cf6ce** and are
withdrawn as method-comparison evidence. Raw files are preserved unchanged.
See [the audit](../2026-09-08/audit/README.md) for its findings and
[RESUME.md](../../../RESUME.md) for current remaining work.

## Annotation/refiner matrix

[matrix.tsv](matrix.tsv) records 500 initializations per template per cell,
with three templates and an object set described by the run as 100 objects.

| Annotation | Refiner | Initializations | Refined files | Yield |
|---|---|---:|---:|---:|
| old | grasp | 1500 | 21 | 1.4% |
| old | grasp_admm | 1500 | 275 | 18.3% |
| new | grasp | 1500 | 26 | 1.7% |
| new | grasp_admm | 1500 | 325 | 21.7% |

These are refinement counts, not physics-validated lifts or student success.
The TSV does not contain per-candidate identities, seeds, or the full object
manifest; preserve that limitation when interpreting it.

## Evolution

[evo.tsv](evo.tsv) records six rounds, described by the run as twelve templates
over sixty objects. The final row contains 5085 refined slots and 1637 stored
success entries under the earlier evaluation setting. Reused filenames mean
round totals are neither independent new-grasp counts nor a sum of all distinct
grasps ever generated.

The later local success directory contained 4495 entries after a reported kp=50
reevaluation. That reevaluation is not tabulated in evo.tsv. Treat 4495 as a
historical artifact count until a fresh replay with immutable candidate IDs and
the complete evaluation configuration confirms it. Do not replace the TSV's
1637 with 4495 or use either as arm/policy performance.

## Static scene filtering

[scene_filter.log](scene_filter.log) accounts for all 4495 inputs:

| Historical log category | Count |
|---|---:|
| Blocked by table **or** closes on nothing | 2435 |
| Kept by the static sweep | 1662 |
| Object not in the scene | 398 |
| Total | 4495 |

The log does not separate table collisions from contact failures. It also does
not establish why the 398 objects were absent; do not relabel them as measured
scale mismatches. “Kept” allows an adjusted roll/depth pose and is not proof of
arm reachability or sustained lift.

The filter used grasp_qpos when this log was collected; it was later changed to
squeeze_qpos in 9ded65b. That change does not alter the historical log. The
standalone filter also lacks the collector's explicit scale guard. Dexonomy can leave a contact gap at grasp_qpos and
close at squeeze_qpos. The filter's 37.0% retention is therefore a historical
program output, not a validated current feasibility rate.

## Withdrawn arm execution baseline

[scene_baseline.log](scene_baseline.log) records 60 attempted candidates and
zero lifts: 42 “nothing closes,” 17 arm failures, and 1 off-pose failure.
Its sample contains 52 Power Disk and 8 Power Sphere candidates.

Orientation overwrite, contact-state scoring, and sampling issues invalidate
using this run to estimate corrected executor performance or to conclude that
Dexonomy grasps cannot be executed. It did not execute all 1662 filtered entries.
It is not the baseline that RL “has to beat.”

Collect new logs after the corrections, with a bounded scene, fixed scales,
stratified candidate manifest, complete settings, and all failure denominators.
The paper currently makes no numerical policy-performance claim from these logs.
