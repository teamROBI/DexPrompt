<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 확대 실험: 예측 후보 93개, align seed 0·1·2, 두 오프셋 조건

물체 1개(`core_bottle_6444875e3217bf891f5f48a891d827bd`, scale 0.11),
템플릿 `33_Inferior_Pincer`. 탐색적 확대이며 물체 간 일반화 검증이 아니다.
한쪽 조건에만 seed를 추가하지 않았고, 필터 출력 수는 맞추지 않았다.

## 고정한 것

- GraspGenX 예측 1회, 후보 93개. zero 조건은 같은 후보의 복제본으로
  `grip_offset`만 0이다(후보 매핑 sha256 `a351f449654cd914…`).
- 템플릿 라이브러리: `evo_hx5_d20`에서 `cp -rL`로 실체화한 118개.
  내용 카탈로그 해시 `1ecf098bb859206d`, 네 실험 모두 동일. 기본 템플릿
  sha256 `6b50ec3aa3734f50…`. 선택 규칙은 내용 해시 정렬 + seed 난수.
- 필터: roll 12 × depth ±30 mm 5단계, IK 120회·damping 0.05, thumb 50 mm.
- 정렬: `--in-plane 1`, `--opt-steps 25`, `--keep 100`(= op=init `n_final`).
- 실행기: 기본 `ExecutionConfig`(lift_force null, hold 4 N), 실행 seed 0,
  yaw 1개, 64×64. 이동량 제한이나 제어 변경은 넣지 않았다.

## 주 지표 — 예측 후보 93개 기준

| 조건 | 필터 통과 | 정련 도달 | 실행 진입 | **실행 완주** | **완주 + 6방향 유지** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보정 offset | 57 | 24 | 15 | **7** | **3** |
| offset 0 | 40 | 23 | 16 | **14** | **5** |

**정련까지 도달한 후보 범위는 24 대 23으로 거의 같다.** 차이는 그 뒤의 선별과
실행에서 벌어진다. 정련 *파일* 수(32 대 60)와 정련에 도달한 *후보* 수는 다른
값이므로 섞지 않는다.

완주 후보 — 보정: grasp_7, 14, 31, 36, 38, 41, 50 /
zero: grasp_10, 20, 51, 53, 64, 66, 73, 74, 75, 77, 83, 84, 86, 90.
6방향까지 유지 — 보정: grasp_38, 41, 50 / zero: grasp_20, 51, 66, 75, 77.

**10개 후보에서 관측한 방향(보정 3/10 대 zero 1/10)이 93개에서는 뒤집혔다.**
오프셋 보정의 이점은 필터 단계에 있고(통과 후보 57 대 40), 정련 도달 범위는
사실상 같으며(24 대 23), 차이는 선별·실행에서 생긴다. 이 실험 하나로 최종
우열을 확정하지 않는다.

## 보조 지표 — 단계별 입력·통과·실행 횟수

필터(조건당 변형 5580 = 후보 93 × roll 12 × depth 5):

| 조건 | out_of_reach | ik_failure | collision | thumb_far | passed | kept 변형 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 보정 | 0 | 3672 | 398 | 41 | 1469 | 165 |
| zero | 0 | 3617 | 452 | 850 | 661 | 112 |

정렬(조건당 3회, seed 0·1·2). 괄호는 패딩 이전 통과 수:

| 조건 | seed 자세 | pose | loss | collision | qp | init 파일 | 재도달 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 보정 | 165 | 165 | 27 / 23 / 22 | 67 / 66 / 77 | 66 / 64 / 59 | 100 × 3 | 48 등 |
| zero | 112 | 112 | 15 / 14 / 13 | 48 / 39 / 51 | 39 / 36 / 37 | 100 × 3 | 51 등 |

`init 파일 100`은 `--keep 100`(op=init `n_final`) 상한이다. 정렬 자체는
**165개와 112개 모두 수행되고**, 그 결과에서 최대 100개를 골라 파일로 남긴다.
"100개만 초기화했다"는 뜻이 아니다. 두 조건에 같은 출력 예산을 주었으므로
이 비교는 유효하며, 변형이 많은 보정 조건이 더 많이 잘리는 것은 사실이지만
그것이 최종 수율을 떨어뜨렸는지는 예산을 바꾼 별도 실험 없이 알 수 없다.

정련 이후:

| 조건 | 정련 파일 | select 통과 | 실행 | 완주 | 상승(중앙값) | 최대 손가락 힘 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 보정 | 32 | 18 | 18 | 9 | 14.64 cm | 4.1–21.3 N |
| zero | 60 | 32 | 32 | 24 | 14.96 cm | 3.5–22.1 N |

중복 파지는 한 번만 실행했고, 이번 확대에서는 seed 간 중복이 없었다.
select 단계 탈락 사유는 보정에서 `open_hand_inside_object` 12,
`hand_through_table` 2(32개 중 18개 통과), zero에서
`open_hand_inside_object` 26, `hand_through_table` 2(60개 중 32개 통과)다.
정련에는 성공했지만 여기서 탈락한 파지도 "정련 도달"에는 포함된다.
귀속은 `*-distinct.json`의 `parents`에 파지별로 모두 남는다.

## 실행 실패의 분해

접근 실패를 분모에서 빼지 않는다. 아래는 실행 시도 기준이다.

| 조건 | 접근 실패 | 닫기 실패 | 리프트 실패 | 완주 |
| --- | ---: | ---: | ---: | ---: |
| 보정 (18회) | 7 | 1 | 1 | 9 |
| zero (32회) | 7 | 0 | 1 | 24 |

접근 실패는 후보마다 여섯 방향을 시도한 뒤의 결과이며, 시도별 사유는 다음과
같다(보정 / zero): `ik_failure` 16 / 9, `quality_approach_displacement` 18 / 15,
`quality_penetration` 7 / 13, `quality_table_collision` 1 / 5.
이는 **현재 실행기의 경로 검증 실패**이며 물리적 도달 불가능 판정이 아니다.

## 별도 품질 지표 — 6방향 외력 평가

`scripts/audit_grasp_eval.py`가 조기 중단만 제거하고 ±x·±y·±z를 모두 돌린다.
임계값 50 mm·15°, 마찰 `[0.6, 0.02]`, 물체 질량 0.1 kg.
완주한 파지 기준: 보정 9개 중 4개, zero 24개 중 12개가 여섯 방향을 유지했고,
후보 단위로는 3 대 5다.

**이 통과는 시뮬레이터에서 한 가지 마찰·질량 설정으로 여섯 방향의 외력을
견뎠다는 관측이며, 수학적 force-closure의 증명이 아니다.** 방향 집합도 여섯
개의 축 방향에 한정되고, 임계값도 이 설정의 값이다.

## 한계

- 물체 1개·타입 1개다. 물체 간 일반화는 검증하지 않았다.
- `--keep 100` 출력 예산이 변형이 많은 조건을 더 잘라낸다(165 → 100). 같은
  예산에서의 비교로는 유효하며, 예산을 바꾼 실험은 하지 않았다.
- 조건 간 실행 횟수가 다르다(18 대 32). 주 지표를 후보 단위로 보는 이유다.
- 실행 완주는 이 실행기·이 접근 방식의 결과다. 접근 실패 14건(보정 7, zero 7)
  은 경로 생성 문제이며 파지 자체의 판정과 분리해서 읽어야 한다. 사유 집계의
  합이 14보다 큰 것은 한 시도가 여섯 방향을 모두 시험하기 때문이다.
- 정렬 이동량은 측정하지 않았고 제한도 걸지 않았다.

## 기록

- [chain-offset.json](chain-offset.json), [chain-zero.json](chain-zero.json)
  — 후보별 전체 사슬과 요약(`scripts/report_grasp_chain.py` 출력)
- [filter-comparison.json](filter-comparison.json) — 필터 단계 수치와 설정
- [offset-distinct.json](offset-distinct.json), [zero-distinct.json](zero-distinct.json)
  — 실행한 파지, 원본 sha256, 부모 후보 전부
- `select-*.verdicts.json` — select 단계 파일별 판정
- [eval-audit-offset/](eval-audit-offset), [eval-audit-zero/](eval-audit-zero)
  — 방향별 외력 기록

재현:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/compare_grip_offset.py --template 33_Inferior_Pincer \
  --objects out/offset-compare/objects --observation out/offset-compare/observation \
  --predicted out/offset-compare/predicted --out out/offset-compare-93 --max-per-object 93
# 조건 × seed 마다: new_tmpl 을 cp -rL 로 준비 → align_grasps --seed → op=grasp_admm
#   → collect_teacher_dataset select → collect → audit_grasp_eval
python3 scripts/report_grasp_chain.py --template 33_Inferior_Pincer --object <obj> \
  --exp big1_offset_s0 --exp big1_offset_s1 --exp big1_offset_s2 \
  --selection out/offset-compare-93/manifests/offset-s0.json ... \
  --executed out/offset-compare-93/executed-offset \
  --eval-audit out/offset-compare-93/eval-audit-offset \
  --predictions out/offset-compare/predicted/33_Inferior_Pincer/<obj>.yml \
  --out out/offset-compare-93/chain-offset.json
```

## 다음에 정해야 할 것

오프셋 비교는 여기서 마무리한다. 다음은 검증된 파지를 고정한 작은 RL 준비이고,
기존 접근 실행기는 비교 기준으로 유지한다. 데이터 확대는 같은 병의 seed·예산
탐색이 아니라 다른 물체·다른 파지 타입으로 간다. `--keep` 예산 실험은 필요할
때 별도로 하며 RL의 선행 조건이 아니다.
