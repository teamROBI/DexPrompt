# 후보 귀속과 템플릿 선택 고정, 그리고 재실행

요청한 순서대로 진행했다: ① 후보 ID와 실제 템플릿 선택 보존 → ② 같은 입력 2회
실행으로 재현성 확인 → ③ 성공 파지의 op=eval 판정을 방향별로 기록 → ④ 확대는
아직 하지 않았다. 물체 1개, 템플릿 `33_Inferior_Pincer`, 예측 후보 10개.

## 0. 이전 보고 정정: 진화 템플릿 라이브러리가 깨져 있었다

`evo_hx5_d20/new_tmpl/33_Inferior_Pincer`의 118개 항목은 **그 실험의
`grasp_data`를 가리키는 상대 심볼릭 링크**다. `cp -r`로 복사하면 링크만 복사되어
새 실험 디렉터리에서 전부 끊긴다. 즉 [이전 문서](../grip-offset-pipeline/README.md)의
seed 0–5 표는 "두 조건 모두 진화 템플릿 사용"이 아니라 **기본 템플릿 1개만**
사용한 결과다. Dexonomy 로더는 읽기 실패를 로그로만 남기고 계속 진행한다.

`cp -rL`로 다시 만들고 모두 실측했다. 그 표의 refine 수치(보정 2, zero 4)는
철회한다. 아래가 실제 라이브러리를 쓴 수치다. 새 `choose_templates`는 템플릿을
**내용 해시로** 읽고 고르므로 이 실패가 다시 조용히 지나가지 않는다.

## 1. 보존한 사슬

`align_grasps`가 물체마다 `<object>.provenance.json`을 쓴다.

```
prediction_id (grasp_5)
  → variant_id (grasp_5_r270_d30, roll·depth·grip_offset·thumb_gap)
    → template (파일 sha256 + 선택한 접촉점 index + n_evo)
      → init 파일 (0_11_grasp.npy)  ← 필터 단계별 판정과 함께
        → refine 파일 (같은 이름)
          → select 판정 (kept / hand_through_table / open_hand_inside_object …)
            → 실행 episode
```

`op=init`는 최종 배치의 행 번호로 파일 이름을 짓는다. 그래서 matcher 출력에
`seed_id` 열을 넣어 필터의 재정렬·패딩을 함께 따라가게 하고, 마지막 순서로
파일 ↔ 변형을 잇는다. 자세 이동량은 이 귀속과 무관하다(이동 제한은 넣지 않았다).

## 2. 템플릿 선택을 고정했다

Dexonomy `HandTemplateLoader`는 비동기 버퍼에서 `torch.randint`로 템플릿을
고르고, 파일 목록은 `set()`에서 나온다. `--seed`만으로는 재현되지 않는다.
이제 정렬 전에 seed 자세마다 (템플릿 파일, 접촉점 index)를 정해 기록하고,
`get_batched_data`를 실행 범위 안에서 그 선택으로 대체한다. 카탈로그는 경로가
아니라 **내용 해시 순서**라 라이브러리를 어느 디렉터리로 복사해도 같은 선택이
나온다. 접촉점은 로더 자신의 `_load_data`를 쓰되 그 안의 난수만 고정한다.

**같은 입력 2회 실행(seed 0):** 템플릿 선택·접촉점·재도달 수·refine 수가 모두
같고, refine 결과 파일이 sha256까지 동일했다(양쪽 조건 모두).

## 3. init 파일 수는 통과 수가 아니다

`valid_index_padding`이 최소 개수를 맞추려 탈락 후보를 되돌린다. 이제 패딩
**이전** 판정을 필터별로 남긴다(보정 arm, 입력 18개):

| align seed | pose | loss | collision | qp | fps | init 파일 | 재도달 | refine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 5 | 10 | 9 | 18 | 18 | 10 | 7 |
| 1 | 18 | 2 | 8 | 7 | 18 | 18 | 10 | 4 |
| 2 | 18 | 1 | 10 | 7 | 18 | 18 | 7 | 1 |

zero arm(입력 8개): loss 3/0/0, collision 5/3/2, qp 2/0/3, refine 2/0/1.
`loss`(접촉 거리·법선 잔차)가 가장 큰 탈락 지점이고, init 파일 수는 항상 입력
수와 같다 — 전부 패딩으로 되돌아온 것이다. `qp_error` 값도 함께 저장한다.

## 4. 예측 후보까지 이어진 결과

3개 seed의 refine 결과를 중복(sha256) 제거 후 전부 실행했다. 실행 설정은
동일하다(기본 `ExecutionConfig`, 실행 seed 0, yaw 1개, 64×64).

| 조건 | 필터 보존 | refine(중복 제거) | select 통과 | 실행 완주 | 완주한 예측 후보 |
| --- | ---: | ---: | ---: | ---: | --- |
| 보정 offset | 18 | 7 | 7 | **3** | grasp_1, grasp_6, grasp_7 |
| offset 0 | 8 | 1 | 1 | **1** | grasp_5 |

refine까지 도달한 예측 후보는 보정 6개(grasp_1·5·6·7·8·9), zero 1개(grasp_5)다.
select 단계 탈락 사유도 파일별로 남는다(보정에서 `hand_through_table` 2건,
`open_hand_inside_object` 2건; zero에서 1건).

실행 결과:

| 조건 | 예측 | 변형 | 결과 | 상승 | 최대 힘 |
| --- | --- | --- | --- | ---: | ---: |
| 보정 | grasp_6 | r90_d30 | completed | 14.76 cm | 20.56 N |
| 보정 | grasp_7 | r60_d15 | completed | 14.74 cm | 12.26 N |
| 보정 | grasp_1 | r330_d15 | completed | 10.72 cm | 14.08 N |
| 보정 | grasp_5 | r270_d30 | lift_timeout | 4.88 cm | 15.64 N |
| 보정 | grasp_8/9/5 | — | no_validated_home_path | — | — |
| zero | grasp_5 | r30_d30 | completed | 15.37 cm | 8.69 N |

## 5. op=eval 방향별 판정

`eval_fc`는 ±x·±y·±z 여섯 방향의 외력을 걸고 **첫 실패에서 멈춘다**. 그래서
어느 방향이 왜 실패했는지 남지 않는다. `scripts/audit_grasp_eval.py`는 조기
중단만 제거하고(실행 범위 안에서) 여섯 방향을 모두 돌려 물체 이탈을 기록한다.
임계값 50 mm·15°, 마찰 [0.6, 0.02].

| 파지 | +x | -x | +y | -y | +z | -z | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 기준선 Pincer (0_11) | 4.2 mm | 3.7 | 3.7 | 3.0 | 3.3 | 4.7 | **6/6 유지** |
| 보정 grasp_6 | 1.1 | 1.3 | 0.9 | 1.1 | 0.7 | 1.2 | **6/6 유지** |
| 보정 grasp_1 | 0.6 | 1.3 | 1.0 | 1.1 | 0.7 | 1.2 | **6/6 유지** |
| 보정 grasp_7 | 1.1 | **65.4** | 0.9 | **438.4** | **45.7 / 36.1°** | 5.7 | 3/6 |
| zero grasp_5 | **41.1°** | **16.6°** | **26.2°** | **173.0 mm** | **57.3°** | **67.7°** | 0/6 |

- 실행기를 완주한 4개 중 2개(보정 grasp_1·grasp_6)는 여섯 방향을 모두 견딘다.
  즉 **op=eval 0**이었던 이전 보고는 깨진 템플릿 라이브러리로 만든 파지에
  대한 것이었고, 지금은 통과하는 파지가 나온다.
- zero arm의 유일한 완주 파지는 15.37 cm를 들지만 **여섯 방향 전부 실패**한다.
  대부분 회전 이탈(41°, 57°, 68°)이다. 들어올리기 성공과 force-closure는
  다른 기준이라는 것을 이 한 사례가 그대로 보여준다.
- 이전 문서에서 조사 대상으로 삼았던 파지(`offcmp5_offset_s0` 0_0)는 +x 94°,
  +y 106 mm, -y 154 mm, +z 98°로 실패한다. 기록은 `eval-audit/`에 있다.

## 6. 한계

- 물체 1개·타입 1개·예측 후보 10개·align seed 3개다. 조건 간 최종 우열을
  주장하지 않는다. 특히 zero arm은 refine 결과가 1개뿐이다.
- `no_validated_home_path` 3건은 실행기의 접근 경로 생성 실패이며 파지 품질과
  분리해서 세어야 한다.
- 이 문서의 모든 수치는 진화 템플릿 라이브러리를 복사해 사용한 조건이다.
  라이브러리는 `evo_hx5_d20`에서 왔고, 그 안에는 이 물체로 진화한 템플릿이 없다.
- 자세 이동량 제한은 넣지 않았다. 다음 실험 변수다.

## 7. 기록과 재현

- [stage_counts_v2.json](stage_counts_v2.json) — 조건 × 실행별 필터 판정·수치
- [executed_summary_v2.json](executed_summary_v2.json) — 실행 결과와 귀속
- [provenance-offset-s0.json](provenance-offset-s0.json),
  [provenance-zero-s0.json](provenance-zero-s0.json) — 전체 사슬 1회분
- [v2-offset-distinct.json](v2-offset-distinct.json),
  [v2-zero-distinct.json](v2-zero-distinct.json) — 실행한 파지와 원본 sha256
- `v2-*-s*.verdicts.json` — select 단계의 파일별 판정
- [eval-audit/](eval-audit) — 방향별 op=eval 기록

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/align_grasps.py --objects out/offset-compare/objects \
  --reachable out/offset-compare/reachable-offset --template 33_Inferior_Pincer \
  --exp-name offcmp8_offset_s0 --seed 0     # new_tmpl은 cp -rL 로 준비할 것
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/audit_grasp_eval.py --grasp <grasp_data/....npy> --out <out.json>
```

전체 테스트 **285 passed**. 변경 파일 Ruff 통과.

## 8. 다음

요청대로 ④ 확대(예측 후보 93개)로 넘어갈 준비가 됐다. 확대 전에 정할 것:
align seed 개수, 조건별 입력 수 불균형(18 대 8)을 어떻게 보고할지, 그리고
`no_validated_home_path`를 접근 계획 문제로 따로 집계할지.
