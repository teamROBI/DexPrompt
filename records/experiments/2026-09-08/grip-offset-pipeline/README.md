<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 필터 다음 단계: align → refine → 접근 → 리프트

> **정정 (이후 확인).** 이 문서의 3·4절 수치는 진화 템플릿 라이브러리가
> 끊긴 심볼릭 링크였던 상태에서 나온 것이다. 즉 기본 템플릿 1개만 사용됐다.
> refine 수치(보정 2, zero 4)와 "op=eval 0"은 철회한다. 실제 라이브러리로 다시
> 측정한 결과는 [grasp-provenance](../grasp-provenance/README.md)에 있다.
> 1·2절(기준선 비트 단위 재현, 고친 버그 3개)은 그대로 유효하다.

[필터 단계 비교](../grip-offset-compare/README.md)에서 보존한 후보를 그대로 이어
실행했다. 물체 1개(`core_bottle_6444875e3217bf891f5f48a891d827bd`, scale 0.11),
템플릿 `33_Inferior_Pincer`, 예측 후보 10개, 조건 2개(보정 offset / offset 0).

## 1. 성공한 Pincer 기준 실행 재현

`out/candidate_screen_all`에서 유일하게 완주했던 후보
(`evo_hx5_d20/grasp_data/33_Inferior_Pincer/.../floating/scale011/0_11_grasp.npy`)를
단독 manifest로 다시 실행했다. 결과는 **비트 단위로 동일**하다.

| 항목 | 원본 | 재현 |
| --- | --- | --- |
| episode sha256 | `d83891fd5b06…` | `d83891fd5b06…` |
| ticks | 955 | 955 |
| 상승 | 14.70 cm | 14.70 cm |

`right_state`, `action`, `timestamp`, `contact_groups`, `target_height_min`이
모두 완전히 일치한다. 실행기·물리·설정이 그대로임을 확인한 기준선이다.

## 2. 이어붙이는 과정에서 고친 것 세 가지

세 가지 모두 "조용히 0"이 되는 종류였다. 각각 회귀 테스트를 추가했다.

1. **DGN 레이아웃의 Dexonomy scene config를 못 찾았다.** `align_grasps`는
   `<objects>/<name>/scene_cfg.npy`만 봤다. DGN은 세트 루트의
   `scene_cfg/<object>/floating/scale011.npy`에 둔다. 이제 두 레이아웃을 모두
   찾고, 심볼릭 링크로 만든 단일 물체 디렉터리는 실제 자산 위치를 따라간다.
   못 찾으면 예외로 실패한다.
2. **op=init 결과를 못 찾았다.** Dexonomy는 scene id로 디렉터리를 만들고 DGN의
   scene id는 `<object>/floating/scale011`이라 한 단계 더 깊다. `glob("*.npy")`가
   아무것도 못 찾아 18개 seed에서 18개 파일이 만들어졌는데 "0 aligned"로
   보고했다. `rglob`으로 고쳤다.
3. **템플릿 추첨에 시드가 없었다.** Dexonomy `HandTemplateLoader`는 seed 자세마다
   `np.random.randint`로 접촉 템플릿을 고른다. 같은 입력이라도 실행마다 다른
   템플릿에 정렬되고 refine 결과가 달라진다. `align_grasps --seed`를 추가해
   고정하고, 정렬 기록에 seed·scene config·템플릿 라이브러리를 남긴다.

또한 성공한 기준선과 같은 조건을 주기 위해, 두 조건 모두 `evo_hx5_d20`의
진화된 템플릿 라이브러리(`new_tmpl/33_Inferior_Pincer`, npy 118개)를 각 실험
디렉터리에 **복사**해서 사용했다. 기존 라이브러리는 수정하지 않았다.

## 3. 단계별 생존 수 (align seed 0–5, 각 조건 6회)

두 조건 모두 같은 설정이다: `--in-plane 1`, `--opt-steps 25`, `--keep 100`,
IK 200회·damping 0.05, 실행기는 기본 `ExecutionConfig`(lift_force `null`,
hold 4 N, close 600 ticks), 실행 seed 0, yaw 1개, 64×64.

| 조건 | 필터 보존 | init(=seed) | 재도달 | refine(grasp_admm) | op=eval 성공 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보정 offset | 18 | 18 × 6 = 108 | 51 (47%) | **2** | 0 |
| offset 0 | 8 | 8 × 6 = 48 | 28 (58%) | **4** | 0 |

seed별 refine 수 — 보정: 1, 0, 1, 0, 0, 0 / zero: 0, 0, 0, 2, 2, 0.
적어도 하나가 refine된 seed는 양쪽 모두 6회 중 2회다.
seed를 고정하지 않았다면 이 표는 실행마다 달라진다.

refine된 파일 중 서로 다른 grasp은 보정 1개(2개 파일이 sha256 동일),
zero 2개(4개 파일)다. 중복은 독립 증거가 아니므로 접어서 실행했다.

## 4. 접근 → 닫기 → 리프트 → 유지

| 조건 | 실행 | 완주·수용 | 상승 | 최대 손가락 힘 | 종료 시 접촉군 |
| --- | ---: | ---: | --- | --- | --- |
| 기준선(Pincer) | 1 | 1 | 14.70 cm | 5.94 N | 2/2 |
| 보정 offset | 1 | 1 | 15.08 cm | 10.47 N | 2/2 |
| offset 0 | 2 | 2 | 14.39 / 13.95 cm | 12.15 / 20.54 N | 2/2 |

refine을 통과한 grasp은 양쪽 모두 실행기를 완주했다(3/3). 원래 후보 10개 중
몇 개가 완주했는지는 **이 데이터로 답할 수 없다**: 정렬 최적화가 자세를 크게
옮기기 때문이다. 실행된 보정 쪽 grasp의 정렬 후 자세는 가장 가까운 필터
보존 자세(`grasp_5_r300_d15`)와도 62.0 mm·41.7° 떨어져 있다. Dexonomy의 init
필터가 순서를 섞으므로 파일 이름으로도 추적되지 않는다.

## 5. 이번 결과가 말하는 것과 말하지 않는 것

- 오프셋 보정의 이점은 **필터 단계에서 측정됐다**(통과 88 대 27, 후보 6/10 대
  3/10, 엄지 거리 0 mm 대 최소 19.3 mm). [앞 문서](../grip-offset-compare/README.md).
- **정렬 이후 단계에서는 두 조건의 차이를 보이지 못했다.** 표본이 매우 작고
  (refine 2 대 4), refine 비율은 오히려 zero 쪽이 높다. 이 표로 "보정이
  refine까지 개선한다"고 말할 수 없고, 반대도 말할 수 없다.
- op=eval(흔들기 평가)은 양쪽 모두 0이다. 기준선 후보도 원래 `grasp_data`에서
  가져온 것이며 `succ_grasp`가 아니다. 실행기 완주와 op=eval 통과는 다른 기준이다.
- 실행기를 완주한 3개는 모두 두 개의 required group을 끝까지 유지했다. 다만
  이 템플릿의 required group은 2개(엄지·검지)이며 다섯 손가락 조건이 아니다.
- 정렬 최적화가 자세를 크게 옮기므로, 예측 후보 → 최종 grasp의 귀속은 현재
  파이프라인에서 보장되지 않는다. 후보 단위 비교를 계속하려면 정렬 단계에
  이동량 제한이나 귀속 기록이 필요하다.

## 6. 기록

- [stage_counts.json](stage_counts.json) — 조건 × seed별 단계 수치
- [executed_summary.json](executed_summary.json) — 실행 결과 4건
- [manifest-baseline-pincer.json](manifest-baseline-pincer.json),
  [manifest-offset-distinct.json](manifest-offset-distinct.json),
  [manifest-zero-distinct.json](manifest-zero-distinct.json) — 실행한 후보와 원본 sha256
- [execution-settings.json](execution-settings.json) — 실행기 설정·품질 기준·소스 해시
- 원본 출력: `out/offset-compare/`, `third_party/dexonomy/output/offcmp5_*`

재현(조건 하나, seed 하나):

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/align_grasps.py --objects out/offset-compare/objects \
  --reachable out/offset-compare/reachable-offset --template 33_Inferior_Pincer \
  --exp-name offcmp5_offset_s0 --seed 0
cd third_party/dexonomy && .venv/bin/dexrun op=grasp_admm hand=hx5_d20 exp_name=offcmp5_offset_s0
cd - && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py collect \
  --manifest out/offset-compare/manifests/offset-distinct.json \
  --out out/offset-compare/executed-offset-distinct --max-candidates 4 --max-yaws 1 \
  --seed 0 --width 64 --height 64
```

## 7. 다음

1. 표본을 늘린다. 같은 물체에서 예측 후보 10 → 전체 93개, align seed도 늘려
   refine 수를 두 자리로 만든 뒤에야 조건 간 비교가 의미를 갖는다.
2. 정렬 이동량을 기록·제한해서 예측 후보 → 최종 grasp 귀속을 복구한다.
   지금은 62 mm·41.7° 이동이 관측된다.
3. op=eval 통과 0을 별도로 조사한다. 실행기 완주와 Dexonomy 평가가 갈리는
   지점이 어디인지 확인한다.
