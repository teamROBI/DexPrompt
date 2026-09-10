# 소규모 비교 실험: 보정 grip offset 대 offset 0

병 1개(`core_bottle_6444875e3217bf891f5f48a891d827bd`, scale 0.11),
템플릿 `33_Inferior_Pincer`, 후보 10개. **필터 단계까지의 결과다.**
파지 성공률이 아니라 "도달 가능하고 충돌 없고 엄지가 닿을 거리에 도착한 후보"
수다. 성공 여부는 align → refine → 접근 → 리프트에서 결정된다.

## 조건 세 가지를 어떻게 지켰는지

**1. 예측은 한 번만.** GraspGenX는 1회만 실행했다(93개 제안, 93개 통과).
zero 조건은 그 결과를 복사해서 `grip_offset`만 0으로 바꾼 것이다.
`scripts/compare_grip_offset.py`가 복사본을 다시 읽어 `grasps` 매핑이
원본과 완전히 같은지 확인하고, 후보 매핑의 sha256을 기록한다
(`comparison.json`의 `candidate_digest` = `a351f449654cd914…`).
두 파일에서 다른 키는 `grip_offset`, `grip_offset_applied`, `derived_from` 뿐이다.
후보 ID 순서·자세·confidence를 별도로 대조해 모두 동일함을 확인했다.

**2. 모든 단계가 같은 단일 물체 장면.** `out/offset-compare/objects/`에 병 1개만
심볼릭 링크로 두고 observe_scene·predict_grasps·filter_grasps 모두 이 디렉터리를
넘겼다. 비교 스크립트가 실행 전에 필터가 만들 모델을 직접 만들어 관측 기록과
대조한다: scene 파일 sha256, 물체 집합, 물체별 world 변환(위치·회전),
mesh scale, 그리고 로봇의 명명된 hinge/slide 관절 63개 값. 하나라도 다르면
아무것도 실행하지 않고 종료한다. 이번 실행에서는 전부 일치했다.
(관측에 scene 해시와 로봇 관절 상태를 적도록 `observe_scene`에 추가했다.
필터가 그 뒤에 적용하는 손가락 open pose는 의도된 차이라 비교에서 제외한다.)

**3. 설정 고정.** roll 12, depth ±0.03 m 5단계, IK 반복 120, damping 0.05,
thumb 임계값 0.05 m, 후보 10개. IK 초기값은 두 조건 모두 매 시도마다 장면의
rest 자세다. 같은 값을 두 실행에 그대로 전달한다(`comparison.json`의 `settings`).

## 결과

후보 10개 × (roll 12 × depth 5) = 변형 600개, 조건당.

| 조건 | out_of_reach | ik_failure | collision | thumb_far | passed | kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 보정 offset | 0 | 473 | 39 | 0 | **88** | 18 |
| offset 0 | 0 | 524 | 19 | 30 | **27** | 8 |

후보 단위로 보면 적어도 하나의 변형이 살아남은 후보가
**보정 6/10 대 zero 3/10**이다.

| 조건 | 후보별 최근접 엄지 거리(mm) |
| --- | --- |
| 보정 offset | 0.0, 0.0, 0.0, 0.0, 0.0, 16.9 |
| offset 0 | 19.3, 21.3, 26.6 |

`thumb_far`가 보정에서 30 → 0으로 사라지고, 살아남은 후보의 엄지가 물체에
닿는다(0.0 mm). offset이 고치려던 실패 양상 —"손가락은 닿고 엄지는 못 닿음"—
과 일치한다. 보정된 offset은 115 mm이며 축별로
`[-0.0946, -0.0494, 0.0438] m`이다.

IK 자체도 개선됐다. IK를 통과한 변형은 **76 → 127개**다. 다만 국소 IK 실패는
여전히 두 조건 모두에서 가장 큰 탈락 요인이다(524/600, 473/600). 보정이 IK
성공을 늘렸다는 것과, 남은 실패의 대부분이 접근 자세 생성·국소 IK 해법의
문제라는 것은 둘 다 사실이다. 이 수치를 물리적 도달 불가능성의 증거로 읽지
않는다.

충돌 수는 19 → 39로 늘었지만, **검사 대상 자체가 늘었다.** IK 성공 이후의
충돌 비율은 19/76 = **25.0%** 대 39/127 = **30.7%**다. 충돌 수가 두 배라는
사실만으로 위험이 두 배가 됐다고 읽으면 안 된다. 방향성은 있다: 보정이 손을
물체와 테이블 쪽으로 당기므로 자세당 충돌 확률은 5.7%p 높아졌다.
막은 접촉은 zero에서 `world x finger_r_link4` 14, `world x arm_r_link7` 5이고,
보정에서는 `world x arm_r_link6` 20, `world x hx5_r_base` 10,
`world x arm_r_link7` 5, `world x arm_r_link5` 4다(`world`에 테이블 geom이
포함된다). zero에서는 손가락이 테이블에 닿고, 보정에서는 팔뚝이 걸린다.

후보 단위로는 포함 관계다. zero에서 통과한 `grasp_1`, `grasp_5`, `grasp_6`은
보정에서도 모두 통과했고, 보정에서 `grasp_7`, `grasp_8`, `grasp_9`가 추가됐다.
엄지 거리 0.0 mm는 접촉이 가능한 위치에 도착했다는 뜻일 뿐, 필요한 접촉력이나
안정적인 파지를 보장하지 않는다.

## 한계

- 물체 1개, 템플릿 1개, 후보 10개(제안 93개 중 상위 confidence 10개)다.
  통계가 아니라 단일 사례 비교다.
- `kept`는 후보당 최대 3개로 제한되므로 `passed`보다 정보가 적다.
- 필터의 충돌 검사는 대상 물체와 로봇 내부 접촉을 면제한다. 무충돌 인증이 아니다.
- 파지 성공률은 측정하지 않았다. 엄지가 닿을 위치에 도착하는 것과 잡아 드는
  것은 다른 주장이다.

## 재현

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/observe_scene.py --objects out/offset-compare/objects \
  --out out/offset-compare/observation
OPENBLAS_NUM_THREADS=1 modules/graspgenx/.venv/bin/python \
  scripts/predict_grasps.py --template 33_Inferior_Pincer \
  --observation out/offset-compare/observation --objects out/offset-compare/objects \
  --out out/offset-compare/predicted
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/compare_grip_offset.py --template 33_Inferior_Pincer \
  --objects out/offset-compare/objects --observation out/offset-compare/observation \
  --predicted out/offset-compare/predicted --out out/offset-compare
```

기록: [comparison.json](comparison.json) (양쪽 조건의 단계별·후보별 수치),
[grip_offset.json](grip_offset.json) (예측 단계의 offset 산출 기록),
[stages-offset.json](stages-offset.json), [stages-zero.json](stages-zero.json).

## 다음

같은 10개 후보로 align → refine → 접근 → 리프트를 두 조건 모두 실행하고,
성공한 Pincer 실행을 기준선으로 함께 재현한다. 필터 통과 수가 늘었다는 것과
파지가 성공한다는 것은 다른 주장이므로 분리해서 기록한다.
