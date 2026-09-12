<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# grip offset: 렌더링에 쓰인 물체 자세를 기록해서 사용

사용자 요청 범위: 수정과 회귀 테스트까지. 비교 실험(offset 0 대 보정 offset)은
코드 검토 후 같은 예측 후보를 고정해 따로 진행한다. 이 문서에는 파지 성공률이
없다. 좌표 계산 검증 결과다.

## 고친 내용

`predict_grasps`는 물체 자세를 관측에서 읽지 않고 scene 설정에서 다시 만들었다.
그래서 세 가지가 함께 틀렸다.

1. 모든 물체에 **공통 테이블 슬롯** 하나를 썼다. 실제로는 물체가 y로 0.18 m씩
   벌어져 서 있다.
2. **안착 회전**을 빼먹었다. DGN 물체는 tabletop pose로 눕혀 세우므로 회전이
   centroid를 수 cm 옮긴다.
3. mesh centroid에 **scale을 적용하지 않았다.** DGN mesh는 정규화되어 있고
   크기는 0.05–0.20의 scale로 들어간다.

그리고 DGN 레이아웃에는 `mesh.obj`가 없어서 `if (asset / "mesh.obj").exists()`
조건이 거짓이 되고, offset이 **조용히 [0, 0, 0]으로 기록**됐다. 측정한 값과
구분되지 않았다.

이제 `observe_scene`이 렌더링한 그 상태에서 물체마다
`world_transform`(4x4), `asset`, `mesh`, `scale`을 `meta_data.json`에 적는다.
scale은 asset에서 다시 계산하지 않고 **컴파일된 모델의 mesh scale**에서 읽는다.
경로는 저장소 안이면 저장소 기준 상대 경로로 적는다. 심볼릭 링크로 붙인
sibling 체크아웃을 resolve하면 특정 머신의 홈 디렉터리가 박히기 때문이다.

`predict_grasps.true_centre`는 그 기록만 사용한다. centroid에 scale을 **정확히
한 번** 곱한 뒤 world로 옮긴다(회전 이후에 곱하거나 두 번 곱하면 결과가 달라진다).

기록이 없거나 변환이 유효하지 않으면 `MissingPlacement`로 실패한다. 자동
zero offset 대체는 없다. offset 0은 "GraspGenX의 그립이 이미 물체와 일치한다"는
주장이므로 명시적인 `--zero-offset`으로만 선택한다. 결과는
`out/predicted/<template>/grip_offset.json`에 물체별
`status`(`measured` / `zero_offset` / `no_grasps` / `failed`)와 이유로 남고,
`failed`가 하나라도 있으면 종료 코드가 1이다. 물체별 YAML에도
`grip_offset_applied`가 함께 들어간다.

## 변경 파일

- `scripts/observe_scene.py` — `placements`, `mesh_scale`, `recorded_path`,
  `write(..., placement=...)`
- `scripts/predict_grasps.py` — `MissingPlacement`, `true_centre` 재작성,
  `grip_offset`, `unmeasured`, `--zero-offset`, `grip_offset.json`.
  사용하지 않게 된 `--scene`과 `yaml as _yaml` 임포트를 제거했다.
- `tests/test_grip_offset.py` — 신규 24개

## 검증

기존 legacy(icra) 장면을 실제로 렌더링하고, 옛 계산과 새 계산의 물체 중심을
비교했다. 공통 슬롯에 서 있는 물체 하나만 맞았다.

| 물체 | 이동량 |
| --- | --- |
| ChocolatePudding | 180.0 mm |
| GranolaBars | 0.0 mm |
| Mustard | 180.0 mm |

DGN 물체 2개(`core_bottle_1071fa4c…`, `core_bottle_109d55a1…`)도 렌더링했다.
scale이 각각 0.08과 0.14로 기록됐고 두 물체의 자세가 서로 다르게 남았다.
두 번째 물체는 centroid가 body 원점에서 x로 31 mm 떨어져 있다. 옛 경로에서는
이 물체에 offset이 아예 계산되지 않았다.

회귀 테스트가 다루는 것: 기록된 회전이 centroid에 적용되는지, 물체별 자세가
따로 기록되는지, scale이 asset이 아니라 모델에서 오는지, scale이 정확히 한 번
곱해지는지(미적용·제곱 모두 실패하도록), legacy와 DGN 레이아웃 양쪽에서
mesh를 찾는지, 관측→예측 왕복이 MuJoCo `mj_local2Global` 결과와 일치하는지,
정보가 없거나 4x4가 아니거나 scale이 없을 때 실패하는지, `--zero-offset`이
기본값이 아니고 실패와 구분되어 기록되는지, 경로가 저장소 상대인지.

전체 테스트: **260 passed** (MuJoCo 3.3.3 / Dexonomy 환경).
`scripts/predict_grasps.py`의 Ruff 경고 3건(E741 1건, B905 2건)은 이번에
건드리지 않은 줄에 있던 기존 경고다.

재현:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  third_party/dexonomy/.venv/bin/python -m pytest -ra
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  third_party/dexonomy/.venv/bin/python scripts/observe_scene.py --out out/observation
```

## 다음

같은 관측과 같은 예측 후보·roll·depth·IK seed에서 offset 0과 보정 offset을
비교한다. 실행 결과는 [../grip-offset-compare/](../grip-offset-compare/README.md)에 있다.
후보 동일성을 보장하려고 `--zero-offset` 재실행 대신 저장한 예측을 복제해서
offset만 0으로 바꿨다. 성공한 Pincer 실행을 기준선으로 함께 재현한다. 이번 수정만으로
접근 실패율이 해결됐다고 주장하지 않는다.
