# GraspGenX → Dexonomy 자세 전달 수정

사용자 요청: 먼저 기존 경로의 연결 오류를 고치고 검증한다. 학습과 새 파지
수집은 실행하지 않았다. 이번 결과는 좌표 전달 검증이며 파지 성공률이 아니다.

## 확인한 오류와 수정

1. `filter_grasps`는 roll/depth/grip_offset을 적용한 손목 자세를 검사하지만,
   `align_grasps`는 원래 예측의 rotation/position만 읽었다. 이제 공통
   `icra.grasp_frames` 변환으로 필터의 검사 자세를 복원한다.
2. Dexonomy `sample_init_pose`의 출력은 손목 또는 GraspGenX canonical frame이
   아니라 **템플릿 접촉 좌표계**다. `HandTemplateLoader`는 템플릿을 샘플링할 때
   접촉점을 선택하므로, object sampler에서 이 변환을 미리 확정할 수 없다.
   이제 `HandObjMatcher.forward` 직전에 실제 `rot_c2h` / `trans_c2h`를 사용한다.
   외부 Dexonomy 소스는 수정하지 않고 실행 범위 안에서만 함수를 교체한다.
3. 공통 테이블 원점만 빼던 변환을 실제 MuJoCo 물체별 위치·회전으로 바꿨다.
   정렬 결과에 저장된 Dexonomy 물체 자세도 역변환에 사용한다.
4. Dexonomy 출력은 이미 `hx5_r_base` 자세인데, 재검사에서 GraspGenX의 URDF
   오프셋을 다시 적용하고 있었다. 이 중복 오프셋을 제거했다.
5. 기본 추가 roll 수는 4 → 1로 바꿨다. 기본값은 필터가 검사한 자세를 유지한다.
   명시적인 `--in-plane > 1`의 추가 자세는 입력 IK 검사를 통과했다고 볼 수 없다.

좌표계 표기에서 `T_A_B`는 B의 좌표를 A로 옮긴다. 입력 손목을 `H`,
현재 물체를 `O`, 선택된 접촉 좌표계를 `C`, Dexonomy 장면을 `D`라고 하면:

```text
T_O_H = inverse(T_W_O) @ T_W_H_filtered
T_O_C = T_O_H @ inverse(T_C_H)
matcher translation = translation(T_O_C) / object_scale
T_W_H_aligned = T_W_O @ inverse(T_D_O) @ T_D_H_aligned
```

## 진단 및 재실행 동작

`align_grasps`는 `out/alignment/<exp>_<hand>/<template>/<object>.json`에
입력 보정 손목 자세, 실제 물체 자세, 정렬 후 손목 자세, IK 위치·회전 오차,
충돌 목록과 보존 여부를 기록한다. `full_path_validated`는 false다.
정렬 후의 IK 검사와 기존 정적 장면 충돌 검사는 유지했다. 이 충돌 함수는
대상 물체 및 로봇 내부 접촉을 면제하므로 전체 무충돌 인증으로 해석하지 않는다.

기존 init_data가 있는 출력 경로에는 재실행하지 않고 새 `--exp-name`을 요구한다.
기존 결과를 새 결과로 집계하거나 삭제하는 것을 막기 위한 동작이다. 함수 교체와
작업 디렉터리는 예외가 나도 복구한다. Dexonomy가 matcher 오류를 내부에서
삼켜도 호출한 단계는 실패하도록 했다.

## 검증

신규 회귀 테스트는 필터 보정, 서로 다른 회전·깊이 후보 보존, 물체별 위치·회전,
템플릿별 접촉 변환, 비단위 object scale, 출력 중복 오프셋, 예외 복구를 검사한다.
실제 Dexonomy `HandObjMatcher.forward`도 1회 평가했다. 이 테스트는
`backward`와 optimizer step을 호출하면 실패하도록 하며 최적화를 실행하지 않는다.

최종 전체 테스트: **236 passed** (MuJoCo 3.3.3 / Dexonomy 환경).
수정한 Python 파일들의 Ruff 검사와 `git diff --check`도 통과했다.

저장된 후보 704개를 실제 템플릿 접촉 좌표계로 변환하고 손목으로 복원했다.
검사에 사용한 입력/코드 해시와 요약은 [audit.json](audit.json)에 저장했다.

- 최대 위치 오차: `2.2987007359244158e-08 m`.
- 최대 회전행렬 원소 오차: `4.968529516480302e-06`.
- 원본 접촉 좌표계 직교성 오차: 최대 `5.868859979694982e-06`.
  원본 템플릿 법선의 수치 오차 때문에 첫 1e-6 행렬 오차 기준을 넘었다.
  데이터를 몰래 정규화하지 않고 이 오차를 보고하며, 최종 행렬 기준은 1e-5다.
  이는 각도(rad)가 아니라 행렬 원소 오차다. 위치 기준은 1e-6 m다.

재현:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  third_party/dexonomy/.venv/bin/python scripts/audit_grasp_alignment.py \
  --out out/graspgenx-frame-audit.json
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  third_party/dexonomy/.venv/bin/python -m pytest -ra
```

704개 검사는 현재 scene과 기존 flat object asset을 사용한 **좌표 왕복 검사**다.
기존 IK 판정의 재인증, 접촉 정렬 최적화, 접근·리프트 실행은 포함하지 않는다.
정렬 최적화가 도달성을 유지하도록 하는 제약도 이번에는 추가하지 않았다.

## 다음 실험

1. 한 물체의 소수 후보로 필터를 다시 실행하고 새로운 실험 이름으로 정렬한다.
   기본 `--in-plane 1`을 유지한다. 과거 결과와 설정이 달라졌으므로 후보 개수와
   필터 탈락 사유부터 다시 기록한다.
2. 새 alignment JSON에서 입력과 정렬 후 손목 자세, IK 잔차를 비교한다.
   정렬로 도달성을 잃는 비율과 처음부터 IK에 실패하는 비율을 구분한다.
3. 정련 후에도 손목·접촉 상태와 도달성을 다시 확인하고, 동일한 현재 실행기로
   홈 접근 → 닫기 → 리프트 → 접촉 유지 조건을 검사한다.
4. 이미 성공한 Pincer 실행을 비교 기준으로 함께 재현한다. 이 결과를 보고
   접근 계획 또는 RL 보정의 필요성을 판단한다. 과거 0/192나 현재 81% IK 실패를
   이 수정만으로 해결했다고 주장하지 않는다.
