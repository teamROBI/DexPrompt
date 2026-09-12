<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 캔 시범의 시각 student 연결 검증

2026-09-10. 학습 없이, 채택된 캔 전체 시범 한 건을 기존 visual-v2 경로에 연결했다.
Teacher 생성·정련·제어 코드는 변경하지 않았다. 실제 DINO를 포함한 전체 모델 검사까지 통과했다.

## 데이터와 물리 재현

- 원본: `out/multiobject/can/executed-all/episodes/teacher_0008_yaw_000.npz`.
- 새 데이터: `out/student-can-visibility-v2`.
- 설정: `configs/visual_student_256.json` — native 256×256, 머리 카메라 아래로 20°.
- 1,251 tick 전체의 pre-action 측정 관절 상태·시간과 post-action 접촉/침투 등
  물리 기록을 원본과 대조했다. 상태 불일치 0, 최대 물리 신호 차이 0.
- RGB 영상 2개를 제외한 원본 배열 15개와 품질 판정이 정확히 같다.
  108.7 mm 상승과 종료 시 5/5 접촉군 유지라는 teacher 실행 결과를 보존했다.
- 재생으로 동작을 재현했으며 새 teacher 후보를 생성하거나 독립 성공 시범을 추가한 것은 아니다.

## 카메라와 목표 기억

같은 native 256 해상도에서 원래 머리 카메라 방향과 변경 방향을 전체 궤적으로 비교했다.

| 지표 | 프레임 수 |
| --- | ---: |
| 원래 방향의 머리 mask 비어 있음 | 35 |
| 변경 방향의 머리 mask 비어 있음 | 0 |
| 변경 방향의 머리·손목 모두 비어 있음 | 0 |
| 변경으로 새롭게 보인 머리 프레임 | 35 |
| 변경으로 보이지 않게 된 머리 프레임 | 0 |

최초 머리 참조는 tick 0, mask 1,347픽셀이다. 최초 손목 참조는 tick 44,
mask **3픽셀**이다. 현재 GoalMemory는 첫 양성 mask를 참조로 고정하므로, 손목 참조가
풍부한 형상 정보를 담았다고 볼 수 없다. 이번에는 참조 선택 규칙을 바꾸지 않았다.
캔에서는 두 카메라가 동시에 가려지는 프레임이 없었다. 완전 가림 검증은 이전 병 데이터의
근거이며, 이번 캔 실행에서 새로 검증했다고 주장하지 않는다.

## 데이터 준비 검사

- 검증된 cache: `out/student-can-visibility-v2-cache`, 820,307,128 bytes.
- 일반 preflight: 종료 코드 2. 유일한 사유는 독립 validation split 없음.
- 명시적 train-only diagnostic preflight: 종료 코드 0. 학습 실행은 없음.
- 처음 목표가 보이지 않아 잘라낸 prefix는 0프레임.

## 재현 명령

새 출력 디렉터리를 지정해야 한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  third_party/dexonomy/.venv/bin/python scripts/prepare_visual_dataset.py \
  --data out/multiobject/can/executed-all \
  --out out/student-can-visibility-v2 \
  --visual-profile configs/visual_student_256.json --episode teacher_0008_yaw_000

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-student/bin/python \
  scripts/train_hx5_student.py prepare-cache \
  --data out/student-can-visibility-v2 --out out/student-can-visibility-v2-cache

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-student/bin/python \
  results/2026-09-10/can-visual-student/verify_student.py
```

`verify_student.py`는 backward·Adam/AdamW 생성·가중치 다운로드를 금지하고 실제 로컬
DINO를 불러온다. 온라인 GoalMemory를 tick 순서대로 갱신한 입력과 같은 시점의
학습 데이터 입력을 대조한 뒤, 초기/마지막 프레임으로 전체 모델을 검사한다.


## 실제 전체 모델 검사 결과

- 사전학습 DINOv2 ViT-B/ViT-L을 로컬 가중치로 로딩하고 동결 상태를 확인했다.
- 최초 tick 0·손목 참조 획득 tick 44·마지막 tick 1250의 온라인/오프라인 관측 텐서가
  모두 정확히 같다. 전체 1,251 tick을 순서대로 관측해 목표 기억을 갱신했으며,
  텐서 대조 지점은 세 곳이다.
- 실제 전체 모델은 tick **0과 1250**을 입력받았다. 손실 **1.3167665**, 행동 출력
  **[2, 64, 27]**, 모두 유한. 모델 state_dict 변경 없음.
- backward 0회, optimizer step 0회, 다운로드 0회. CPU 1 thread, 모델 검사 약 128.6초.
- 손실은 무학습 정책의 수치 건전성 확인이며, 파지 성능이나 학습 수렴의 근거가 아니다.
- 새 검증 스크립트 Ruff와 git diff --check 통과. 이번 작업은 production 코드를
  변경하지 않았으므로 기존 테스트 모음은 반복 실행하지 않았다.

## 완료 범위와 다음

병에 이어 캔에서도 **검증 시범 → visual-v2 데이터/cache → 실제 DINO 전체 모델** 연결이
성립했다. P5의 카메라 입력만을 사용한 학습 정책의 전체 폐루프 완주는 아직 미완료다.
캔 1건으로 배치/물체 일반화를 주장하지 않으며 재현본을 train/val로 나누지 않는다.

다음은 병·캔을 이용한 작은 시각 BC 진단의 데이터 묶음과 폐루프 평가 조건을 고정하는
것이다. 학습 시작은 사용자에게 맡긴다. 첫 양성 3픽셀 손목 참조의 품질 문제는 참조
선택 규칙을 별도로 비교할 항목으로 남긴다. 조건을 바꾸면 온라인/오프라인 인과성 검사를
함께 반복해야 한다. Teacher의 물체·파지 확대는 별도 진행한다.

근거: [물리/배열 대조](data-check.json), [가시성](render-report.json),
[전체 모델](full-model.json), [목표 기억 대조](verification.json),
[일반 준비 검사](preflight.json), [진단용 준비 검사](diagnostic-preflight.json).
