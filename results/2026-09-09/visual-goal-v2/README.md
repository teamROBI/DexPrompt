# 시각 입력 v2: 카메라 프로필·wrist mask·목표 기억

2026-09-09. **요청한 1–3번을 수집부터 student 추론까지 구현했다. 학습은 하지 않았다.**
새 경로는 `configs/hx5_student.visibility.local.json`의 `observation_version=2`다.
기존 v1 데이터/모델 경로는 유지하며, 서로 다른 버전의 state dict를 섞지 않는다.

## 1. 원본 해상도와 카메라 구도

`configs/visual_student_256.json`은 native 256×256, head 하향 20°를 명시한다.
`collect_teacher_dataset.py collect --visual-profile ...`로 선택하며, 기본 전역 장면이나
기존 수집 명령의 카메라를 바꾸지 않는다. 프로필은 새 모델에 한 번 적용하고 실제
카메라 quaternion·시야각·해상도를 기록한다. archived MJB에도 최종 카메라가 들어가며
replay에서는 이미 적용된 회전을 다시 더하지 않는다.

`prepare_visual_dataset.py`는 검증된 MJB·초기 integration state·float64 action을
재생해 새 디렉터리에만 시각 데이터를 만든다. 모든 tick에서 원래 관절 상태·timestamp와
모든 physics substep의 접촉·힘·변위·침투 집계를 대조한다. 차이가 한계를 넘으면
중단하며, 완료한 episode만 기존 품질 판정을 다시 통과시켜 sealed dataset으로 쓴다.
카메라 비교는 동일한 256 해상도로 전체 궤적에서 수행한다.

검증은 같은 병의 승인된 파지 3개, 총 **2,919프레임**이다. 한 물체 안의 자세 차이이며
다물체 실험으로 세지 않는다. 아래 표는 원래 빈 프레임만 골라낸 결과가 아니다.

| 시범 | 원래 head 소실 | 하향 head 소실 | 하향 head+wrist 모두 소실 | 새로 보임 / 새로 안 보임 |
|---|---:|---:|---:|---:|
| teacher_0002 | 0 | 0 | 0 | 0 / 0 |
| teacher_0006 | 416 | 157 | 157 | 261 / 2 |
| teacher_0016 | 358 | 356 | 30 | 13 / 11 |
| 합계 | **774** | **513** | **187** | **274 / 13** |

원래 head+wrist 모두 소실은 278프레임이었다. 개선과 함께 **새로 잃는 head 관측
13프레임도 포함**했다. 프로필이 모든 시점에서 더 좋다는 주장은 하지 않는다.
물리 신호 최대 오차는 **0**, 저장 관절 상태 불일치 **0**, 품질 판정도 그대로다.

추가로 **실제 신규 teacher 수집기**에 프로필을 넣어 고정 후보 1개를 실행했다.
974 tick, `completed`, 승인 1/1. 새 수집의 RGB·양쪽 mask·action·상태·품질 신호 등
**모든 저장 배열이 해당 재렌더링 시범과 정확히 일치했다.** 파일 배치/메타데이터가 다른
NPZ의 파일 sha가 동일하다는 뜻은 아니다. [검사 기록](collection-check.json).

## 2. 두 카메라의 목표 입력

`icra.visual_observation.capture()`가 같은 pre-action 상태에서 head/wrist RGB와 각
카메라의 물체·부위 mask를 얻는다. 부위는 각 카메라의 depth/외부 파라미터로 별도 투영한다.

- `wrist_mask`, `wrist_part_mask`: wrist 영상과 같은 H×W의 uint8 mask.
- `target_pixels`: head 물체·head 부위·wrist 물체·wrist 부위 순서의 int32 4개.
- `target_visible`: 위 순서의 실제 가시성 0/1. 현재 mask가 0이면 그대로 0이다.
- object 요청에서 양쪽 부위 mask는 0이다. part 요청에서는 별도 부위 주석과
  기존의 물리적 접촉 준수 검사가 계속 필요하다.

schema-3의 명시적 시각 확장으로 구현했다. validator가 shape·dtype·부분집합·픽셀 수와
가시성의 실제 일치를 검사하고, cache는 이 필드를 원본 해시와 묶어 보존한다.
v2 학습에서 누락된 wrist mask를 "측정한 0"으로 채우지 않는다.

모델은 현재 wrist RGB 특징과 wrist 물체/부위 mask를 결합한다. 현재 가시성과 mask의
면적 비율도 별도 token으로 전달한다. simulator mask는 oracle grounding이다.
이미지에서 자동으로 목표를 찾는 segmentation 모델을 학습한 것은 아니다.

## 3. 요청별 목표 기억

`GoalMemory`는 요청마다 카메라별 첫 유효 관측을 한 번 저장한다. 유효성은 해당 요청
mask가 최소 1픽셀 보이는 조건이며, 고품질 인식의 보장은 아니다. object 요청은 물체,
part 요청은 해당 부위의 가시성을 사용한다. 기준은 다음과 같다.

- 배경을 지운 물체 RGB crop와 물체/부위 mask. 정사각형 padding 후 224로 전처리.
- 원래 영상에서의 정규화된 bbox. 모양이 같은 물체의 초기 위치를 구별하기 위한 정보.
- 해당 관측의 측정 관절 상태. world object pose나 teacher grasp pose가 아니다.

각 카메라의 기준 crop은 공유 frozen ViT-B로 인코딩한다. 물체/부위 mask로 가중 집계한
특징 4개, 기준 bbox·측정 상태·유효성 token, 현재 가시성 token을 action transformer에
연결했다. 기준 관절도 train-only state 통계로 정규화한다. 기존의 현재 head ViT-B와
wrist ViT-L은 그대로 사용한다.

**현재 mask와 기억은 분리된다.** 가려진 현재 mask를 과거 픽셀이나 물체 단독 렌더로
덮어쓰지 않는다. 재등장해도 기준을 덮어쓰지 않고, 다른 카메라가 처음 목표를 관측하면
그때부터 해당 카메라의 기준을 추가한다. 새 요청/revision과 reset 때 기억 및 남은 행동
chunk를 제거한다. 새 요청의 기준을 양쪽 모두 얻지 못했다면 명령 생성 전에 명시적으로
실패하며, 이전 목표의 기억을 재사용하지 않는다.

학습도 같은 규칙이다. `EpisodeChunks[t]`는 각 최초 관측 시점이 **t 이하**인 기준만
넣는다. 미래에 얻는 wrist 기준은 `goal_valid=0`이다. 목표가 처음부터 보이지 않는
prefix는 진단용 chunk 구성에서 제외할 수 있지만, home 학습 preflight는 이를 거부한다.
카메라가 나중에 대상을 발견한 기록을 reset 때 이미 알고 있던 것처럼 만들지 않는다.

실제 시범의 첫 기준 관측 tick:

| 시범 | head | wrist | 제외된 prefix |
|---|---:|---:|---:|
| teacher_0002 | 0 | 326 | 0 |
| teacher_0006 | 0 | 415 | 0 |
| teacher_0016 | 0 | 418 | 0 |

이 3개 시범의 초기·wrist 최초 관측·양쪽 가림·마지막 상태 등 **11개 관측에서 온라인
기억과 학습 입력의 모든 tensor가 정확히 일치**했다. 단위 검사에서는 미래 wrist 영상만
바꿔도 앞선 샘플이 바뀌지 않는 것, 현재 두 부위 mask가 모두 0이어도 과거 요청이 다르면
encoder 조건이 구별되는 것, 요청 변경/reset 때 이전 정보가 사라지는 것을 확인했다.
입력의 구별 가능성은 학습된 행동/성공 차이를 증명하지 않는다.

## 실제 DINO와 통합 검사

- 실제 pretrained DINOv2 B/L strict load, 두 backbone 모두 frozen.
- 실제 데이터의 **현재 head/wrist mask가 둘 다 0인 샘플 두 개**를 선택했다.
- 전체 모델의 손실 **1.2262066603**, 행동 **[2,64,27]**, 모두 유한.
- 모델 state dict 불변. backward·optimizer·다운로드 호출 **0**.
- CPU 1 thread의 전체 모델 검사 약 **123초**.
- 기존 실행·데이터·추론 검사와 새 검사 **142개 통과**, 변경 Python 파일 Ruff 통과.
  결과는 [tests.log](tests.log), 모델 검사는 [full-model.json](full-model.json),
  [verification.json](verification.json)에 있다.

실제 MuJoCo 합성 장면에서 반대 방향의 두 카메라에 부위 mask가 각각 올바른 쪽으로
투영됨을 검사했다. 이 검사는 실제 의미 부위의 성공 파지 시범 수집을 대신하지 않는다.
현재 데이터는 object 지시만 포함한다.

## 다음 작업에 사용할 경로와 명령

- 프로필: `configs/visual_student_256.json`
- 모델 설정: `configs/hx5_student.visibility.local.json`
- 검증 시범: `out/student-visibility-v2`
- 검증 cache: `out/student-visibility-v2-cache` (약 1.91 GB)
- 실제 신규 수집 확인: `out/student-visibility-v2-collection-check`
- 전체 모델 검사: `out/student-visibility-v2-full-check`

새 teacher 수집은 기존 명령에 프로필을 더한다. 출력 경로는 새 디렉터리여야 한다.

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
third_party/dexonomy/.venv/bin/python scripts/collect_teacher_dataset.py collect \
  --manifest out/rl-approach/candidate.json --out out/NEW_visual_teacher \
  --visual-profile configs/visual_student_256.json --max-candidates 1 --max-yaws 1
```

기존 시범을 새 영상으로 만드는 명령:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
third_party/dexonomy/.venv/bin/python scripts/prepare_visual_dataset.py \
  --data out/offset-compare-93/executed-offset --out out/NEW_visual_dataset \
  --visual-profile configs/visual_student_256.json \
  --episode teacher_0002_yaw_000 --episode teacher_0006_yaw_000 --episode teacher_0016_yaw_000
```

기본 preflight는 독립 validation 물체가 없어 **종료 코드 2**다. 그 외 준비 항목과
초기 기준 관측 조건은 통과했다. 한 물체 진단임을 명시하는 경우에만
`--allow-train-only-diagnostic`를 더한다. 이 옵션의 별도 preflight는 실제로 종료 코드 0을
확인했다. 학습 실행은 사용자가 결정한다.

```bash
.venv-student/bin/python scripts/train_hx5_student.py preflight \
  --config configs/hx5_student.visibility.local.json --data out/student-visibility-v2 \
  --cache out/student-visibility-v2-cache
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-student/bin/python \
  scripts/train_hx5_student.py dry-run --full-model --device cpu \
  --config configs/hx5_student.visibility.local.json --data out/student-visibility-v2 \
  --cache out/student-visibility-v2-cache --sample-index 1240 --sample-index 2067 \
  --out out/NEW_visual_forward
```

재실행 가능한 no-optimization 검사는 `verify_student.py`에 있다. 코드/프로필/데이터
기록을 보존하며 기존 산출물을 덮어쓰지 않는다. source MJB의 카메라에 이미 시각 프로필이
있으면 재렌더 도구는 상대 회전을 다시 적용하지 않고 거부한다.

**남은 검증:** 실제 카메라 보정 및 다물체 전체 경로, 의미 부위 시범, 학습 후 가려짐
상태의 목표 유지와 전체 파지 완주. 1–3번의 구현 완료를 P5의 학습/폐루프 성공 완료로
세지 않는다. 진행 기준은 [PAPER_TODO](../../../PAPER_TODO.md)와
[입력 계약](../../../configs/STUDENT_GOAL_CONTRACT.md)에 반영했다.
