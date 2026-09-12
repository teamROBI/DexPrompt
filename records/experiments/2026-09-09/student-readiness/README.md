<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 시각 student 준비 상태 및 오류 검토

2026-09-09. Claude의 후보 생성·실행·수집과 병행하는 student 측 작업이다.
새 teacher 실험·학습·렌더링·GPU 사용은 없었다. 기존 실행기/수집기 코드는 변경하지 않았다.

**결론:** 실제 시범 → 검증된 cache → 관측/행동 chunk → 작은 diffusion 모델의 forward와
행동 생성이 연결된다. 하지만 실제 DINO를 사용하는 전체 모델 학습 준비는 아직 완료되지
않았다. `torchvision`, DINO source/weights 설정, 독립 검증 물체가 필요하다.

## 현재 실행 환경

인터프리터: `third_party/dexonomy/.venv/bin/python`.

| 의존성 | 실제 import 결과 |
|---|---|
| torch | 2.6.0+cu124 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| omegaconf | 2.3.1 |
| torchvision | ModuleNotFoundError |

`configs/hx5_student.json`의 DINO source·head weights·wrist weights는 모두 null이다.
프로젝트의 `third_party`, `data`, 설정 및 일반 Torch/Hugging Face cache 경로에서 해당
소스/가중치를 찾지 못했다. 서버 전체에 없다고 단정하지 않는다. 기존 공유 환경의 패키지를
설치/변경하거나 가중치를 다운로드하지 않았다.

현재 데이터/cache를 지정한 preflight는 **종료 코드 2, training_ready=false**를 반환했다.
위 4개 준비 항목과 empty validation split을 사유로 보고한다.
검사 결과: [preflight.log](preflight.log).

## 실제 데이터 연결 검사

원본: `out/offset-compare-93/executed-offset`, schema 3 승인 시범 9개, 총 **8,817프레임**.
병 하나의 object-conditioned 전체 실행이며 train=9, val=0, test=0이다. 같은 물체의
시범을 임의로 validation으로 옮기지 않았다. 독립 물체 검증으로 해석할 수 없다.

별도 출력: `out/student-review-20260909/cache`, 약 **244 MiB**(255,805,716 array bytes).
cache 해시 `83fac9e4a454dd646bae1efe60aebc87337c8a54e5e515387981a3be55965564`.
원본 episode와 cache member 해시 정합성은 기존 검증기를 사용했다.

[check_inputs.py](check_inputs.py)로 확인한 내용:

- 9개 시범 각각의 첫·중간·마지막 프레임에서 같은 시각의 측정 관절과 해당 시각부터의
  행동 chunk가 원본과 일치한다. 마지막 chunk는 다음 episode로 넘어가지 않는다.
- 원본 action은 float64, 모델 tensor는 float32다. 검사한 chunk의 최대 변환 오차는
  약 **1.19e-7 rad**다. 원본 시범은 변경하지 않았다.
- 전체 timestamp 간격과 control_dt의 최대 차이는 약 **1.11e-14초**다.
- 모델 입력은 RGB/mask·wrist RGB·측정 관절·부위 mask·부위 지시 여부뿐이다.
  phase·teacher pose·접촉 정답은 입력 dictionary에 들어가지 않는다.
- 동일 실제 관측에서 두 개의 다른 픽셀 부분집합을 요청 mask로 넣으면 부위 채널만 바뀐다.
  이는 channel routing fixture이며 실제 의미 부위 주석/물리적 접촉 성공은 아니다.

작은 모델의 무학습 forward:

- 실제 승인 시범과 verified memmap 사용.
- random patch encoder fixture, 축소 모델 123,835 parameters. 실제 pretrained DINO가 아니다.
- loss finite, 출력 **[2, 8, 27]**, model state unchanged.
- backward/optimizer steps **0**.
- [forward.log](forward.log), [입력 검사](../../../../out/student-review-20260909/input-check.json).

## 발견하고 수정한 문제

### 1. 무효 패딩 행동값이 유효 구간의 손실에 영향을 줌

`compute_loss`가 마지막 손실 항에만 valid mask를 적용했다. Transformer self-attention은
무효 위치의 입력도 읽으므로 padding 값을 바꾸면 유효 위치의 예측도 변했다.

실제 축소 attention 모델, 같은 가중치·관측·유효 행동·난수에서 무효 행동만 0→100으로
바꾼 진단의 loss는 **1.25426018 → 1.25360596**으로 달랐다. 이 숫자는 학습 결과가
아니라 무효 값에 대한 민감도 재현이다. 수정은 noising 전에 무효 행동을 정규화 공간의
고정값 0으로 치환한다. 무효 위치에는 잡음이 남지만 임의 패딩 라벨 정보는 들어가지 않는다.
수정 후 noise assignment on/off 모두 유효 손실이 정확히 같다.

### 2. Epoch loss를 배치 평균들의 단순 평균으로 집계

배치 크기와 유효 action 수가 다른 마지막 배치가 큰 배치와 같은 가중치를 받았다.
예를 들어 유효 target 8개에서 loss 1, target 1개에서 loss 9이면 기존 평균은 5지만,
전체 유효 target의 평균은 **17/9**다. `DiffusionLossMean`으로 train/validation 로그를
유효 action 수에 따라 집계하도록 수정했다. 한 action의 27차원 수는 모든 항에서 같다.
Optimizer의 배치별 갱신 방식은 바꾸지 않았다.

### 3. 설치 흔적만으로 의존성 정상 판정

preflight의 `find_spec`만으로는 설치됐지만 import가 실패하는 torch/torchvision 조합을
탐지하지 못한다. 실제 import와 버전을 기록하고 예외를 준비 미완료 사유로 반환하도록 했다.
`torchvision::nms` 오류를 주입한 회귀 검사에서 실패로 판정한다.

변경 코드: `src/icra/student.py`, `scripts/train_hx5_student.py`.
회귀 검사: `tests/test_student.py`, `tests/test_student_inference.py`.
새 모델 소스와 맞지 않는 과거 diffusion checkpoint는 기존 해시 검사가 막는다.
현재 접근 BC/PPO는 다른 모델 코드이므로 이 변경 대상이 아니다.

## 목표 조건과 가려짐에서 남은 문제

새 부위 요청 시 queued action을 버리는 것과, 부위가 가려져도 `part_present=1`을 유지하는
것을 회귀 검사로 확인했다. 입력/행동 계약은
[STUDENT_GOAL_CONTRACT.md](../../../../docs/target-contract.md)에 정리했다.

9개 시범 중 4개에는 head camera의 목표 물체 mask가 0인 구간이 있다:

| Episode | 전체 프레임 | 목표 mask=0 |
|---|---:|---:|
| teacher_0006_yaw_000 | 971 | 565 |
| teacher_0008_yaw_000 | 988 | 80 |
| teacher_0009_yaw_000 | 972 | 12 |
| teacher_0016_yaw_000 | 974 | 433 |

합계 1,090/8,817프레임이다. head mask가 비었다고 wrist 영상에서도 물체가 안 보인다고
단정하지 않는다. 소실 원인이 가림·시야 이탈·해상도 중 무엇인지는 이번 검사에서 분리하지
않았다. 다만 현재 정책은 현재 관측만 사용하고 요청 ID는 모델에 임베딩되지 않는다.
동일 RGB·관절 상태에서 서로 다른 요청 mask가 모두 사라지면 목표를 구별할 정보가 없다.
이를 teacher 실패로 재분류하거나 시범에서 조용히 제거하지 않았다. 추적/기억/목표 표현과
가려짐 범위 평가는 다물체 조건부 시각 전이 단계에서 정해야 한다.

원본 영상은 **64×64**다. 518×518으로 확대해도 손잡이 등 세부 부위 정보가 늘어나지는 않는다.
새 부위 시범을 만들 때 visible pixel 수와 목표 소실 구간을 확인해야 한다.

## 검사 및 다음 연결

수정 전 패딩 회귀 2건은 실제로 실패했다. metric helper와 깨진 dependency 판정 검사도
수정 전 실패를 기록했고 수정 후 통과한다. 기록: [regressions-before.log](regressions-before.log).

최종 관련 테스트는 **88 passed, 1 skipped**이며 [tests.log](tests.log)에 보존한다.
Conditioning의 실제 렌더
검사는 GPU 작업과 분리하려고 생략했으며, 전체 프로젝트 테스트를 재실행한 것은 아니다.
변경 파일 Ruff, `git diff --check`, 문서 링크 검사도 통과했다.

다음 순서:

1. Claude가 새 다물체 시범을 만들면 동일 schema/배치 귀속과 RGB/mask 가시성을 확인한다.
2. 실제 DINO source/가중치 경로를 지정하고 호환 torchvision이 있는 student 환경에서
   같은 preflight 및 `dry-run --full-model`을 실행한다. 현재 fixture 성공으로 대신하지 않는다.
3. 실제 부위 두 개의 시범과 물리적 접촉 준수를 확보한다. 현재 픽셀 fixture는 사용하지 않는다.
4. 독립 물체 분할과 작은 전체 시각 실행을 검증한 뒤 학습 규모를 늘린다.

재현 명령(출력 경로는 새 디렉터리로 지정):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  third_party/dexonomy/.venv/bin/python scripts/train_hx5_student.py preflight \
  --data out/offset-compare-93/executed-offset \
  --cache out/student-review-20260909/cache

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  third_party/dexonomy/.venv/bin/python scripts/train_hx5_student.py dry-run \
  --data out/offset-compare-93/executed-offset \
  --cache out/student-review-20260909/cache
```
