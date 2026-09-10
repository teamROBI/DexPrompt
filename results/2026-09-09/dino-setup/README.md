# 공식 DINOv2 다운로드 및 실제 전체 모델 검증

2026-09-09. 사용자가 직접 다운로드를 승인한 뒤 공식 소스·가중치를 준비했다.
기존 Dexonomy 실험 환경은 변경하지 않고 별도 `.venv-student`에 설치했다.
**학습, backward, optimizer 갱신은 하지 않았다.**

## 준비한 파일

공식 [DINOv2 저장소](https://github.com/facebookresearch/dinov2)의 backbone 다운로드를
사용했다. 현재 student 설정에 맞는 **register 없는** ViT-B/14 및 ViT-L/14다.

| 항목 | 저장 경로 | 크기/버전 |
|---|---|---|
| DINOv2 소스 | `out/student-assets/dinov2` | `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, detached HEAD |
| Head encoder | `out/student-assets/dinov2_vitb14_pretrain.pth` | 346,378,731 bytes |
| Wrist encoder | `out/student-assets/dinov2_vitl14_pretrain.pth` | 1,217,586,395 bytes |
| 사용 설정 | `configs/hx5_student.local.json` | 기본 설정에 로컬 소스·가중치 경로 지정 |
| 실행 환경 | `.venv-student/bin/python` | Python 3.10.12 |

다운로드 URL과 전체 SHA256은 [assets.json](assets.json)에 있다. 임시 `.partial` 파일을
`torch.load(weights_only=True)`로 읽어 patch/position embedding 규격과 register 토큰 부재를
확인한 뒤 최종 파일명으로 바꿨다. 후속 실제 모델 생성에서 `load_state_dict(strict=True)`도
통과했다. SHA256은 다운로드한 파일의 재현 식별자이며 별도 공식 서명 검증을 뜻하지 않는다.

주요 패키지는 torch **2.6.0+cu124**, torchvision **0.21.0+cu124**, numpy **2.2.6**,
scipy **1.15.3**, omegaconf **2.3.1**, mujoco **3.3.3**다.
PyTorch/torchvision 조합은 [공식 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/)
기준으로 맞췄다. 전체 버전은 [requirements.lock](requirements.lock)에 고정했다.
`uv pip check`에서 44개 패키지의 의존성 정합성을 확인했다.

설치에 사용한 UV cache는 `/tmp/icra-student-uv`다. 환경은 약 5.1 GiB, 소스/가중치는
약 1.5 GiB다. cache와 환경은 hardlink를 공유할 수 있으므로 두 `du` 값을 단순 합산하지 않는다.
`.venv-student/`와 대형 산출물이 Git에 들어가지 않도록 처리했다.

## 실제 검증 결과

[verify_full_model.py](verify_full_model.py)가 다운로드·backward·Adam/AdamW 생성을 금지한
상태에서 기존 `dry_run` 경로를 호출했다. CPU 한 스레드에서 약 **96.2초**였다.

- ViT-B/14: **86,580,480 parameters**, embedding width 768, strict load 통과, 동결 확인.
- ViT-L/14: **304,368,640 parameters**, embedding width 1024, strict load 통과, 동결 확인.
- 기본 크기의 전체 student: **554,994,715 parameters**, 입력 크기 518, action horizon 64.
- 기존 승인 병 시범 9개의 verified cache를 사용하고 그 중 batch 2로 forward/생성을 검사했다.
- 행동 출력 **[2, 64, 27]**, loss와 출력 모두 finite, state_dict 전체 변화 없음.
- 초기 diffusion loss **1.40470147**. DINO는 사전학습 가중치이고 action 모델은 초기화
  상태다. 이 수치를 학습 성과나 파지 성공으로 해석하지 않는다.
- 확인 중 다운로드·backward·optimizer steps **0**. GPU 사용 없음.
- xFormers 미설치 경고는 있었으며 PyTorch 기본 attention 경로로 실행을 완료했다.

근거: [full-model.json](full-model.json), [encoders.json](encoders.json),
[full-model.log](full-model.log).

새 환경에서 관련 회귀 테스트 **88 passed, 1 skipped**: [tests.log](tests.log).
실제 렌더 검사는 생략했다. CPU forward 검증은 GPU 학습 메모리·속도, 실시간 추론 성능,
실제 파지 성공이나 전체 시뮬레이터 replay 검증을 대신하지 않는다.

## Preflight의 두 결과

| 조건 | 결과 | 의미 |
|---|---|---|
| 기본 preflight | 종료 코드 2, training_ready=false | 남은 사유는 validation split 부재 하나 |
| `--allow-train-only-diagnostic` 명시 | 종료 코드 0, training_ready=true | 단일 물체의 학습 전용 진단에 한한 준비 확인 |

소스·가중치·의존성 문제는 해결됐다. 데이터는 여전히 병 하나, train=9/val=0/test=0이다.
시범을 임의로 나눠 독립 물체 검증처럼 만들지 않았다. 학습 전용 옵션을 일반 학습의 기본값으로
바꾸지 않았으며, 이 준비 과정에서 학습을 시작하지도 않았다.

## 실행 방법

**저장소 루트에서**, 새 환경과 local 설정을 함께 사용한다. 기본 `hx5_student.json`의
null 경로를 사용하거나 기존 Dexonomy Python을 사용하면 이번 준비가 적용되지 않는다.

```bash
# 일반 학습 준비 검사: 현재는 독립 검증 데이터가 없어 종료 코드 2가 정상
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv-student/bin/python scripts/train_hx5_student.py preflight \
  --config configs/hx5_student.local.json \
  --data out/offset-compare-93/executed-offset \
  --cache out/student-review-20260909/cache

# 실제 pretrained DINO를 포함한 전체 모델 무학습 검사
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  .venv-student/bin/python scripts/train_hx5_student.py dry-run --full-model \
  --config configs/hx5_student.local.json \
  --data out/offset-compare-93/executed-offset \
  --cache out/student-review-20260909/cache
```

환경 재생성 시 `.venv-student`를 새로 만든 뒤 다음 고정 목록을 설치한다:

```bash
uv pip install --python .venv-student/bin/python \
  --index-strategy unsafe-best-match \
  --index https://download.pytorch.org/whl/cu124 \
  --default-index https://pypi.org/simple \
  -r results/2026-09-09/dino-setup/requirements.lock
```

두 공식 index에서 정확히 고정한 버전을 해석하도록 사용한 옵션이다.
이 환경의 모델 소스는 assets.json의 commit으로 고정하고, 가중치는 같은 URL과 SHA256으로
대조한다. 새로운 환경에서 다시 검사한 뒤 학습에 사용한다.
