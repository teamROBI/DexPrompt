# 형상·목표 조건부 DP3 실행 가능성 검사

2026-09-12 23:03 KST 시작. **완료 (23:14:12 KST): 원본 재생 3/3, 학습 정책 평가 5/9 완주.**
사용자가 DP3 비교 실행을 요청했다. Claude의 `stage1_bc` 구현·GPU 2 작업과 분리하며
이 코드는 `scripts/stage1_dp3/`와 해당 snapshot에만 작성했다. 공통 코드·물리 기준은 수정하지 않았다.

## 질문과 범위

정확한 현재 물체 점군·로봇 상태·Dexonomy 파지 목표에서, DP3가 전체 팔·손 행동을
학습해 **학습한 조건**에서 완주하는가? 형상에서 목표를 스스로 선정하는 실험이 아니며,
새 물체 일반화·VLA 성능·BC+PPO 대비 우월성의 검증도 아니다.
Claude 실험은 75시범 및 다른 상태 입력이므로 현재 두 실행을 단일 요인 비교라 하지 않는다.

## 고정 자료

데이터 계약 원장의 A 물리·단일 물체·현재 controller·중앙(yaw 0)에서 물체당 1개를 골랐다.
기존 신호에서 `contact_acquisition_guard_v2`를 통과하는 시범 중 기록된 품질 여유가 가장 큰
항목, 동률은 episode SHA 순이다. 선택은 학습 전에 고정했고 세부 탈락도 selection.json에 남겼다.
총 3시범·3,089프레임. 과거에 관측한 성공 시범이므로 독립 test가 아니다.

| 물체 | 후보 SHA | episode SHA | tick |
| --- | --- | --- | ---: |
| AlphabetSoup | `34815d4f21e5` | `409a5d4a75d5` | 1024 |
| Peaches | `7a8cdecf4042` | `9aa6657866eb` | 1047 |
| Raisins | `af15fdf0bce4` | `0ef25c7b5eea` | 1018 |

## 모델과 관측

- 공식 소스: https://github.com/YanjieZe/3D-Diffusion-Policy,
  commit `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`.
- 공식 DP3의 PointNet encoder, ConditionalUnet1D, compute_loss/predict_action,
  LinearNormalizer와 EMA를 사용한다. DDIM 100 training timesteps/10 inference steps,
  sample(x0) prediction, horizon 16/history 2/execute 8.
- 네트워크 폭 `[256,512,1024]`는 공식 YAML `[512,1024,2048]`보다 작다.
  encoder output 64. HX5 관측·행동에 맞춘 **DP3 적응판**이지 원 논문 벤치마크 재현은 아니다.
- 공식 파일의 사용되지 않는 import 두 개(pytorch3d, torchvision)만 실행 사본에서 제거했다.
  각 이름이 import 외에 사용되지 않음을 검사했고 전후 해시는 upstream-adaptation.json에 기록.
  공식 원본 checkout은 수정하지 않았다.
- 점군: 컴파일된 **visual mesh** 표면 512점 + table 상면 128점. scale은 컴파일 mesh에 이미
  적용되어 있다. 매 tick 행동 직전 geom 위치/회전으로 world 변환하며 초기 물체 자세를 반복하지 않는다.
  RGB, mask, 카메라 렌더, 과거 실험의 이미지 크기에 의존하지 않는다.
- 상태 125차원: 관절 위치27·속도27·직전 최종 명령27·현재 손 루트 위치/회전12·요청 목표32.
  요청 목표는 world 파지 root12와 손가락 squeeze20. 원장에 있는 adapted_final은 사용하지 않는다.
  phase·경과 시간·결과·미래 실제 상태·행동 후 접촉은 정책 입력에 없다.
- 출력 27차원 최종 actuator 명령을 직접 사용한다. actuator 명령 범위 clip만 적용하고 중력 보상,
  teacher 접촉 피드백, IK를 정책 실행에 추가하지 않는다.

## 순서와 판정

1. 전체 전송 파일 SHA 확인 → 공식 모델의 loss·sampling 검사(backward/optimizer 0회).
2. 세 보관 모델과 초기 상태에서 기록 명령 재생. 모든 tick의 관절·접촉·힘·침투·높이 신호를
   원본과 비교하며, 상태 오차≤1e-6 rad, 힘≤1e-5 N, 기하 신호≤1e-8 m, 접촉 불일치0을 요구한다.
   이것은 CPU간 재현 수치 검사이며 물리 품질 한계를 변경하는 것이 아니다.
   bit 동일 여부를 따로 기록하고 phase-free 양성 대조 성공도 요구한다. 미통과 시 학습 금지.
3. 통과 시 행동 직전 점군/상태를 별도로 저장해 학습. seed42, batch64, AdamW lr1e-4,
   decay1e-6, warmup500/cosine, gradient clip1, **6,000 updates**. 세 episode 균등 샘플링.
   원본 시범과 clip 경계를 넘어 action window가 섞이지 않으며 첫/끝은 경계값 padding.
4. EMA **최종** checkpoint만 각 3조건×seed1000–1002 = 9회 평가. 결과를 보고 checkpoint를 고르지 않는다.
   각 원본 tick 길이가 실행 예산이다. 접촉 상실과 품질 위반이 성공으로 처리되지 않음을 회귀 검사로 확인.
   contact_acquisition_guard_v2: 전 접촉군 0.5s 획득 후 접근 이동 판정을 종료;
   종료 시 10cm 이상 상승 및 전 접촉군/힘 유지가 0.5s 이상, 궤적 품질 위반 없음.
   기록된 teacher phase를 learned rollout에 덮어씌우지 않는다.

이 모델은 learned action을 바로 적용하므로 현재 teacher의 접촉 피드백이 자동으로 유지되는
구조는 아니다. 실제 완주 여부를 보고하고, 실패를 데이터 생성 성공으로 해석하지 않는다.

## 자원과 실행 위치

- lecun GPU **3**, CPU thread 1, 렌더 없음, nice10. 기존 사용자·Claude 프로세스 미변경.
- tmux `codex-stage1-dp3` (SSH가 끊겨도 계속 실행).
- `/data1/yspark/projects/dexprompt/stage1-dp3-v1/`
  - `package/`: 영상 없는 3시범, 고유 모델3개, 초기 상태, 코드·plan·hashes. 전송 약626 MB.
  - `deps/`: 별도 zarr2.18.3/numcodecs0.13.1/asciitree/fasteners/termcolor.
  - 기존 upstream-action-v1/deps의 diffusers0.32.1 등은 읽기 전용 재사용, 공용 venv 변경 없음.
  - `model-check.json`: 실제 공식 모델 입력·손실·샘플 출력 검사.
  - `run/`: status.json, stages.json, replay.json, features/, progress.tsv, final.pt,
    evaluation.json, FINAL.md. 실패 exit/timeout은 파지 실패와 분리.
- model-check 통과 후 실행기 전체 상한4시간; replay40분/train150분/evaluate40분.
  조건 미충족 시 후속 단계 중단. `run/STOP`은 학습과 평가 시작 경계에서 확인된다.
- 시작 전 검사: 새 회귀 테스트4개 통과; 보관 모델3개를 시간 진행 없이 열어 point640×3,
  state125 및 점군 치수 확인. 실제 모델 forward/loss/sampling 유한, 출력[2,8,27],
  파라미터 약66.77M, backward/optimizer0회. 전체 suite 재실행은 하지 않았다.

## 현재 상태 확인

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa_yspark lecun
cat /data1/yspark/projects/dexprompt/stage1-dp3-v1/run/status.json
tmux attach -t codex-stage1-dp3
```

완료 후 tmux 세션은 종료됐다. 상태와 결과는 보관된 JSON을 기준으로 확인한다.

## 실행 확인

replay.json: 세 물체 모두 모든 tick의 right_state와 접촉·힘·높이·침투 신호 오차0,
양성 대조 contact_acquisition_guard_v2 3/3 완주. 각각 약20·18·16초.
서로 다른 CPU에서도 이 세 기록은 bit 동일하게 재현됐다.
train-status-snapshot.json은 회수 시점의 학습 진행 기록이고 최종 결과가 아니다.
최종 평가와 FINAL.md는 원격 실행기가 학습 완료 후 자동 작성한다.

## 최종 결과

학습 6,000회 갱신 완료(316.75초). 마지막 train loss는 0.00003536이며,
폐루프 성공 여부는 별도 시뮬레이션으로 판정했다. 최종 EMA만 평가했다.

| 물체 | 완주 / 실행 | 관찰 |
| --- | ---: | --- |
| AlphabetSoup | 2/3 | 성공은 종료까지 1.20–1.42초 조건 유지. 실패는 최대 상승 69.9 mm |
| Peaches | 0/3 | 최대 상승 146.4–148.6 mm, 중간 조건 유지 2.30–2.42초였지만 종료 시 유지 조건 미충족 |
| Raisins | 3/3 | 최대 상승 145.7–146.3 mm, 종료까지 3.18–3.38초 조건 유지 |
| 합계 | **5/9** | 학습한 3조건 × 샘플링 seed 3개 |

9회 모두 품질 위반과 actuator 범위 clipping은 없었다. Peaches의 종료 시 조건 상실이
접촉·높이 중 무엇에서 시작했는지는 이 요약 지표만으로 분리하지 않는다.

**확인한 것:** 정확한 형상·상태와 주어진 Dexonomy 목표에서 학습된 정책이 팔·손 전체를
직접 제어해 일부 학습 조건을 완주했다. 실행 중 기존 접촉 피드백 제어기나 IK로 마무리하지 않았다.
**아직 확인하지 않은 것:** 새 배치·새 물체 일반화, 목표/유형 자동 선정, 기존 생성기보다 높은 수율,
MLP BC/PPO보다 높은 성능, 최종 VLA 성능. 원본 재생은 3/3 성공이므로 현재 결과로 기존보다
좋은 생성기라고 주장하지 않는다. 정책 학습 seed는 42 하나이고 9회는 추론 샘플링 반복이다.

결과 파일: evaluation.json, training-result.json, stages.json, status.json, FINAL.md.
최종 checkpoint SHA256: `cf8c59d613e02bc79c0678a01a646f0eb02c0b40c5e069dbb5748551e16f167d`.
실행 결과를 VLA 학습용 완전한 시범으로 인수하거나 기존 데이터에 병합하지 않았다.
