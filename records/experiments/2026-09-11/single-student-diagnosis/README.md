<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 단일 시범 student: 국소 잡음 제거와 전체 행동 생성의 차이

2026-09-11. lecun 단일 병 학습의 사전 지정 최종 epoch 13만 읽었다.
학습 관측 frame **0/200/450/700**, 잡음 seed 1000/1001/1002를 고정했다.
새 학습·물리 실행은 없고, `backward`와 `AdamW.step`은 호출 시 예외로 차단했다.
기존 모델·normalizer·전처리·checkpoint는 수정하지 않았다.

## 직접 확인한 내용

1. 실제 학습 입력을 넣어도 첫 명령이 틀린다. frame 0의 최대 관절 오차
   **0.414/0.918/0.860 rad**가 앞선 폐루프 결과와 float 정밀도 안에서 일치한다.
   frame 200/450에서도 각각 0.454–0.561 / 0.485–0.642 rad다.
   frame 700은 0.118–0.441 rad로 차이가 있지만 완주를 측정한 것은 아니다.
2. **정답 행동에 작은 잡음을 넣어 복원하는 능력은 있다.** 다음은 3개 잡음 seed의
   평균 chunk RMSE이며, 정답 chunk를 먼저 제공하는 진단이다. 실제 행동 생성 성능과
   혼동하지 않는다.

| 학습 관측 frame | 약한 잡음 t=0 | 중간 t=25 | 강한 잡음 t=49 |
| --- | ---: | ---: | ---: |
| 0 | 0.00363 rad | 0.03241 rad | 0.25752 rad |
| 200 | 0.00378 rad | 0.02044 rad | 0.28089 rad |
| 450 | 0.00186 rad | 0.01699 rad | 0.32525 rad |
| 700 | 0.00128 rad | 0.01624 rad | 0.27285 rad |

3. 실제 `predict_action()`의 역확산 경로도 계측했다. 처음 t=49에서 clean action을
   복원할 때 **전체 좌표의 97.7–98.2%**가 정규화 범위 [-1,1] 밖이라 clipping된다
   (seed 1000, 관측 4개). 팔 7관절만 봐도 **98.0–99.6%**다. 이는 물리 actuator
   한계에 대한 saturation 비율이 아니라 **샘플러 내부의 정규화 좌표 clipping**이다.
   action_scale ≥0.01 rad인 유효 차원으로도 확인했으며 이번 checkpoint의 27개
   차원은 모두 해당한다. 거의 안 움직이는 관절만 집계해 생긴 숫자는 아니다.
4. 낮은 epsilon loss의 한계가 남아 있다. frame 0, t=49의 평균 epsilon MSE는
   0.00466인데, alpha_bar≈9.71e-7에서 이를 clean action으로 환산하면 큰 오차가 난다.
   같은 noised teacher 입력에서 normalized clean action을 항상 0으로 가정하는
   기준선의 epsilon MSE는 약 3.92e-7이다. **강한 잡음에서 epsilon loss가 작다는
   사실만으로 유용한 행동을 예측한다고 할 수 없다.** 이 기준선은 정규화 중심을
   내보내므로 파지를 할 수 있다는 뜻도 아니다.
5. 관측 encoder 출력을 0으로 바꾼 진단에서 frame 0의 첫 행동은 최대 0.788 rad
   달라지고 teacher 오차는 1.202 rad로 커졌다. 따라서 관측에 대한 민감성은 있다.
   모든 condition token을 없앤 분포 밖 진단이므로 이미지·mask·관절 중 무엇을
   올바르게 사용하는지, 시각 grounding이 됐는지는 이 검사로 알 수 없다.

## 결론과 다음 실험 제안

확인한 것은 **국소적인 잡음 제거는 되지만 순수 잡음에서 유용한 행동을 생성하지
못하는 상태**다. 작은 epsilon loss만을 보고 데이터 확대나 학습 시간 증가로 해결될
것이라고 판단하지 않는다. clipping을 제거하면 해결된다거나 schedule 구현이
틀렸다는 결론도 아니다. 앞선 정확한 epsilon oracle 검사는 통과했다.

다음 단일 요인 후보는 **출력/학습 목표를 epsilon에서 clean action(x0)으로 바꾸는
비교**다. 학습 목표와 그에 맞는 sampler 복원식은 함께 바꿔야 하며, 기존 epsilon
checkpoint에 x0 해석만 씌워 평가해서는 안 된다. 모델 구조·DINO·단일 시범·normalizer·
학습 seed·6,331 update 예산·품질 기준을 유지하고 별도 config/checkpoint로 비교한다.
전처리 변경·새 물체 데이터·rate 제한·관측 제거를 동시에 도입하지 않는다.

이 변경은 **아직 구현하거나 학습하지 않았다**. 착수 시 먼저 정확한 x0 oracle과
샘플러 회귀 검사를 만들고, 새 학습은 lecun의 빈 GPU에서 tmux로 실행한다.
초기 행동 오차뿐 아니라 teacher 인계 없는 동일 과제의 전체 완주로 판정해야 한다.
HOPE 자산 준비는 이 문제와 별도로 진행하며, 새 데이터가 원인 해결의 증거는 아니다.

## 재현 및 산출물

- `scripts/diagnose_single_student.py`: strict checkpoint load, 실제 EpisodeChunks와
  `predict_action`을 사용한다. 4개 관측 모두 학습 과제 안의 진단이다.
- [diagnosis.json](diagnosis.json): 최초 진단.
- [diagnosis-active.json](diagnosis-active.json): 동일 조건에 팔/유효 차원 clipping
  계측만 추가한 확인. 원래 통계와 모델 출력은 동일 조건으로 보존한다.
- lecun `out/single-student-diagnosis-v1`, `-v2`: 각각 종료 코드 0, completed=true.
  GPU 0(UUID `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`)이 비어 있는 것을 확인한 뒤
  CPU/BLAS 1스레드·nice 15·20분 상한·별도 tmux로 실행했다. 작업은 모두 종료됐다.
  GPU 1–5와 다른 사용자의 작업은 변경하지 않았다.
