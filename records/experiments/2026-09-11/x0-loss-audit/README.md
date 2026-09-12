<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# x0 손실 분해와 낮은 잡음 가중치 비교 준비

**검사한 입력에서는 팔 오차가 손실의75–79%를 차지한다.** 손가락 차원이 많아서
팔의 실제 손실 기여가 묻힌다는 설명은 지지되지 않는다. 낮은 잡음 t0 손실은
검사한 t25·t49보다 크며, 이전 생성 경로 진단과 함께 다음 비교 대상으로 삼았다.
새 학습은 하지 않았고 **별도 옵션 구현·테스트·lecun 실제 DINO 사전 검사까지 완료**했다.

## 기존 손실이 실제로 계산하는 것

학습 시점의 `student.py`와 `train_hx5_student.py`가 당시 run 해시와 같은지 확인한 뒤
[frozen-source](frozen-source)에 보존했다. 이후 추가한 옵션과 기존 계산을 구분한다.

기존 x0 손실은 정규화된 raw 출력과 정규화된 정답의 제곱 오차를 모든 유효 행동·27관절에
균등 평균한다. **clipped 출력이나 물리 rad 오차를 학습 손실로 쓰지 않는다.**

- 잡음 timestep은 예제마다0–49 균등 추첨: 각2%, t0–9 합계20%.
- 패딩 없는64행동 묶음에서 첫8행동의 좌표 비중은12.5%, 첫8개 팔7관절은3.24%.
  이는 동일 제곱 오차를 가정한 계수 비중이지 실제 손실·gradient 비중이 아니다.
- 정규화 scale로 나누므로 같은 rad 오차라도 관절별 손실은 다르다.
- 패딩은 분자와 분모에서 제외한다. 분모는 `valid.sum()*27`이다.
- 실제 학습은 batch2·noise assignment 켜짐·dropout0.1인 train mode다.
  저장된 진단은 독립 noise·eval mode다. 따라서 아래 값은 **학습 로그의 재현이 아니라,
  같은 손실식에 대한 고정 관측 평가**이며 전체974관측·전체50timestep 평균도 아니다.

## 저장된72개 raw 예측의 분해

8관측×3seed×t0/25/49의 raw x0를 재사용했다. 추가 모델 forward는0회다.
네 영역의 합이 원래 normalized MSE와 일치하도록 검증했다. 비율은 각 영역의 SSE를
해당 timestep의 전체 SSE로 나눈 값이며, 시점별 비율의 단순 평균이 아니다.

| timestep | 평균 normalized MSE | 팔 전체 기여 | 첫8 팔 기여 | 첫8 손 기여 | 나머지56행동 기여 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0012397 | 79.2% | 9.7% | 1.0% | 89.3% |
| 25 | 0.0008609 | 78.2% | 7.3% | 2.5% | 90.3% |
| 49 | 0.0008493 | 75.3% | 9.0% | 1.6% | 89.4% |

실제 첫8팔 기여는7–10%로 좌표 비중3.24%보다 크다. 나머지56행동이 전체 손실의
89–90%를 차지하지만, 그것만으로 앞8행동 가중치 변경이 해결책이라고 결론내리지 않는다.
t0 손실은 t25보다약1.44배, t49보다약1.46배다. 측정하지 않은 t1–24·26–48이 같은
양상이라고 가정하지 않는다. 이 수치는 gradient norm·최적화 난이도의 측정도 아니다.

## 다음 단일 요인: 낮은 잡음 손실 가중치

선택적 설정 `low_noise_loss={steps:10, relative_weight:4.0}`을 추가했다.
기존 설정에는 키가 없으므로 동작은 그대로다. 새로운 config만 이 옵션을 켠다.

- t0–9: 가중치2.5. t10–49:0.625. 상대비4:1, 균등 timestep에서 평균 가중치1.
- 식: `sum(weight[t] * squared_error * valid) / (27 * sum(valid))`.
- batch 가중치 합으로 분모를 바꾸지 않는다. 그러면 batch1에서 효과가 사라진다.
- timestep 추첨·noise assignment·팔/손/offset 가중치·학습률·horizon·샘플러는 바꾸지 않는다.
- 평균 계수는 같지만 실제 weighted loss 및 gradient norm이 같아지는 것은 아니다.
  변경 후 train_loss 숫자를 기존 숫자와 직접 비교하지 않는다.
- t0–9와 상대비4는 **사전 고정한 가설값**이다. 최적값을 찾았거나 전체 낮은 잡음
  구간의 오류가 입증됐다는 뜻이 아니다. 효과는 향후 동등 예산 학습으로 확인한다.

[다음 실험 계획](../../../plans/STUDENT_LOW_NOISE_LOSS_2026-09-11.md)에 비교·완료 조건을 고정했다.

## 구현과 검사

- `src/icra/student.py`: x0 전용 선택적 가중치, 값 검증, 기본 손실 경로 유지.
  sampler와 state_dict 구성은 미변경.
- `scripts/train_hx5_student.py`: 경량 dry-run에서 절대 timestep cutoff를 보존하도록
  이 옵션이 있을 때만 학습 diffusion 단계 수를 유지한다. 실제 전체 모델 설정은 그대로다.
- `configs/hx5_student.single-bottle-x0-low-noise.lecun.json`: 기존 x0 config와 옵션1개만 다름.
- **관련76테스트 통과**: 가중 loss/gradient, t9/t10 경계, batch1, 패딩 gradient0,
  RNG 불변·sampler 출력 불변, 부적절한 설정 거부, 경량 dry-run, 기존 x0 oracle·inference 등.
  합성 소형 텐서 gradient 검사는 수행했으나 실제 모델 학습이나 optimizer 갱신은 없다.
- [기본값 보존](default-preservation.json): 수정 전 소스와 비교해 epsilon/x0 각각
  train/eval loss와 샘플 출력이 소형 fixture에서 bit 동일함을 확인했다.
- 새 코드·tests Ruff 통과, diff whitespace 검사 통과. 전체 repository suite는 미실행.
  기존 trainer 파일 전체의 오래된 스타일을 일괄 수정하지 않았다.

## lecun 준비 — 학습 없음

기존 `out/student-x0-code-v1`은 그대로 두고 **새 코드 사본**
`/home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1`을 만들었다.
기존 checkpoint는 strict source 검증을 통과하는 원래 코드 사본으로 평가한다.
새 소스에서 과거 checkpoint를 `allow-code-change`로 우회해 읽지 않았다.

GPU1 tmux `dexprompt-low-noise-check-v1`에서 preflight와 실제 DINO full-model dry-run 완료,
종료 코드0. 실제 시범의 frame0·200 배치에서 random initialization loss1.1956645,
행동 `[2,64,27]`, 유한성·모델 불변·backward/optimizer0을 확인했다.
이 값은 기존 학습 checkpoint의 성능이 아니다. [검사 결과](lecun-check/results/dry-run/dry_run.json).

기존 cache SHA `bad6aba5e5b24320…`가 일치하며 데이터는 미변경.
실행 중인 자체 tmux는 없고 GPU1은 해제됐다. 아직 새 학습 run/checkpoint는 생성하지 않았다.
원격 job은 `/data1/yspark/projects/dexprompt/student-checks/x0-low-noise-check-job-v1`,
결과는 같은 부모의 `x0-low-noise-preflight-v1`에 보존한다.

## 산출물과 재집계

[audit.json](audit.json)에72개 개별 손실과 영역별 기여, 사용 소스 해시를 저장했다.
`audit_x0_loss.py --raw results/2026-09-11/x0-denoising-trace/raw
--run results/2026-09-11/x0-denoising-trace/checkpoint-run.json
--source-root results/2026-09-11/x0-loss-audit/frozen-source --out <새 폴더>`로 재집계한다.
원래 source 해시를 검증하므로 새 옵션 구현 후에는 보존된 source-root를 지정한다.
Claude teacher/Milk/multiccd 작업과 fixed7·새5 자료는 변경하지 않았다.
