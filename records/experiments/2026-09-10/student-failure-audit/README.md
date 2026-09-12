<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 시각 student 초기 실패 감사

2026-09-10. 기존 epoch 2 평가를 보존하고, **새 학습 없이** 데이터·명령·물리 재현과
고정 체크포인트의 초기 출력을 확인했다. 감사한 계약에서 실패를 설명할 구현 오류는
발견하지 못했다. 모델이 첫 명령부터 시범을 크게 벗어난다는 것이 직접 확인된 문제다.
모델·학습·전처리·teacher 제어·품질 기준은 변경하지 않았다.

## 입력과 범위

- Dataset: `out/student-bottle-can-v1`, 시범 4개, 4,170프레임, 모두 학습 데이터.
- Cache: `out/student-bottle-can-v1-cache`. 이번에는 봉인 manifest·파일 크기·헤더와
  데이터 연결을 검사했다. 전체 캐시 바이트 해시는 학습 사전 검사에서 검증된 것으로,
  이번에 전부 다시 해시했다고 주장하지 않는다.
- 고정 checkpoint: `epoch_0002.pt`, SHA256
  `8d6ec064fa6e528f4699d63a2804b40ab376756d4e622c5f31ac2bdc7364550b`.
- 원래 평가: `out/lecun-student-eval-v1/results`의 12개 기록.
- CPU 감사: `scripts/audit_student_failure.py`, 결과 `out/student-failure-audit-v3`.
  v1/v2는 같은 진단에 자기접촉 분리와 pre-action 정렬 검사를 순차 추가한 중간 기록이다.
- GPU 진단: `scripts/diagnose_student_denoising.py`, `out/student-denoising-audit-v1`.
  lecun GPU 0이 비어 있음을 확인한 뒤 `dexprompt-diagnosis` tmux, CPU 1스레드,
  nice 15로 실행했다. GPU 1–5나 다른 사용자 프로세스는 변경하지 않았다.
  `backward`와 `AdamW.step`은 호출하면 예외가 나도록 차단했고 종료 코드 0이다.

## 데이터와 실행 계약

1. 학습 소스 8개의 해시가 run 기록과 일치한다. 전체 학습 데이터에서 다시 계산한
   action/state normalizer가 run 기록과 정확히 같다. GPU strict checkpoint load에서도
   저장 normalizer·확산 schedule buffer와 metadata 일치를 확인했다.
2. 27개 이름 순서, 실제 actuator가 연결된 joint ID, absolute joint position,
   0.02초 제어 주기/0.002초 물리 주기가 일치한다. 추가 중력 보정은 하지 않는다.
3. 각 시범의 시작·100 tick·wrist 참조 획득·마지막 frame, 합계 **16개 상태**에서
   `EpisodeChunks`와 실제 `StudentActor`가 만드는 모든 입력 tensor가 정확히 같다.
   참조 기억은 과거 관측에서만 채웠다. 이 검사는 DINO forward를 관측 probe로 대체했다.
   raw label의 현재 frame 시작과 끝 padding도 확인했다.
4. 실제 actor의 실행 queue는 0–7번 명령을 순서대로 쓴 뒤 새 chunk로 넘어간다.
   캔을 포함한 초기 head/wrist mask·관절 입력이 학습 frame과 같으며, RGB 차이는
   캔 wrist의 **196,608 채널 값 중 하나가 1/255** 다른 것뿐이다. 병은 RGB도 모두 같다.
   이 작은 차이만으로 캔 실패 원인을 단정하지 않는다.
5. 저장 student 명령 12개를 archived MJB/reset에서 CPU로 재생했다. 접촉·힘·침투·
   물체 이동·높이 등 저장된 **모든 물리 배열의 최대 차이가 0**이었다.
   시범 명령의 초기 구간도 재생해 저장 pre-action 관절과 시간 정렬을 검사했다.
   병 각 17 tick·캔 118 tick에서 관절 차이 0, 전체 timestamp 간격 오차는 최대
   1.11e-14초였다. 전체 4시범 성공 재생 대조군은 이전 lecun 검사 결과를 참조한다.

## 언제 무엇이 잘못 움직였나

아래는 병 첫 시범과 캔을 각각 같은 seed 1000–1002로 본 수치다.
첫 명령 오차는 같은 초기 관측에 대응하는 시범의 첫 명령과 비교한다.
이후에는 정책과 시범의 상태가 달라지므로 동일 tick 차이를 정답 행동 오차로 해석하지 않는다.

| 관측 | 병 | 캔 |
| --- | --- | --- |
| 첫 명령의 최대 관절 오차 | 0.708–1.451 rad | 0.821–1.690 rad |
| 종료 전 최대 명령 변화/20ms | 1.168–1.451 rad | 1.337–1.704 rad |
| 같은 길이 시범의 최대 명령 변화/20ms | 약 0.0128 rad | 약 0.0137 rad |
| 첫 외부 침투 | 엄지/검지 말단–병 | 중지/약지 말단–테이블 |
| 종료까지 실행 tick 수 | 17 / 13 / 16 | 100 / 118 / 77 |

병 seed 1000은 시작부터 `arm_r_joint5` 명령이 시범보다 0.708 rad 크고,
seed 1001은 `arm_r_joint1`이 1.187 rad 작다. seed 1002의 joint5 오차는 1.451 rad다.
단위 변환 없이 큰 절대 목표가 실제 actuator에 전달된 결과를 물리 재생으로 확인했다.
명령 변화 수치는 관절의 측정 속도가 아니다.

원시 contact에는 home에서 이미 있던 손가락끼리의 약 0.55–0.68mm 겹침도 포함된다.
이를 첫 물체 충돌로 세지 않았다. 현재 home/품질 판정은 로봇–물체/테이블/방해물
침투를 검사하며 자기접촉을 같은 항목에 포함하지 않는다. 성공 기준을 이번에 바꾸지 않았다.
student의 품질 검사는 물리 stepping 뒤 종료하는 검사이므로 2mm 제한이
teacher의 미리보기 제어처럼 모든 순간의 침투를 2mm 이하로 막는다는 뜻도 아니다.
캔 seed 1000에서는 테이블 접촉 뒤 같은 tick에 물체 침투도 생겼고,
판정 우선순위 때문에 종료 이름은 `penetration`이다.

## 12개 기록의 중복

병 3시범은 MJB SHA·초기 integration state SHA·목표 조건·필요 접촉군·품질 한계가
같다. 예산만 974/971/974 tick으로 일부 다르다. 모두 17 tick 이전에 실패해 예산
차이는 이번 결과에 영향을 주지 않았고, 같은 seed의 명령과 모든 물리 배열이 정확히 같다.
따라서 **기록 12개는 서로 다른 관측·물리 초기 상태 2개 × seed 3개의 궤적 6개**로
구분해 보고한다. 향후 완주 근처 결과는 서로 다른 시간 예산까지 통일/명시해야 한다.

반면 병 teacher 시범은 중복이 아니다. 같은 초기 입력에서 첫 행동 차이는 최대
0.0055–0.0127 rad이고, 이후 경로 차이는 크다. 서로 다른 시범이 있다는 사실만으로
diffusion 실패 원인이 다중 경로 평균화라고 결론내리지 않는다. 현재 첫 명령 오차는
이 초기 teacher 간 차이보다 훨씬 크다.

## 추론 횟수와 확산 복원 진단

같은 초기 입력·checkpoint·seed로 16회 추론을 다시 실행했을 때 첫 8명령이 원래
폐루프 기록과 정확히 일치했다. 이어 **추론 횟수만 50**으로 바꿨다.
물리 실험은 아니며 50회 조건의 성공률을 측정한 것으로 쓰지 않는다.

| 초기 상태 | 16회: 첫 명령 최대 오차 범위 | 50회: 같은 지표 |
| --- | --- | --- |
| 병 | 0.708–1.451 rad | 0.905–1.331 rad |
| 캔 | 0.821–1.690 rad | 0.978–1.576 rad |

50회에서도 큰 오차가 남는다. 16회 샘플링을 50회로 바꾸는 것만으로 초기 시범
복원이 해결되지는 않았다. 기본 설정은 16회로 유지했다.

시범 chunk에 알려진 잡음을 넣고 예측 잡음에서 clean action을 복원하는 별도 진단도 했다.
병 초기 상태에서 t=49의 epsilon MSE는 0.00520이지만 clean action RMSE는 0.453 rad였다.
이때 alpha_bar는 약 9.71e-7이다. 낮은 평균 epsilon loss가 작은 actuator 오차를
보장하지 않는 구체적 사례다. **단일 역확산 단계의 오차를 전체 실패의 유일 원인이나
schedule 구현 오류로 해석하지 않는다.** 오직 초기 상태 2개와 고정 잡음에서 측정했다.

정확한 잡음을 반환하는 합성 oracle을 실제 `predict_action()`에 연결하면
16/50회 모두 알려진 절대 관절 chunk를 복원한다. 따라서 검사한 범위에서는
역확산 인덱스·부호·최종 역정규화의 단순 구현 오류가 발견되지 않았다.

## 다음 작업과 완료 조건

다음은 **고정 병 시범 하나를 카메라 정책으로 재현하는 제한된 학습 진단**이다.
지금의 전체 모델·전처리·성공 기준을 유지하고 별도 데이터/설정/출력 경로를 사용한다.
한 과제 진단의 목적은 실제 카메라 모델이 시범을 폐루프로 재현할 수 있는지 확인하는
것이며, 데이터 다양성이나 일반화 성과로 세지 않는다. 새 학습은 아직 시작하지 않았다.

- 학습 전에 단일 시범 데이터·초기 정답 chunk·업데이트 예산·평가 seed를 고정한다.
- epoch loss 외에 고정 관측의 실제 명령 오차를 기록하고, 체크포인트별 폐루프 결과를 본다.
- 초기 명령부터 틀리는 경우와 초기 명령은 맞지만 누적 오차로 실패하는 경우를 구분한다.
- 통과 조건은 teacher 인계 없이 home→접근→닫기→10cm 이상 지속 lift와 기존 접촉/품질 기준.
- 추가 학습, action 평활화/rate 제한, 목적함수 변경, 제어 기준 변경을 한꺼번에 도입하지 않는다.

검증: student 데이터·캐시·추론·준비·oracle 관련 **52개 테스트 통과**.
새 진단 스크립트 2개와 oracle 테스트의 Ruff 통과. 전체 저장소 테스트를 다시 돌린 것은 아니다.
상세 수치는 [CPU 감사](audit.json), [고정 모델 추론 진단](denoising.json)에 저장한다.

CPU 감사 재현(새 `--out` 필요):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa \
  .venv-student/bin/python scripts/audit_student_failure.py \
  --data out/student-bottle-can-v1 --cache out/student-bottle-can-v1-cache \
  --evaluation out/lecun-student-eval-v1/results \
  --run data/student-runs/bottle-can-bc-v1/run.json --out out/student-failure-audit-repeat
```
