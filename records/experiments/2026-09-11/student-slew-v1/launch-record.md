<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# G3 고정 명령 변화 제한 평가 — 실행 시작

사용자의 2026-09-11 “우리것도 바로 시작” 지시에 따라
`plans/STUDENT_NEXT_SINGLE_FACTOR.md`의 무학습 평가를 lecun에서 시작했다.

- tmux: `dexprompt-x0-slew-v1`
- 실행기: `scripts/run_student_slew_lecun.py` (원격 student-checks에 복사).
- 정책: 기존 x0 epoch 13, seed 1000/1001/1002, 각 최대 974 tick.
- 변경: 고정 관절별 slew limit 적용 여부 하나. 모델·가중치·데이터·품질 조건 불변.
- GPU: `GPU-5b42e5c8-12cf-4520-c591-a377d5e2f035`.
- 원격 상태/로그: `/data1/yspark/projects/dexprompt/student-checks/x0-slew-job-v1/`
- 원격 평가/trace: `/data1/yspark/projects/dexprompt/student-checks/x0-slew-v1/`
- 1시간 상한, 실패·시간초과 시 해당 subprocess 그룹만 종료. 재시도·재튜닝 없음.
- 평가기·한계·checkpoint·manifest SHA를 실행기가 검증한다. backward/optimizer 호출 차단.

해시 확인 후 실제 GPU 정책 실행 진입을 확인했다. **결과는 아직 회수하지 않았다.**
관련 limiter/추론 테스트 18개 통과, 실행기 Ruff 통과. 새 학습은 0회다.

## 결과 회수 시 지킬 것

1. 상태 JSON과 `evaluation.json`, 세 trace를 새 로컬 결과 폴더에 회수한다.
2. seed·checkpoint·한계·source 검사 결과를 확인한다. 각 최대 974 tick이며, 품질 위반으로
   그 전에 종료된 실행은 정상적인 실패 결과다. 짧다는 이유로 분모에서 빼지 않는다.
3. G1과 같은 팔/손 상태 오차·초기 64 tick·명령 경계 변화·접촉·리프트 지표를 비교한다.
4. 완주 여부가 개선되면 이 고정 후처리의 효과로 해석한다. 이것만으로 원래 실패의
   유일/필수/충분 원인이 속도였다고 단정하지 않는다. 제한은 실행 상태·후속 예측도 바꾼다.
5. 한계/시드/체크포인트를 변경하거나 추가 학습을 자동으로 시작하지 않는다.

같은 세 seed의 이전 평가 소요는 805.7초였으며 현재 서버에서의 완료 시간 보장은 아니다.
