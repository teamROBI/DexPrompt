# DemoFunGrasp 공개 구현 확인 — 당일 가능성 판단

2026-09-12. 사용자 요청에 따라 공식 코드를 clone하고 핵심 실행 경로만 확인했다.
설치·원 논문 재현·HX5 이식·새 학습·새 물리 실행은 하지 않았다.

- 공식 저장소: https://github.com/BeingBeyond/DemoFunGrasp
- 프로젝트: https://beingbeyond.github.io/DemoFunGrasp/
- 로컬: `third_party/demofungrasp`
- commit: `116b74d13ab59aadf7a2edde7f6ba80823b7ba73`
- 코드/설정/README를 받았다. 대용량 `assets/`, `checkpoint/`는 sparse checkout에서 제외했다.
  기존 Dexonomy/student 환경에는 패키지를 설치하지 않았다.

## 코드로 확인한 적용 지점

| 코드 | 확인한 동작 | 우리 적용 |
| --- | --- | --- |
| `tasks/grasp.py:generate_reaching_plan_idx` | 물체 초기 좌표에서 시범 손목의 이동·회전, 손 관절 목표의 배율·잔차를 편집한다. 리프트 변위는 원 시범에서 유지한다. | Dexonomy reference를 수정하는 작은 행동 공간의 참고. 정확한 프레임·단위를 먼저 맞춘다. |
| `algo/ppo_onestep/ppo.py:run` | 에피소드 시작에 정책 행동을 한 번 샘플한 뒤 reference controller로 전체 동작을 실행해 보상을 얻는다. | 매 tick 7관절 접근 PPO와 다른 문제. 긴 행동열을 처음부터 탐색하지 않는다. |
| `algo/ppo_onestep/storage.py:compute_returns` | return은 실행 보상, advantage는 reward−value다. | one-step RL의 직접 구현 근거. 기존 장기 PPO를 그대로 복제할 이유가 없다. |
| `tasks/train/PPOOneStep.yaml` | 물체 point cloud/PointNet과 정책·가치 모델, nsteps=1 설정. | privileged 자동 시범 생성 단계의 참고. 시각 VLA와 구분한다. |
| `README.md`, `scripts/train_*` | Isaac Gym Preview4/IsaacGymEnvs, Shadow/Inspire 손과 별도 자산 사용. | HX5/MuJoCo에 체크포인트를 바로 넣을 수 있다는 근거는 없다. |

유형(style)은 조건으로 입력된다. 자동 유형 선택까지 이미 해결한 모델이라고 해석하지 않는다.
내부 reference 추종/로봇 제어가 필요하므로 IK나 충돌 제약이 사라지는 방식도 아니다.
공개 구현의 고정 손 목표 추종을 우리 close/lift에 통째로 복사하지 않는다.
우리 고정 squeeze 대체 실험에서는 성공 대조군까지 깨졌으므로 기존 접촉 피드백을 유지한다.

## 오늘의 판단

문헌과 공개 구현은 **시범 편집 RL로 데이터를 만들고 시각 정책에 전이하는 방법의 근거**다.
HX5에서의 수율, Dexonomy 결합의 효과, 미지 물체 일반화와 논문 기여는 아직 별도 검증 대상이다.
오늘은 원 구현의 새 시뮬레이터 전체 설치보다 기존 실행기에 편집 인터페이스를 붙이는 쪽을 우선한다.
검증 절차는 [당일 작업서](../../../plans/TODAY_FEASIBILITY_2026-09-12.md)를 따른다.
