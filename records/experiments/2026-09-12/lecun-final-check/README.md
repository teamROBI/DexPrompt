# lecun 접속 복구와 기존 최종 결과 확인

2026-09-12. 사용자 SSH agent로 접속했다. 새 학습·평가·패키지 설치는 실행하지 않았다.

- 원격 `hostname`: lecun. 기본 tmux 소켓에는 실행 세션이 없다. 다른 프로세스 전체의 종료를 확인한 것은 아니다.
- `student-checks/single-bottle-x0-low-noise-job-v1/status.json`: completed, running=false.
- 사전 고정 최종 epoch13: seed1000–1002, 접촉 준수 완주 **0/3**, 지속 리프트 **0/3**. 모두974tick, 접촉 획득 없음. `failure_reason=null`은 성공을 의미하지 않는다.
- `student-checks/low-noise-comparison-v1/README.md`: A/B 모두 epoch1/3/6/9/13에서0/3. 낮은 잡음 변경이 폐루프 파지 성공으로 이어지지 않았다.
- 기록 명령 대조군은 접촉 준수 리프트에 성공했다. 따라서 현 시각 student의 알려진 과제 전이가 아직 확인되지 않은 상태다. 전체 연구나 DemoFunGrasp 편집 방법의 실패로 확대하지 않는다.
- `student-checks/hope-twelve-job-v1/checks/release.json`: 통합 완료, 12시범/12477프레임/3물체/5후보/9물체·배치 조건. full-model batch2, optimizer step0. 데이터 준비이며 다물체 학습 성공이 아니다.

원격 기준 경로: `/data1/yspark/projects/dexprompt/`.
최종 평가 발췌와 원본 해시는 `final-evaluation.json`에 있다.
다음은 최종 정책의 초기 행동과 실제 적용 명령을 짧게 비교하는 것. 평가 전체나 학습을 반복하지 않는다.
