<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 2026-09-11

| 작업 | 확인한 것 |
| --- | --- |
| [milk-state-crosscheck](milk-state-crosscheck/README.md) | Claude 보관 초기상태 진단 완료 확인. 원시 속도 최대/RMS 일치·원본 해시 보존. LD는 접근 경로 실패이며 파지 유지 실패 대조군이 아님. 추가 실행 없음 |
| [hope-twelve-student](hope-twelve-student/README.md) | 승인된 기존7+새5를 별도 통합·검증하고 새후보2개 DINO forward 검사 시작. low-noise A/B 자동 보고 감시도 별도 tmux 시작. 새 학습 없음 |
| [milk-handoff-crosscheck](milk-handoff-crosscheck/README.md) | Milk B 새후보1/8 성공, 원본·재현3종 파일해시 일치 직접 확인. 기존A 데이터와 분리, student 인수 미실시 |
| [student-low-noise-training](student-low-noise-training/README.md) | 승인 후 lecun GPU1에서 낮은잡음 손실가중치 학습 시작. 6,331updates·고정epoch평가 자동 연결, 결과 대기 |
| [x0-loss-audit](x0-loss-audit/README.md) | raw72예측 손실 분해. 팔 기여75–79%, 선택적 낮은잡음 가중치 준비. 관련76테스트·lecun 실제DINO dry-run 통과, 학습 미실행 |
| [x0-denoising-trace](x0-denoising-trace/README.md) | x0 직접 예측72개·생성 경로24개, 기존 출력 동일. 마지막 호출에서 팔8행동 오차24/24 증가. 낮은 잡음 추정도 불완전. 새 학습 없음 |
| [student-sampler-steps](student-sampler-steps/README.md) | 고정24쌍 추론16→50 비교. 첫8팔행동 오차24/24 악화, 기존16 유지. 새 학습·물리 실행 없음 |
| [teacher-observation-diagnosis](teacher-observation-diagnosis/README.md) | 기록된 teacher 관측8시점×3seed 진단 완료. 입력 전처리8/8 일치, 기존12개 오차 재현, 초기 팔 명령 오차 지속. 모델 변경·학습·물리 실행 없음 |
| [hope-diversity-crosscheck](hope-diversity-crosscheck/README.md) | Claude 새 후보 승인6건/후보3개·재현5건 확인. 실제 중앙 실패 phase close8/lift1. 저장신호 student guard6/6 위반 없음. 이후 재현5건 시각 인수 완료 |
| [student-slew-v1](student-slew-v1/README.md) | 고정 명령 변화 제한 평가 완료0/3. 초기 팔 오차 감소, 지속 접촉획득 없음. seed1001의 침투 조기 종료도 실패에 포함 |
| [hope-diversity-student-intake](hope-diversity-student-intake/README.md) | 새5시범/5,011프레임 검사 완료. 물리 오차0, recorded5/5·stationary0/5, preflight 통과. 기존7과 분리 보관, 새 학습 없음 |
| [hope-pilot](hope-pilot/README.md) | HOPE 6개 자산 생성·텍스처 연결·30개 초기 배치 검증. 세 물체 15배치가 초기 장면 검사 통과. 파지 시범 생성 전 |
| [single-student-diagnosis](single-student-diagnosis/README.md) | 고정된 단일 시범 최종 정책의 관측 4개를 무학습 진단. 작은 잡음 제거는 되지만 순수 잡음에서의 행동 생성 실패가 남음 |
| [squeeze-branch](squeeze-branch/README.md) | 동일한 lift 시작 상태에서 접촉 피드백 닫기를 고정 teacher squeeze 목표 추종으로 바꾼 6회 비교. A는 원본을 오차 0.0으로 재현하고 두 조건의 팔 명령 해시가 같다. B는 세 건 모두 목표에 도달하지만 도달 시점에 필요한 접촉이 없다(그릇 최대 상승 139.6→20.9 mm, 105.0→50.8 mm, 양성 대조군 캔 성공도 깨짐). 이 전환 조건의 실패까지만 결론. HOPE 수집에 적용할 변경 없음, 그릇 튜닝 종료 |
| [single-student-x0](single-student-x0/README.md) | x0 예측 방식 구현 후 lecun에서 6,331회 갱신 완료. 사전 고정 epoch 1/3/6/9/13 모두 폐루프 완주 0/3 |
| [ten-hour-codex](ten-hour-codex/README.md) | x0 후속 평가·명령 감사와 HOPE 시범 인수 자동화 완료. 이후 고정 7시범 인계로 연결 |
| [hope-centre](hope-centre/README.md) | HOPE 3물체 teacher 전체 실행 완료. **중앙 승인 3/36**(물체당 1/12: AlphabetSoup·Peaches Power Disk, Raisins Large Diameter), **배치 승인 5/12**(AlphabetSoup 3/4, Peaches 1/4, Raisins 1/4). 승인 8건 전부 동일 조건 재현 episode sha256 동일, 배치 상대좌표 15/15 확인. 고유 teacher 후보 3. Peaches·Raisins 승인은 품질 여유가 얇음. 디스크 가득 참(05:15)으로 Raisins 첫 수집 중단→격리 후 재실행. Peaches Large Diameter는 정렬 자세 전부 테이블 겹침으로 정련 대상 0. 실행 계약 오류(평면 자산 teacher 정체 검사) 수정, 예측 111→140·그릇 정련 10개 정정 |
| [hope-geometry-groups](hope-geometry-groups/README.md) | HOPE 28개 시각 메시에서 근접 형상 21쌍 선별. 상품명과 형상 다양성을 분리하기 위한 사전 감사 |
| [hope-student-intake](hope-student-intake/README.md) | teacher 성공과 student 획득 계약을 함께 통과한 7시범/7,466프레임 확정. AlphabetSoup 중앙만 계약 차이로 제외 |
| [student-next-diagnosis](student-next-diagnosis/README.md) | x0 최초 폐루프 이탈을 동일 최초 입력의 첫 팔 명령으로 국소화. 승인 HOPE 7시범/7,466프레임 통합·lecun 전송, 7+7 물리 대조군과 물체별 실제 DINO 3배치 통과. 후속 고정 slew-limit 평가 결과는 student-slew-v1에 별도 기록. 새 학습 없음 |

이전 통합: [Teacher 8시범·student 입력 7시범, 최종 x0 0/3](consolidated-status/README.md),
[x0 최초 이탈·7시범 lecun 연결·다음 단일 요인 실험](student-next-diagnosis/README.md).

최신 추가: [G3 완료0/3](student-slew-v1/README.md), [새5시범 검사 완료](hope-diversity-student-intake/README.md).
기존7과 분리 보관한 검사 통과 자료는 합계12시범/12,477프레임이며 새 학습은 없다.
