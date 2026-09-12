<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 2026-09-10

- [single-bottle-student](single-bottle-student/README.md): lecun 단일 시범 6,331회 갱신 완료. 사전 지정 최종 epoch 13 완주 0/3, 중간 4개 checkpoint도 각각 0/3. 09-11 결과 확인.
- [hope-data-readiness](hope-data-readiness/README.md): 공식 HOPE 28개 메시 확보·치수 점검. full은 mm, eval은 cm로 배포본 단위 차이 확인. 기존 3개 자산 변환은 맞지만 렌더 material은 없음. 6물체·다배치 수집 계획과 준비 조건 기록.

| 실험 | 무엇을 확인했나 |
| --- | --- |
| [table-refinement](table-refinement/README.md) | 정련(ADMM)에 테이블을 넣는 소규모 A/B. 정적 검사의 테이블 관통 기각 4→0, 실행 도달 후보 3→7, 완주는 1→1. 실패 지점이 접촉 획득에서 유지로 이동. 기존 캔 성공은 바이트 단위로 보존 |
| [overnight-teacher-expansion](overnight-teacher-expansion/README.md) | 새 물체 3개(머그·그릇·망치)에서 고유 후보 32개 실행, 완주 0. 병목을 단계별로 규명: 정렬이 테이블 여유를 깨뜨리는 것은 매처 반복 수를 줄이면 되돌릴 수 있으나, 그릇·망치는 정련이 전부 **요구 접촉 미달성**으로 거부한다. 캔은 yaw 전달 검증 뒤 0°와 +15°에서 완주(재현 해시 동일) |
| [contact-diagnosis](contact-diagnosis/README.md) | 요구 접촉 미달성의 원인 분리. 매핑·좌표·접촉 판정·scale에 오류 없음(양성·음성 동일 통과). 종료 시점에 침투 0·관절 여유 0.13–0.25 rad. 반복 수만 50→400으로 바꾸자 손가락 2가 6건 중 5건에서 회복(9.7–13.5 mm → 0.78–0.93 mm) — 최적화 예산 문제의 직접 증거. 손가락 5는 미해결 |
| [init-sensitivity](init-sensitivity/README.md) | 템플릿·접촉 목표를 고정하고 **초기 손 자세만** 바꾼 비교(후보 8 × 초기화 5 = 40). 음성 **0/30**, 양성 **9/10**. 음성은 전부 contact에서 탈락하고 QP는 실행되지 않음. 미충족에 손가락 5가 30/30 포함. 같은 변형 폭이 양성에서는 결과를 바꾸므로 (병 tilt_plus가 collision 탈락) 변형이 무의미하게 작지는 않다. 정련 통과 0 → 동적 실행 없음 |
| [finger5-flexion](finger5-flexion/README.md) | 손가락 5 PIP 초기 굴곡만 바꾼 비교(후보 8 × 초기값 5, 미실행 2). 원본 복사 기준선은 이전 반복 400과 완전 일치. 음성 정련 **0/28**, 손가락 5는 한 번도 성립하지 않음(최소 2.187 mm). 캔 +0.30이 5/5 충족 후 QP 탈락. 여섯 후보 미세조정 보류, 다음 축은 손가락 5를 요구하지 않는 유형을 제안 |
| [penetration-and-scale](penetration-and-scale/README.md) | 침투 측정과 정정: 실행 중 침투는 0.81–2.00 mm(상한 2 mm)이며 1.5 mm는 압력 완화 기준. 논문 그림의 망치 '관통'은 후보 `03fa5b0e` 종료 자세의 가림으로, 실측 +0.826 mm 여유. 그릇 0.08/0.05 = 정확히 1.6. scale×matcher 2×2에서 0.08×matcher4만 정련·실행에 도달(10회) 완주 0 |
| [bowl-drop](bowl-drop/README.md) | 그릇을 들어 올리다 놓치는 원인 진단(제어 수정·학습 없음). 세 건 모두 archived 재생이 오차 0.0으로 일치. 두 그릇은 14.0/10.5 cm까지 올린 뒤 **hold에서** 접촉을 잃고 회복 못 함(지속적 상실 1회씩), 캔은 끊김 70회에도 0회. 그릇은 손가락 2·4가 0.3–0.7 N·0.1–0.2 mm로만 접촉하고, 닫기는 **엄지가 목표를 지나치고 중지·소지가 못 미친 채** 끝남(관절 여유 0.89–1.40 rad 남음, 캔은 최대 0.28 rad). 힘은 상실 **이전에 감소**하고 명령은 더 감기고 있었음. 회전을 행렬로 재계산(9.4/15.4/11.5°)했고 인과 서술은 정정. squeeze 비교 프로토콜은 설계만 |
| [can-visual-student](can-visual-student/README.md) | 채택된 캔 시범 한 건을 학습 없이 시각 student 경로에 연결하고 물리 재현·DINO 검사까지 확인 |
| [student-training-ready](student-training-ready/README.md) | 목표 참조 품질, 병·캔 4개 시범의 진단 묶음, 폐루프 평가 연결. 리프트 중 밀림 오판 수정, 기록 명령 4/4·제자리 0/4. 학습 없음 |
| [teacher-expansion-crosscheck](teacher-expansion-crosscheck/README.md) | 원시 53개 시도 대조. 머그 전 접촉 12/32, 종료 접촉 오분류 1건 수정. 캔 +15° 해시 확인, 불가능성 단정 정정. 렌더 테스트 순서 의존성 수정, OSMesa 전체 421개 통과 |
| [visual-student-bc-v1](visual-student-bc-v1/README.md) | 사용자 승인 후 병·캔 시각 BC 시작. GPU 0 한 장, 스텝 휴지·메모리 제한, tmux 실행. 100 updates 유한 손실 확인, 폐루프 성능 미평가 |
| [init-sensitivity-crosscheck](init-sensitivity-crosscheck/README.md) | 실제 저장 목표 40개 재검사 통과. 검사 코드 오류 수정, 현재 original 기준 접촉 상실 3→5. 주 결과 0/30·9/10 유지. 새 정련·student 실행 없음 |

- [server-migration](server-migration/README.md): 새 서버용 코드·6.02GB 자산 묶음, 경로 분리 모델 로드와 물리 대조군 4/4 재현.

- [lecun-student-eval](lecun-student-eval/README.md): 새 3090 환경 검증 통과, 시각 정책 12회 기록에서 완주 0. 다음은 초기 행동·중복 조건 감사.

- [student-failure-audit](student-failure-audit/README.md): 무학습 계약·명령·물리 재생 감사 완료. 초기 명령의 큰 오차와 병 평가 중복 확인. 16→50회 추론에도 오차 잔존, 관련 테스트 52개 통과.
