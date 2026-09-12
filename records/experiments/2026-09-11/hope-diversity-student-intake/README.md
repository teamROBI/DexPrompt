<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 새 HOPE 후보의 student 데이터 검증 — 5/5 통과

**5시범/5,011프레임 모두 통과했다.** 새 teacher 후보 2개, 물체 2개, 물체–배치5개다.
새 학습은 하지 않았다. 원본/재현 파일 확인, 전체 시각 재생, cache·입력 audit,
recorded/stationary 물리 대조군, preflight를 마쳤다. 최종 종료코드0이고 자체 tmux는 모두 종료했다.

| 물체 | 배치 | 프레임 | 지속 접촉 획득 tick | 최대 상승 | 데이터 검사 |
| --- | --- | --- | --- | --- | --- |
| AlphabetSoup | centre | 979 | 629 | 144.26 mm | 통과 |
| AlphabetSoup | yaw_plus15 | 983 | 632 | 144.72 mm | 통과 |
| AlphabetSoup | x_minus20mm | 981 | 631 | 144.40 mm | 통과 |
| Raisins | centre | 1,046 | 666 | 128.61 mm | 통과 |
| Raisins | yaw_plus15 | 1,022 | 665 | 140.94 mm | 통과 |

모든 프레임에서 측정 관절 상태 불일치0, 물리 신호 최대 오차0. 기록 명령 대조군5/5가
teacher 접촉 준수 리프트를 완주했고 종료접촉5/5를 유지했다. stationary는5/5 실패했다.
이는 **시범/평가기 검증**이며 학습된 시각 정책의 성공률이 아니다.

native256×256, 머리카메라 하향20°의 기존 profile을 유지했다. profile 적용 후 head
목표 mask가 빈 프레임0/5,011, 양쪽 camera 모두 빈 프레임도0이었다. AlphabetSoup X−20은
기존 head pose에서266프레임이 비었지만 profile 적용 후0이었다. 새 camera 튜닝은 하지 않았다.
입력 audit는 각 시범의 home·참조 획득·대표 tick 등5지점 전처리를 비교했고 모든 home
reference가 사용 가능했다. 각 시범은 train-only 진단이며 독립 일반화 test는 없다.

## 보관과 다음 작업

- 원격 데이터/cache: `/data1/yspark/projects/dexprompt/student-checks/diversity-intake-egl-v1/`
  아래 개별 시범 디렉터리. **이미 lecun에 있으며 추가 전송이 필요 없다.**
- [확정 목록·증거](evidence/release.json), [원시 단계 기록](evidence/intake/results.json).
  첫 시범의 DINO 경로 복구는 확정 목록에서 별도로 표시했다.
- 로컬 회수 evidence.tar.gz SHA:
  `875bc9be05bf55cd414a3d0c61795be68ee8b8a555b48f69350274ea9c0f0b01` (양쪽 일치).
- 기존 고정7시범/7,466프레임과 분리 보관했다. 합치면 데이터 검사 통과 자료는
  **12시범/12,477프레임**이지만 통합 bundle이나 학습 입력은 변경하지 않았다.
- 이번5에 대한 새 DINO full-model forward는 하지 않았다. 기존7의 실제 DINO 검사와
  이번의 데이터/preflight 검사를 구분한다.
- AlphabetSoup의 얇은 품질 여유는 그대로이며 한계를 완화하지 않았다. 자세·마찰 변화에
  대한 강건성은 이 검사로 증명하지 않는다.
- 다음 우선순위는 [G3 결과](../student-slew-v1/README.md)에 따른 단일 병 정책의
  teacher 관측 기반 행동 생성 진단이다. 자료가 늘었다는 이유만으로 큰 학습을 시작하지 않는다.

## 실행 중 수정과 보존 기록

아래는 시작 시점부터의 기록이다. “진행 중”은 역사적 상태이며 최종 판정은 위의5/5다.

2026-09-11. 기존 고정 7시범을 보존하고, 별도 경로에서 새 후보의 재현된 5시범을 검사한다.

고정 입력: AlphabetSoup 후보1443cca6의 centre/yaw+15/X−20,
Raisins 후보3e8fa799의 centre/yaw+15. 후보438146b5의 centre는 재현 기록이 없어 제외했다.
HANDOFF/ledger는 각각 SHA 22e6c645…/c13269dc…이며, 5건의 원본·재현본에 대해
실제 episode/model/initial state SHA와 후보·배치 귀속을 확인했다.

## 실행

`scripts/intake_hope_diversity.py`: 원본/재현 검증 → acquisition guard → 전체 시각 재생 →
cache → 입력 정렬 audit → recorded/stationary 물리 대조군 → preflight.
이전 watcher와 달리 preflight의 종료 코드뿐 아니라 training_ready 값도 확인한다.
완료 전에는 student-ready로 선언하지 않는다. 학습 및 DINO 모델 forward는 하지 않는다.

로컬 tmux icra-diversity-intake-v1에서 첫 render를 시작했으나 load average 약194/CPU32,
nice15 프로세스가 2분여 동안 CPU 1초 미만을 받아 진행이 지연됐다. 자체 parent/child만
중단했다. 산출물과 interrupted.json은 /data1/yspark/sim/icra/hope-diversity-student-intake-v1에
보존한다. 과학적 실패/성공으로 집계하지 않는다.

lecun에는 같은 MuJoCo3.3.3/osmesa가 있어 격리된 코드와 자료를 전송했다. 원본 archive
SHA d5fb0a93c5c04145e023e874f970a733f082b7aafd6f3b1ceca49b584b1339d0,
코드 archive SHA c5a7aab53ab3e5972b9543983a23cb377283c25d948cc652442acf0586c11f0f.
양쪽에서 해시 일치 확인. manifest 내부 상대 경로는 그대로이며 ledger collection의
명시된 원본 prefix만 새 source 경로로 대응한다. 원본 파일 내용은 변경하지 않는다.

- tmux: dexprompt-diversity-intake-v1
- 실행기/로그: /data1/yspark/projects/dexprompt/student-checks/diversity-intake-job-v1
- 결과: /data1/yspark/projects/dexprompt/student-checks/diversity-intake-v1
- 3시간 상한. stage timeout 시 해당 process group 종료, 후속 단계 차단. 다른 시범은 별도 집계.
- 기존 fixed7, x0 checkpoint, Claude teacher 출력은 수정하지 않는다.

## 렌더링 실행 환경 변경

lecun osmesa에서도 약1프레임/초여서 완주 전에 해당 자체 작업만 중단했다. 원본 CPU
검사 산출물은 그대로 보존한다. GPU1(432d6dc7…)이 빈 것을 확인했고, 같은 MJB에서
20프레임 정적 렌더링 시험은0.805초였다. GPU0의 타 작업은 건드리지 않았다.

동일 frozen5 입력과 physics/카메라 profile을 유지하고 **EGL device1**을 명시한 새
실행으로 바꿨다. 픽셀값이 osmesa와 bit-identical이라고 주장하지 않는다. 새 데이터의
물리 재생, 입력 정렬, 목표 가시성, 대조군은 이 EGL 실행에서 모두 다시 검사한다.

- 현재 tmux: `dexprompt-diversity-intake-egl-v1`
- 현재 결과: `/data1/yspark/projects/dexprompt/student-checks/diversity-intake-egl-v1`
- code: 기존 job의 `code-egl/`; log/exit: `run-egl.log`, `exit-code-egl`.
- runner SHA: `926585937adf25458742aa658f0f52870ac6bbb87de9470e4474122ab7b771a3`.
- `--renderer` 기본값은osmesa, EGL은 명시적 device 없으면 오류. 선택은 plan.json에 기록.
- 관련 회귀 테스트12개 통과(이중으로 입증되지 않은 시범 제외, 잘못된 모델 재현 거부,
  전송 경로 대응, preflight false 거부, 조기 종료 trace 포함), 변경 파일 Ruff 통과.

## 첫 preflight의 실행 경로 누락과 복구

첫 AlphabetSoup centre의 render/audit/controls는 통과했으나 preflight는 exit2였다.
격리된 code-egl/out/student-assets 연결이 빠져 DINO source/head weights/wrist weights만
missing으로 나왔으며, 데이터/캐시 오류는 없었다. 기존 lecun의 검증된 student-assets에
심볼릭 링크를 추가하고 `preflight-assets-linked/`에 새 결과를 남겨 training_ready=true를
확인했다. 원본 preflight 실패와 원시 results.json은 덮어쓰지 않는다.

`scripts/report_diversity_student_intake.py`는 이 복구가 DINO 경로 누락에만 해당하고
cache SHA가 동일하며 optimizer0인지 검사한다. 데이터 오류나 다른 cache를 복구로
덮을 수 없다. 원시 결과의 실패와 복구 후 데이터 적격 여부를 따로 기록한다.
완료 후 release.json/evidence.tar.gz를 만드는 tmux `dexprompt-diversity-report-v1`을
연결했다. 결과는 job 디렉터리에 생긴다. 보고 프로그램은 학습을 실행하지 않는다.

기존 fixed7 목록 SHA f13daac296a4a851… 및 bundle manifest SHA d9e98792ad218562…는
작업 전과 같음을 다시 확인했다. 이번5의 성공이 기존 student 정책 성공을 뜻하지 않는다.

상태: 2/5 검사 통과(centre는 경로 복구 포함), 세 번째 물리 대조군 검사 중.
