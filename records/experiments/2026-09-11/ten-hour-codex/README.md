<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Codex 10시간 실행 기록

사용자: "우리도 그럼 10시간동안 계속 작업을 진행하자. 나한테 별도 승인받지 않고 계속 작업을 진행하면돼."
작업창: 2026-09-11 **02:41:16–12:41:16 KST** (종료 Unix1789098076).
상위 계획: `plans/TEN_HOUR_SPLIT_2026-09-11.md`. 이 기록은 실행 중이며 최종 결과가 아니다.

## 최신 상태 — 04:50

x0 학습은 **6,331회 완료**했다. epoch 1/3/6 각 3회에서 완주는 없고 epoch 9를
평가 중이다. 최종 epoch 13이 주 지표이며, 중간 모델을 사후 선택하지 않는다.

새 HOPE teacher 성공은 AlphabetSoup와 Peaches 중앙 각 1건이다. 둘 다 독립적인
controller 재실행의 episode 해시가 일치한다. 데이터 인수는 서로 다르다.

| 물체 | 시각/기억 검사 | 기존 student 기록 명령 대조군 | 데이터 인수 |
| --- | --- | --- | --- |
| AlphabetSoup | 1,192 tick 변환, 물리 차이 0 | 획득 확정 전 닫기 이동으로 실패 | 보류 |
| Peaches | 1,006 tick 변환, 목표 참조 0/443 | 14.81 cm, 종료 5/5, 연속 2.78초 접촉 준수 상승 | 통과 |

Peaches의 최대 침투는 1.961 mm, 최대 손가락 힘은 28.89 N이다. 기존 한계 안이지만
여유가 작으며 같은 조건 재현이 물리 강건성이나 실물 성공을 보장하지 않는다.
정지 대조군은 높이 0, 접촉 0으로 실패한다. 데이터와 재현 귀속 보고서는
`out/hope-student-readiness-0421.json`에 있다.

Peaches의 14파일/235,326,449바이트를 lecun으로 복사해 전 파일 SHA를 대조했다.
`dexprompt-hope-peaches-check`는 CPU 검사 후 기존 x0와 후속 진단이 종료될 때까지
GPU를 기다린다. 실제 DINO를 포함한 forward 검사만 수행하며 새 학습은 하지 않는다.

추가 배치 인수 감시도 tmux에서 시작했다. Claude의 기존 네 배치 실행만 기다리며
teacher 생성이나 동적 시도를 추가하지 않는다. 동일 후보 재현을 확인한 뒤 현재
student 계약에 맞지 않는 archive를 먼저 거절하고, 나머지는 순차 CPU로 전체 변환한다.
이 사전 거절은 물리 재생·양성 대조군을 대체하지 않는다. 관련 검사 17개와 Ruff 통과.
출력은 `out/hope-placement-intake-v1`이며 data1을 사용한다.

아래는 작업 중 남긴 시간순 기록이다. 과거의 진행 중/ready 0 표시는 해당 시점 값이다.

## 운영 중인 작업

- lecun `dexprompt-x0-bc`: 기존 x0 학습과 사전 지정5checkpoint 평가. 모델/학습 설정 그대로.
- lecun `dexprompt-x0-follow`: 위 작업 완료 후 epoch13 최종3seed 판정을 읽는다.
  완주≥1이면 x0/epsilon 최종을 seed3000–3009 각10회 비교한다. 완주0이면 x0 대응
  저장 관측 진단 및 실제 평가 명령 재생 감사를 실행한다. 새 학습은 시작하지 않는다.
  결과: `/home/yspark/projects/DexPrompt/out/ten-hour-student-followup-v1`.
- 로컬 `dexprompt-visual-intake`: HOPE execute exit0·timeout false와 완료 manifest가
  둘 다 생긴 물체만 인수한다. 물체별 첫 accepted시범1개(최대3)를 visual-v2로
  재렌더→sealed cache→causal input 검사→기록 명령/정지 대조군→diagnostic preflight.
  출력: `out/hope-student-intake-v1` → `/data1/yspark/sim/icra/hope-student-intake-v1`.
  이는 Claude의 동일 조건 teacher controller 재실행 확인을 대신하지 않는다.
  그 재현 확인 전에는 최종 데이터 채택 여부를 별도로 남긴다. 새 DINO 학습 없음.

세 작업 모두 tmux이며 네트워크 유실에 독립적이다. 후속/인수는 위 마감 시각을 사용한다.
실행 중 객체/관측 코드와 과거 checkpoint는 수정하지 않았다. teacher 파일/시범도 변경하지 않았다.

## 추가 구현 및 검사

- `scripts/diagnose_single_student.py`: checkpoint의 prediction_type에 따라 clean/epsilon을
  복원한다. x0를 epsilon으로 해석하지 않도록 하고 두 좌표의 MSE를 구분한다.
  정확한clean/noise 합성 사례6개 및 알 수 없는모드 거부 검사 통과.
- `scripts/follow_student_comparison.py`: 최종 checkpoint/6331 updates/정확한seed목록/
  완료상태/품질판정을 검증한 뒤 분기한다. 예외·누락·중복seed·source변경을0%로 처리하지 않는다.
- `scripts/audit_visual_intake.py`: 데이터 품질 재검사, cache전체해시검증, 전체tick을
  순서대로 진행한 goal memory와 오프라인reference획득tick 비교. 실제 전처리는
  첫참조/200/450/마지막 및 추가참조 획득tick에서 비교한다. 전체tick의518px텐서를
  모두검사했다고 주장하지 않는다. action chunk/padding 정렬도 표본에서확인한다.
- `scripts/watch_visual_intake.py`: 정련/선별 출력만으로 데이터 인수를 시작하지 않는다.
  실패/timeout/부분journal을 완료로 읽지 않는 검사 통과.

새 회귀검사21개(진단7+후속6+입력3+인수감시5) 통과.
관련 기존student준비/목표기억을 포함한 검사43개도 통과했다. 전체repo suite가 아니다.
변경 파일 Ruff 통과. 실제x0 학습 파일/모델계약은 이 도구 추가와 분리되어 있다.

## 기존 데이터로 검증한 인수 결과

- 캔1251tick: accepted1, teacher1, 고유정책초기입력1. headreference0,
  wristreference395. 온라인/오프라인reference 일치, 표본0/200/395/450/1250 전처리일치.
- 병·캔혼합4시범: teacher파일4개, **고유정책초기입력2개**.
  병3개는 초기입력이 같지만 첫action chunk가 다르다. 이는 입력/라벨분포의 관측이며
  diffusion이 다양한 경로를 학습할 수 없다는 뜻이나 실패의 인과증거가 아니다.
- 검사는 신규DINO 실행이나 신규물리시뮬레이션을 하지 않았다. 저장data의 품질 증거와
  인과적전처리검사이며 과거 full replay검증과 구별한다.

## 디스크 공간 조치

로컬root잔여1.8GB를발견했다. 완료된우리cache3개를`/data1/yspark/sim/icra/student-caches`
로복사해91개파일의SHA256을원본과대조했다. 약5.47GB의동일데이터를확인후기존경로를
symlink로유지하고기존물리복사본을제거했다. root여유약6.8GB확보. 타인의파일삭제없음.
새시각data는처음부터/data1사용. 이전manifest/시범/모델내용은변경없음.

## 대기 및 다음 확인

x0의실제전체평가와HOPE첫동적완주를대기중이다. 파지성공이나새시범확보로집계하지않는다.
후속결과가생기면추가진단결과및물체별인수기록을이문서에갱신한다.

## 추가: 평가 분할의 근접 형상 감사

공식HOPE28개메시를고정샘플로검사해21개근접쌍을기록했다.
Peaches/Corn 등6개, AlphabetSoup/TomatoSauce 등은이름이다르더라도형상독립test로
바로세지않도록묶어서검토한다. 이미개발에사용한7개와근접그룹을합친14개를개발용
제외목록으로보존했다. 최종split을만들거나나머지14개의형상독립성을보장하지않는다.
`../hope-geometry-groups/README.md` 참조. 새로운action시범/학습은없다.

## 03:28 — HOPE 입력 오류와 인수 감시 보완

AlphabetSoup 최초 12개는 모두 `teacher source scene identifies a different target object`
검사에서 거절됐으며 episode_id가 없었다. **물리 실행 0회**이므로 파지 실패 0/12로
해석하지 않는다. DGN의 `<object>/floating/scaleNNN.npy`만 가정한 경로 검사가
HOPE의 `<object>/scene_cfg.npy`를 거부했다. 확인 중 Claude가 이미 동일 오류와 wrapper의
계수/재개 처리를 수정했다. Codex는 공유 collector를 중복 수정하지 않았으며, 수정본으로
보존된 AlphabetSoup 후보 12개 전부의 `checked_candidate` 통과를 직접 확인했다.

우리 감시기는 최신 execute 기록과 실제 episode_id 존재를 함께 확인하도록 보완했다.
이전 실패 결과가 수정된 manifest는 다시 인수하고, 이미 인수 통과한 물체는 중복 처리하지
않는다. v1의 조기 `no_accepted_episode` 결과는 입력 오류 자료로만 보존한다.
현재 tmux `dexprompt-visual-intake-v2`, 출력 `out/hope-student-intake-v2`를 사용한다.
v1은 STOP으로 정상 종료했다(코드 0). 물리 실행이나 teacher 생성은 중복하지 않았다.

보완 테스트를 삽입하다 테스트 함수의 일부가 잘못 배치돼, 테스트가 감시기 소스 파일에
JSON을 쓰는 Codex 실수가 있었다. 실행 중 프로세스는 이미 읽은 코드를 사용하고 있어
그 실행에는 영향이 없었다. 테스트 배치를 고치고 감시기를 복구한 뒤 새 프로세스를 띄웠다.
인수 감시 7개 + teacher scene identity/collection 관련 검사를 합쳐 **36개 통과**,
변경한 두 파일 Ruff 통과. 이 오류를 성공적인 테스트 실행으로 숨기지 않는다.

lecun x0는 3,930/6,331 updates, 학습 추정 잔여 약45분. 폐루프 성공 결과는 아직 없다.

## 데이터 채택의 마지막 확인

`scripts/report_hope_student_readiness.py`는 인수 통과만으로 ready를 만들지 않는다.
원본 teacher manifest와 episode 해시, 동일 candidate의 controller 재실행 해시·성공 판정,
시각 데이터 candidate 귀속, render/audit/controls/preflight 결과를 함께 대조한다.
재현 기록이 없으면 대기, 다른 후보나 다른 해시이면 오류다. 관련 회귀 검사6개 통과.
첫 실행 `out/hope-student-readiness-initial.json`에서는 새 ready 시범0개이며, 이는 아직
시각 인수를 기다리는 상태다. 이 도구의 ready는 진단용 데이터 준비이며 독립test나
새 DINO 학습 성공을 의미하지 않는다. 실제 DINO 무학습 실행은 lecun 후속 확인으로 분리한다.

## 03:58 — 첫 HOPE 성공과 인수 보류

AlphabetSoup 중앙 실제12회 중1회가 완주했고 동일 후보 재실행 episode 해시가 일치했다.
[첫 시범 교차 확인](../hope-first-demonstration/README.md) 및 신호/영상 표본을 남겼다.
시각 변환1192tick 물리차이0, 목표 기억 ref0/387 및 입력 검사 통과다.
그러나 student controls가 tick621(close)의 XY이동10.00194mm에서 중단됐다.
[계약 감사](../student-acquisition-contract/README.md): 기존 병3·캔1은위반없음,
AlphabetSoup만0.5초 전체접촉 확정(tick712) 전에 닫기 중 변위제한을 넘는다.
Teacher는close에서그변위를허용하지만student guard_v2는확정전모두접근으로본다.
물리재현오류/시각오류가아니라계약차이이며, 새teacher1·새student-ready0으로분리한다.
현재x0실험과품질/제어/평가기준은바꾸지않았다. 새데이터학습없음.
