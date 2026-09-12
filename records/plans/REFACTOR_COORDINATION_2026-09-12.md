# 코드 구조 정리 사전 공지

2026-09-12. 사용자 요청: Claude 작업을 방해하지 않도록 코드 구조 변경 전에 알릴 것.
최초 공지와 후속 확인을 구분해 아래에 기록한다.

## 후속 확인과 실제 변경

사용자가 전달한 Claude 응답으로 공지 확인 및 편집 중인 파일이 없음을 확인했다.
target-switch의 실제 import/설정 목록과 Codex 인수의 의존성을 합쳐 고정했다.
비활성 scripts 117개, tests 86개와 과거 문서는 기능별/기록 폴더로 이동했다.
최상위 스크립트30개, src/icra 및 관련 테스트·설정70개는 내용과 경로를 유지했다.
구조와 명령 호환성은 [현재 구조](../../docs/layout.md), 해시 고정 목록은
`records/maintenance/2026-09-12-layout/protected.json`을 따른다.
아래는 이동 전 사전 공지의 기록이다.

## 현재까지 한 일

- 구조와 import 의존성을 조사했다. src/icra 29개, scripts 최상위 Python 147개,
  tests 최상위 Python 90개를 확인했다.
- 코드 파일 이동·패키지명 변경·import 경로 변경은 아직 하지 않았다.
- 앞선 공간 정리는 별도다. 완료된 out/teacher-expansion, out/offset-compare-93,
  out/candidate_screen_all은 /data1에 보관하고 기존 경로를 심볼릭 링크로 유지했다.
  908개 파일의 해시 대조를 통과했으며 실험 내용은 유지됐다.

## 현재 실행과 충돌하지 않는 범위

- 지금은 파일 분류·의존성 조사·이전 경로와 새 경로의 대응표를 준비한다.
- 실행 중인 teacher/student 작업이 참조하는 스크립트와 공용 src/icra 모듈은
  이번 구조 정리에서 이동하거나 이름을 바꾸지 않는다.
- 특히 collect_episodes.py, collect_teacher_dataset.py, replay_hx5_student.py,
  execution.py, scene.py, recording.py 및 진행 중 run_target_switch.py 경로를 보존한다.
- 기존 CLI 명령과 import를 바꾸는 실제 이전은 관련 실행이 종료된 경계에서 한다.
  이전 경로 호환 진입점, import 검사, CLI 검사와 관련 회귀 검사를 함께 준비한다.
- 과거 reports/manifests에 기록된 경로·소스 해시는 새로운 코드 값으로 덮어쓰지 않는다.
- src/icra 이름을 dexprompt로 바꾸는 작업은 이번 범위에 포함하지 않는다.

## Claude에 전달할 내용

Codex가 프로젝트 폴더 구조 정리를 준비하고 있습니다. 현재 코드 이동이나 import 변경은
없습니다. 현재 다물체 teacher 작업은 기존 경로로 계속 진행해 주세요.
이번 실행에서 편집 중인 파일, 실행기가 나중에 불러올 스크립트, 공용 코드 수정 필요 여부를
보고에 적어 주세요. 구조 정리는 Codex가 담당하므로 동시에 폴더나 모듈 이름을 바꾸지 말아 주세요.
공용 코드 변경이 필요하면 대상과 이유를 먼저 공유해 충돌을 피하겠습니다.
이 공지를 확인하느라 진행 중인 실험을 중단하거나 다시 실행할 필요는 없습니다.

참고: scripts 사이 직접 import가 135개이고, 111개 스크립트가 부모 디렉터리 깊이로
루트 경로를 계산한다. 단순 파일 이동은 실행을 깨뜨릴 수 있으므로 전면 이동부터 하지 않는다.
