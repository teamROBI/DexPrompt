# 작업 공간 정리

2026-09-12. 사용자 요청에 따라 재생성 가능한 캐시를 정리하고 완료된 과거 실험을
대용량 디스크에 보관한다. 전체 Git 변경을 초기화하거나 미추적 소스를 삭제하지 않는다.

## 삭제·보관

- 프로젝트의 pytest/Ruff 캐시, scripts/src/tests의 Python bytecode 캐시:
  269파일, 2,456,532 bytes 삭제. 필요하면 자동 생성된다.
- 루트 `debug.xml`, `MUJOCO_LOG.TXT`: `root-debug/`로 보관.
  자동 생성 파일이 루트 미추적 목록을 어지럽히지 않도록 .gitignore에 추가했다.
- yspark의 uv 캐시: `uv cache prune --offline`으로 미참조 항목만 정리했다.
  결과는 21파일/5.5KiB. 약 36GB의 캐시 전체를 삭제한 것은 아니다.
  설치 환경과 캐시를 잇는 심볼릭 링크는 검사한 프로젝트 가상환경 3개에서 없었다.

## 과거 실험 저장 위치 정리

| 기존 경로 (계속 사용 가능) | 보관 위치 |
| --- | --- |
| `out/teacher-expansion` | `/data1/yspark/sim/icra/archive-outputs-20260912/teacher-expansion` |
| `out/offset-compare-93` | `/data1/yspark/sim/icra/archive-outputs-20260912/offset-compare-93` |
| `out/candidate_screen_all` | `/data1/yspark/sim/icra/archive-outputs-20260912/candidate_screen_all` |

위 세 폴더는 9월 8~10일 종료된 실험이다. 실행 중 student 인수의 입력/출력과 분리했으며,
Git 추적 파일·상대 심볼릭 링크·폴더 밖으로 공유된 hard link가 없음을 확인했다.
폴더를 rsync -aH로 복사한 뒤 원본·복사본 전 파일의 SHA256, 크기, mtime, mode 및
심볼릭 링크 대상을 대조한다. 원본이 복사 중 변하지 않았는지도 다시 확인한다.
검증 성공 후 기존 경로를 보관 위치에 대한 심볼릭 링크로 바꾸고 중복 사본을 제거한다.
실험 내용은 삭제하지 않으며 기존 경로로 계속 읽을 수 있다.

완료 여부·실제 공간은 `relocation.json`을 기준으로 한다. `completed=false`이면 진행 중이다.
파일별 검증 원장은 `*-files.json`, 복사 로그는 `*-copy.log`다.
대용량 디스크로 옮기는 작업이므로 전체 저장 용량 절감과 루트 디스크 공간 확보를 구분한다.

## 보존

teacher/student 원본과 체크포인트, HANDOFF, 논문·참고 논문, 코드, 과거 실패 보고 및
진단 로그, 가상환경, SSH 설정, lecun 및 다른 사용자 자료는 삭제하지 않았다.
진행 중 `codex-followup-intake`는 중단하지 않았다.

## 완료 결과

- 세 폴더 이동 완료. 정규 파일 908개 전부의 SHA·메타데이터 대조 통과.
- 기존 세 경로가 보관 위치로 정상 연결되는지 확인했다.
- 루트 가용 공간: 16,504,958,976 → 43,003,310,080 bytes (약 24.68 GiB 확보).
  `df -h` 표기로 약 16G → 41G, 프로젝트의 로컬 점유는 64G → 39G다.
- 저장 위치를 옮긴 실험 원본은 /data1에 그대로 있으므로 전체 서버에서 25GB를 삭제했다는 뜻은 아니다.
- 완료 확인 시 student 인수는 다음 Peaches 시범을 재렌더 중이었고 계속 실행됐다.

## 되돌리기

해당 경로를 사용하는 작업이 없는 상태에서 충분한 루트 공간을 확보한 뒤, 보관 폴더를
새 로컬 폴더로 rsync -aH 복사·검증하고 심볼릭 링크를 그 로컬 폴더로 교체하면 된다.
보관 원본을 먼저 삭제하지 않는다.
