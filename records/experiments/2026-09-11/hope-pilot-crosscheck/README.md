<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# HOPE 중앙 파일럿 중간 교차 확인

2026-09-11. 실행 중인 Claude wrapper/teacher 파일은 수정하지 않고 읽기 확인했다.
새 실험이나 전체 테스트를 수행하지 않았다. 현재 성공한 단계 결과를 폐기할 근거는 없다.

## 확인된 결과와 정정

- AlphabetSoup / Power Disk / align seed0: 102 filtered outputs → 100 init outputs →
  팔 재도달 69 → 정련 파일 54 → 정적 선별 후보39. init 출력 수는 실제 init filter 통과
  수와 다르다. 정련54와 후보39는 실제 파일/manifest 수와 일치.
- placement/observe/predict/filter/align/refine/select의 해당 기록 모두 exit0·timeout false.
  refine645.3초, select11.6초. 원시 단계 기록은 stages-snapshot.json.
- 보고된 예측111은 현재 원시 기록과 불일치: stages.json의 predict/filter 및
  predicted/10_Power_Disk/AlphabetSoup.yml의 grasps mapping은 **140**.
  현재 예측 파일 SHA256 `16ff361e2a4208b66ccb557f0b0e4b0d6e44581ed3cee665c7714779a937248e`.
  111이 별도 실행을 가리키는지 보고자가 분리/정정해야 한다. 임의 재실행은 하지 않는다.
- "그릇·망치는 모든 설정 정련0"은 역사 기록과 불일치. 큰 그릇 scale0.08/matcher4에서
  정련10·실행10·완주0이었다. HOPE54는 새로운 정적 후보 확보이며 동적 우위를 뜻하지 않는다.
- align_grasps.scene_config가 asset/scene_cfg.npy를 먼저 선택한다는 설명은 코드와 맞다.
  이 경우 --scene-cfg 생략과 실제 table entry 기록은 합리적이다.
- squeeze-branch의 일일 index 행은 현재 존재하며 유지한다.

## 후속 실행 전 보완할 wrapper 경로

`scripts/run_hope_pilot.py` 기준:
1. placement 실패 반환값을 저장만 하고 observe로 넘어간다(165–169줄).
2. align/refine의 반환값을 검사하지 않고 다음 단계/파일 수 확인으로 넘어간다(212–230줄).
   timeout/예외로 일부 파일만 생겨도 정상 정련 파일처럼 select할 수 있다.
3. subprocess.run(timeout=...)은 직접 자식만 종료하는 경로이며 dexrun 하위 worker의
   process group 종료 보장이 없다. stdout/stderr는 tail 일부만 기록하고 timeout 시
   부분 출력도 기록하지 않는다. 원본 로그를 파일로 보존하고 자신의 process group을
   종료/회수하는 회귀 검사가 필요하다.
4. stage 오류가 있어도 main이 0을 반환할 수 있다. 정상 no_candidates와 실행 오류의
   전체 상태/종료 코드도 구분해야 한다.

이는 코드상 잠재 실패 처리 문제다. 확인한 seed0 결과에서 timeout이 발생했다는
뜻은 아니다. 이미 완료한 정상 결과는 보존하며, 실행 중 Python은 소스를 수정해도
자동 반영되지 않으므로 다음 안전한 단계부터 수정 wrapper를 적용할 절차가 필요하다.
재시작 시 기존 output을 무조건 삭제하거나 완료 stage를 무조건 다시 실행하지 않는다.

## 실행 후보 선택 규칙과 시간

build_expansion_manifest는 전체 내용 SHA 순서로 dedup 후12개를 선택한다. 결정론적이지만
유형/seed 균등 배분은 아니다. 우리의 새10시간 계획에 들어간 균형 순환 제안은 이미
돌고 있는 실험의 규칙과 다르므로 **이번에는 기존 hash 규칙을 유지**하고 유형/seed별
선택 수를 보고하도록 계획을 정정했다. 유형 성능의 동등 예산 비교로 주장하지 않는다.

one_object는 한 물체의2유형×3seed를 마치면 그 물체의 동적 실행을 수행한다.
18개 전체 정련을 끝낼 때까지 첫 실행을 기다리는 구조가 아니다. seed0의약11분으로
추정한 총3–4시간은 정련 시간만의 추정이며 필터/접근/재현/환경 차이에 따라 늘어난다.
정련 수율이 높은 현재 결과를 동적 완료로 연결하는 것이 다음 판단의 핵심이다.
