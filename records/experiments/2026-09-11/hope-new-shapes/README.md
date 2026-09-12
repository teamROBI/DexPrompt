<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 새로운 형상의 HOPE teacher 시범

2026-09-11, **완료 (새 형상 시범 0건 — 초기 장면 계약에서 4/4 탈락, 생성·동적 미실행)**. T0 20:54:12 KST. 새 생성(예측·정렬·정련) 시작 마감 00:54(T+4h),
새 동적 후보 시작 마감 01:54(T+5h), 강제 종료 02:54(T+6h).

기존 용기 후보의 접촉 제어 튜닝은 범위 밖이다. 제어·접촉군·품질·성공 기준, 기존 모델·데이터,
student/visual/mask/goal-memory, lecun의 Codex 작업은 바꾸지 않는다.

- 출력: `/data1/yspark/sim/icra/hope-new-shapes-v1` (tmux `icra-shapes`, nice 15·BLAS 1스레드,
  동시 최대 2프로세스, 예측만 GPU 2).
- 도구: [run_hope_new_shapes.py](../../../../scripts/experiments/run_hope_new_shapes.py)
  (`missing`/`assets`/`static`/`select`/`dynamic`/`report`). 정적 단계는 검증된
  [run_hope_pilot.py](../../../../scripts/run_hope_pilot.py)를, 동적 실행·원장은
  [run_hope_candidate_diversity.py](../../../../scripts/experiments/run_hope_candidate_diversity.py)의 실행기를 그대로 쓴다.

> **정정 (2026-09-11, 사용자 검토 반영).**
> 1. 탈락 사유는 각속도만이 아니다. **Spaghetti(1.2 s 선속도 0.0715 m/s)와 Butter(0.0393 m/s)는 선속도
>    기준(≤ 0.005 m/s)도 넘었다.** Milk·Mayo는 선속도 기준 안(0.0032, 0.0007 m/s)이고 각속도만 넘었다.
> 2. Milk의 4 s 시점 각속도(0.029 rad/s)는 **그 한 시점에서는 기준 아래**다. 그러나 3–4 s 구간 최대
>    0.235 rad/s로 마지막 0.5 s 지속 기준은 통과하지 못했다. 한 시점 통과와 지속 안정화는 다르다.
>    아래 본문의 "대기 시간 부족이 원인이 아니다"는 단정이므로 철회한다 — 대기 시간의 효과는 이 관찰로
>    결정하지 않는다.
> 3. 접촉점 수(조각당 1개, 실패 물체는 1개)와 안정화 실패의 관계는 **원인 가설**이다. 관찰된 연관이며
>    인과는 확인하지 않았다. 확인은 [multiccd 비교](../hope-multiccd-ab/README.md)에서 별도로 한다.

## 1. 이전 보고 정정

[hope-candidate-diversity](../hope-candidate-diversity/README.md) 맨 위에 정정문을 넣었다.
새 중앙 실패 9건의 저장 phase를 다시 읽어 **실제 종료 phase는 close 8, lift 1**임을 확인했다
(lift 진입은 AlphabetSoup LD `0ebc6a25` 하나). "닫기 5 / 운반 4"는 접촉 성립 여부에 따른 분류
(`contacts_never_all_taken` 5 / `contacts_taken_then_lost` 4)이며, "운반" 4건 중 3건은 close 중
접촉을 잃은 것이므로 리프트 하중 유지 실패로 해석하지 않는다.

## 3. 개발용 물체 선정 (검사 전 순서 고정)

근거: [geometry 감사](../hope-geometry-groups/README.md)의 근접 그룹과 `development-exclusions.json`,
공식 full 메시(mm)에서 직접 계산한 단면·받침·무게중심(`selection/shape-measures.json`),
세 방향 실루엣(`selection/silhouettes.png`). 모든 치수는 공식 단위 그대로이며 scale을 바꾸지 않는다.

| 기존 성공 형상 | 단면 (중간 높이) | 높이 |
| --- | --- | ---: |
| AlphabetSoup·Peaches (원통 캔) | 지름 약 65 mm 원 | 84 / 58 mm |
| Raisins (얇은 상자) | 85 × 38 mm 직사각형 | 123 mm |

후보군(아직 새 형상 test로 승인되지 않은 14개)에서 기존 성공과 **같은 계열은 제외**했다:
Cookies·Popcorn(38–40 mm 두께 상자, Raisins 계열), Tuna·Mushrooms·Parmesan·Yogurt(지름 65–68 mm
원통/컵, 캔 계열). 남은 계열과 사전 순서:

| 순서 | 물체 | 형상군 | 단면 (10% / 50% / 90% 높이) | 받침 면적 | 무게중심 높이 | 파지 가능한 면 | 정식 유형 (최대 2) |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | **Milk** | 박공 지붕 사각 종이팩 (근접: OrangeJuice) | 70×72 / 71×73 / 70×16 mm (지붕 능선) | 48.2 cm² | 88 mm | 70–72 mm 평행 옆면, 높이 190 mm | Large Diameter, Prismatic 4 Finger |
| 2 | **Spaghetti** | 긴 납작 막대 (근접 없음) | 247×28 / 249×28 / 250×28 mm | 66.1 cm² | 25 mm | 28 mm 두께 평행면, 높이 50 mm, 길이 250 mm | Prismatic 4 Finger, Prismatic 2 Finger |
| 3 (대체) | Butter | 얇은 납작 상자 (근접: CreamCheese) | 102×23 / 102×24 / 102×24 mm | 22.8 cm² | 27 mm | 24 mm 두께 평행면 | Prismatic 4 Finger, Prismatic 2 Finger |
| 4 (대체) | Mayo | 짜는 병 (BBQSauce·Mustard 계열) | 63×40 / 53×40 / 36×35 mm | 18.1 cm² | 67 mm | 폭 40–63 mm 곡면, 위로 좁아짐 | Large Diameter, Medium Wrap |

- **기존 안정화 실패를 반영했다.** Mustard(받침 16.4 cm², 무게중심 64 mm)·BBQSauce(20.2, 66)는 병,
  ChocolatePudding(23.2, 24)은 얇은 작은 상자로 1.2초 후 각속도 기준에 실패했다. Mayo는 병 계열과,
  Butter는 ChocolatePudding과 받침·높이 비가 비슷해 대체 순위를 뒤로 두었다. 그래도 순위만으로
  통과를 가정하지 않고 모두 같은 검사를 거친다.
- Milk를 쓰면 근접 그룹 OrangeJuice도 개발 쪽으로 묶여 최종 새 형상 평가에 쓸 수 없다. Butter를
  쓰면 CreamCheese도 같다. 이번 선정 물체는 **개발용**이며 독립 test가 아니다. 나머지 미승인
  물체(Cookies, MacaroniAndCheese, Popcorn, Tuna, Mushrooms, Parmesan, Yogurt, SaladDressing 등)와
  근접 그룹은 이번에 건드리지 않고 최종 평가 설계에 남긴다(단, 계열이 기존 성공과 겹치는 것은
  "새 형상" 평가로 쓸 수 없음을 함께 적는다).
- 유형은 형상만 보고 정했다: 사각 기둥은 감싸기(Large Diameter)와 네 손가락 평면 대향(Prismatic 4),
  얇은 막대·상자는 평면 대향(Prismatic 4·2), 병은 감싸기(Large Diameter·Medium Wrap). 템플릿은
  계획 파일에 해시로 고정한 init_tmpl 파일만 쓴다.

초기 검사(기존 계약): 공식 메시 해시, 치수 오차 ≤ 0.001 mm, CoACD 충돌 조각, 텍스처(MTL·이미지),
Dexonomy 로더 대 실행기 질량·관성 일치(obj_mass 0.1 밀도 규칙), scene의 테이블 항목, 빌더 대 실행기
초기 자세, 안정화(이동 ≤ 5 mm·회전 ≤ 5°·선속도 ≤ 0.005 m/s·각속도 ≤ 0.05 rad/s), 로봇–물체/테이블
초기 침투 ≤ 1 µm, 두 카메라 중 하나 이상 목표 참조 가능, 텍스처가 600 step 물리를 바꾸지 않음.
두 개가 통과하면 멈추고, 최대 4개까지만 확인한다.

## 4·5. 생성·선택 계획과 예산 (동적 실행 전 고정)

- 물체당 유형 2개, 요구 접촉군 전체 유지, align seed 0/1/2, refine 400, 테이블 포함 scene_cfg와
  테이블 정련 훅, matcher 4·예측 100개(기존 HOPE 설정), 예측은 물체·유형별 1회 저장 후 재사용,
  중앙 배치만.
- 선택: 물체당 고유 후보 최대 8개, 유형별 4개씩 source 해시 순, 부족한 유형은 다른 유형으로 보충,
  round-robin. 모든 parents 보존, 같은 파일 중복은 한 번만.
- 대조군: AlphabetSoup 중앙·Peaches 중앙의 기존 성공(원본 episode 해시)을 각 1회. 둘 다 재현돼야
  새 동적 실행을 시작한다.
- 성공하면 물체별 실행 순서상 처음 두 고유 성공 후보를 각각 1회 재현.
- 예산: 미재현 보완 1 + 대조군 2 + 새 후보 16 + 성공 재현 4 = **최대 23회**. 시뮬레이션을 시작한
  실행은 인프라 실패·재시도라도 차감한다. 남은 시간·예산을 유형·scale·seed·제어 추가로 채우지 않는다.

## 2. Raisins `438146b5` 재현

직전 세션의 manifest(`Raisins-2-438146b501.json`)·scene·수집 인자 그대로 1회 실행했다(21:12:57,
120.2 s, exit 0, 인프라 오류 없음). **accepted·completed, 세 해시 모두 원본과 동일:**

| 해시 | 원본 = 재현 |
| --- | --- |
| episode | `668548823a64c48d…` |
| 컴파일 모델 | `89121bacba2e3304…` |
| 초기 상태 | `96713c98c9b3c533…` |

별도 HANDOFF: `/data1/yspark/sim/icra/hope-new-shapes-v1/missing-reproduction/HANDOFF.json`.
기존 `out/hope-pilot/HANDOFF.json`과 `hope-candidate-diversity-v1/HANDOFF.json`은 수정하지 않았다
(후자의 `reproduction_note`는 그대로 두고, 이번 파일이 그 재현 결과를 담는다). 차감 1/23.

## 3. 초기 검사 결과 — 4개 모두 미통과

사전 순서대로 네 물체를 모두 만들어 확인했다(두 개가 통과하지 않아 대체 후보까지 확인, 상한 4).
**빌드·치수·텍스처·질량/관성·테이블 항목·초기 침투·가시성은 모두 통과했고, 안정화 속도 기준
(1.2 s 시점 각속도 ≤ 0.05 rad/s)에서 네 물체 모두 실패했다.**

| 물체 | 빌드 | 충돌 조각 | 치수 오차 | 질량 (Dexonomy = 실행기) | 안정화 이동 / 회전 | 1.2 s 선속도 | 1.2 s 각속도 | 초기 침투 | 머리 카메라 목표 픽셀 | 판정 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| Milk | 49.0 s | 10 | 6e-6 mm | 0.10902 kg 일치 | 0.30 mm / 0.33° | 0.0032 m/s | **0.146** | 0 / 0 | 1587 | 미통과 |
| Spaghetti | 2.6 s | 1 | 6e-6 mm | 0.11898 kg 일치 | 3.06 mm / 0.51° | **0.0715** | **0.844** | 0 / 0 | 993 | 미통과 |
| Butter | 4.7 s | 1 | 2e-6 mm | 0.11012 kg 일치 | 3.30 mm / 2.86° | **0.0393** | **0.550** | 0 / 0 | 384 | 미통과 |
| Mayo | 27.5 s | 3 | 6e-6 mm | 0.11008 kg 일치 | 0.15 mm / 0.69° | 0.0007 m/s | **0.140** | 0 / 0 | 574 | 미통과 |

(모두 손목 카메라에는 보이지 않고 머리 카메라로 목표 참조 가능 — 기존 기준은 둘 중 하나. 텍스처가
600 step 물리를 바꾸지 않음도 4/4 확인.)

**따라서 이번 범위에서 새 형상 물체는 확보하지 못했다.** 지시대로 5번째 물체를 보거나, scale·마찰·
안정화 시간·기준을 바꾸지 않았다. 정적 생성(예측·정렬·정련)·후보 선택·대조군·새 동적 실행은 모두
**미실행**이다(대조군도 새 동적 실행이 없으므로 실행하지 않음). 차감은 2번의 1회뿐이다.

### 안정화 실패의 관찰 (진단만, 예산 외)

파일럿이 Mustard 등 세 물체에 쓴 것과 같은 방식으로, 같은 scene·제어로 4초까지 0.1초 간격으로
관찰했다(`assets/settling-tail.json`). 통과 물체 AlphabetSoup를 기준으로 함께 봤다. 다섯 경우 모두
1.2 s 상태가 초기 검사와 정확히 같다.

| 물체 | 각속도 0.5 / 1.2 / 2 / 4 s (rad/s) | 4 s 선속도 | 1.2→4 s 이동 | 물체–테이블 접촉점 (3–4 s) | 테이블에 닿은 충돌 조각 | 마지막 0.5 s 기준 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Milk | 0.351 / 0.146 / 0.103 / 0.029 (3–4 s 최대 0.235) | 0.0020 | 0.00 mm | **1** | 1 / 10 | 실패 |
| Spaghetti | 0.472 / 0.844 / 0.400 / 1.233 | 0.0280 | 0.32 mm | **1** | 1 / 1 | 실패 |
| Butter | 1.196 / 0.550 / 1.138 / 0.694 | 0.0144 | 0.10 mm | **1** | (1 / 1) | 실패 |
| Mayo | 0.216 / 0.140 / 0.215 / 0.199 | 0.0017 | 0.03 mm | **1** | (1 / 3) | 실패 |
| AlphabetSoup (통과) | 0 / 0 / 0 / 0 | 0 | 0.00 mm | **15** | 15 / 63 | 통과 |

(Butter·Mayo의 "닿은 조각" 괄호 값은 직접 세지 않았고, 조각마다 접촉점 1개라는 Milk·Spaghetti·
AlphabetSoup의 측정과 접촉점 1개에서 추론한 값이다.)

확인된 사실:

- 네 물체는 **넘어지거나 미끄러져 멀어지지 않는다**(1.2→4 s 이동 0.03–0.32 mm). 그런데 각속도가
  줄지 않거나 오히려 커진다 — 제자리에서 계속 흔들리는 상태다. 1.2 s를 4 s로 늘려도 기준을
  통과하지 않으므로, 대기 시간 부족이 원인이 아니다(파일럿의 세 물체와 같은 결론).
- 실행기의 테이블은 **box geom**이고 MuJoCo의 convex 다중 접촉(`multiccd`)은 **꺼져 있다**. 테이블에
  닿은 충돌 조각마다 접촉점이 **정확히 1개**다. 실패 물체는 바닥에 닿은 조각이 1개라 접촉점이 1개,
  AlphabetSoup은 15개 조각이 닿아 15개다.
- 이전 파일럿까지 합친 HOPE 10개에서, 통과한 3개는 충돌 조각이 많고(63·45·7) 실패한 7개는 적다
  (1·1·1·2·2·3·10).

해석(가설, 개입 실험 안 함): 평평한 바닥이 한 convex 조각으로 테이블 box에 닿으면 기본 설정에서
지지점이 1개뿐이라 물체가 그 점 위에서 계속 흔들리는 것으로 보인다. 이는 물체 형상이 물리적으로
불안정하다는 뜻이 아니라, **현재 충돌 분해 + 접촉 생성 설정에서 이 형상들이 받는 지지점 수**의
문제일 가능성이 크다. 다만 multiccd를 켜거나 CoACD 분해를 바꿔 확인하는 것은 물리·자산 계약 변경이라
이번 범위에서 하지 않았다. 따라서 "해당 물체로는 파지 불가"라고 결론내리지 않는다.

## 결론과 다음 판단

- **새 형상 teacher 시범: 0건.** 개발용 후보 4개(Milk, Spaghetti, Butter, Mayo)가 기존 초기 장면 계약의
  안정화 기준에서 모두 탈락해 생성·동적 실행 단계에 들어가지 못했다. 인프라 실패는 없었다(빌드 4/4
  exit 0, 검사 오류 0). `no_candidates`나 정련 0이 아니라 **초기 장면 단계의 계약 미통과**다.
- 이번 결과로 확인된 것: 현재 계약에서 HOPE 새 형상 확대의 병목은 파지 이전, **평평한 바닥 물체의
  테이블 지지 접촉**이다. 이는 PAPER_TODO P3 "서로 다른 형상 3개" 항목의 선행 조건이다.
- 사용자 결정이 필요한 선택지(실행하지 않음):
  1. **물리 계약 변경 검토**: `multiccd` 활성화 같은 접촉 생성 설정. 모든 기존 성공(8건+)의 재현이
     바뀔 수 있어, 켜기 전후로 기존 대조군을 전부 다시 확인하는 별도 실험이 필요하다.
  2. **자산 계약 변경 검토**: 바닥이 여러 조각으로 나뉘도록 충돌 분해 규칙을 바꾸기(물체별 조정이 아니라
     공통 규칙으로). 질량·관성 계약과 기존 자산 재검증이 필요하다.
  3. 계약을 유지한 채 다른 계열에서 후보를 다시 고르는 것은, 이번 관찰상 평평한 바닥 물체 대부분이 같은
     이유로 탈락할 가능성이 커 권하지 않는다.
- 선정한 4개는 개발용으로 소모된 것으로 기록한다(근접 그룹 OrangeJuice·CreamCheese 포함). 최종 평가
  형상 분할에서 제외해야 한다.

## 계산과 상태

| 항목 | 값 |
| --- | --- |
| 물리 실행 차감 | **1 / 23** (Raisins 재현 1). 대조군·새 후보·성공 재현 0 |
| 자산 빌드·검사 | 4개(상한 4), 빌드 합계 약 84 s, 검사 오류 0 |
| 진단 관찰 | 4 s 관찰 5물체(동적 예산 외, 파지 실행 아님) |
| 실행 시간 | 21:12:57–21:16:47(체인), 이후 진단. 6시간 중 약 40분 사용 |
| 남은 프로세스·tmux | 없음(확인) |

산출물(`/data1/yspark/sim/icra/hope-new-shapes-v1`): `missing-reproduction/HANDOFF.json`,
`assets/object-plan.json`(순서·유형 사전 고정), `assets/checks.json`, `assets/settling-tail.json`,
`assets/checks/<object>/centre/`(scene·head/wrist/mask·inspection 이미지), `assets/objects/<object>`
(빌드된 자산), `selection/`(단면 측정·실루엣), `generation-plan.json`(적격 0), `HANDOFF.json`(승인 0),
`ledger.json`, `chain-status.tsv`, `run.log`.

검사: `tests/test_hope_new_shapes.py`(6), `tests/test_hope_candidate_diversity.py`,
`tests/test_hope_pilot_wrapper.py` 합계 28개 통과, 신규 파일 Ruff 통과. 전체 suite는 실행하지 않았다.
Teacher accepted는 student-ready가 아니며, 이번 세션의 새 accepted는 Raisins 재현 확인 1건(기존 episode)뿐이다.
