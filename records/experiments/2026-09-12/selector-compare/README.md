# 후보 선정 비교 — 같은 풀·같은 예산에서 해시 순 / 무작위 / 학습 선정기

2026-09-12, **완료 (평가 14:23–14:47).** 대조 3/3 재현, 라벨 48회(accepted 9), lecun 학습 3개 모델, 평가 41회와
재현 2회, 물리 94/102. **이 후보 풀에서 학습 선정기가 기준선보다 성공 후보를 더 많이 고르지 못했다**
(AlphabetSoup 2/8·2/8·2/8, Peaches 0/8·0/8·0/8, Raisins 학습 3/8 = 해시 3/8, 무작위 2/8 — 학습 = 해시 선택 동일).
T0 13:41:01 KST, 종료 상한 15:59(새 물리 실행 시작 마감 15:50).
출력: `/data1/yspark/sim/icra/selector-compare-v1`. 도구: [run_selector_compare.py](../../../../scripts/experiments/run_selector_compare.py),
[grasp_selector.py](../../../../scripts/grasp_selector.py). 기존 자료·HANDOFF·물리 기준·student 코드·Codex 작업은 수정하지 않는다.

## 갱신 (14:20): lecun 접속 해결, 학습 재개

사용자가 준 명령으로 연결됐다. 기본 `~/.ssh/config` 항목은 이 키를 쓰지 않았고,
`-o IdentitiesOnly=yes -i ~/.ssh/id_rsa_yspark`와 지정된 agent socket을 함께 줘야 인증된다.

- 학습 묶음 3개와 `grasp_selector.py`를 lecun `/data1/yspark/projects/dexprompt/selector-compare-v1`로 복사했다.
  양쪽 해시가 같다.
- 14:21:19에 lecun tmux `selector-train`에서 CPU로 학습을 시작했다(`CUDA_VISIBLE_DEVICES=""`, nice 15).
- 해석기는 Codex가 쓰는 `/home/yspark/projects/DexPrompt/.venv-student/bin/python`(torch 2.6.0)을 읽기 전용으로 썼다.
  설치·수정은 하지 않았다. GPU와 Codex 작업은 건드리지 않았다.

아래 "차단" 절은 13:41 당시 기록이다.

## 차단: lecun 학습 불가 (13:41, 14:20 해결)

`ssh lecun`(147.47.200.171:19423, 사용자 yspark)이 **`Permission denied (publickey)`**로 거부됐다. Codex가 앞서 기록한 것과 같은
인증 문제다. 학습은 lecun에서만 하도록 지정했고, 인증이 안 되는 동안 다른 서버에서 대신 학습하거나 접속을 반복하지 않는다는
규칙이 있으므로:

- **오늘은 선정기 학습, 학습 선정기의 순위, 평가 비교(최대 48회)와 재현(최대 3회)을 실행하지 않는다.**
- 평가를 기준선만으로 줄여 억지로 진행하지 않는다(요청한 비교가 아니므로).
- 학습과 무관한 부분은 끝까지 진행한다: 후보 추출·분할 고정, 대조 3회, 라벨 수집 48회, 학습 입력 묶음과 재개 명령.

## 1. 후보와 조건 고정 (실행 전, 13:45 `plan.json`)

- A 물리 각 물체의 **미실행** 후보에서 고정 seed(20260912)로 32개를 무작위 추출해, 추출 순서의 앞 16개를 라벨 수집, 나머지 16개를
  평가 풀로 나눴다. 후보 해시는 `plan.json`과 `evaluation-candidates.json`에 저장했다.
- 사실상 동일 후보 검사: grasp·squeeze 관절 목표의 최대 차이가 1e-3 미만이면 같은 묶음으로 보고, 묶음은 한쪽에만 둔다.
  세 물체 모두 해당 쌍 **0개**다(가장 가까운 쌍도 0.0087 이상 차이).
- 조건: 중앙·단일 물체·yaw 0, 기존 제어·접촉 피드백·품질·성공 기준 그대로.
- 평가 풀의 결과는 학습·튜닝·체크포인트 선택에 쓰지 않는다. 학습 묶음을 만들 때 평가 후보가 들어가지 않았는지 코드로 검사한다.

| 물체 | 미실행 풀 | 추출 | 라벨 16 (유형) | 평가 16 (유형) |
| --- | ---: | ---: | --- | --- |
| AlphabetSoup | 179 | 32 | PD 8 · LD 8 | PD 10 · LD 6 |
| Peaches | 50 | 32 | PD 16 | PD 16 |
| Raisins | 32 | 32 (전부) | LD 16 | LD 16 |

Raisins는 미실행 후보가 정확히 32개라 전부 사용한다. 무작위 추출은 해시 순 선택 편향을 줄이지만, 풀 자체가 Dexonomy 정련과
기존 선별(손이 이미 물체 안 제외)을 통과한 후보이므로 **모든 선택 편향이 없어진 것은 아니다.**

**학습 설정(결과 확인 전 고정)**: 폭 128 모델, Adam lr 1e-3, weight decay 0, full batch, **400회 갱신**, seed 0, 마스크된 BCE
(전체·접근·닫기·carry head), **마지막 갱신을 최종 체크포인트로 사용**(조기 종료·체크포인트 선택 없음), 점수는 전체 성공 head의
sigmoid. 3개 모델은 각각 평가 물체의 형상군 전체를 학습에서 뺀다. 학습 데이터는 A·중앙·단일 물체의 독립 라벨(기존 + 새 수집)만이며,
배치·clutter·재현본·B 물리는 제외한다.

**순위 규칙(고정)**: 해시 순 = source sha256 오름차순. 무작위 = `default_rng(20260913)` 순열(물체 순서대로). 학습 선정기 = 해당 형상군을
빼고 학습한 모델의 점수 내림차순(동점은 해시 순). 세 순위를 모두 저장·해시한 뒤에만 평가를 시작한다.

**예산**: 물리 최대 102 = 대조 3 + 라벨 48 + 평가 최대 48 + 재현 최대 3. 시뮬레이션을 시작한 실행은 접근 실패·인프라 중단도 차감.

## 2. 대조와 라벨 수집 (13:45:12–14:07, 동시 2개 실행)

**대조 3/3 재현**: 물체별 기존 성공 후보(AlphabetSoup `057c04e1`, Peaches `203b95f1`, Raisins `0ddf3447`)가 모두 accepted이고
원본 episode 해시와 같았다. 세 물체 모두 라벨 수집을 진행했다.

**라벨 48회** (물체당 16, 인프라 오류 0, 미실행 0, 모두 시뮬레이션 시작·차감):

| 물체 | accepted | 종료 결과 분포 | 실행 시간 합 |
| --- | ---: | --- | ---: |
| AlphabetSoup | **0 / 16** | closure_timeout 7, no_quality_feasible_command 4, lift_timeout 3, 접근 경로 실패 2 | 856 s |
| Peaches | **1 / 16** (`7a8cdecf40`) | closure_timeout 7, no_quality_feasible_command 6, 접근 경로 실패 2, completed 1 | 776 s |
| Raisins | **8 / 16** | completed 8, no_quality_feasible_command 4, lift_timeout 2, closure_timeout 2 | 850 s |
| 합계 | **9 / 48** | | 2,482 s |

실패는 모두 라벨로 남겼다(접근 경로 실패 포함). 성공 수를 맞추려고 추가 수집하지 않았다.
물리 실행 **51 / 102** (대조 3 + 라벨 48).

물체별 성공률이 크게 다르다(0%, 6%, 50%). 사전 비교 설계에는 다음과 같은 한계가 있다.
- AlphabetSoup 평가 풀도 비슷하게 성공 후보가 거의 없다면, 어떤 방법도 top-8에서 성공을 고를 수 없다.
- Raisins는 성공이 흔해 방법 간 차이가 작을 수 있다.

이는 결과가 아니라 설계 한계로 기록한다.

## 3. 학습 입력 준비

- 라벨 48행을 기존 원장과 합쳐 학습 원장을 새로 만들었다(`train-ledger/`, 173행, 입력 원장 대조 일치).
- `export-bundle`: A·중앙·단일 물체·독립 라벨만 사용한다. 배치·clutter 32행, 재현 15행, B 13행, 미관측 17행은 제외했다.
- 평가 후보는 제외 목록으로 막고 코드로 검사했다. 학습 원장에 들어간 평가 후보는 **0개**다.
- 형상군별 묶음 3개(각 64행 = 나머지 두 물체의 중앙 단일 독립 라벨):
  - `bundle-AlphabetSoup.pt` `ab36cbbf…`
  - `bundle-Cherries.pt`(Peaches 형상군) `fb038d56…`
  - `bundle-Raisins.pt` `f5dbe38e…`
- 묶음은 표면점 캐시를 포함해 메시 파일 없이 학습할 수 있다. 학습 설정은 `grasp_selector.TRAINING`에 고정했고,
  `train-bundle`은 `--confirm-training` 없이는 거부한다.

## 4. lecun 학습 (14:21:19–14:21:57)

사전 고정 설정 그대로 3개 모델을 학습했다. 각 400회 갱신, 마지막 갱신을 체크포인트로 사용했고, 평가 풀 결과는 쓰지 않았다.

| 모델 (학습에서 뺀 형상군) | 학습 행 | 첫 / 마지막 train loss | 체크포인트 sha256 | 소요 |
| --- | ---: | --- | --- | ---: |
| AlphabetSoup+TomatoSauce | 64 | 2.753 / 0.00274 | `0dcf146a1f7f…` | 12 s |
| Peaches 형상군(Cherries+…) | 64 | 2.786 / 0.00233 | `a5c619e9e8ae…` | 13 s |
| Raisins | 64 | 2.769 / 0.00177 | `b582788a31e8…` | 13 s |

- 세 체크포인트를 로컬로 가져와 lecun과 해시를 대조했다(일치).
- 마지막 train loss가 0.002 안팎으로, 64행을 거의 완전히 맞췄다. **과적합 가능성이 크다.**
  설정은 결과 확인 전에 고정했으므로 바꾸지 않았다.

## 5. 순위 고정 (14:22:59, 평가 시작 전)

`rankings.json`에 세 방법의 전체 순위와 학습 점수를 저장했다. 평가는 이 뒤에 시작됐다.

| 물체 | 학습 점수 범위 | 고유 점수 | 학습 top-8 = 해시 top-8 | top-8 겹침 (학습∩해시 / 학습∩무작위 / 해시∩무작위) | 평가할 합집합 |
| --- | --- | ---: | --- | --- | ---: |
| AlphabetSoup | 7.7e-19 – 0.972 | 14/16 | 아니오 | 3 / 5 / 4 | 14 |
| Peaches | 1.9e-21 – 0.00085 | 10/16 | 아니오 | 4 / 3 / 5 | 14 |
| Raisins | **0 – 0** (모두 정확히 0) | **1/16** | **예 (전체 순서 동일)** | 8 / 3 / 3 | 13 |

- **Raisins에서는 학습 선정기와 해시 순을 구별할 수 없다.** Raisins를 뺀 모델의 점수가 16개 모두 0.0으로 underflow돼,
  사전 규칙의 동점 처리(해시 순)가 순위를 정했다.
  - 이 모델은 AlphabetSoup·Peaches의 64행으로 학습됐고, 그중 성공은 4행이다.
  - 출력이 포화돼 Raisins(Large Diameter만)에 대해 전부 실패로 예측했다.
  - Raisins 행의 "학습 선정기" 결과는 해시 순 결과와 같은 것으로 보고한다.
- Peaches도 점수가 매우 작고 일부 동점(고유값 10/16)이 해시 순으로 풀렸다. AlphabetSoup만 점수가 실제로 퍼져 있다.

> **정정 (2026-09-12, 사후 — 위 원문은 기록으로 남김)**: 위 표의 '고유 점수' 14/16(AlphabetSoup)·10/16(Peaches)과
> 'Peaches 일부 동점이 해시 순으로 풀렸다'는 서술은 틀렸다. `rankings.json`에 저장된 확률은 두 물체 모두 16개가 서로 다르며
> (동점 없음), 반올림한 표시값을 센 오류다. 전부 같은 값이었던 것은 Raisins(1/16, 모두 0.0)뿐이다.
> Raisins의 0.0은 모델 출력이 동일해서가 아니라, 서로 다른 raw logit(-241.7 ~ -193.7, 16개 모두 다름)을 float32 sigmoid가
> 전부 0으로 만든 수치 underflow다. raw logit 순위로 다시 계산한 사후 정정 비교는
> `/data1/yspark/sim/icra/teacher-followup-5h-20260912-v1/{rankings-logit.json,comparison-logit.json,FINAL.md}`에 따로 있고,
> 이 문서의 순위·결과를 대체하지 않는다. 진단: `results/2026-09-12/selector-logit-audit/`.
- 합집합 41회(최대 48). 세 방법의 결과는 이 합집합 실행을 공유한다.

## 6. 같은 평가 풀에서의 비교 (14:23:00–14:47:24, 합집합 41회 + 재현 2회, 인프라 오류 0)

각 방법이 평가 후보 16개 중 top-8을 골랐다. 중복 후보는 한 번만 실행해 결과를 공유했다. 판정은 실행기 자체 기준
(hold·필수 접촉군·품질)이고, 미관측 없이 방법마다 8개가 모두 실행됐다.

| 물체 | 해시 순 | 고정 seed 무작위 | 학습 선정기 | 평가 풀에서 실행한 후보 중 성공 | top-8 구성 (셋 공통 / 학습 고유 / 해시 고유 / 무작위 고유) |
| --- | ---: | ---: | ---: | ---: | --- |
| AlphabetSoup | **2 / 8** | **2 / 8** | **2 / 8** | 2 / 14 | 2 / 2 / 3 / 1 |
| Peaches | 0 / 8 | 0 / 8 | 0 / 8 | 0 / 14 | 2 / 3 / 1 / 2 |
| Raisins | 3 / 8 | 2 / 8 | 3 / 8 (**해시 순과 같은 선택**) | 4 / 13 | 3 / 0 / 0 / 5 |
| 합계 | 5 / 24 | 4 / 24 | 5 / 24 | 6 / 41 | |

- **AlphabetSoup**: 세 방법이 같은 두 성공 후보(`99f5c885db`, `34815d4f21`)를 모두 top-8에 넣었다. 성공 수는 같다.
  - 학습 top-8의 실패 6개는 lift_timeout 5·closure_timeout 1이다. 종료 단계는 hold 4·lift 3·close 1로, 다른 두 방법
    (hold 3·lift 2·close 3)보다 뒤 단계에서 끝난 것이 많다.
  - 이는 서술일 뿐이며 성공이 아니다.
- **Peaches**: 평가 풀에서 실행한 14개 중 성공이 없어, 어떤 방법도 성공을 고를 수 없었다.
- **Raisins**: 학습 선정기의 점수가 모두 0.0이어서 선택이 해시 순과 완전히 같다. 3/8 대 무작위 2/8은 학습 효과가 아니라
  해시 순과 무작위의 차이다.
- **재현**: 학습 선정기가 고른 첫 성공 1회씩. AlphabetSoup `99f5c885db`, Raisins `81839469d4` 모두 accepted이고 episode 해시가 원본과 같다.
  Peaches는 성공이 없어 재현하지 않았다.
- 비용: 평가 41회 2,625 s(동시 2개 실행). 전체 물리 94/102(대조 3 + 라벨 48 + 평가 41 + 재현 2).

### 판단

- **이 제한된 후보 풀에서 학습 선정기가 기준선보다 성공 후보를 더 잘 고른다는 근거는 없다.**
  - AlphabetSoup·Peaches는 세 방법이 같았다.
  - Raisins는 학습 선정기가 사실상 해시 순이었다(출력 포화). 셋 중 비교가 성립한 것은 AlphabetSoup 하나이고, 거기서도 차이가 없었다.
- 원인 후보는 학습 행 64개 중 성공 4개, 과적합(train loss 0.002), 물체 간 유형 겹침이다. 원인은 분리하지 않았다.
- 세 형상만으로 일반화를 주장하지 않는다. 무작위 추출이 모든 선택 편향을 없앴다고 쓰지 않는다.
  평가 풀 결과는 학습·튜닝·체크포인트 선택에 쓰지 않았다.
- 평가 풀 결과에 맞춰 모델·설정·순위를 바꾸지 않았다.
- 새 성공 시범(평가·재현)은 선정 비교용 teacher 실행 기록이다. student-ready로 표시하지 않았다.

## 완료·미완료·실행하지 않은 것

**완료**
- 후보 추출·분할 고정(물체당 라벨 16 / 평가 16, 사실상 동일 후보 0쌍, 해시 저장).
- 대조 3/3 재현, 라벨 48회(9 accepted), 인프라 오류 0.
- 학습 원장과 형상군별 학습 묶음 3개. 평가 후보 누설 0 확인.
- 순위·평가·재현 단계 구현(`run_selector_compare.py rank/evaluate`). 체크포인트가 없으면 거부한다.

- lecun 학습 3개 모델(사전 고정 설정, CPU), 체크포인트 해시 대조.
- 세 순위 저장·고정(평가 전), 합집합 41회 평가, 학습 선정기 첫 성공 재현 2회(모두 동일 해시), 결과 보고(`comparison.json`).

**미완료**
- 없음(요청 범위 기준). Peaches는 성공이 없어 재현 대상이 없었다.

**실행하지 않은 것 (의도적)**
- 평가 결과를 본 뒤의 재학습·설정 변경·순위 변경.
- 남은 물리 예산 8회.
- 학습 선정기의 결과 보고 기능(`report`)은 계획 고정 뒤에 추가한 보고 전용 코드다. 실행 경로와 순위 규칙은 바꾸지 않았다.

**실행 순서 기록** (오늘 14:20 이후 이 순서로 실행했다. 다시 할 때는 새 출력 폴더와 새 cut-off를 쓴다)

1. `/data1/yspark/sim/icra/selector-compare-v1/bundles/`와 `scripts/grasp_selector.py`를 lecun으로 복사하고 해시를 대조한다.
2. lecun에서 별도 tmux로(Codex GPU2 작업은 건드리지 않음, CPU로 충분):
   `python scripts/grasp_selector.py train-bundle --bundle bundle-<AlphabetSoup|Cherries|Raisins>.pt --out checkpoints --confirm-training` 3회.
3. `checkpoints/selector-*.pt`를 `/data1/yspark/sim/icra/selector-compare-v1/checkpoints/`로 되가져온다.
4. `third_party/dexonomy/.venv/bin/python scripts/run_selector_compare.py rank --out /data1/yspark/sim/icra/selector-compare-v1`
   (세 순위를 모두 저장한 뒤)
5. `... run_selector_compare.py evaluate --out /data1/yspark/sim/icra/selector-compare-v1`
   (합집합 최대 48회 + 재현 최대 3회. 스크립트의 cut-off 시각을 새 실행일에 맞게 바꿔야 한다.)

세 형상만으로 일반화를 주장하지 않는다. 무작위 추출이 모든 선택 편향을 없앴다고 쓰지 않는다.
