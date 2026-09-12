# 공식 행동 backend 연결 — 3시간 결정 실험

2026-09-12 KST, 시작12:59 / 종료 상한15:59. 학습은 lecun GPU2, 별도 snapshot/tmux.

## 사전 고정

기존 단일 병 x0 최종0/3에 대해, 공식 DexGraspVLA 행동 Transformer와 diffusers DDIM,
공식 AdamW decay 그룹 규칙을 사용하는 비교 한 건을 수행한다.
고정 출처: https://github.com/Psi-Robot/DexGraspVLA , commit `0144c7928817c5149ae836da2682291360828a21`.
`timm==1.0.12`, `diffusers==0.32.1`, `einops==0.4.1`은 별도 deps 폴더에 설치한다.
공유 환경과 기존 코드 snapshot을 변경하지 않는다.

- 동일 데이터/cache 해시, 관측 v2와 DINO, HX5 27관절, 정규화, horizon64/execute8.
- 동일 x0 목표(공식 scheduler의 sample), masked padding, batch2/seed42/13epoch/6331갱신.
- 고정 최종epoch13, 평가seed1000–1002. 성공은 접촉 획득·리프트·유지·품질 기준 전체.
- backend-check → 기존 명령 대조 → 학습 → 최종 폐루프 평가. 실패 단계 뒤는 진행하지 않는다.
- 종료 상한 안에 평가40분을 남겨 학습에 시간 제한을 둔다. 미완료는 실패0으로 세지 않는다.
- 여러 코드 차이를 함께 공식 경로로 바꾸는 구현 묶음 비교다. 단일 요인 인과 실험이라고 하지 않는다.
- 공식 전체 정책 재현은 아니다. 관측/goal-memory/HX5/action padding/데이터/훈련 예산은 우리 계약이다.
  사전학습 행동 가중치도 가져오지 않는다. 한 과제 성공은 미지 물체·지시 성공이 아니다.

## 코드 대조 발견

1. workspace는 PyTorch 기본 Linear 초기화, 공식 Transformer는 std0.02/zero bias 초기화.
2. timestep grid: workspace 49→0 linspace16개, 공식 DDIM은50/16 leading 간격으로45→0.
3. 공식 optimizer는 ndim>=2만 weight decay, workspace는 모든 trainable parameter에 적용.
4. **고정 timm1.0.12의 RmsNorm fallback은 sample variance를 사용**한다. workspace는 mean square다.
   동일 가중치를 이식한 비교에서 차이가 있었고, 폐기 가능한 검사 모델의 norm만 동일하게 하면
   출력 일치 여부를 확인할 수 있다. 이를 student 실패 원인이라고 확정하지 않는다.
   원본 버전 동작을 보존하며, workspace의 기본 norm/기존 checkpoint는 변경하지 않는다.
5. 공식 설정에서 attention mask와 EMA는 꺼져 있으므로 누락 원인으로 삼지 않는다.

## 판단 규칙

- 정상 완료+새 완주: 공식 action 경로 적용의 제한적 근거, 다음은 작은 배치/목표 변경 검증.
- 정상 완료+완주0: 이 교체만으로 현재 과제는 해결되지 않음. 같은 과제 추가 architecture 튜닝 중단.
- 의존성/구현/시간 제한 실패: 학습 성능 판단 보류, 인프라 실패를0으로 집계하지 않음.

## 경로

lecun `/data1/yspark/projects/dexprompt/` 아래:
- `student-code/upstream-action-v1`: 별도 코드/공식 원본2파일/deps.
- `student-checks/upstream-action-v1`: 단계 exit/timeout·검사·최종 평가.
- `student-runs/upstream-action-v1`: 학습 기록/원본 checkpoint.
- tmux `dexprompt-upstream-action-v1`.

기존 teacher 자료·Claude 작업·물리·접촉 기준은 변경하지 않는다.

## 실행 중 확인

13:07 tmux 시작. backend-check와 preflight, 기록명령/정지 대조를 통과해 학습 진입.
동일 가중치 출력차0.006165는 norm 연산을 동일하게 한 검사에서 정확히0으로 줄었다.
실제 공식 backend를 사용하며, 작은 검사 모델의 condition gradient·backward는 유한하고
optimizer parameter 그룹 누락/중복이 없다. 이 검사는 optimizer 갱신0회다.
로컬 기존 student 검사47개와 upstream 경계 검사3개 통과. 전체 suite는 미실행.


## 자동 완료 인계

Turing tmux `codex-upstream-watch`가60초마다 승인된 원격 작업만 읽는다.
학습·최종 평가 종료 또는15:59 상한에 `FINAL.md`를 한국어로 작성하고 종료한다.
`snapshot.json`은 원격 stage/exit/timeout/진행률/최종 판정, `watcher.json`은 조회 상태다.
비정상/일부 평가를0/3으로 바꾸지 않는 판정 검사3개 통과.
전체 관련 검사53개(기존47+upstream 경계3+보고3), 전체 suite는 미실행.
대화가 끝나도 원격 학습·평가와 로컬 보고 생성은 tmux에서 계속된다.

## 최종 결과 확정 (14:13)

6,331갱신 완료, 마지막 train loss0.000705930577. 최종3회 모두 접촉 획득 없음,
지속 리프트0, 접촉 준수 완주0/3,974tick. 상승은 약4.9246e-7m다.
backend-check/preflight/controls/train/evaluate 모두exit0,timeout없음. 전용 원격 tmux 종료 확인.
[자동 최종 집계](FINAL.md). 공식 action backend 교체만으로 해결되지 않았으며,
전체 원본 정책/사전학습 행동 가중치 재현이나 논문 아이디어 반증으로 해석하지 않는다.
