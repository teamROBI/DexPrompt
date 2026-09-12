# DexPrompt — 성공을 가정한 논문 작성본

사용자의 2026-09-12 요청에 따라, 연구 주장이 모두 성립하고 실험에 성공한 상황을 가정해 쓴
**영문 학술 논문 초안**이다. 현재 진행상황 보고가 아니다.

**이 폴더의 실험 규모·수치·성공 주장·새 학습기 구현 설명은 집필용 가정이다.**
실측 성과, 예측, 달성해야 할 수치 목표 또는 실행 승인으로 사용하지 않는다.
기존 [사실 기반 초안](../dexprompt.tex), [연구 정의](../../RESEARCH_PLAN.md),
[PAPER_TODO](../../PAPER_TODO.md)와 별도로 읽는다.

## 읽을 파일

- [논문 PDF](dexprompt_success.pdf): 7페이지, 영문 2단 조판, 그림4개·표5개·수식6개·참고문헌9개.
- [편집용 Markdown](manuscript.md): 같은 본문을 IDE에서 읽고 검토하는 버전.
- [LaTeX 원고](dexprompt_success.tex): 기존 ieeeconf 템플릿용 소스.
- [예시 수치](illustrative_results.json): 그래프·표에 사용한 가상의 값과 평가 규모 가정.
- [구조화된 원고](manuscript.json): 본문·그림 캡션·표·인용의 공통 입력.

PDF는 ReportLab으로 조판한 검토용 출력이다. LaTeX 컴파일러가 없어 `.tex`를 컴파일한 PDF는 아니다.
Markdown/LaTeX/PDF는 같은 원고 입력에서 생성하며, 페이지 배치는 달라질 수 있다.
실제 투고용 페이지 제한·익명성·스타일 적합성은 아직 판정하지 않았다.

## 논문이 전달하는 이야기

**정확한 형상과 Dexonomy 지식 → 실행 가능성을 학습한 파지 선정 → 작은 시범 수정 RL →
전체 동작 검증 → 시각 행동 BC → 미지 물체와 물체 선택 지시에 대한 파지.**

완성 논문의 문체로 다음 인과관계를 주장하도록 구성했다.

1. 학습된 선정과 시범 편집이 서로 다른 실패 단계를 줄여 자동 시범 생성의 수율을 높인다.
2. 같은 양의 데이터를 학습시켜도 제안 방식의 시범으로 학습한 시각 정책이 더 잘 수행한다.
3. 유형·접촉 지식과 목표 기억을 제거하는 실험이 각 구성요소의 역할을 뒷받침한다.
4. 같은 행동 모델이 목표 grounding을 통해 mask·point·language 지시에 대응한다.
5. 새로운 실제 물체에서도 성능을 유지하되, 이를 모든 물체·모든 환경의 보장으로 확대하지 않는다.

위 주장은 이 문서의 **가정된 결말**이며 아직 실제로 확인한 결과가 아니다.
예시 baseline은 우리 시스템의 통제된 변형이다. 다른 논문이 해당 수치를 보고한 것으로 쓰지 않았다.

## 그림

| 번호 | 내용 | 성격 |
| --- | --- | --- |
| 1 | 두 학습 단계와 privileged/visual 정보 경계 | 개념도 |
| 2 | 파지 선정·bounded editing·접촉 피드백 실행 | 개념도; 후보 점수도 예시 |
| 3 | 생성 수율·BC 데이터 효율·clutter 성능 | **가상 결과 그래프** |
| 4 | 동일 장면에서 mask/point/language로 목표 변경 | 개념도; 실제 rollout 아님 |

`figures/`에는 그림별 PNG·벡터 PDF·SVG를 함께 제공한다. 실제 로봇 사진이나 실험 영상처럼
보이는 합성 장면을 성과 증거로 넣지 않았다. 나중에 실제 실행 이미지로 대체할 수 있다.

## 본문 가정 중 구현 전에 확정할 것

- 후보별 단계 조건부 확률을 학습하는 선정기의 관측·라벨·탐색 편향 처리.
- 32차원 시범 편집의 변수/범위와 one-step PPO의 보상·예산.
- 유형/접촉 ablation에서 고정할 후보 풀·controller·수집 기준.
- 180개 형상 분할, 12,000시범, 5개 학습 seed 등의 실제 확보 가능 규모.
- 목표 grounding/추적, 행동 모델, 실물 평가의 구체적 구현과 고정 조건.

기존 접촉 제어를 고정 squeeze 추종으로 바꾸거나 현재 데이터에 B 물리를 섞는 계획은 아니다.
새 학습을 실행하거나 실제 진행 문서의 완료 항목을 바꾸지 않았다.

## 다시 생성

프로젝트 루트에서 실행한다. Python 환경에는 NumPy, Matplotlib, Pillow, ReportLab이 필요하며,
PDF 조판은 시스템의 Liberation Serif 글꼴을 사용한다.

```bash
python3 paper/hypothetical-success/draw_figures.py
python3 paper/hypothetical-success/author_content.py
python3 paper/hypothetical-success/build_draft.py
```

- 본문 수정: `author_content.py`를 수정한 뒤 뒤의 두 명령을 실행한다.
- 수치 수정: `illustrative_results.json`을 수정하고 표·그래프뿐 아니라 본문의 숫자도 함께 검토한다.
- 예시 수치가 실제 데이터로 바뀌어도 자동으로 실측 논문으로 승격하지 않는다. 실험 원장과 연결해
  별도 검증한 뒤 사실 기반 원고에 반영한다.

LaTeX 환경이 있는 곳에서는 원본 PDF를 덮어쓰지 않고 다음과 같이 빌드할 수 있다.

```bash
cd paper/hypothetical-success
mkdir -p latex-build
TEXINPUTS=..: pdflatex -interaction=nonstopmode -halt-on-error -output-directory=latex-build dexprompt_success.tex
TEXINPUTS=..: pdflatex -interaction=nonstopmode -halt-on-error -output-directory=latex-build dexprompt_success.tex
```

## 검사 기록

- PDF 7페이지를 렌더링하고 페이지 구성과 그림/표 배치를 확인했다.
- 모든 페이지의 가상 원고 표기, 캡션과 예시 수치 표기를 확인했다.
- 예시 성공+실패 분모, 수율, 평균·표준편차, 그림/표 번호와 인용 key를 검사했다.
- 기존 사실 기반 초안과 RESEARCH_PLAN/PAPER_TODO/RESUME는 전후 SHA-256이 같다.
- [검사 상세](validation.json)에 검사 결과를 남겼다. 실제 학습/물리/하드웨어 실험은 실행하지 않았다.
