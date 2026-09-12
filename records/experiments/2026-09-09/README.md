<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# 2026-09-09

고정 과제(병 1개 · `33_Inferior_Pincer` · yaw 0)에서 **접근을 시범으로 전이하고**,
그 위에서 RL이 무엇을 더하는지 확인한 날이다. 여기의 결과는 모두 privileged
접근 정책(팔 7관절)이며 시각 student가 아니다.

| 실험 | 무엇을 확인했나 |
| --- | --- |
| [approach-ppo-v1](approach-ppo-v1/README.md) | PPO 단독 200만 스텝: 인계 0회. 정지한 것은 아니지만 접근 자세·경로를 배우지 못했고(최근접 시점 자세 오차 130°), 실패 집합 안에서 보상이 평평하다는 것을 보상 분해로 확인 |
| [approach-bc-v1](approach-bc-v1/README.md) | 검증된 시범 9개(4,030쌍)의 행동 복제로 home에서 완주. 탐색 잡음 크기의 영향(std 0.05 대 0.37)과 critic warm-up 없는 이어가기의 붕괴 |
| [overnight-plan.md](overnight-plan.md) · [overnight-bc-ppo](overnight-bc-ppo/README.md) | BC를 보존하는 PPO 설정 탐색. 두 설정 모두 붕괴, 한 체크포인트만 BC를 넘었고 학습 seed 1에서 재현되지 않음 |
| [selection-and-kl](selection-and-kl/README.md) | 정책 선택 자동화(원본 BC 포함), 갱신 후 KL 측정·취소, 감시–중지 연결 |
| [kl-and-seeds](kl-and-seeds/README.md) | 규칙을 고정한 학습 seed 3개 비교. 최종 평가에서 seed 0의 선택 정책 38/40, BC 33/40, 나머지 두 seed는 동률 또는 BC 반환 |
| [cpu-multiobject-screen](cpu-multiobject-screen/README.md) | 새 물체 5개의 정적 통과 검사. 기본 놓임 자세에서 0/100, 다른 놓임 자세에서 일부 통과(동적 미검증) |

현재 병목은 [PAPER_TODO.md](../../../PAPER_TODO.md)의 P3(다물체 시범 확보)이며,
이날의 접근 전이는 그 위에서 쓰일 도구다. 여기서 얻은 정책은 고정 과제
한 개에서만 검증됐고, 물체 일반화나 시각 입력 전이는 확인하지 않았다.
