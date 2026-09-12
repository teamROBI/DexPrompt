<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# First RL runs — historical interpretation audit

The retained [filtered.tsv](filtered.tsv) and [unfiltered.tsv](unfiltered.tsv)
record the original runs, described as five million steps, eleven workers,
and six objects each. The filtered selection had 19 candidates and the other
30. They are different task distributions, not paired trials.

Historical summaries (means over the first/last fifth of logged rows):

| Logged quantity | Filtered: first → last | Unfiltered: first → last |
|---|---|---|
| Reward | 0.543 → 0.637 | 0.402 → 0.525 |
| reached | 0.215 → 0.330 | 0.113 → 0.169 |
| digits | 0.0024 → 0.0023 | 0.0003 → 0.0026 |
| lift_rate | 0.0007 → 0.0020 | 0.0006 → 0.0003 |

## What these columns actually measured

* digits was the mean weighted, gated reward term over rollout steps. It is
  neither an ungated contact fraction nor a fraction of episodes that grasped.
* reached was sampled at steps after the running closest distance crossed a
  threshold. It is not an episode-normalized reach success rate.
* lift_rate was cumulative successes divided by completed training episodes.
  Averaging late cumulative ratios is not the success rate of the final policy.
* Success required sustained height, without contact compliance; accidental
  elevation could be counted. Reward and success now have different definitions.

The old trainer did not seed Torch, ignored its EPISODE constant, and treated
time limits as true terminals. The shape objective also rewarded squeezing while
still approaching. Missing orientation reward was one hypothesis among several,
not an identified sole cause of failure.

## Conclusions the runs do not support

They do not establish useful grasp learning, a benefit from the filter, or that
rejected poses are impossible to execute. Filtered did not beat unfiltered in
every column (its late digits value was lower). An open hand failing to fit at a
closed grasp pose only rejects the selected open-then-close heuristic; a different
trajectory can change the hand posture during approach.

The experiment is privileged-state PPO guided by Dexonomy goals, not visual
student training. Fresh corrected runs require raw physical metrics, matched
candidate manifests, seeded initialization, and separate evaluation. See
[the audit](../audit/README.md). Raw logs remain unchanged.
