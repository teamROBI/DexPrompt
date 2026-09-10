# Orientation runs — historical interpretation audit

[filtered.tsv](filtered.tsv) and [unfiltered.tsv](unfiltered.tsv) retain the
second runs. They were described as five million steps, eleven workers, and
six objects each. No seed replicates or separate final-policy evaluations were
recorded here.

## Multiple changes, not an orientation-only ablation

The update added absolute/progress orientation reward, tightened the contact
reward gate from position to position-and-orientation, AND added orientation
error to the observation. Torch initialization was not seeded, so an identical
CLI environment seed did not make the policy initialization identical.
The reward's effect cannot be separated from those other changes in these runs.

Historical final-fifth summaries:

| Run | Reward | reached | digits | lift_rate |
|---|---|---|---|---|
| Baseline, filtered | 0.637 | 0.330 | 0.0023 | 0.0020 |
| Baseline, unfiltered | 0.525 | 0.169 | 0.0026 | 0.0003 |
| Orientation, filtered | 0.533 | 0.118 | 0.0014 | 0.0033 |
| Orientation, unfiltered | 0.437 | 0.025 | 0.0000 | 0.0010 |

Reward definitions differ, so comparing their numerical size does not measure
progress on a fixed objective. digits is a weighted/gated per-step reward;
changing the gate changes the measurement. reached is a step-weighted running
flag, and lift_rate is cumulative training success under the old height-only
criterion. None is a separate final-policy benchmark.

The logs do not show that orientation is unnecessary, that a filter is superior,
or that teacher-guided RL is ineffective in general. Nor do they validate the
visual knowledge-transfer hypothesis. The current audit corrects measurements,
termination/bootstrap, seeding, approach/closure goals, and success criteria.
A new run tests that revised executor as a whole; an isolated orientation
ablation would require holding other code, inputs, tasks, and initialization
constant. See [audit verification](../audit/README.md). Original logs are preserved.
