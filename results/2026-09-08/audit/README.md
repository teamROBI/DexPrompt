# Audit verification, 2026-09-08

This is code-validation evidence, not a learned grasping result. Historical
experiment logs in adjacent directories were preserved unchanged.

## Bounded checks

* All 35 tests passed. The suite covers the original hand checks plus GAE terminal/timeout
  handling, potential shaping, contact-compliant success, open/close goals,
  ladder support/NA behavior, pre-action recording, and generation-stage failure.
* [PPO smoke](ppo_smoke.tsv): one candidate, one worker, CPU, seed 0,
  64 control steps, eight-step time limit; eight episodes, zero sustained lifts.
  The run exercised spawned workers, time-limit bootstrap, optimization, metrics,
  manifest creation, and the final checkpoint.
* [Separate evaluation](evaluation_smoke.json): deterministic actor, two fresh
  resets on that training candidate; zero contact-compliant sustained lifts.
  This is not held-out-object evaluation or evidence of policy learning.
* [Ladder smoke](ladder_smoke.tsv): one candidate; rung 0 passed 1/1,
  rung 1 passed 1/1, rung 2 failed 0/1. Temporary execution XML compiled.
  A one-candidate diagnostic does not estimate population table feasibility.
* Real environment reset/stepping produced finite 140-dimensional observations
  and correctly reported time-limit truncation. One compiled object weighed
  0.09528843 kg despite the nominal 0.1 kg setting.
* Synthetic three-frame Zarr serialization preserved schema v2, pre-action
  alignment metadata, timestamps/control_dt, and episode boundaries. A separate
  regression test verified pre-action images/states and unchanged phase duration.
  Successful full-scene demonstrations with this recorder remain to be established.

## What changed in interpretation

The old contact statistic was a gated/weighted reward, reach was step-weighted,
and lift rates were cumulative training statistics with a height-only success
criterion. The new raw contact and contact-compliant lift metrics cannot be
numerically compared as if their definitions were unchanged.

The revised approach/closure goals and reward are engineering hypotheses. They
need a fresh learning experiment with fixed tasks and separate evaluation.
The corrected hand-only ladder does not implement arm rung 3. The privileged
PPO trainer still does not export visual demonstration episodes or train the
DexGraspVLA-style student.

## Local artifacts and reproduction

The full smoke manifests/checkpoints are at out/rl/audit-smoke-v3; separate
evaluation is at out/rl/audit-evaluation-v3. These are ignored local artifacts.
The manifest records code/candidate hashes, candidate/yaw selection, scene,
weights, and training settings; it does not bundle every external asset/dependency.
See [SETUP.md](../../../SETUP.md) for fresh-output commands and
[HANDOVER.md](../../../HANDOVER.md) for outstanding research work.

The final finite-action guard also rejects positive/negative infinity before
clipping; the v3 smoke manifests predate that guard-only correction. Their
evaluation was performed with their matching environment code. Current evaluation
intentionally rejects a checkpoint with a different environment hash; use a fresh
smoke output or the matching revision.

The paper passed static environment/brace/citation checks (seven citation keys).
No LaTeX engine was available, so PDF compilation and visual layout inspection
were not performed. The manuscript remains a research draft with explicit TODOs.
