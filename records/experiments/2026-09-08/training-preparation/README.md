<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# Teacher execution and student preparation, 2026-09-08

This task implements the pipeline up to user-launched training. No optimizer
update, backward pass or student training was run. The agreed teacher is still
Dexonomy; the new student is a workspace-owned 27D DINOv2/RDT-style architectural
adaptation, not a replacement teacher or an imported trained DexGraspVLA policy.

## Implemented path

- Original-source validation binds teacher hash, object, grasp type, HX5 identity,
  configurations, contact groups, quaternion convention and executed scale.
- Each candidate has a freshly simulated complete home approach. Dynamic execution
  commands actuators; IK and two-tick control previews restore all integration state.
- Contact/force and bounded finger-geometry feedback close the hand. Every physics
  substep contributes conservative contact, force, penetration, height and
  disturbance measurements. Failures remain outside training splits.
- Schema-3 NPZ/JSON episodes carry pre-action head RGB/object mask, wrist RGB,
  27 measured joints and executed targets. Compiled MJB models and complete
  initial states preserve simulator replay. Source snapshots and hashes are saved.
- Optional manually annotated object-local part regions define both image masks
  and physical contact membership. This does not implement automatic language
  grounding or supply missing semantic labels.
- Dataset merging, interruption recovery, object/candidate-disjoint splits,
  train-only normalization, action chunks, verified memory-mapped training caches,
  student preflight, forward-only checks, user-invoked training and checkpoint
  inference/evaluation are implemented.

See [the workflow](../../../../docs/training.md),
[dataset contract](../../../../docs/teacher-data.md), and
[student configuration](../../../../docs/student.md).

## Actual fixed-task collection

The original bottle / 11_Power_Sphere / scale 0.11 / yaw 0 candidate is retained.
The full dynamic local +X path is validated again. The engineering limits remain
2 mm target penetration, 30 N per whole digit, and 10 mm approach displacement;
robot-table collision tolerance is 1 micrometre. Physical gains/friction/force
limits were not changed to pass these checks.

The final `out/teacher-preparation-v4` completed 1124 control intervals
(22.48 simulated s).
It reached five contact groups and an interval-minimum rise of 8.4247 cm, but
stopped with `no_quality_feasible_command` before the required sustained 10 cm
hold. Peak penetration was 1.99975 mm, peak whole-digit normal force 29.1299 N,
and maximum approach XY displacement 0.0292 mm. No robot-table penetration was
measured. **Accepted training episodes: 0.** The validator correctly retains the
episode in diagnostics and reports no training/validation split.

The final stored-command replay in `out/teacher-preparation-audit-replay-v4` matches
all 1124 pre-action joint states, timestamps and physics-quality signals with
**maximum error 0**, with no contact-bit mismatches. It reproduces the rejected
quality result. Images are not independently revalidated by this numerical
replay; a separate real reset/render check verified the archived observation
interface at `[64,64,4]`, `[64,64,3]` and `[27]`.

The v1 preparation attempt exposed a scalar-versus-isotropic-three-vector scale
handling bug, which was fixed. v2 is an intermediate integration run. v3 archived
float32 action labels while issuing float64 actuator commands; v4 preserves
float64 commands for exact simulator replay. The final physical measurements
equal v3's, but only the v4 archive supplies the full-precision commands.

The selection CLI was also run against the existing teacher/object roots with
a limit of 3. It produced `out/teacher-preparation-candidates-v1.json` after
rejecting 104 table-intersecting and 10 open-fit candidates. These are static
selection outcomes, not three successful executed grasps.

## Existing-code errors corrected

Recorder reset now clears time, activation, applied forces and solver warmstart.
Observation-only `mj_forward`, IK and rejected previews cannot change integration
state. The collector CLI's `paths` name collision no longer prevents every launch.
Arm reach uses the first joint and a conservative chain-length bound, rather than
a wrist-centered sampled radius. Segmentation checks geom type and object
descendants. Contact control reconciles rejected/scaled proposals to prevent
unexecuted corrections accumulating. Successful execution metadata no longer
contains a truthy failure reason that would reject every positive episode.

The legacy palm diagnostic used a version-sensitive percentage of contact points.
It now selectively removes palm collision and checks improved tracking while
retaining finger collisions and actuator settings. This changes the test oracle,
not the simulator defaults. The independent [physics audit](../contact-physics/README.md)
also discloses object mass and finger-object torsional-friction mixing.

## Limits before user training

The fixed constrained task has no accepted full-hold demonstration yet. Collection
must produce accepted examples across independent object groups before an
object-disjoint training/validation experiment is possible. The current interpreter
also lacks torchvision and configured DINOv2 source/ViT-B/ViT-L weights. Preflight
reports these missing prerequisites and refuses to start learning. Tiny random
patch-encoder fixtures verify software interfaces only; actual pretrained-DINO
execution and learned student success are not established.

## Verification evidence

- Final complete suite after the cross-review: **201 passed** in the
  simulation/Torch interpreter (MuJoCo 3.3.3), using
  `OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python -m pytest -ra`.
  The same command with `.venv/bin/python` (MuJoCo 3.12) gives
  **169 passed, 5 skipped**; Torch-dependent tests are skipped because Torch is
  not installed there. These are two environments, not independent experiments.
- Real full collection, conservative dataset validation and full-precision
  stored-command replay completed on the fixed task.
- Merging the rejected run retains its MJB, initial state and source archives;
  the merged archive also passed a real reset/render check. A valid dataset with
  zero accepted examples correctly returns not-ready rather than starting training.
- A final forward-only fixture returns `[2,8,27]` actions with finite loss and
  unchanged model parameters; optimizer steps and backward calls remain zero.
  Cache tests exercise prepare-cache → frame/chunk loading → forward without
  reading whole NPZ episodes during random sampling.
- Scoped Ruff, diff checks and static TeX structure/citation checks pass. No
  LaTeX engine is installed, so the paper was not compiled to PDF.

Small evidence copies are `collection_manifest.json`, `replay_summary.json`,
`audit_replay_summary.json`, `reset_render_check.json`, `merged_reset_check.json`,
`student_forward_check.json`, and `student_preflight.json`.
`audit_replay_summary.json` records the replay after all cross-review fixes;
the earlier replay is retained for provenance. The canonical simulator/data
bundles remain under out/.

## Additional bug-priority cross-review

After the user explicitly prioritized errors over further implementation, each
pipeline area was reviewed by an agent that had not primarily authored it.
Concrete failing regressions established additional defects:

- Cache-manifest checks alone did not bind an image array to its source NPZ.
  Relabelling train/validation cache arrays could therefore bypass integrity
  checks. Full preflight now verifies each cached member against the authoritative
  source episode, preventing that leakage route.
- Dataset phase validation checked only first occurrences. A return from hold
  to transit, or an unknown phase, could still pass. It now enforces the known
  phase vocabulary and monotonic approach/closure/lift/hold progression.
- Failed-entry identity checks were weaker than accepted-entry checks. Object,
  candidate and condition aliases now must match provenance in both paths.
- Part compliance ignored contacts made by robot links outside the required
  groups. Positive-force contact outside the requested region now invalidates
  compliance even for other fingers, palm or arm. Student replay also requires
  the matching live part annotation and home-start provenance.
- A target name containing `table` could be mistaken for the supporting table.
  Target membership now takes precedence when classifying contact pairs.
- Unsupported/overlapping contact groups and missing collision geometry were
  rejected too late, and squeeze root changes were silently discarded. They now
  fail before execution; equivalent quaternion signs remain valid. Malformed
  candidate entries no longer stop collection of subsequent candidates.
- Checkpoint loading could overwrite fixed normalization and diffusion schedule
  buffers after metadata validation, including zero/NaN scales. Those buffers now
  must match the recorded configuration; invalid execute-step counts and seeds
  are rejected before training readiness can be reported.
- The optional RL branch counted hold at control endpoints only. Intermediate
  contact loss, zero force, height dips or table collisions now interrupt hold.
  Its checkpoint source hashes also include the shared simulation-state helper.
  A real two-step environment smoke check returned finite 140D observations and
  the expected time-limit termination without any training.
  This legacy RL success criterion remains narrower than the schema-3 data gate:
  it does not impose that gate's whole-digit force and target-penetration limits.
  An RL success alone does not qualify a visual training demonstration.

These are software correctness fixes, not evidence that clean grasp execution,
actual DINO inference, model learning or hardware transfer has been validated.
Future implementation tasks and their acceptance criteria are in
[PAPER_TODO.md](../../../../PAPER_TODO.md) (the current replacement for NEXT_STEPS).
