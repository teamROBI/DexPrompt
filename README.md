# DexPrompt — Dexonomy knowledge transfer for visual dexterous grasping

**새 RTX 3090 서버에서 시작:** [설치·파일 전송·tmux 평가 안내](SERVER_3090.md).
코드는 Git으로, 기존 시범·DINO 가중치·학습 체크포인트는 별도 검증 가능한 묶음으로 옮깁니다.

This project studies whether Dexonomy's grasp types, hand configurations, and
contact structure can improve a DexGraspVLA-style visual policy through synthetic
demonstrations. The intended task is grasping a requested object or part among a
few distractors, without new real-robot grasp demonstrations for the ROBOTIS
FFW-SH5 and its HX5-D20 hand.

This is an active research implementation, not a validated end-to-end system.
The first embodiment has 7 right-arm and 20 finger action dimensions. Submission
year and deadlines are undecided and do not determine the research scope.

## Intended pipeline

~~~text
Training
Dexonomy templates and object-specific grasp synthesis
  -> candidates, grasp types, and contact information
  -> scene-feasible approach / close / sustained lift
  -> images, target conditions, robot states, and action demonstrations
  -> imitation learning of a DexGraspVLA-style visual controller

Deployment
Text / Point / Mask -> object and optional part grounding
  -> head and wrist images + target condition + robot state
  -> visual controller -> arm and finger actions
~~~

Dexonomy is the training-time teacher: an existing grasp synthesis expert, not
necessarily a neural policy whose weights are distilled. Its geometric knowledge
does not itself provide language understanding or semantic part labels. The
proposed main path does not query Dexonomy at deployment.

The primary scope is **part-conditioned dexterous grasping in mild clutter**.
An object selection lets the system choose an appropriate feasible region; an
explicit part request constrains that choice. A point selecting a watering can
differs from a point specifying contact on its handle. Handle grasping does not
establish pouring ability. Heavy clutter and additional hands are extensions.

## Implementation status

| Component | Current state |
|---|---|
| Hand preparation | Template retargeting, HX5-D20 assets, contact annotation, and per-type calibration exist; arbitrary-hand automation is not established. |
| Dexonomy synthesis | Initialization, ADMM refinement, evaluation, and evolution tooling exist. Stored successes need execution-specific revalidation. |
| Execution and recording | A fresh-home collector screens complete dynamic approaches and uses contact feedback with multi-tick quality previews. Schema-3 export retains per-physics-step evidence, teacher provenance, compiled models and initial states. Verified demonstrations exist on a bottle and a can, including a second can yaw. More objects and grasp diversity remain a bottleneck ([multi-object record](results/2026-09-09/multiobject/README.md), [expansion](results/2026-09-10/overnight-teacher-expansion/README.md)). |
| Privileged RL | The arm's approach is learnable on the fixed task: behaviour cloning from nine verified demonstrations completes from the home pose, and PPO continued from it beat that baseline on one training seed of three ([approach transfer](results/2026-09-09/approach-bc-v1/README.md), [seed comparison](results/2026-09-09/kl-and-seeds/README.md)). This is a 7-joint privileged policy on one object, not the visual student. |
| Part conditioning | Explicit manual object-local box regions provide separate visible part masks and physical contact checks. Automatic text/point grounding and semantic part recognition remain pending. |
| Student | The full DINO + 27-dimensional diffusion student completed its first BC run: four bottle/can demonstrations, three epochs, 6,255 updates. Closed-loop GPU evaluation is pending; training loss is not evidence of grasp success. See [training record](results/2026-09-10/visual-student-bc-v1/README.md) and [server migration](SERVER_3090.md). |
| Evaluation | Controlled teacher ablations, held-out policy tests, and real-robot transfer remain pending. |

Scene search and GraspGenX are alternative candidate generators and diagnostic
tools. They do not replace Dexonomy as the agreed main teacher. RL is an optional
executor improvement if corrected scripted or feedback execution cannot provide
adequate demonstrations, not a prerequisite or a proven necessity.

## Data and training interface

The current training path uses schema-3 episode NPZ files, provenance JSON, and
a sealed dataset manifest. Accepted episodes live in episodes/; rejected
rollouts stay in diagnostics/. Each episode records:

~~~text
action, right_state       (T, 27)          absolute targets, measured joint positions
rgbm                      (T, H, W, 4)    head RGB + target-object mask
right_cam_img             (T, H, W, 3)    wrist RGB
timestamp, control_dt     (T,)            pre-action time and 0.02 s interval
part_mask                 (T, H, W)       optional separately annotated part
contact/force/depth/height evidence        conservative interval minima/maxima
~~~

Images and states precede the action. Quality evidence covers every physics step
during its 20 ms interval, so a transient contact loss or force peak is retained.
Positive demonstrations require a clear home start, the complete approach, and
a terminal contact-preserving hold while satisfying whole-trajectory quality
limits. Legacy Zarr stores lack this evidence and remain diagnostic.

The workspace-owned student adapts frozen dual DINOv2 encoders and RDT-style
action diffusion to 27 joints, separate object/part masks, and training-only
normalization. It predicts 64 targets and executes a configurable prefix, default
8, before observing again. This is an architectural adaptation, not an upstream
policy-checkpoint conversion or an implemented language planner.

See [teacher data preparation](configs/TEACHER_DATASET.md) and
[student configuration and preflight](configs/HX5_STUDENT.md). Training is left
to the user. A passing code fixture is not training readiness: accepted data,
local pretrained DINO assets, and a full-model forward check are still required.

Scripted demonstrations can train a student that reobserves and replans during
execution. Recovery ability depends on training coverage and must be tested;
scripted data do not logically force an open-loop policy.

## Evidence and navigation

Pre-fix arm execution rates and the static-contact “19.5 mm tolerance” claim are
withdrawn. They do not show that pose-based execution fails or that RL is needed.
Stored grasp counts are not learned-policy success rates.

* [PAPER_TODO.md](PAPER_TODO.md): where the work stands against the goal, what
  counts as evidence for each part, and the next tasks. Start here.
* [Training preparation workflow](configs/TRAINING_PREPARATION.md): collection,
  quality validation, image cache, forward checks and the user's training command.
* [Preparation verification](results/2026-09-08/training-preparation/README.md):
  actual collection/replay, corrected errors and remaining training prerequisites.
* [ROADMAP.md](ROADMAP.md): hypotheses, related work, method, and experiments.
* [SETUP.md](SETUP.md): dependencies and bounded smoke runs.
* [Historical measurements](results/2026-09-07/README.md): logs and interpretation limits.
* [Audit verification](results/2026-09-08/audit/README.md): bounded code checks and remaining limits.
* [Contact physics audit](results/2026-09-08/contact-physics/README.md): reset/solver-state bugs, force semantics, and timestep-sensitive contact preservation.
* [2026-09-09 results](results/2026-09-09/README.md): approach transfer by behaviour
  cloning, the PPO comparisons around it, and the multi-object screen.

Every experiment keeps its own directory under `results/<date>/`, including the
runs that failed and the conclusions later withdrawn. Corrections are written
into the report that made the claim rather than removing it.

### Documents

| File | What it is for |
| --- | --- |
| [PAPER_TODO.md](PAPER_TODO.md) | Live status against the research goal, and the next tasks |
| [ROADMAP.md](ROADMAP.md) | Research question, prompt semantics, method and evidence plan |
| [SETUP.md](SETUP.md) | Checkouts, generated assets, bounded smoke runs |
| [configs/](configs) | Workflow guides: dataset schema, training preparation, student setup |
| [results/](results) | One directory per experiment, with its records and reproduction commands |
| [paper/](paper) | Manuscript draft and the venue rules that govern its claims |

`HANDOVER.md`, `NEXT_STEPS.md` and `RESUME.md` were session snapshots of an
earlier state -- a fixed Power-Sphere candidate that stalled at 8.42 cm, and a
student pipeline with no accepted demonstrations. Both statements are now wrong,
and what they were for is covered by PAPER_TODO.md and the reports under
`results/`. They were removed rather than left to be read as current; the text
remains in the git history.
* [Contact-feedback control](results/2026-09-08/contact-feedback/README.md): repeated five-group lifts, force/penetration measurements, and a failed constrained-control test.
* [Contact and full-approach validation](results/2026-09-08/contact-approach/README.md): corrected candidate-specific approach reproduces full execution; contact preservation fails.
* [Fixed student experiment](results/2026-09-08/fixed-student/README.md): synthetic close/lift, visual imitation, and closed-loop replay; no part/contact-transfer claim.
* [Paper draft](paper/dexprompt.tex) and [writing rules](paper/rules.md).

“No new real-robot grasp demonstrations” permits existing human-annotated
templates and pretrained foundation models. It does not mean no human
information, no calibration, or already demonstrated zero-shot sim-to-real.
