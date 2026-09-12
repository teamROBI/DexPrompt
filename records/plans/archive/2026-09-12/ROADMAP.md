# 과거 상태 — 현재 실행 지시 아님

2026-09-12 문서 정리 전 내용이다. [현재 계획](../../../../RESEARCH_PLAN.md)을 우선한다.

# Research direction and evidence plan

## Research question and scope

Can an existing dexterous grasp expert's type and contact knowledge improve a
visual policy's requested-part grasping when supervision comes from synthetic
execution instead of new real-robot grasp demonstrations?

Dexonomy is the teacher; synthetic demonstrations are the transfer medium;
a DexGraspVLA-style controller is the student. The initial setting is the
FFW-SH5 right arm and HX5-D20 hand on a table with a few distractors.
The contribution must concern effective knowledge transfer and measured benefit.
Connecting two existing systems or changing the teacher is insufficient.

ICRA is the preferred venue fit for an embodied method with controlled simulation
and hardware evidence. ICLR would require a broader learning insight supported
beyond this pipeline. This is a scope judgment, not an acceptance prediction or
lower standard of rigor. No submission year or deadline is fixed.

## Prompt semantics

| Request | Intended behavior | Required evidence |
|---|---|---|
| Object text, object mask, or selection point | Select the object and choose an accessible, appropriate region. | Target correctness and sustained lift; assess preferred-region selection separately. |
| Explicit part text or part mask | Constrain the grasp to that region. | Correct target, sustained lift, and compliant contacts. |
| Contact point | Constrain contact to a defined neighborhood in a specified frame. | Region adherence; distinguish this mode from object selection. |

For “grasp the watering can,” an accessible handle is a desired semantic
preference, not a Dexonomy guarantee. For “grasp the handle,” it is a task
constraint. Grounding and training examples must express that difference.
Successful part grasping does not prove functional use such as pouring.
“Anything” is a long-term ambition: demonstrate held-out geometries, parts, and
layouts within the robot's workspace and hand capabilities, and define how
infeasible requests are treated.

## Prior work and positioning

* [Dexonomy](https://arxiv.org/abs/2504.18829) supplies taxonomy-based grasp
  synthesis, configurations, and contacts. Reuse this expertise; the main teacher
  is not a new policy learned from scratch.
* [DexGraspVLA](https://arxiv.org/abs/2502.20900) supplies the hierarchical
  planner/controller reference. Its visual controller does not take raw text;
  object masking alone cannot distinguish different part requests on one object.
* [ClutterDexGrasp](https://arxiv.org/abs/2506.14317) learns a simulation RL
  teacher and distills it into a partial-point-cloud DP3 student. It already
  addresses closed-loop target-oriented grasping with sim-to-real transfer.
* [DemoFunGrasp](https://arxiv.org/abs/2512.13380) uses one grasp demonstration
  and demonstration-editing RL conditioned on style and affordance. It is close
  prior work for region/style control, not pure learning from scratch.
* [UniDexGrasp](https://arxiv.org/abs/2303.00938) connects generative grasp
  proposals to execution policies. Pose-to-policy transfer itself is not new.
* [DexVLG](https://arxiv.org/abs/2507.02747) addresses semantic dexterous grasp
  prediction. Part/language-conditioned grasp poses are not new by themselves;
  distinguish pose prediction from executed visual-policy behavior.
* [GraspVLA](https://arxiv.org/abs/2505.03233) supports synthetic action
  pretraining. Rendering and domain randomization matter; pretrained visual
  features alone do not establish sim-to-real transfer.

These are supporting precedents and comparison requirements. There is no basis
for an “unoccupied synthetic + target-conditioned + closed-loop corner” claim.

## Method implementation and remaining experiments

1. **Prepare the teacher.** Retarget annotated templates and refine
   object-specific grasps with Dexonomy. Preserve source type, object identity,
   scale, frames, pregrasp/grasp/squeeze configurations, and available contacts.
   Current anatomical correspondences are hand-specific; cross-embodiment claims
   require another hand and its evaluation.
2. **Associate goals and grasps.** Explicit manual object-local box regions now
   define shared labels for visible pixels and physical contact points. The
   collector checks scale/bounds and requires the annotated teacher groups to
   contact the requested region. These are supplied annotations, not semantic
   labels inferred by Dexonomy. Automatic text/point grounding and efficient
   region-compatible candidate selection remain open.
3. **Produce executable demonstrations.** Validate approach, closure, and
   sustained lift with the arm and table. Preserve teacher contacts and type
   where feasible; record adaptations, original/adjusted candidate IDs, and
   rejection reasons. The implemented collector tests complete home approaches,
   constrains feedback commands with multiple control intervals of simulation
   preview, and exports only trajectories passing schema-3 quality gates as
   positives. A lift after changing the grasp does not automatically demonstrate
   preservation of the original knowledge, so record what was preserved rather
   than assuming the type survived.
4. **Train the student, when prerequisites pass.** A workspace-owned
   DexGraspVLA-style adaptation now combines frozen dual DINOv2 features,
   separate object/part/presence channels, normalized 27-joint state, and an
   RDT-style diffusion action transformer. It predicts 64 absolute joint targets
   at 50 Hz; the default executed prefix is 8. Training, checkpoint inference,
   and evaluation entry points exist. Text and points enter through grounding.
   Dexonomy is training-only under this design.
5. **Measure transfer.** Evaluate requested-part success and contact/type fidelity
   at teacher, demonstration, and student stages. Good poses or demonstrations
   alone do not establish a good learned policy.

IK and scripted phases are the baseline executor. An experimental privileged
PPO executor and hand-only ladder now exist; their audit checks validate code
interfaces, not successful learning or the proposed visual knowledge transfer. Feedback/replanning or residual
RL are options if a fresh corrected baseline shows a limitation. If RL is used,
include its compute/interaction cost and compare against corrected non-RL
execution. Scripted trajectories can train a visually closed-loop student;
recovery and perturbation coverage still need explicit testing.

## Experiments that decide the claim

| Question | Controlled comparison | Readout |
|---|---|---|
| Does diversity help? | Same student/budget, limited or type-unaware candidates vs diverse Dexonomy candidates. | Held-out requested-part lift success and verified diversity. |
| Does preserving knowledge help? | Pose-only execution vs the implemented contact/type-informed transfer mechanism. | Contact/type fidelity and downstream success. |
| Is the condition used? | Same object/scene, change only the requested part; object-only vs part input. | Behavior change and contact compliance. |
| Where are errors introduced? | Oracle part grounding vs predicted grounding. | Separate grounding, executor, and student failures. |
| Does it generalize? | Held-out objects/part combinations; vary distractors. | Per-split success, disturbance, and uncertainty. |
| Does it transfer to hardware? | Frozen sim-trained student, disclosed calibration. | Goal-conditioned success without new real grasp demonstrations. |

Match student architecture, optimization budget, scene distribution, and training
episode counts for teacher ablations. Report synthesis attempts, simulation steps,
filtering/rendering cost, and training compute as well: equal episode counts
do not imply equal collection cost. Direct teacher execution is a separate
diagnostic baseline. Alternative grasp sources must each produce executed
episodes before their students can be fairly compared.

Split by source object geometry **before synthesis**. Different scales, textures,
or views of the same mesh must not cross train/test. Define held-out categories,
part/object combinations, and layouts explicitly. Use multiple seeds and
confidence intervals; retain failed attempts and their denominators.

The primary metric is the joint event: correct object, sustained lift, and
required part-contact compliance. Report each component, table/robot collisions,
and distractor movement separately. A minimum required-digit check does not
certify an exact taxonomy label. Fix contact tolerance, lift height, hold
duration, feasibility rules, and failure categories before held-out evaluation.
Test feedback behavior under controlled disturbances rather than inferring it
from architecture.

Heavy clutter is secondary robustness evaluation. Additional hands, downstream
tool use, and more student families should follow only for a specific claim.
Report real-data fine-tuning separately from the no-new-real-demonstration setting.

## Evidence gates, without a calendar

1. **Reproducible execution:** keep scale, frames, geometry, and documented
   actuation consistent; reproduce a known candidate from hand-only to table to
   full-arm execution. Rerun the corrected baseline before replacing the executor.
2. **Small end-to-end student:** record a clean small set and make the adapted
   student reproduce training tasks. This tests learning and interfaces, not
   generalization. Do not postpone it for indefinite generator optimization or
   the separate DexLearn branch.
3. **Knowledge-transfer test:** validate the implemented contact/part gating and
   student interfaces with matched ablations on held-out tasks. Revise the method or claim if gains
   disappear under fair controls.
4. **Embodied evidence:** evaluate part counterfactuals, mild clutter,
   perturbations, and hardware. Write results when reproducible.

Where each gate stands, and what would close it, is kept in
[PAPER_TODO.md](../../../../PAPER_TODO.md); the measurements are in `results/`, one directory
per experiment, including the runs that failed. [paper/rules.md](../../../../paper/rules.md)
governs claims in the evolving draft.
