# Training preparation and user launch

Dexonomy remains the geometric teacher. The implemented main path is existing
teacher exports → validated dynamic home approach and contact-feedback execution
→ quality-gated image/action episodes → 27D DexGraspVLA-style diffusion student.
The implementation does not train a replacement teacher. The preparation task
has not run student training, backward passes, or optimizer updates.

Implementation and research readiness are different checks. The collection and
student interfaces exist, but the historical five-contact lift exceeds the
penetration/force limits and loses full contact support in a smaller-timestep
replay. It must not be used as a clean positive. Actual training also needs local
DINOv2 source/weights and compatible Torch/torchvision, which are not bundled.

## 1. Select existing teacher exports

Use the exact object asset root associated with the teacher exports. A selection
manifest preserves original object/type/pose/contact annotations and teacher hash.

```bash
OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py select \
  --grasps third_party/dexonomy/output/evo_hx5_d20/succ_grasp \
  --objects data/objects/evoall --limit 32 \
  --out out/teacher_candidates.json
```

`--screen-yaws` optionally applies an IK feasibility screen. Selection is not
full-path validation. A single fixed diagnostic candidate is already pinned in
`results/2026-09-08/fixed-student/selection.json`.

## 2. Collect full executions

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python \
  scripts/collect_teacher_dataset.py collect \
  --manifest out/teacher_candidates.json --out out/teacher_run01 \
  --max-candidates 32 --max-yaws 3
```

Each candidate tests its own signed-axis home path, then dynamically executes
closure, lift and hold. The opening posture is initialized only at home; IK and
command previews restore the full integration state. A two-control-tick preview
rejects closure commands exceeding the disclosed limits. This is a bounded
search and feedback baseline; failure does not prove that no feasible motion
exists. Camera/action recording uses 50 Hz with physics substep checks.

Closure keeps the 2 N preload until lifting starts; the default carrying target
is 4 N rather than the previous 8 N. Required-site force regulates closure, with
whole-digit overload taking precedence over searching for a missing site. During
lifting, penetration relief begins at `penetration_margin` times the 2 mm hard
limit (default 0.75, hence 1.5 mm). Relief of the combined command also adjusts
persistent geometry offsets; its maximum unloading step is 0.001 rad per joint.
These feedback targets are not additional acceptance guarantees. During contact
acquisition, existing support is preserved before trying a whole-hand release.
During carrying, arm progress and finger changes are previewed separately, and
unloading is tried before freezing the previous command. Contact loss pauses
ascent while finger recovery remains active. Hard quality limits are unchanged.

`--directions x` restricts a diagnostic to one previously studied direction; it
does not establish a universal approach direction. `--execution-config` accepts
an `icra.execution.ExecutionConfig` JSON override, whose values are saved in
provenance. Default physical gains, friction and force limits are unchanged.

`lift_force` is an optional extra force threshold for starting the lift. Its
default `null` disables only that extra gate: all required contacts, the existing
0.1 N minimum, part compliance and 25 consecutive ready intervals still apply.
Set `"lift_force": 1.5` to reproduce the stricter diagnostic. This setting does
not change dataset acceptance. Frozen paired configurations and four same-object,
same-type candidates are in
[the candidate comparison protocol](../results/2026-09-08/candidate-screen/README.md).

Add `--distractor /path/to/one/object` for each intended distractor. Add
`--part-annotation /path/to/annotation.json` for a manually annotated requested
part; [the annotation guide](TEACHER_DATASET.md) specifies coordinates and scale.
Automatic language/point grounding remains external. The same annotation must
identify the selected object; an object mask cannot stand in for its part mask.

Outputs include `attempts.json`, `episode_entries.json`, a sealed `manifest.json`,
`simulation/*.mjb`, complete initial states, and source snapshots. Accepted NPZs
go to `episodes/`; failures go to `diagnostics/`. Pre-execution failures retain
metadata and rejection reasons even when no camera/action episode exists.
Use a fresh output directory for each run and retain rejected attempt counts.
If collection was interrupted after complete episode files were written, seal
those durable records with `python scripts/prepare_teacher_dataset.py recover
out/interrupted_run`. Recovery does not invent missing frames or resume physics
from an unfinished episode.

## 3. Validate and form independent splits

```bash
.venv/bin/python scripts/prepare_teacher_dataset.py validate out/teacher_run01
.venv/bin/python scripts/prepare_teacher_dataset.py prepare \
  --input out/teacher_run01 --input out/teacher_run02 \
  --out out/teacher_training_data --split-by object --seed 0
.venv/bin/python scripts/prepare_teacher_dataset.py validate \
  out/teacher_training_data --require-ready
```

Validation recomputes quality from stored evidence and checks episode integrity,
condition identity, boundaries and leakage. A terminal half-second hold must
preserve every required contact with positive normal force above a 10 cm lift.
Whole-episode limits include 2 mm target penetration, 30 N summed normal force
per whole digit, table/obstacle collision and distractor disturbance. These are
engineering simulation-quality limits, not calibrated hardware ratings.

An empty positive set or one object with no independent validation group is
reported as not training-ready. Repeating that same scene cannot create held-out
objects. Candidate-disjoint splitting is available but tests a weaker claim.

## 4. Check the actual student, without learning

Configure a local copy of `configs/hx5_student.json` following
[HX5_STUDENT.md](HX5_STUDENT.md). First materialize a verified memory-mapped cache:

```bash
python scripts/train_hx5_student.py prepare-cache \
  --data out/teacher_training_data --out out/teacher_training_cache
```

This conversion does not train. It avoids decompressing an entire episode's
images for each randomly sampled training frame. The cache is bound to the sealed
dataset and array hashes; it must be regenerated if that dataset changes.
Then run:

```bash
python scripts/train_hx5_student.py preflight \
  --config configs/hx5_student.local.json --data out/teacher_training_data \
  --cache out/teacher_training_cache
python scripts/train_hx5_student.py dry-run --full-model \
  --config configs/hx5_student.local.json --data out/teacher_training_data \
  --cache out/teacher_training_cache --device cuda:0 --out out/student_full_forward_check
```

Neither command creates an optimizer. The full-model command loads real local
DINO weights and tests loss/action shapes and finite outputs. Without weights,
`dry-run --out out/student_fixture_check` tests a small random patch fixture only;
its report explicitly distinguishes that from a full pretrained-model check.

## 5. Training is the user's action

After the previous checks pass, the user can run:

```bash
python scripts/train_hx5_student.py train \
  --config configs/hx5_student.local.json --data out/teacher_training_data \
  --cache out/teacher_training_cache --device cuda:0 --out out/hx5_student_run01
```

The student consumes RGB, object/optional part masks and 27 measured joints.
Privileged teacher contacts, simulator state and phase labels serve quality
validation; they are not student observations. Normalization uses only training
episodes; action chunks do not cross episode boundaries. Predicted actions are
absolute actuator targets, already including the executor's gravity bias in their
supervision, so deployment must not add that compensation a second time.
Use held-out closed-loop evaluation after training; diffusion loss is not grasp
success or proof of transferred contact knowledge.

To verify a collected teacher trajectory before any training, replay its exact
commands without rendering or loading a policy:

```bash
OPENBLAS_NUM_THREADS=1 third_party/dexonomy/.venv/bin/python \
  scripts/replay_teacher_dataset.py --data out/teacher_run01 \
  --episode teacher_0000_yaw_000 --out out/teacher_command_replay
```

The replay checks pre-action states, timestamps and every stored physics-quality
signal against the archived model and initial state. It does not independently
verify the stored images. `replay_hx5_student.py --check-reset-only` separately
checks archived model/state restoration and camera rendering without learning.
