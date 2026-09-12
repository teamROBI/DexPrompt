<!-- research-scope-2026-09-12 -->
> **현재 범위 (2026-09-12):** [두 단계 연구 정의](../RESEARCH_PLAN.md)를 따른다.
> ① 정확한 형상 기반 Dexonomy 파지 선정·시범 적응 학습/자동 수집, ② 물체 선택 조건부 시각 행동 BC.
> 아래 part API/부위 마스크는 선택적 호환 기능이며 필수 과제가 아니다. mask·DINO는 DexGraspVLA 설계와 양립한다.
> 날짜가 붙은 실행/미실행 상태와 명령은 당시 기록이다. 현재 완료 여부와 재개는 RESUME의 원격 산출물로 확인한다.

# HX5 시각 BC 구현과 설정

현재 핵심 과제는 물체 선택 조건부 행동 BC다. 아래 part 채널은 선택적 호환 기능이다.
2026-09-10 이후 lecun에서 학습을 실행했다. 최신 작업 위치와 확인된 상태는
[RESUME](../RESUME.md)를 따른다. 다음은 날짜별 검증 설정 및 재현용 API 설명이다.

## 2026-09-10 병·캔 진단 설정 기록

당시 사용한 설정은
`configs/hx5_student.bottle-can.local.json`, 데이터는 `out/student-bottle-can-v1`, cache는
`out/student-bottle-can-v1-cache`다. 참조 품질 검사·통합 데이터·전체 평가 명령은
[학습 준비 기록](../records/experiments/2026-09-10/student-training-ready/README.md)에 고정했다.
해당 준비 검사 자체는 학습하지 않았다. 이후 학습 이력과 구분한다. 아래는 기존 v2 설정과
일반 API 설명이며, 재현 시 해당 실험 config의 참조 기준을 따른다.

기존 시각 입력 v2 구현/검증: `.venv-student/bin/python`과
`--config configs/hx5_student.visibility.local.json`을 사용한다. 수집에는
`collect_teacher_dataset.py collect --visual-profile configs/visual_student_256.json`을
추가한다. 이 프로필은 원본 256×256와 head 하향 20°를 명시적으로 선택하며, 기존
전역 카메라 설정을 바꾸지 않는다. 실제 장착/보정값과 다물체 검증은 별도다.

v2는 head/wrist 물체·부위 mask, 카메라별 가시성, 요청별 목표 기준 관측을 연결한다.
요청마다 각 카메라에서 처음 보인 목표 crop·bbox·측정 관절 상태를 보존하고, 새 요청이나
reset 때 초기화한다. 기존 설정의 첫 가시성은 요청 mask가 최소 1픽셀 있는 조건이며, 고품질 목표
인식을 보장하는 임계값은 아니다. 첫 wrist 관측을 그보다 이른 학습 프레임에 넣지 않는다.
새 요청의 기준 관측이 양쪽 모두 없으면 명령을 내지 않는다. 버전은 모델 config의
`observation_version: 2`, `goal_image_size: 224`로 고정한다.

검증 데이터는 `out/student-visibility-v2`, cache는 `out/student-visibility-v2-cache`다.
실제 DINO 전체 모델에서 양쪽 현재 mask가 모두 0인 관측으로도 유한 손실과 행동 출력을
확인했다. 이는 당시 무학습 검사의 결과이며 이후 학습 정책의 성능을 뜻하지 않는다. [구현/검증 보고서](../records/experiments/2026-09-09/visual-goal-v2/README.md),
[입력 계약](target-contract.md)에 실행 명령과 남은 조건을 고정했다.

```bash
.venv-student/bin/python scripts/student/train_hx5_student.py preflight \
  --config configs/hx5_student.visibility.local.json --data out/student-visibility-v2 \
  --cache out/student-visibility-v2-cache
.venv-student/bin/python scripts/student/train_hx5_student.py dry-run \
  --config configs/hx5_student.visibility.local.json --data out/student-visibility-v2 \
  --cache out/student-visibility-v2-cache --full-model --device cpu \
  --sample-index 1240 --sample-index 2067 --out out/NEW_visual_v2_forward
```

해당 작은 진단 데이터의 기본 preflight에는 독립 validation 물체가 없다는 사유가 남았다.
`--allow-train-only-diagnostic`는 한 물체 진단을 명시적으로 허용한다. 목표가 초기에
보이지 않아 기준 관측을 얻을 수 없는 에피소드는 이 옵션과 무관하게 home 학습에서
거부한다. v2는 wrist mask 없는 구 데이터를 자동으로 채워 사용하지 않는다.
`scripts/data/prepare_visual_dataset.py`로 원본 물리를 재현하며 새 디렉터리에 재렌더링하거나,
새 수집 프로필로 시범을 모아야 한다. 일반 v1 모델과의 차이는 실험 조건으로 구분한다.

아래는 기존 v1 경로의 설명과 명령이다. v2에 사용할 때는 위 config와 v2 데이터/cache를
지정한다. v1 구조·기존 데이터 입력은 유지되며, 두 버전의 정책 state dict는 서로
호환되지 않는다. 기존 체크포인트의 코드 해시 검사는 그대로 유지한다.

Local v1 setup verified on 2026-09-09: use `.venv-student/bin/python` together with
`--config configs/hx5_student.local.json`. Both official pretrained DINOv2
backbones strictly load, and the full student completed a forward-only check
with actual accepted data. See [setup and evidence](../records/experiments/2026-09-09/dino-setup/README.md).
That dated setup check did not train the student. Later training is tracked in
RESUME; independent geometry splits remain necessary for generalization claims.

The HX5 training entrypoint is implemented in `scripts/student/train_hx5_student.py`.
Actual training readiness requires accepted schema-3 demonstrations, a verified
memmap cache, and configured local DINOv2 code/weights. Preflight and dry-run
are forward-only checks; the separate train command updates the action model.
Do not interpret successful setup as successful closed-loop visual grasping.

`hx5_student.json` configures a workspace-owned adaptation of the DexGraspVLA
controller: frozen DINOv2 ViT-B/14 head and ViT-L/14 wrist encoders, mask feature
fusion, a 12-layer RDT-style action diffusion transformer, 64 future absolute
joint targets, and a 50-step cosine diffusion schedule. Actions and states contain
7 right-arm and 20 HX5 finger joints in the dataset manifest order. The proposed
receding-horizon execution uses the first 8 targets before receiving another
observation; the model returns all 64 targets. No robot is commanded by training.

This implements the controller architecture, not the complete upstream language
planner or an upstream pretrained policy checkpoint conversion. It reimplements
the RDT attention blocks and DDIM sampler in native PyTorch. Unlike the reference,
it preserves MuJoCo RGB ordering, reads 27 state values, handles variable patch
counts, normalizes state from training episodes only, and excludes padded targets
from the loss. The DDIM schedule spans the full configured diffusion interval;
this differs from the reference scheduler's default timestep spacing. Image
augmentation and the upstream attention-dropout mask ablation are not implemented.
Language/point inputs must already have been grounded to the supplied object or
annotated part; this controller does not infer those labels.

The head RGB object mask always remains a distinct channel. Optional part masks
and a part-presence indicator enter separate mask encoder channels. Part examples
must carry dataset provenance and measured requested-part contact compliance.
An unannotated object mask is never silently used as a part mask.

Setup and checks:

1. Install `requirements-student.txt` into the environment you choose for training.
   Use a mutually compatible PyTorch/torchvision build for your GPU. The project
   does not install packages or fetch weights automatically.
2. Obtain a local official DINOv2 checkout and the matching `dinov2_vitb14` and
   `dinov2_vitl14` pretrained state dictionaries. In a copy of `hx5_student.json`,
   set `vision.source`, `vision.head_weights`, and `vision.wrist_weights` to these
   local paths. Loading is strict, local-only, and never falls back to random vision.
3. Prepare and validate schema-3 accepted demonstrations. The default split is
   object-disjoint. Repeated frames or resets of one candidate cannot supply
   independent validation groups. The old fixed-student and penetration-violating
   diagnostic trajectories are not accepted automatically.
4. Materialize the accepted student inputs into an explicit new cache directory.
   This prepares data only and performs no learning:

```bash
python scripts/student/train_hx5_student.py prepare-cache \
  --data out/teacher_dataset --out out/teacher_student_cache
```

5. Run preflight and a forward-only check before launching training:

```bash
python scripts/student/train_hx5_student.py preflight \
  --config configs/hx5_student.local.json --data out/teacher_dataset \
  --cache out/teacher_student_cache
python scripts/student/train_hx5_student.py dry-run \
  --config configs/hx5_student.local.json --data out/teacher_dataset \
  --cache out/teacher_student_cache \
  --full-model --device cuda:0 --out out/student_full_forward_check
```

Preflight validates all persisted quality signals, episode hashes, metadata,
split leakage, local dependency availability, and the model paths. It returns
exit code 2 when prerequisites are missing. The full forward check additionally
loads the actual DINOv2 models and checks finite loss and action predictions with
no backward call, optimizer, or parameter updates. Local DINOv2 code imports can
still reveal extra dependency or checkpoint compatibility problems at this step;
file-only preflight does not prove those models can execute.

The cache contains separate read-only `.npy` memory maps for accepted images,
states, actions, timing and optional part masks. Its manifest binds the original
dataset manifest, source episode archives, and every cached array by SHA-256.
Preflight verifies array hashes once; shuffled samples then read only a selected
image/state and action chunk. Cache preparation validates source quality and
streams each compressed NPY member to disk. It consumes uncompressed storage,
whose byte count is reported. Rejected diagnostics and physics arrays are excluded.
Training requires `--cache`; the previous two-episode NPZ fallback could repeatedly
decompress entire image sequences under random sampling. Uncached access remains
available for small forward-only diagnostics. Cache files are never generated,
modified or silently repaired during training.

For a small CPU architecture/shape fixture independent of pretrained weights:

```bash
python scripts/student/train_hx5_student.py dry-run --out out/student_fixture_check
```

That command uses synthetic observations and a small random patch encoder in the
same diffusion implementation. It cannot establish full DINOv2 execution,
successful grasping, useful learned actions, or dataset quality. Supplying `--data`
uses accepted real records for the forward check while retaining the explicitly
labelled fixture encoder, unless `--full-model` is also passed.

The user can start training once preflight and the full-model check pass:

```bash
python scripts/student/train_hx5_student.py train \
  --config configs/hx5_student.local.json --data out/teacher_dataset \
  --cache out/teacher_student_cache \
  --device cuda:0 --out out/hx5_student_run01
```

Each run requires a new output directory. It saves configuration, train-only
normalization, ordered joints, dataset/cache/code/pretrained-weight hashes, per-epoch
loss, and checkpoints with optimizer/scheduler state. The current entrypoint
starts a fresh run; automatic resume and distributed training are not implemented.
The batch size is 8 by default and should be adjusted to available GPU memory.
Validation reports diffusion loss, not task success; closed-loop evaluation is
still necessary after training. `--allow-train-only-diagnostic` permits an explicit
one-group memorization run but does not turn it into a generalization experiment.

After user-launched training, evaluate recorded observations separately from actual
task execution:

```bash
python scripts/student/evaluate_hx5_student.py \
  --checkpoint out/hx5_student_run01/epoch_0124.pt --data out/teacher_dataset \
  --cache out/teacher_student_cache \
  --split test --device cuda:0 --out out/student_offline_test
python scripts/student/replay_hx5_student.py \
  --checkpoint out/hx5_student_run01/epoch_0124.pt --data out/teacher_dataset \
  --episode YOUR_HELD_OUT_EPISODE_ID --device cuda:0 --out out/student_closed_loop
```

The first command reports diffusion loss and first-action error on recorded
observations. The second restores the exact archived MJB model and complete
initial integration state, then commands the student in closed loop. It reports
terminal contact-supported lift, teacher/part compliance, penetration, force,
and actuator saturation. It uses the collection MuJoCo version and validates MJB,
initial-state and annotation hashes. Repeated runs share the same physical reset
and use different diffusion sampling seeds; they are not object/pose perturbations.
Evaluation contract `contact_acquisition_guard_v2` checks target XY displacement
until all required contacts exceed the force threshold continuously for the
configured minimum hold duration. Contact evidence is used only for evaluation;
the policy receives no teacher phase labels or contact feedback. Once acquisition
is measured, carrying displacement is permitted. Other quality limits apply
throughout. The old whole-rollout XY guard incorrectly rejected two recorded
successful demonstrations; its results are not interchangeable with this contract.
Its safety intervention
aborts on a measured violation and applies no teacher contact/IK corrections.
Full-split recorded-observation evaluation requires `--cache`; uncached evaluation
requires an explicit `--max-samples` limit and is labelled a small diagnostic.

Both evaluations check checkpoint/data joint ordering and training overlap.
`--allow-training-overlap` explicitly labels a memorization diagnostic. Local DINO
paths may be relocated with a `--vision-paths` JSON containing `source`,
`head_weights`, and/or `wrist_weights`; weight hashes must still match. Changes to
model/preprocessing code require explicit `--allow-code-change` and are reported.

Saved simulator resets can already be checked without training or a checkpoint:

```bash
python scripts/student/replay_hx5_student.py --data out/teacher_dataset \
  --episode YOUR_ARCHIVED_EPISODE_ID --check-reset-only --out out/student_reset_check
```

`icra.student_inference.load_checkpoint` and `StudentActor` expose the same policy
interface for additional simulation evaluations. Call `actor.reset()` for every
environment reset. `command_recorder` sends absolute actuator targets directly:
the demonstrated arm targets already include gravity compensation. It clips to
the simulator actuator limits and records clipping in `last_command_clipped` and
`saturated_ticks`, then leaves physics advancement/contact measurements to the
caller. Never apply the teacher's gravity compensation a second time.
