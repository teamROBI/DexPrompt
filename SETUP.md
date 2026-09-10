# Setup and bounded smoke runs

**기존 student 체크포인트를 새 서버에서 평가하려면** [SERVER_3090.md](SERVER_3090.md)를 먼저 읽으세요.

This is a research workspace with external checkouts and generated local assets.
The current collection-to-student workflow is documented in
[the training preparation guide](configs/TRAINING_PREPARATION.md). Collection,
dataset validation, preflight and forward-only checks do not start learning.
A fresh-machine bootstrap still requires the external assets listed below.
[PAPER_TODO.md](PAPER_TODO.md) distinguishes implemented code from measured success.

## Checkouts and generated assets

The local layout is:

~~~text
third_party/dexonomy                    -> sibling Dexonomy checkout
third_party/robotis_mujoco_menagerie     upstream robot geometry/model
third_party/dexgraspvla                 student reference implementation
third_party/dexlearn                    optional point-cloud training branch
third_party/graspvla                    optional reference
modules/graspgenx                       -> sibling GraspGenX checkout
data                                   -> bulk storage
~~~

For a fresh workspace, clone the required repositories only if their destination
does not already exist:

~~~bash
mkdir -p third_party modules
git clone https://github.com/JYChen18/Dexonomy.git ../Dexonomy
ln -s ../../Dexonomy third_party/dexonomy
git clone https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie third_party/robotis_mujoco_menagerie
git clone https://github.com/Psi-Robot/DexGraspVLA.git third_party/dexgraspvla
git clone https://github.com/NVlabs/GraspGenX.git ../GraspGenX
ln -s ../../GraspGenX modules/graspgenx
~~~

GraspGenX's proposal route is optional, but current calibration/search code also
uses its gripper asset directory. Optional DexLearn experiments use
https://github.com/JYChen18/DexLearn. Follow each upstream checkout's installation
instructions and record its commit; an unpinned clone is not a reproducible run.

The upstream ROBOTIS source is retained. The simulated robot is modified at build
time in src/icra/scene.py, including collision geometry and actuation. Generated
Dexonomy hand assets/configuration are also local additions. Preserve those
alongside source revisions.

Create data as a directory or a link to an existing storage location. Do not
replace an existing data link. Generated models under model/graspgen, scene.yaml,
templates/calibrated grippers, and outputs are ignored or external and must be
copied or regenerated explicitly when moving machines.

## Environments

Keep incompatible dependencies in separate environments and run stages in their
own processes. From the project root, the project package can be installed with:

~~~bash
uv venv --python 3.10 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -e ".[dev]"
~~~

pyproject.toml declares only the core MuJoCo/NumPy dependencies plus development
tools. Asset generation, rendering, synthesis, and recording need additional
dependencies from their respective environments; this install alone is not
sufficient. The existing workspace uses Dexonomy's interpreter for most such
scripts:

~~~text
third_party/dexonomy/.venv/bin/python
third_party/dexonomy/.venv/bin/dexrun
~~~

Install Dexonomy using its checkout instructions, and configure Torch/CUDA for
the actual driver. Versions recorded on the development machine included
Torch 2.6.0+cu124, MuJoCo 3.3.3 in Dexonomy, and warp-lang 1.8.0. These are local
environment observations, not universal compatibility requirements. Separate
MuJoCo versions still require matching recorded model/control settings when
comparing outcomes.

Use a working EGL driver for GPU rendering and set MUJOCO_GL=egl. Verify an
actual rendered frame; a successful import does not test the renderer.
A CPU operation setting does not guarantee all Dexonomy dependencies avoid CUDA.

DexLearn is optional and is not the planned visual student. Its local setup used
nflows 0.14 and a DensePointNet shim after a MinkowskiEngine build failure.
That local failure does not imply MinkowskiEngine can never build, and the
replacement's accuracy has not been compared. DexGraspVLA has its own
requirements. The workspace now supplies a 27-dimensional architectural
adaptation in `icra.student`, with a user-invoked training entrypoint. Install
`requirements-student.txt` in the chosen training environment and configure
local DINOv2 source and weights as described in [HX5_STUDENT.md](configs/HX5_STUDENT.md).
The current simulation interpreter lacks torchvision and configured DINO weights;
passing a tiny fixture forward check does not remove these requirements.

## Objects and scene manifests

Dexonomy's object assets are available from
[its dataset repository](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy).
The local workspace uses DGN_5k_processed.tar.gz and DGN_5k_vision.tar.gz.
The original Shadow joint trajectories cannot be executed directly on HX5-D20;
this does not prevent transferring their grasp knowledge through retargeting.

The local raw directory had 5751 object entries, while valid_split listed
4557 train and 1140 test entries. Do not treat those as a partition of all raw
entries; use the manifests and check required files. Stable-pose counts should be
read per object, not assumed to be six or eight universally. Split source
geometries before synthesis; rescaling a training mesh does not make a test object.

src/icra/objects.py reads both DGN processed-object directories and the layout
written by scripts/build_object_assets.py. DGN synthesis configs and execution
assets must refer to the same object, scale, and frame. The collector now rejects
known scale disagreements. Generation and loading now share the OBB-based scale
selection in objects.load; old generated sets can still disagree.

A dataset root is not a scene. `scene.build` includes every supplied directory,
at default 0.18 m spacing. The new `collect_teacher_dataset.py` builds one target
scene per candidate and adds only explicit `--distractor` directories. The
legacy collector still needs a bounded flat object root.

## Hand and scene prerequisites

Before retargeting or recording, verify the following exist:

* Exported hand URDF and its meshes under model/graspgen/hx5_d20_right.
* Dexonomy target-hand right.xml plus source Shadow annotations.
* Exported full-robot URDF and the saved model/graspgen/ffw_sh5/scene.yaml.
* Calibrated per-type gripper assets under the configured GraspGenX asset root.
* Object meshes, convex decomposition, scale, and stable-pose data.

Relevant preparation scripts are export_hand_urdf.py, export_ffw_urdf.py,
onboard_dexonomy_hand.py, decompose_palm.py, and scene_setup_gui.py.
Inspect their arguments and local mesh requirements before running them.

There is a bootstrap dependency: retarget_templates.py loads the target hand's
right.xml, while onboard_dexonomy_hand.py writes that model and reads a pose
file. A bare checkout must first provision the target model and a bootstrap pose
file (or restore the existing generated assets). The retargeting sequence below
is for an already provisioned hand, not a clean bootstrap:

~~~bash
third_party/dexonomy/.venv/bin/python scripts/retarget_templates.py \
  --poses out/retargeted_poses.json
third_party/dexonomy/.venv/bin/python scripts/onboard_dexonomy_hand.py \
  --from robotis --poses out/retargeted_poses.json
(cd third_party/dexonomy && .venv/bin/dexrun op=tmpl hand=hx5_d20)
third_party/dexonomy/.venv/bin/python scripts/calibrate_gripper.py
~~~

Inspect thumb opposition, contact normals, joint limits, and palm collision
decomposition. Applying scene.apply_init is necessary before pose/reach checks.
Do not change the hardware mounting to compensate for camera appearance.

## Small execution diagnostic

After the prerequisites, create a single-object root using an existing legacy
object asset. This keeps search and execution on the same asset directory.
Use a fresh run name and destination each time:

~~~bash
mkdir -p out/smoke_objects
ln -s "$PWD/third_party/dexonomy/assets/object/icra/Mustard" out/smoke_objects/Mustard

third_party/dexonomy/.venv/bin/python scripts/search_scene_grasps.py \
  --objects out/smoke_objects --template 2_Small_Diameter \
  --exp-name smoke_search_review --limit 1

third_party/dexonomy/.venv/bin/python scripts/collect_episodes.py \
  --objects out/smoke_objects --exp-name smoke_search_review \
  --max-episodes 6 --out out/smoke_episodes_review --video out/smoke_clips_review
~~~

The legacy Mustard asset must exist; otherwise prepare a single object with
build_object_assets.py first. This scene-search smoke run tests an alternative
generator and the recorder, not the Dexonomy knowledge-transfer hypothesis.
No success count or runtime is promised. A video is useful inspection evidence,
not a held-out policy evaluation.

For the main teacher path, generate candidates with Dexonomy initialization,
grasp_admm, and eval under an explicit object/config manifest, then record against
the same bounded execution assets. generate_grasps.py expects scene_cfg.npy
files, whereas build_object_set.py writes floating scale-specific configs;
their defaults are not interchangeable.

Use fresh experiment outputs and inspect stage logs. build_object_set.py now
accepts a source override and follows project paths; evolve_grasps.sh logs and
stops on stage failure. The static filter now uses squeeze_qpos, and generation
wrappers propagate failed stages. These fixes do not revalidate historical
success directories. The standalone filter is still a static candidate screen,
not an arm-execution certificate.

## Privileged RL executor checks

Use a fresh output directory for every run. The following is a short interface
check, not enough training to assess grasp learning:

~~~bash
MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python scripts/train_grasp_rl.py \
  --grasps third_party/dexonomy/output/evo_hx5_d20/succ_grasp \
  --objects data/objects/evoall --objects-limit 1 --tasks-per-object 1 \
  --workers 1 --steps 64 --episode-steps 8 --device cpu \
  --out out/rl/my-smoke
MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python scripts/evaluate_grasp_rl.py \
  --checkpoint out/rl/my-smoke/policy.pt --episodes 2 --out out/rl/my-evaluation
~~~

Evaluation uses the checkpoint's candidate manifest and fresh reset seeds.
It is not held-out-object evaluation. Old observation/reward checkpoints require
their original code; matching array dimensions does not imply semantic compatibility.
The trainer has no visual-demonstration exporter yet. RL and visual-student
training remain distinct stages.

Ladder stages 0,1,2 can be run with stage_ladder.py on a small fixed set.
Stage 3 is unimplemented and is rejected. NA denotes skipped/invalid outcomes;
retain the status column and count only valid attempted trials in denominators.

## Record and validate before training

Use `collect_teacher_dataset.py` and schema 3 for the intended student. It
archives a compiled MuJoCo model, complete initial integration state, source
snapshots, original/adapted teacher goals, calibrated physics, pre-action images,
and conservative quality signals from every physics step. Accepted trajectories
and failures are kept separately. `prepare_teacher_dataset.py` validates and
combines these runs with object-disjoint splits. Legacy schema-2 Zarr lacks the
physics/provenance evidence and cannot automatically become clean positives.

`rgbm` retains the object mask; optional `part_mask` remains separate. The same
explicit part-region annotation is checked against rendered pixels and physical
contact points. This is an annotation interface, not automatic text grounding.
See [TEACHER_DATASET.md](configs/TEACHER_DATASET.md) for the schema and annotation
format. Training starts only through the explicit `train` subcommand after data,
dependencies, and actual pretrained-model forward checks pass.

The earlier small training-set reproduction is historical evidence only.
The [fixed student experiment](results/2026-09-08/fixed-student/README.md) provides
collect/train/evaluate commands using the existing Dexonomy interpreter. It trains
a torch-only recurrent visual BC baseline on one 350-frame close/lift trajectory,
starting at the open grasp pose. Its diagnostic data fail complete teacher contact
compliance and must not be mixed into a clean transfer dataset without disclosure.

## Paper build

From paper/, with a LaTeX distribution installed:

~~~bash
pdflatex -interaction=nonstopmode -halt-on-error dexprompt.tex
pdflatex -interaction=nonstopmode -halt-on-error dexprompt.tex
~~~

The bibliography is inline. The draft contains TODOs and is not submission-ready;
choose a venue/cycle before applying its final format and page requirements.

## Contact preservation and full approach

The [contact/approach report](results/2026-09-08/contact-approach/README.md) gives
commands for validate_teacher_execution.py and replay_teacher_execution.py.
It pins the previous teacher candidate, screens approach directions, checks
contacts at every physics step, and retains failed executions. A selected +X
home-to-lift trajectory works; complete teacher contact preservation does not.


Contact-feedback control and the final/gentle/guarded experiment commands are in
[the follow-up report](results/2026-09-08/contact-feedback/README.md). It uses the
same existing interpreter and approach replay bundle. Preserve the force and
penetration labels; the demonstrated contact-lift success is not clean-data approval.
