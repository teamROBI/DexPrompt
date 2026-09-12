<!-- research-scope-2026-09-12 -->
> **현재 범위 (2026-09-12):** [두 단계 연구 정의](../RESEARCH_PLAN.md)를 따른다.
> ① 정확한 형상 기반 Dexonomy 파지 선정·시범 적응 학습/자동 수집, ② 물체 선택 조건부 시각 행동 BC.
> 아래 part API/부위 마스크는 선택적 호환 기능이며 필수 과제가 아니다. mask·DINO는 DexGraspVLA 설계와 양립한다.
> 날짜가 붙은 실행/미실행 상태와 명령은 당시 기록이다. 현재 완료 여부와 재개는 RESUME의 원격 산출물로 확인한다.

# Teacher-execution data contract

The schema-3 dataset is the input to `train_hx5_student.py`. It contains one NPZ
and provenance JSON per episode, and a sealed `manifest.json`. Accepted episodes
live under `episodes/`; measured failures live under `diagnostics/` and never enter
a positive visual-BC training split. Failures remain essential labels/rewards for
the proposed Stage 1 selector and adaptation learner, under a separate training
contract; this schema guide does not yet implement that learner. Legacy schema-1/2 Zarr data cannot establish the required contact,
force or penetration evidence and is not automatically promoted to positives.

## Recorded fields

Every array has the same leading episode length `T`. Observations precede the
action they supervise. Controls run at 50 Hz; timestamps advance exactly 0.02 s.
Actions are absolute joint-position targets in the manifest's ordered 27 joints.

| Array | Shape | Meaning |
| --- | --- | --- |
| `rgbm` | T,H,W,4 uint8 | Head RGB and target-object mask |
| `right_cam_img` | T,H,W,3 uint8 | Wrist RGB |
| `right_state`, `action` | T,27 float | Pre-action measured positions and executed command |
| `timestamp`, `control_dt` | T float | Pre-action simulation time and command duration |
| `phase` | T string | Transit, approach, closure, lift and hold labels |
| `contact_groups` | T,G bool | Every required group contacted throughout the physics interval |
| `group_forces_min` | T,G float | Minimum normal force per required group over that interval |
| `digit_force_max` | T,5 float | Maximum summed normal force on each finger, including unannotated links |
| `robot_object_penetration_max` | T float | Maximum robot–target penetration during the interval |
| `robot_table_penetration_max` | T float | Maximum robot–table penetration during the interval |
| `target_height_min` | T float | Minimum absolute target height during the interval |
| `object_xy_displacement_max` | T float | Maximum target displacement from its initial XY position |
| `robot_obstacle_penetration_max` | T float | Required when distractors exist |
| `non_target_displacement_max` | T float | Required when distractors exist; checked for the entire episode |
| `part_mask` | T,H,W uint8 | Optional separately annotated visible requested part |
| `part_contact_compliant` | T bool | Mandatory measured contact-region compliance for a part request |

For a part request, every required contact group must carry positive normal force
inside the annotated region. Any positive robot–target contact force outside that
region invalidates compliance, including palm, arm and proximal finger links that
are absent from the teacher's required groups. The minimum group-force threshold
is checked separately by the trajectory quality gate.

All quality arrays aggregate **post-action physics steps**, retaining transient
contact losses and collisions between camera frames. A final-state snapshot is
insufficient. Positive episodes must start clear at home, include the whole
approach and end in at least 0.5 s of uninterrupted contact-preserving hold above
0.10 m. Default limits are 2 mm robot–object penetration, 30 N per finger, and
10 mm approach/non-target displacement. These are disclosed simulation-quality
thresholds, not calibrated HX5 hardware safety ratings.

Provenance identifies the original Dexonomy file and SHA-256, object/candidate/type,
required contact bodies, source-code and geometry hashes, measured initial state,
and both original and adapted grasp targets. Object conditioning cannot acquire a
part label merely by copying its object mask.

## Validate and split without training

```bash
.venv/bin/python scripts/data/prepare_teacher_dataset.py validate out/teacher-run
.venv/bin/python scripts/data/prepare_teacher_dataset.py prepare \
  --input out/teacher-run-a --input out/teacher-run-b \
  --out out/teacher-training-data --split-by object --seed 0
.venv/bin/python scripts/data/prepare_teacher_dataset.py validate \
  out/teacher-training-data --require-ready
# If collection was interrupted before manifest.json was finalized:
.venv/bin/python scripts/data/prepare_teacher_dataset.py recover out/interrupted-teacher-run
```

The preparation command requires fresh output, verifies source hashes and measured
quality, and recomputes disjoint groups after combining runs. Default splits are
object-disjoint; candidate-disjoint is an explicitly weaker alternative that may
share objects. Repeated episodes or aliases of one teacher candidate remain in the
same split. One object cannot provide object-disjoint validation. Exit code 2 means
the dataset is valid but lacks accepted training/validation groups; it is not a
request to loosen contact-quality limits. Changing the seed does not create new
independent objects. No preparation or validation command starts learning.
Merging copies and rebases hashed simulator models, initial integration states and
archived source files so the prepared dataset remains replayable. Recovery seals
only complete, validated NPZ/JSON pairs and cross-checks the collector's episode
journal; it never replaces an existing manifest or treats a failed initial path
with no executed frames as a positive episode.

## Explicit part annotations

`icra.conditioning.PartRegion` supports manually supplied unions of object-local
axis-aligned boxes. Coordinates are in metres **after** the object's executed
scale. Its `validate_geometry(bounds, scale)` checks the annotation against the
actual asset before collection. Regions cannot tile the entire object bounds.
The same region labels both rendered pixels and physical contact points; an
image mask alone never proves the fingers contacted the requested part.

An annotation looks like this toy cube example, which is not a project object:

```json
{
  "schema_version": 1,
  "units": "m",
  "coordinate_frame": "object",
  "object_id": "toy_cube",
  "part_id": "right_half",
  "object_scale": 0.2,
  "object_bounds": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]},
  "regions": [{"min": [0.0, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}],
  "text_aliases": ["right half"]
}
```

This is a manual annotation interface, not automatic language grounding or an
anatomical part classifier. Projection currently supports the scene's centred
perspective cameras. Camera/depth conventions follow the
[MuJoCo camera reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)
and [MuJoCo's metric-depth renderer](https://github.com/google-deepmind/mujoco/blob/3.3.3/python/mujoco/renderer.py).
An independent projection-sensor test and a real EGL-rendered cube check verify
pixel centres, vertical orientation, visible-object masking and camera-plane depth.
