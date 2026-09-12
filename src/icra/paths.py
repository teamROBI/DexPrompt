"""Every path this project uses, resolved from the repository root.

Scripts here reach into three places outside their own tree -- a GraspGenX
checkout, a Dexonomy checkout, and a few gigabytes of checkpoints and generated
grasps that have no business on the repo's disk. Those were spelled out as
absolute paths under one particular home directory, which makes the scripts
unrunnable anywhere else and the results awkward to reproduce.

They are reached from the repository root instead:

    modules/graspgenx     -> ../../GraspGenX    (symlink: sibling checkout)
    third_party/dexonomy  -> ../../Dexonomy
    third_party/dexlearn, graspvla, dexgraspvla (cloned, read-only references)
    data                  -> wherever the bulk storage is  (machine-specific)

The split between the two directories is whether we change the code.

``third_party`` is taken as given. Dexonomy is there because we only ever *add*
to it -- a hand's assets under ``assets/hand/`` and one config file -- and never
touch its source.

``modules`` is for checkouts we patch. GraspGenX is the only one: it carries the
joint-limit fix, without which its playback drives the hand's joints hundreds of
degrees past their range. The patch itself stays in ``patches/`` so the checkout
can be re-cloned and the change re-applied rather than quietly becoming part of
it.

Both are symlinks rather than clones because each carries a multi-gigabyte
virtual environment with mutually incompatible pins -- GraspGenX needs torch
<2.7 on cu124, Dexonomy pins mujoco 3.3.3, this project's own venv is a third.

``data`` cannot be a relative link -- it points at whichever disk has room -- so
it is gitignored, and :func:`check` reports it missing rather than failing
obscurely several steps later.

Import ``paths`` and use the constants; do not rejoin strings from ``__file__``
in individual scripts, which is how the absolute paths accumulated the first
time.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root: this file is ``<root>/src/icra/paths.py``.
ROOT = Path(__file__).resolve().parents[2]

# -- external checkouts, via third_party/ ----------------------------------

THIRD_PARTY = ROOT / "third_party"
#: Checkouts we patch; see the module docstring for why this is not third_party.
MODULES = ROOT / "modules"

GRASPGENX = MODULES / "graspgenx"
DEXONOMY = THIRD_PARTY / "dexonomy"

#: Read-only references. Nothing here is run from this repo; they are checked out
#: so the code being followed is at hand and pinned.
DEXLEARN = THIRD_PARTY / "dexlearn"
GRASPVLA = THIRD_PARTY / "graspvla"
DEXGRASPVLA = THIRD_PARTY / "dexgraspvla"

#: Interpreters. Each checkout carries its own environment: GraspGenX needs
#: torch <2.7 on cu124, Dexonomy pins mujoco 3.3.3, and this project's own venv
#: is a third. Nothing is gained by trying to merge them.
GRASPGENX_PYTHON = GRASPGENX / ".venv/bin/python"
DEXONOMY_PYTHON = DEXONOMY / ".venv/bin/python"
DEXONOMY_RUN = DEXONOMY / ".venv/bin/dexrun"

#: Dexonomy's per-hand asset directory, written by onboard_dexonomy_hand.py.
DEXONOMY_HAND = DEXONOMY / "assets/hand/hx5_d20"

# -- models, in the repo ----------------------------------------------------

MODEL = ROOT / "model"
#: Vendored upstream URDFs, kept beside the generated ones for a readable diff.
HAND_URDF_SOURCE = MODEL / "hx5_d20_right.urdf.orig"
ROBOT_URDF_SOURCE = MODEL / "ffw_sh5_follower.urdf.orig"

GENERATED = MODEL / "graspgen"
HAND_URDF = GENERATED / "hx5_d20_right/gripper.urdf"
ROBOT_URDF = GENERATED / "ffw_sh5/ffw_sh5.urdf"
#: Scene layout saved by scene_setup_gui.py.
SCENE = GENERATED / "ffw_sh5/scene.yaml"

MESHES = MODEL / "assets"

#: Hand poses, one per grasp type, found by ``scripts/find_grasp_poses.py``.
#:
#: This replaces GraspGenX's gripper config as the source of the hand's poses.
#: That config holds exactly two -- ``open`` and ``close`` -- which cannot
#: describe a hand that wraps a bottle one way and pinches a ball another, and
#: any joint it leaves at zero in both can never be reached by interpolating
#: between them. On this hand that silently pinned the knuckles, whose 0..2.0 rad
#: range is the largest it has. The poses here are searched for against the
#: objects each grasp type is meant to handle.
GRASP_POSES = GENERATED / "hx5_d20_right/grasp_poses.json"

# -- generated bulk, via data/ ---------------------------------------------

DATA = ROOT / "data"
#: Object meshes plus the convex decomposition and scene config Dexonomy needs.
OBJECTS = DATA / "objects"
#: Dexonomy's successful grasps, which is what DexLearn trains on.
GRASPS = DATA / "grasps"
#: Rendered DexGraspVLA episodes.
EPISODES = DATA / "episodes"
#: Scratch for individual pipeline runs.
RUNS = DATA / "runs"

#: Small outputs worth looking at directly; gitignored but on the repo disk.
OUT = ROOT / "out"


def check(*required: Path) -> list[str]:
    """Report which of ``required`` is missing, as human-readable lines.

    Returns an empty list when everything resolves. Callers should print the
    lines and stop, rather than proceeding to fail somewhere less obvious -- a
    missing ``data`` symlink otherwise surfaces as a confusing error inside
    whichever subprocess happened to need it first.
    """
    problems = []
    for path in required:
        if path.exists():
            continue
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        hint = ""
        if not DATA.exists() and DATA in path.parents:
            hint = "  (create the data symlink: ln -s /path/with/room data)"
        elif THIRD_PARTY in path.parents or MODULES in path.parents:
            hint = "  (expects GraspGenX and Dexonomy checked out beside this repo)"
        problems.append(f"missing: {shown}{hint}")
    return problems


def snapshot_library(destination: Path) -> None:
    """Copy the complete importable library into a NEW frozen code snapshot."""
    import shutil

    shutil.copytree(
        ROOT / "src/icra", Path(destination), ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


# Archive keys are immutable. Resolve their spelling without changing their hash.
SOURCE_ALIASES = {
    "src/icra/arm.py": "src/icra/robots/arm.py",
    "src/icra/hand.py": "src/icra/robots/hand.py",
    "src/icra/scene.py": "src/icra/sim/scene.py",
    "src/icra/simulation.py": "src/icra/sim/simulation.py",
    "src/icra/objects.py": "src/icra/sim/objects.py",
    "src/icra/placement.py": "src/icra/sim/placement.py",
    "src/icra/object_visuals.py": "src/icra/sim/object_visuals.py",
    "src/icra/execution.py": "src/icra/teacher/execution.py",
    "src/icra/contact_control.py": "src/icra/teacher/contact_control.py",
    "src/icra/grasp_frames.py": "src/icra/teacher/grasp_frames.py",
    "src/icra/probe.py": "src/icra/teacher/probe.py",
    "src/icra/stages.py": "src/icra/teacher/stages.py",
    "src/icra/dataset.py": "src/icra/data/dataset.py",
    "src/icra/recording.py": "src/icra/data/recording.py",
    "src/icra/outcomes.py": "src/icra/data/outcomes.py",
    "src/icra/conditioning.py": "src/icra/vision/conditioning.py",
    "src/icra/visual_observation.py": "src/icra/vision/visual_observation.py",
    "src/icra/goal_memory.py": "src/icra/vision/goal_memory.py",
    "src/icra/student.py": "src/icra/learning/student.py",
    "src/icra/student_cache.py": "src/icra/learning/student_cache.py",
    "src/icra/student_data.py": "src/icra/learning/student_data.py",
    "src/icra/student_inference.py": "src/icra/learning/student_inference.py",
    "src/icra/student_upstream.py": "src/icra/learning/student_upstream.py",
    "src/icra/approach_env.py": "src/icra/learning/approach_env.py",
    "src/icra/grasp_env.py": "src/icra/learning/grasp_env.py",
    "src/icra/ppo.py": "src/icra/learning/ppo.py",
    "src/icra/dexlearn_shim.py": "src/icra/integrations/dexlearn_shim.py",
    "scripts/align_grasps.py": "scripts/teacher/align_grasps.py",
    "scripts/build_expansion_manifest.py": "scripts/teacher/build_expansion_manifest.py",
    "scripts/calibrate_gripper.py": "scripts/teacher/calibrate_gripper.py",
    "scripts/collect_clutter.py": "scripts/teacher/collect_clutter.py",
    "scripts/collect_episodes.py": "scripts/teacher/collect_episodes.py",
    "scripts/collect_teacher_dataset.py": "scripts/teacher/collect_teacher_dataset.py",
    "scripts/filter_grasps.py": "scripts/teacher/filter_grasps.py",
    "scripts/observe_scene.py": "scripts/teacher/observe_scene.py",
    "scripts/predict_grasps.py": "scripts/teacher/predict_grasps.py",
    "scripts/grasp_selector.py": "scripts/teacher/grasp_selector.py",
    "scripts/build_stage1_bc_handoff.py": "scripts/data/build_stage1_bc_handoff.py",
    "scripts/intake_hope_diversity.py": "scripts/data/intake_hope_diversity.py",
    "scripts/prepare_followup_intake.py": "scripts/data/prepare_followup_intake.py",
    "scripts/prepare_visual_dataset.py": "scripts/data/prepare_visual_dataset.py",
    "scripts/build_object_assets.py": "scripts/assets/build_object_assets.py",
    "scripts/scene_setup_gui.py": "scripts/assets/scene_setup_gui.py",
    "scripts/audit_mask_visibility.py": "scripts/diagnostics/audit_mask_visibility.py",
    "scripts/audit_visual_intake.py": "scripts/diagnostics/audit_visual_intake.py",
    "scripts/check_followup_placements.py": "scripts/diagnostics/check_followup_placements.py",
    "scripts/check_hope_pilot.py": "scripts/diagnostics/check_hope_pilot.py",
    "scripts/check_placement.py": "scripts/diagnostics/check_placement.py",
    "scripts/followup_final_report.py": "scripts/diagnostics/followup_final_report.py",
    "scripts/report_hope_student_readiness.py": (
        "scripts/diagnostics/report_hope_student_readiness.py"
    ),
    "scripts/prepare_hope_pilot.py": "scripts/experiments/prepare_hope_pilot.py",
    "scripts/run_hope_pilot.py": "scripts/experiments/run_hope_pilot.py",
    "scripts/run_target_switch.py": "scripts/experiments/run_target_switch.py",
    "scripts/run_teacher_followup_5h.py": "scripts/experiments/run_teacher_followup_5h.py",
    "scripts/evaluate_student_suite.py": "scripts/student/evaluate_student_suite.py",
    "scripts/replay_hx5_student.py": "scripts/student/replay_hx5_student.py",
    "scripts/train_hx5_student.py": "scripts/student/train_hx5_student.py",
    "scripts/watch_visual_intake.py": "scripts/ops/watch_visual_intake.py",
    "scripts/evolve_grasps.sh": "scripts/teacher/evolve_grasps.sh",
    "scripts/setup_student_server.sh": "scripts/ops/setup_student_server.sh",
    "scripts/launch_followup_visual_intake.sh": "scripts/ops/launch_followup_visual_intake.sh",
    "scripts/run_followup_visual_intake.sh": "scripts/ops/run_followup_visual_intake.sh",
    "scripts/launch_teacher_followup_5h.sh": "scripts/ops/launch_teacher_followup_5h.sh",
    "scripts/analyse_student_closed_loop.py": "scripts/diagnostics/analyse_student_closed_loop.py",
    "scripts/assemble_hope_student_bundle.py": "scripts/data/assemble_hope_student_bundle.py",
    "scripts/audit_contact_physics.py": "scripts/diagnostics/audit_contact_physics.py",
    "scripts/audit_grasp_alignment.py": "scripts/diagnostics/audit_grasp_alignment.py",
    "scripts/audit_grasp_eval.py": "scripts/diagnostics/audit_grasp_eval.py",
    "scripts/audit_guard.py": "scripts/diagnostics/audit_guard.py",
    "scripts/audit_student_failure.py": "scripts/diagnostics/audit_student_failure.py",
    "scripts/audit_x0_loss.py": "scripts/diagnostics/audit_x0_loss.py",
    "scripts/build_followup_handoff.py": "scripts/teacher/build_followup_handoff.py",
    "scripts/build_object_set.py": "scripts/assets/build_object_set.py",
    "scripts/build_selector_ledger.py": "scripts/teacher/build_selector_ledger.py",
    "scripts/build_tabletop_scene_cfg.py": "scripts/teacher/build_tabletop_scene_cfg.py",
    "scripts/bundle_hope_handoff.py": "scripts/data/bundle_hope_handoff.py",
    "scripts/check_hope_bundle_lecun.py": "scripts/ops/check_hope_bundle_lecun.py",
    "scripts/check_hope_student_bundle.py": "scripts/diagnostics/check_hope_student_bundle.py",
    "scripts/check_hope_student_lecun.py": "scripts/ops/check_hope_student_lecun.py",
    "scripts/check_placement_transfer.py": "scripts/ops/check_placement_transfer.py",
    "scripts/check_upstream_action.py": "scripts/diagnostics/check_upstream_action.py",
    "scripts/check_yaw_transfer.py": "scripts/ops/check_yaw_transfer.py",
    "scripts/collect_approach_demos.py": "scripts/rl/collect_approach_demos.py",
    "scripts/collect_dataset.py": "scripts/data/collect_dataset.py",
    "scripts/collect_hand_edit.py": "scripts/teacher/collect_hand_edit.py",
    "scripts/collect_multiccd.py": "scripts/teacher/collect_multiccd.py",
    "scripts/collect_scene_grasps.py": "scripts/teacher/collect_scene_grasps.py",
    "scripts/collect_wrist_edit.py": "scripts/teacher/collect_wrist_edit.py",
    "scripts/compare_baseline_reproduction.py": (
        "scripts/diagnostics/compare_baseline_reproduction.py"
    ),
    "scripts/compare_grip_offset.py": "scripts/diagnostics/compare_grip_offset.py",
    "scripts/compare_multiccd.py": "scripts/diagnostics/compare_multiccd.py",
    "scripts/compare_student_sampler_steps.py": (
        "scripts/diagnostics/compare_student_sampler_steps.py"
    ),
    "scripts/contact_feedback_experiment.py": "scripts/experiments/contact_feedback_experiment.py",
    "scripts/decompose_palm.py": "scripts/assets/decompose_palm.py",
    "scripts/diagnose_bowl_drop.py": "scripts/diagnostics/diagnose_bowl_drop.py",
    "scripts/diagnose_contacts.py": "scripts/diagnostics/diagnose_contacts.py",
    "scripts/diagnose_proprio_retrieval.py": "scripts/diagnostics/diagnose_proprio_retrieval.py",
    "scripts/diagnose_single_student.py": "scripts/diagnostics/diagnose_single_student.py",
    "scripts/diagnose_student_denoising.py": "scripts/diagnostics/diagnose_student_denoising.py",
    "scripts/diagnose_teacher_observations.py": (
        "scripts/diagnostics/diagnose_teacher_observations.py"
    ),
    "scripts/diagnose_x0_denoising.py": "scripts/diagnostics/diagnose_x0_denoising.py",
    "scripts/evaluate_approach_policy.py": "scripts/rl/evaluate_approach_policy.py",
    "scripts/evaluate_grasp_rl.py": "scripts/rl/evaluate_grasp_rl.py",
    "scripts/evaluate_hx5_student.py": "scripts/student/evaluate_hx5_student.py",
    "scripts/export_ffw_urdf.py": "scripts/assets/export_ffw_urdf.py",
    "scripts/export_hand_urdf.py": "scripts/assets/export_hand_urdf.py",
    "scripts/filter_scene_feasible.py": "scripts/teacher/filter_scene_feasible.py",
    "scripts/find_grasp_poses.py": "scripts/assets/find_grasp_poses.py",
    "scripts/fixed_student_experiment.py": "scripts/experiments/fixed_student_experiment.py",
    "scripts/follow_student_comparison.py": "scripts/ops/follow_student_comparison.py",
    "scripts/generate_grasps.py": "scripts/teacher/generate_grasps.py",
    "scripts/grasp_inspector_gui.py": "scripts/teacher/grasp_inspector_gui.py",
    "scripts/make_student_slew_limits.py": "scripts/data/make_student_slew_limits.py",
    "scripts/milk_state_diagnosis.py": "scripts/teacher/milk_state_diagnosis.py",
    "scripts/onboard_dexonomy_hand.py": "scripts/assets/onboard_dexonomy_hand.py",
    "scripts/package_student_transfer.py": "scripts/ops/package_student_transfer.py",
    "scripts/perturb_initialisation.py": "scripts/experiments/perturb_initialisation.py",
    "scripts/plot_approach_training.py": "scripts/diagnostics/plot_approach_training.py",
    "scripts/prepare_student_diagnostic.py": "scripts/data/prepare_student_diagnostic.py",
    "scripts/prepare_teacher_dataset.py": "scripts/data/prepare_teacher_dataset.py",
    "scripts/probe_grasp_slip.py": "scripts/diagnostics/probe_grasp_slip.py",
    "scripts/probe_hand.py": "scripts/diagnostics/probe_hand.py",
    "scripts/rank_grasps.py": "scripts/teacher/rank_grasps.py",
    "scripts/record_finger5_plan.py": "scripts/experiments/record_finger5_plan.py",
    "scripts/render_grasps.py": "scripts/teacher/render_grasps.py",
    "scripts/render_hand.py": "scripts/assets/render_hand.py",
    "scripts/render_object_pointclouds.py": "scripts/assets/render_object_pointclouds.py",
    "scripts/replay_teacher_dataset.py": "scripts/data/replay_teacher_dataset.py",
    "scripts/replay_teacher_execution.py": "scripts/teacher/replay_teacher_execution.py",
    "scripts/report_bowl_scale_matcher.py": "scripts/diagnostics/report_bowl_scale_matcher.py",
    "scripts/report_can_placement.py": "scripts/diagnostics/report_can_placement.py",
    "scripts/report_contact_residuals.py": "scripts/diagnostics/report_contact_residuals.py",
    "scripts/report_diversity_student_intake.py": (
        "scripts/diagnostics/report_diversity_student_intake.py"
    ),
    "scripts/report_grasp_chain.py": "scripts/diagnostics/report_grasp_chain.py",
    "scripts/report_hope_student_pool.py": "scripts/diagnostics/report_hope_student_pool.py",
    "scripts/report_initialisation_sensitivity.py": (
        "scripts/diagnostics/report_initialisation_sensitivity.py"
    ),
    "scripts/report_iteration_sweep.py": "scripts/diagnostics/report_iteration_sweep.py",
    "scripts/report_low_noise_comparison.py": "scripts/diagnostics/report_low_noise_comparison.py",
    "scripts/report_missing_digit_gap.py": "scripts/diagnostics/report_missing_digit_gap.py",
    "scripts/report_multiobject.py": "scripts/diagnostics/report_multiobject.py",
    "scripts/report_table_refinement.py": "scripts/diagnostics/report_table_refinement.py",
    "scripts/report_teacher_expansion.py": "scripts/diagnostics/report_teacher_expansion.py",
    "scripts/report_variant_gaps.py": "scripts/diagnostics/report_variant_gaps.py",
    "scripts/retarget_templates.py": "scripts/assets/retarget_templates.py",
    "scripts/run_direct_student_lecun.py": "scripts/ops/run_direct_student_lecun.py",
    "scripts/run_guard_audit.py": "scripts/experiments/run_guard_audit.py",
    "scripts/run_hand_edit.py": "scripts/experiments/run_hand_edit.py",
    "scripts/run_hope_candidate_diversity.py": (
        "scripts/experiments/run_hope_candidate_diversity.py"
    ),
    "scripts/run_hope_clutter.py": "scripts/experiments/run_hope_clutter.py",
    "scripts/run_hope_milk_multiccd.py": "scripts/experiments/run_hope_milk_multiccd.py",
    "scripts/run_hope_new_shapes.py": "scripts/experiments/run_hope_new_shapes.py",
    "scripts/run_hope_twelve_checks.py": "scripts/experiments/run_hope_twelve_checks.py",
    "scripts/run_low_noise_student_lecun.py": "scripts/ops/run_low_noise_student_lecun.py",
    "scripts/run_selector_compare.py": "scripts/experiments/run_selector_compare.py",
    "scripts/run_single_student_lecun.py": "scripts/ops/run_single_student_lecun.py",
    "scripts/run_student_slew_lecun.py": "scripts/ops/run_student_slew_lecun.py",
    "scripts/run_upstream_student_lecun.py": "scripts/ops/run_upstream_student_lecun.py",
    "scripts/run_wrist_edit.py": "scripts/experiments/run_wrist_edit.py",
    "scripts/run_x0_student_lecun.py": "scripts/ops/run_x0_student_lecun.py",
    "scripts/search_scene_grasps.py": "scripts/teacher/search_scene_grasps.py",
    "scripts/select_approach_policy.py": "scripts/rl/select_approach_policy.py",
    "scripts/squeeze_branch_compare.py": "scripts/experiments/squeeze_branch_compare.py",
    "scripts/stage_ladder.py": "scripts/rl/stage_ladder.py",
    "scripts/student_server.py": "scripts/ops/student_server.py",
    "scripts/summarise_finger5.py": "scripts/diagnostics/summarise_finger5.py",
    "scripts/summarise_hope_attempts.py": "scripts/diagnostics/summarise_hope_attempts.py",
    "scripts/summarize_teacher_observations.py": (
        "scripts/diagnostics/summarize_teacher_observations.py"
    ),
    "scripts/summarize_x0_denoising.py": "scripts/diagnostics/summarize_x0_denoising.py",
    "scripts/train_approach_bc.py": "scripts/rl/train_approach_bc.py",
    "scripts/train_approach_rl.py": "scripts/rl/train_approach_rl.py",
    "scripts/train_dexlearn.py": "scripts/rl/train_dexlearn.py",
    "scripts/train_grasp_rl.py": "scripts/rl/train_grasp_rl.py",
    "scripts/transfer_hope_bundle_lecun.py": "scripts/ops/transfer_hope_bundle_lecun.py",
    "scripts/validate_teacher_execution.py": "scripts/diagnostics/validate_teacher_execution.py",
    "scripts/verify_approach_env.py": "scripts/diagnostics/verify_approach_env.py",
    "scripts/verify_tree_transfer.py": "scripts/ops/verify_tree_transfer.py",
    "scripts/watch_approach_training.py": "scripts/ops/watch_approach_training.py",
    "scripts/watch_day_feasibility.py": "scripts/ops/watch_day_feasibility.py",
    "scripts/watch_hope_placement_intake.py": "scripts/ops/watch_hope_placement_intake.py",
    "scripts/watch_upstream_feasibility.py": "scripts/ops/watch_upstream_feasibility.py",
}


def source_key(record: dict, relative: str) -> str:
    """Return an existing provenance key, preferring the requested spelling."""
    if relative in record:
        return relative
    canonical = SOURCE_ALIASES.get(relative, relative)
    if canonical in record:
        return canonical
    for old, new in SOURCE_ALIASES.items():
        if new == canonical and old in record:
            return old
    return relative


def source_value(record: dict, relative: str, default=None):
    return record.get(source_key(record, relative), default)
