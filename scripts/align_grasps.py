"""Seed Dexonomy contact alignment from the hand poses retained by the IK filter.

Preserve roll, depth and grip-offset corrections, transform through each object's
actual scene pose, and convert the hand root to the sampled template's contact
frame just before the matcher runs. The sampler alone cannot do that conversion:
HandTemplateLoader selects the contact frame *after* object sampling.

Alignment can still move outside the arm's reach. Recheck the aligned hand root
against IK and static collision, then use op=grasp_admm and op=eval. Neither these
checks nor a small iteration count certify a complete home-to-grasp trajectory.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/align_grasps.py --template palmar_pinch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icra import objects as object_lib  # noqa: E402
from icra import paths  # noqa: E402
from icra import scene as scene_lib
from icra.grasp_frames import aligned_pose, filtered_pose, relative_pose


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=paths.SCENE)
    parser.add_argument("--objects", type=Path, default=paths.DEXONOMY / "assets/object/icra")
    parser.add_argument("--reachable", type=Path, default=paths.OUT / "reachable")
    parser.add_argument("--template", dest="templates", action="append")
    parser.add_argument("--exp-name", default="scene")
    parser.add_argument("--hand", default="hx5_d20")
    parser.add_argument(
        "--opt-steps", type=int, default=25,
        help="Matcher iterations; fewer steps do not guarantee reachability.",
    )
    parser.add_argument(
        "--keep", type=int, default=100,
        help="Grasps the matcher may return per object. Dexonomy's default is 10, "
             "sized for its own blind sampling rather than for seeded poses.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seeds template sampling. Dexonomy draws each seed pose's template "
             "with an unseeded np.random, so two runs of the same input align "
             "to different templates and refine differently.",
    )
    parser.add_argument(
        "--scene-cfg", type=Path, default=None,
        help="Directory of Dexonomy scene configurations laid out as DGN's: "
             "<object>/floating/scale###.npy, or <object>/tabletop/scale###.npy "
             "for a set that carries the table. Only needed when the object "
             "directory does not carry its own scene_cfg.npy.",
    )
    parser.add_argument(
        "--in-plane", type=int, default=1,
        help="Roll variants about the corrected hand root. Default 1 preserves "
             "the filtered pose; extra rolls have NOT passed the input IK check.",
    )
    return parser.parse_args(argv)


def poses_in_object_frame(entries: dict, world_object: np.ndarray, in_plane: int = 1):
    """Filtered hand roots in the object's metric frame (NOT matcher contacts)."""
    if in_plane < 0:
        raise ValueError("in_plane must be nonnegative")
    rotations, translations = [], []
    rolls = np.linspace(0.0, 2 * np.pi, in_plane, endpoint=False) if in_plane else [0.0]
    for entry in entries.values():
        pose = relative_pose(world_object, filtered_pose(entry))
        for roll in rolls:
            cos, sin = np.cos(roll), np.sin(roll)
            turn = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
            rotations.append(pose[:3, :3] @ turn)
            translations.append(pose[:3, 3])
    return (
        np.asarray(rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
    )


def contact_seed_sample(tmpl_sample, obj_sample, rotations, translations):
    """Use the exact sampled contact-to-hand transform and object scale.

    Dexonomy composes T_object_contact @ T_contact_hand to recover the hand
    root. Its seed translations are in unscaled mesh coordinates, whereas
    template contact offsets and our hand translations are in metres.
    """
    import torch

    r_ch = tmpl_sample["rot_c2h"]
    t_ch = tmpl_sample["trans_c2h"]
    r_oh = torch.as_tensor(rotations, dtype=r_ch.dtype, device=r_ch.device)[None]
    t_oh = torch.as_tensor(translations, dtype=t_ch.dtype, device=t_ch.device)[None]
    if r_ch.shape[1:] != r_oh.shape[1:] or t_ch.shape != r_ch.shape[:-1]:
        raise ValueError("template batch does not match the filtered hand seeds")
    scale = obj_sample["obj_scale"].reshape(r_ch.shape[0], 1, 3)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("object scales must be finite and positive")
    r_oc = r_oh @ r_ch.transpose(-1, -2)
    t_oc = t_oh - (r_oc @ t_ch.unsqueeze(-1)).squeeze(-1)
    return dict(obj_sample, rot_o2c_init=r_oc.clone(), trans_o2c_init=(t_oc / scale).clone())


@contextmanager
def matcher_seeds(obj_loader, matcher_type, rotations, translations):
    """Temporarily adapt both sides of Dexonomy's template sampling boundary."""
    sampler = obj_loader.sample_init_pose
    forward = matcher_type.forward
    failures = []

    def seeded_forward(matcher, tmpl_sample, obj_sample):
        import torch

        try:
            sample = contact_seed_sample(tmpl_sample, obj_sample, rotations, translations)
            result = forward(matcher, tmpl_sample, sample)
            # Which seed pose each row came from, carried through the filters:
            # they slice every key alike, so identity survives reordering and
            # padding that the file names alone do not record.
            result["seed_id"] = torch.arange(
                result["grasp_pose"].shape[0], device=result["grasp_pose"].device
            )
            return result
        except Exception as exc:
            failures.append(exc)
            raise

    # Placeholders supply only batch dimensions until the contact frame is known.
    obj_loader.sample_init_pose = lambda mesh, n_point, n_inplane: (
        rotations.copy(), translations.copy()
    )
    matcher_type.forward = seeded_forward
    try:
        yield
    finally:
        obj_loader.sample_init_pose = sampler
        matcher_type.forward = forward
    # operate_init logs and swallows exceptions. A conversion/matcher failure
    # must still fail this stage even if earlier batches wrote some files.
    if failures:
        raise RuntimeError("Dexonomy matcher failed during seeded alignment") from failures[0]


def seed_everything(seed: int) -> None:
    """Fix the template draw, which is otherwise unseeded inside Dexonomy.

    ``HandTemplateLoader`` picks a contact set per seed pose with a bare
    ``np.random.randint``. Two runs of one input then align to different
    templates and refine differently, which is not a difference between the
    inputs being compared.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def produced_order(files: list[Path]) -> list[Path]:
    """op=init's files in the order it wrote them: ``<epoch>_<row>_grasp.npy``.

    Sorted by name they run 0_0, 0_1, 0_10, 0_11, 0_2, and the row number is
    what identifies which filtered batch entry a file came from.
    """
    def position(path: Path):
        parts = path.stem.split("_")
        try:
            return (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            return (10**9, 10**9)

    return sorted(files, key=position)


def initialisations(directory: Path) -> list[Path]:
    """The grasp files op=init wrote for one object, wherever it filed them.

    Dexonomy names the output directory after the scene id, and DGN's scene ids
    carry the pose and the scale: ``<object>/floating/scale011``. Looking only in
    ``<object>`` finds nothing there, which reads as an alignment that produced
    nothing -- 18 seeds in, 18 files on disk, "0 aligned" reported.
    """
    return sorted(directory.rglob("*.npy"))


def template_library(hand: str, template: str, exp_name: str) -> list[Path]:
    """Every template this alignment may draw from, in a fixed order.

    Dexonomy's loader builds its list with ``set()`` and fills an asynchronous
    buffer from worker threads, so which templates are in the buffer -- and
    therefore which one a seed pose is matched against -- depends on thread
    timing rather than on any seed. Sorting fixes the order; :func:`choose_templates`
    then makes the draw itself a recorded decision.
    """
    base = paths.DEXONOMY / "assets/hand" / hand / "init_tmpl" / f"{template}.npy"
    evolved = paths.DEXONOMY / "output" / f"{exp_name}_{hand}" / "new_tmpl" / template
    found = [base] if base.is_file() else []
    return found + sorted(evolved.rglob("*.npy"))


def choose_templates(library: list[Path], count: int, seed: int) -> list[dict]:
    """One template file and one contact point per seed pose, decided up front.

    Ordered and recorded by content hash, not by path. An evolved library is
    copied from experiment to experiment, so two runs are comparable only if
    each pose met the same *file*; sorting by name would make the choice depend
    on which directory the copy happens to live in.
    """
    import hashlib

    if not library:
        raise ValueError("no templates to align against")
    catalogue = sorted(
        (hashlib.sha256(path.read_bytes()).hexdigest(), path) for path in library
    )
    generator = np.random.default_rng(seed)
    chosen = []
    for index in range(count):
        digest, path = catalogue[int(generator.integers(len(catalogue)))]
        data = np.load(path, allow_pickle=True).item()
        contacts = int(np.asarray(data["hand_cpn_w"]).shape[0])
        chosen.append({
            "pose_index": index,
            "path": str(path),
            "sha256": digest,
            "contact_index": int(generator.integers(contacts)),
            "contacts": contacts,
            "n_evo": float(np.asarray(data.get("n_evo", 0)).reshape(-1)[0]),
        })
    return chosen


@contextmanager
def pinned_templates(loader_type, choices: list[dict]):
    """Hand each seed pose the template that was chosen for it.

    Replaces the buffer draw entirely: ``get_batched_data`` otherwise takes
    ``torch.randint`` over whatever the loader threads have managed to read.
    The contact point is pinned the same way -- the loader picks one per file
    with a bare ``np.random.randint`` while deriving the contact frame.
    """
    import torch
    from dexonomy.sim import HandCfg, MuJoCo_VisEnv

    original = loader_type.get_batched_data
    cache: dict[tuple[str, int], dict] = {}

    def load(loader, path: str, contact: int) -> dict:
        if (path, contact) not in cache:
            environment = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=loader.xml_path))
            drawn = np.random.randint
            # The only randomness in _load_data is the contact index; pinning it
            # here uses Dexonomy's own maths rather than restating it.
            np.random.randint = lambda low, high=None, *a, **k: contact
            try:
                cache[(path, contact)] = loader._load_data(environment, path)
            finally:
                np.random.randint = drawn
        return cache[(path, contact)]

    def batched(loader, data_size, device):
        objects, poses = int(data_size[0]), int(data_size[1])
        if poses != len(choices):
            raise ValueError(
                f"{poses} seed poses but {len(choices)} template choices were pinned"
            )
        loaded = [load(loader, c["path"], c["contact_index"]) for c in choices]
        batch = {}
        for key, value in loader.data_buffer.items():
            if not value or not isinstance(value[0], torch.Tensor):
                continue
            stacked = torch.stack([torch.as_tensor(row[key]) for row in loaded])
            batch[key] = stacked.unsqueeze(0).expand(
                objects, *stacked.shape
            ).contiguous().to(device)
        for key, value in loader.extra_info.items():
            batch[key] = value
        return batch

    loader_type.get_batched_data = batched
    try:
        yield
    finally:
        loader_type.get_batched_data = original


@contextmanager
def recorded_filters(filter_type, module, record: dict):
    """Keep what Dexonomy's initialisation filters decided, before the padding.

    ``valid_index_padding`` puts rejected candidates back to reach a minimum
    count, so the number of files op=init writes is not the number that passed
    its contact checks. Each filter is wrapped where it is *called*, so the
    verdict recorded is the one that was used -- two of these filters draw
    random numbers, and running them a second time to observe them would both
    change the answer and change what came after.
    """
    names = ("pose", "loss", "collision", "qp", "fps")
    stages = record.setdefault("filters", [])
    originals = {name: getattr(filter_type, f"_{name}_filter", None) for name in names}
    wrapper = filter_type._wrapped_filter
    errors = record.setdefault("qp_error", [])
    solver = getattr(module, "get_qp_error_batched", None)
    pending: dict[str, dict] = {}

    def judged(name, original):
        def judge(self, data):
            verdicts = original(self, data)
            seeds = [int(v) for v in data["seed_id"].tolist()]
            stage = {
                "filter": name, "seeds_in": seeds,
                "seeds_passed": [
                    seed for seed, ok in zip(seeds, verdicts.tolist(), strict=True) if ok
                ],
            }
            if name == "loss":
                stage["loss_dist_max"] = [
                    float(v) for v in data["loss_dist"].max(dim=-1)[0]
                ]
                stage["loss_normal_max"] = [
                    float(v) for v in data["loss_normal"].max(dim=-1)[0]
                ]
            if name == "qp" and errors:
                stage["qp_error"] = errors[-1]
            pending[name] = stage
            return verdicts
        return judge

    def wrapped(self, data, name, remaining):
        data = wrapper(self, data, name, remaining)
        stage = pending.pop(name, {"filter": name})
        after = [int(v) for v in data["seed_id"].tolist()]
        stage["seeds_after_padding"] = after
        stage["padded_back"] = len(
            [seed for seed in after if seed not in stage.get("seeds_passed", after)]
        )
        stages.append(stage)
        return data

    def measured(*arguments, **keywords):
        forces, error = solver(*arguments, **keywords)
        errors.append([float(v) for v in error])
        return forces, error

    for name, original in originals.items():
        if original is not None:
            setattr(filter_type, f"_{name}_filter", judged(name, original))
    filter_type._wrapped_filter = wrapped
    if solver is not None:
        module.get_qp_error_batched = measured
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is not None:
                setattr(filter_type, f"_{name}_filter", original)
        filter_type._wrapped_filter = wrapper
        if solver is not None:
            module.get_qp_error_batched = solver


def body_pose(data, body):
    if body < 0:
        raise ValueError("target object is absent from the robot scene")
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose


def pose7(values):
    import mujoco

    pose = np.eye(4)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(values[3:7], dtype=float))
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = values[:3]
    return pose


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def scene_config(objects: Path, name: str, scale: float,
                 given: Path | None = None) -> Path:
    """Dexonomy's own configuration for this object, in whichever layout it is in.

    The flat sets keep a ``scene_cfg.npy`` beside the mesh. DGN keeps one per
    published scale under the *set* root, which an object directory alone does
    not lead to -- so the asset is resolved through any symlink and its set root
    searched. Raises rather than returning a path that does not exist: Dexonomy
    reports a missing configuration as an empty result, which reads exactly like
    a hard alignment problem and is how this produced 0 of 18 without a word.
    """
    # DGN names them by hundredths: 0.11 is scale011.npy.
    scaled = f"scale{int(round(scale * 100)):03d}.npy"
    tried = [objects / name / "scene_cfg.npy"]
    if given is not None:
        tried.append(given / name / "floating" / scaled)
        # A table-aware set is laid out the same way under ``tabletop``, so a
        # directory holding one is named the same way on the command line and
        # the refiner is handed a scene that contains the table it will be
        # executed on. Searched after ``floating`` so pointing this at DGN's
        # own directory still gets DGN's own configuration.
        tried.append(given / name / "tabletop" / scaled)
        tried.append(given / scaled)
    # Where the asset really lives: <set>/processed_data/<name> for DGN, whose
    # configurations are at <set>/scene_cfg/<name>/floating/<scale>.
    actual = (objects / name).resolve()
    for root in (actual.parents[1], actual.parents[0], objects, objects.parent):
        tried.append(root / "scene_cfg" / name / "floating" / scaled)
    for path in tried:
        if path.is_file():
            # Absolute, because the caller runs the matcher from Dexonomy's own
            # directory and Dexonomy globs this path: a relative one matches
            # nothing there and is reported as "Object number: 0", which reads
            # exactly like every seed being filtered out.
            return path.resolve()
    raise FileNotFoundError(
        f"no Dexonomy scene configuration for {name} at scale {scale}; looked in "
        + ", ".join(str(path) for path in dict.fromkeys(tried))
        + ". Pass --scene-cfg with the set's scene_cfg directory."
    )


def main(argv: list[str] | None = None) -> int:
    import mujoco
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dexonomy.data.obj_loader as obj_loader
    import dexonomy.op.gen_init as gen_init
    from dexonomy.data.hand_loader import HandTemplateLoader
    from dexonomy.op.gen_init import HandObjMatcher, InitFilter, operate_init
    from filter_grasps import Arm, collisions, resting_contacts

    args = parse_args(argv)
    # Dexonomy's config spells its asset and output paths relative to its own
    # checkout, so ours are resolved before moving there.
    for field in ("scene", "objects", "reachable"):
        setattr(args, field, getattr(args, field).resolve())
    problems = paths.check(args.scene, args.reachable, args.objects)
    if problems:
        print("\n".join(problems))
        print("\nrun observe_scene, predict_grasps and filter_grasps first")
        return 1

    seed_everything(args.seed)
    scene = scene_lib.load(args.scene)
    config_dir = str((paths.DEXONOMY / "dexonomy" / "config").resolve())
    library = (paths.DEXONOMY / "output" / f"{args.exp_name}_{args.hand}" / "new_tmpl")

    templates = args.templates or sorted(
        d.name for d in args.reachable.iterdir() if d.is_dir()
    )
    if not templates:
        print(f"nothing under {args.reachable}")
        return 1

    # Static endpoint checks; complete approach paths are checked by the executor.
    directories = sorted(d for d in args.objects.iterdir() if d.is_dir())
    model = scene_lib.build(scene, directories)
    data = mujoco.MjData(model)
    scene_lib.apply_init(model, data, scene)
    mujoco.mj_forward(model, data)
    arm = Arm(model, data)
    rest = data.qpos.copy()
    object_poses = {
        directory.name: body_pose(data, mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"object_{directory.name}"
        )) for directory in directories
    }
    # What the scene touches before the arm has moved -- the robot standing on its
    # wheels. Counting those as collisions rejects every pose; see
    # filter_grasps.resting_contacts.
    already = resting_contacts(model, data)
    # The fingers have to be in the shape the grasp calls for while this is
    # checked. Left at the scene's starting configuration they are straight, and
    # a straight finger on this hand reaches 236 mm -- through the table on any
    # approach from above.
    finger_address = {}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name and name.startswith("finger_r_joint"):
            finger_address[int(name.removeprefix("finger_r_joint"))] = \
                model.jnt_qposadr[index]

    total = 0
    print(f"{'template':22s} {'object':22s} {'seeds':>7} {'aligned':>9} "
          f"{'reachable':>9}")
    for template in templates:
        for path in sorted((args.reachable / template).glob("*.yml")):
            reachable = yaml.safe_load(path.read_text())
            entries = reachable.get("grasps") or {}
            name = reachable["object"]
            if not entries:
                continue
            world_object = object_poses[name]
            rotations, translations = poses_in_object_frame(
                entries, world_object, args.in_plane
            )
            # One row per seed pose, in the order poses_in_object_frame built
            # them, so a row index names a filtered variant and through it the
            # prediction it came from.
            variants = []
            for label, entry in entries.items():
                for roll_index in range(max(args.in_plane, 1)):
                    variants.append({
                        "pose_index": len(variants),
                        "variant_id": label,
                        "prediction_id": label.split("_r")[0],
                        "extra_roll_index": roll_index,
                        "roll_deg": entry.get("roll_deg"),
                        "depth_mm": entry.get("depth_mm"),
                        "thumb_gap_mm": entry.get("thumb_gap_mm"),
                        "grip_offset": entry.get("grip_offset"),
                    })
            chosen = choose_templates(
                template_library(args.hand, template, args.exp_name),
                len(rotations), args.seed,
            )
            provenance = {"seed": args.seed, "variants": variants, "templates": chosen}
            asset = object_lib.load(args.objects / name)
            if asset is None:
                raise ValueError(f"{args.objects / name} has no geometry to align to")
            config_path = scene_config(args.objects, name, asset.scale, args.scene_cfg)
            produced_dir = (
                paths.DEXONOMY / "output" / f"{args.exp_name}_{args.hand}"
                / "init_data" / template / name
            )
            if initialisations(produced_dir):
                raise FileExistsError(
                    f"{produced_dir} already contains initializations; use a fresh --exp-name"
                )
            with working_directory(paths.DEXONOMY), matcher_seeds(
                obj_loader, HandObjMatcher, rotations, translations
            ), pinned_templates(HandTemplateLoader, chosen), recorded_filters(
                InitFilter, gen_init, provenance
            ):
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    # Dexonomy's config interpolates ${hydra:runtime.choices.op}
                    # and friends, which compose() alone leaves unresolvable, so
                    # the hydra node is asked for and then installed.
                    cfg = compose(
                        config_name="base",
                        return_hydra_config=True,
                        overrides=[
                            "op=init",
                            f"hand={args.hand}",
                            f"exp_name={args.exp_name}",
                            f"tmpl_name={template}",
                            "op.epoch=1",
                            "op.object.n_cfg=1",
                            "op.object.batch_size=1",
                            f"op.matcher.opt_step={args.opt_steps}",
                            # Dexonomy keeps ten grasps per run whatever it was
                            # given, which is right when its own sampler supplies
                            # a thousand blind surface points and wrong when the
                            # seeds are already a filtered, reachable few hundred:
                            # 268 in, 10 out, and the reach recheck then drops
                            # most of those.
                            f"op.filter.general.n_final={args.keep}",
                            "op.object.cfg_path=" + str(config_path),
                        ],
                    )
                    HydraConfig.instance().set_config(cfg)
                    operate_init(cfg)

            produced = initialisations(produced_dir)
            # op=init names its files by position in the final, filtered and
            # padded batch, so the last recorded ordering names their seeds.
            order = provenance["filters"][-1]["seeds_after_padding"] \
                if provenance.get("filters") else []
            for index, produced_path in enumerate(produced_order(produced)):
                if index < len(order):
                    provenance.setdefault("init_files", {})[produced_path.name] = \
                        dict(variants[order[index]], seed_id=order[index],
                             template=chosen[order[index]])
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}"
            )
            kept = 0
            diagnostics = []
            for produced_path in produced:
                grasp = np.load(produced_path, allow_pickle=True).item()
                qpos = np.asarray(grasp["grasp_qpos"])[0]
                # Saved grasp_qpos already describes hx5_r_base in Dexonomy's
                # scene, NOT GraspGenX's URDF world frame. Do not offset it again.
                target = aligned_pose(world_object, pose7(grasp["obj_pose"]), pose7(qpos))
                data.qpos[:] = rest
                for joint, address in finger_address.items():
                    data.qpos[address] = float(qpos[6 + joint])
                mujoco.mj_forward(model, data)
                reached = arm.solve(target, 200, 0.05)
                # The solver moves the arm, not the fingers, but a fresh forward
                # pass is cheap insurance that both are where they are meant to be.
                for joint, address in finger_address.items():
                    data.qpos[address] = float(qpos[6 + joint])
                mujoco.mj_forward(model, data)
                hits = collisions(model, data, body, already)
                actual = body_pose(data, arm.body)
                rotation_cos = (np.trace(target[:3, :3].T @ actual[:3, :3]) - 1) / 2
                diagnostics.append({
                    "file": str(produced_path),
                    "world_hand_pose": target.tolist(),
                    "ik_passed": bool(reached),
                    "ik_position_error_m": float(np.linalg.norm(
                        actual[:3, 3] - target[:3, 3]
                    )),
                    "ik_orientation_error_rad": float(np.arccos(np.clip(rotation_cos, -1, 1))),
                    "collisions": [list(hit) for hit in hits],
                    "kept": bool(reached and not hits),
                })
                if reached and not hits:
                    kept += 1
                else:
                    if os.environ.get("ICRA_DIAG"):
                        why = "ik_failure" if not reached else \
                              f"hits {[h[0] + '|' + h[1] for h in hits[:2]]}"
                        print(f"      dropped: {why}")
                    produced_path.unlink()
            report = paths.OUT / "alignment" / f"{args.exp_name}_{args.hand}" / template
            report.mkdir(parents=True, exist_ok=True)
            provenance["surviving_init_files"] = sorted(
                path.name for path in initialisations(produced_dir)
            )
            (report / f"{name}.provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n"
            )
            (report / f"{name}.json").write_text(json.dumps({
                "world_object_pose": world_object.tolist(),
                "filtered_world_hand_poses": {
                    label: filtered_pose(entry).tolist() for label, entry in entries.items()
                },
                "in_plane": args.in_plane,
                "seed": args.seed,
                "provenance": f"{name}.provenance.json",
                "scene_config": str(config_path),
                "template_library": {
                    "evolved_dir": str(library),
                    "evolved_files": len(list(library.rglob("*.npy"))),
                    "base_dir": str(paths.DEXONOMY / "assets/hand" / args.hand / "init_tmpl"),
                },
                "aligned": diagnostics,
                "full_path_validated": False,
            }, indent=2) + "\n")
            total += kept
            print(f"{template:22s} {name:22s} {len(rotations):7d} "
                  f"{len(produced):9d} {kept:9d}")

    print(f"\n{total} aligned initializations")
    print(f"next: cd {paths.DEXONOMY} && .venv/bin/dexrun op=grasp_admm "
          f"hand={args.hand} exp_name={args.exp_name}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
