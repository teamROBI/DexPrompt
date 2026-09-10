"""Bounded teacher-to-visual-BC experiment; explicitly starts at an open grasp.

Not full approach, part grounding, RL, or the pretrained DexGraspVLA model.
A recurrent small visual policy tests whether synthetic execution can be learned.
Lift and teacher-contact preservation are reported separately; failed contact
preservation demonstrations must not be treated as full teacher-transfer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from collect_episodes import Recorder, in_object_frame, parse_args, write_zarr  # noqa: E402

from icra import scene  # noqa: E402
from icra.recording import record_step  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(row):
    return Recorder(
        parse_args(["--width", "64", "--height", "64"]), scene.load(), [Path(row["directory"])]
    )


def save_state(r):
    from icra.simulation import snapshot

    return snapshot(r.model, r.data)


def restore(r, state):
    from icra.simulation import restore as restore_integration

    restore_integration(r.model, r.data, state)


def initialize(r, row):
    """Explicit curriculum reset: place OPEN hand at original teacher root pose."""
    b = r.object_body(row["object_name"])
    r.reset({b: 0.0})
    r.settle()
    original = r.data.qpos.copy()
    raw = np.load(row["source"], allow_pickle=True).item()
    target = r.object_frame(b) @ in_object_frame(raw, offset=r.offset)
    for seed in ({}, *scene.REACH_SEEDS):
        r.data.qpos[:] = original
        for name, value in seed.items():
            r.data.qpos[r.model.jnt_qposadr[r.model.joint(name).id]] = value
        if r.arm.solve(target, 300, 0.03):
            break
    else:
        raise RuntimeError("fixed teacher pose has no IK solution")
    r.data.qpos[r.addresses[7:]] = row["open_qpos"]
    r.data.qvel[:] = 0
    r.command(r.measured())
    r.mj.mj_forward(r.model, r.data)
    return b, target, save_state(r)


def metric(r, b, row, height):
    touching = set()
    collision = False
    for c in r.data.contact[: r.data.ncon]:
        a, z = (int(r.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        other = z if a == b else a if z == b else -1
        if other >= 0 and c.dist <= 0:
            touching.add(r.model.body(other).name)
        names = (r.model.body(a).name, r.model.body(z).name)
        if c.dist < -0.002:
            for i in range(2):
                if "table" in names[i] and names[1 - i].startswith(
                    ("finger_r_", "hx5_r_", "arm_r_")
                ):
                    collision = True
    groups = [bool(set(g) & touching) for g in row["required"]]
    return [float(r.data.xpos[b, 2] - height), sum(groups), int(collision)]


def summarize(trace, required):
    trace = np.asarray(trace)
    lift_run = strict_run = longest_lift = longest_strict = 0
    for height, groups, collision in trace:
        lift = height > 0.10 and groups >= 1 and not collision
        strict = lift and groups == required
        lift_run = lift_run + 1 if lift else 0
        strict_run = strict_run + 1 if strict else 0
        longest_lift, longest_strict = max(longest_lift, lift_run), max(longest_strict, strict_run)
    return {
        "sustained_lift": longest_lift >= 25,
        "teacher_compliant_lift": longest_strict >= 25,
        "max_height_m": float(trace[:, 0].max()),
        "final_height_m": float(trace[-1, 0]),
        "longest_lift_ticks": longest_lift,
        "longest_teacher_ticks": longest_strict,
        "max_contact_groups": int(trace[:, 1].max()),
        "required_groups": required,
        "table_collision_ticks": int(trace[:, 2].sum()),
    }


def collect(args):
    source = json.loads(args.manifest.read_text())
    row = source["candidates"][0]
    args.out.mkdir(parents=True, exist_ok=False)
    r = build(row)
    try:
        b, target, initial = initialize(r, row)
        height = r.data.xpos[b, 2]
        frames, trace = [], []
        for t in range(350):
            goal = target.copy()
            goal[2, 3] += 0.15 * np.clip((t - 140) / 140, 0, 1)
            saved = save_state(r)
            solved = r.arm.solve(goal, 60, 0.03)
            aq = r.data.qpos[r.arm.address].copy()
            restore(r, saved)
            if not solved:
                raise RuntimeError(f"IK failed at control tick {t}")
            fraction = min(t / 90, 1)
            fingers = (1 - fraction) * np.asarray(row["open_qpos"]) + fraction * np.asarray(
                row["squeeze_qpos"][7:]
            )
            record_step(r, np.r_[aq, fingers], b, frames, 10)
            trace.append(metric(r, b, row, height))
        demo = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
        np.savez_compressed(args.out / "demo.npz", **demo)
        np.save(args.out / "initial_state.npy", initial)
        np.save(args.out / "teacher_trace.npy", trace)
        write_zarr(args.out / "diagnostic_demo.zarr", [demo])
        outcomes = [summarize(trace, len(row["required"]))]
        # Independent simulator resets, identical initial state; not new tasks.
        for _ in range(2):
            restore(r, initial)
            trace = []
            for action in demo["action"]:
                r.step_to(action, 10)
                trace.append(metric(r, b, row, height))
            outcomes.append(summarize(trace, len(row["required"])))
        manifest = {
            "schema_version": 1,
            "candidate": row,
            "scope": ("one object; original teacher pose; open-hand grasp-position "
                      "initialization; no approach or clutter"),
            "policy_dt": float(r.model.opt.timestep * 10),
            "ticks": 350,
            "height_reference": float(height),
            "mujoco_version": r.mj.__version__,
            "initial_state_spec": "mjSTATE_INTEGRATION",
            "scene": scene.load(),
            "teacher_sha256": hashlib.sha256(Path(row["source"]).read_bytes()).hexdigest(),
            "code_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in (
                    "scripts/fixed_student_experiment.py",
                    "scripts/collect_episodes.py",
                    "src/icra/scene.py",
                    "src/icra/arm.py",
                    "src/icra/objects.py",
                    "src/icra/recording.py",
                    "src/icra/simulation.py",
                )
            },
            "expert_and_replays": outcomes,
            "mask_nonzero_frames": int(np.any(demo["rgbm"][..., 3], axis=(1, 2)).sum()),
        }
        dump(args.out / "manifest.json", manifest)
        from PIL import Image

        Image.fromarray(
            np.concatenate([demo["rgbm"][i, ..., :3] for i in (0, 90, 210, 280, 349)], axis=1)
        ).resize((960, 192)).save(args.out / "teacher_head.png")
        print(json.dumps(manifest["expert_and_replays"], indent=2), flush=True)
    finally:
        r.renderer.close()


def policy_class():
    import torch
    from torch import nn

    class VisualBC(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(nn.Linear(7 * 8 * 8, 96), nn.Tanh())
            self.rnn = nn.GRU(96 + 27, 128, batch_first=True)
            self.head = nn.Linear(128, 27)

        def forward(self, images, state, hidden=None):
            x = torch.cat([self.visual(images), state], dim=-1)
            x, hidden = self.rnn(x, hidden)
            return self.head(x), hidden

    return VisualBC


def visual_features(rgbm, wrist):
    """Area downsample both cameras, preserving spatial layout and object mask."""
    images = np.concatenate([rgbm, wrist], axis=-1).astype(np.float32) / 255
    shape = images.shape[:-3]
    images = images.reshape(*shape, 8, 8, 8, 8, 7).mean(axis=(-4, -2))
    return images.reshape(*shape, 448)


def train(args):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    data = np.load(args.data / "demo.npz")
    actions, states = data["action"], data["right_state"]
    center = actions.mean(axis=0)
    scale = np.maximum(actions.std(axis=0), 0.05)
    images = torch.tensor(visual_features(data["rgbm"], data["right_cam_img"])[None])
    q = torch.tensor(((states - center) / scale)[None], dtype=torch.float32)
    targets = torch.tensor(((actions - center) / scale)[None], dtype=torch.float32)
    policy = policy_class()()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    with (args.out / "loss.tsv").open("w") as log:
        log.write("step\tnormalized_mse\n")
        for step in range(args.steps):
            prediction, _ = policy(images, q)
            loss = (prediction - targets).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if step % 100 == 0 or step == args.steps - 1:
                print(f"{step}\t{loss.item():.7f}", flush=True)
                log.write(f"{step}\t{loss.item():.9f}\n")
                log.flush()
    policy.eval()
    with torch.no_grad():
        predicted = policy(images, q)[0][0].numpy() * scale + center
    rmse = float(np.sqrt(np.mean((predicted - actions) ** 2)))
    checkpoint = {
        "model": policy.state_dict(),
        "center": torch.tensor(center),
        "scale": torch.tensor(scale),
        "seed": args.seed,
        "steps": args.steps,
        "data": str(args.data.resolve()),
        "training_action_rmse_rad": rmse,
        "demo_sha256": hashlib.sha256((args.data / "demo.npz").read_bytes()).hexdigest(),
        "model_name": "small recurrent visual behavior cloning; not DexGraspVLA",
    }
    torch.save(checkpoint, args.out / "student.pt")
    print(f"training action RMSE {rmse:.6f} rad", flush=True)


def evaluate(args):
    import torch

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source = Path(ckpt["data"])
    if hashlib.sha256((source / "demo.npz").read_bytes()).hexdigest() != ckpt["demo_sha256"]:
        raise ValueError("demonstrations differ from trained checkpoint")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, sha in manifest["code_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"experiment implementation changed: {name}")
    r = build(manifest["candidate"])
    if r.mj.__version__ != manifest["mujoco_version"] or scene.load() != manifest["scene"]:
        raise ValueError("simulator/configuration differs from collection")
    model = policy_class()()
    model.load_state_dict(ckpt["model"])
    model.eval()
    center, scale = ckpt["center"].numpy(), ckpt["scale"].numpy()
    b = r.object_body(manifest["candidate"]["object_name"])
    initial = np.load(source / "initial_state.npy")
    outcomes = []
    try:
        for trial in range(args.episodes):
            restore(r, initial)
            hidden = None
            frames, trace = [], []
            for _ in range(manifest["ticks"]):
                head, mask = r.view("head", b)
                wrist, _ = r.view("wrist")
                rgbm = np.concatenate([head, mask[..., None]], axis=-1)
                image = visual_features(rgbm, wrist)
                if args.blank_images:
                    image[:] = 0
                state = (r.measured() - center) / scale
                with torch.no_grad():
                    action, hidden = model(
                        torch.tensor(image[None, None]), torch.tensor(state[None, None]), hidden
                    )
                action = action[0, 0].numpy() * scale + center
                action = np.clip(
                    action,
                    r.model.actuator_ctrlrange[r.actuators, 0],
                    r.model.actuator_ctrlrange[r.actuators, 1],
                )
                frames.append(rgbm)
                r.step_to(action, 10)
                trace.append(metric(r, b, manifest["candidate"], manifest["height_reference"]))
            outcomes.append(summarize(trace, len(manifest["candidate"]["required"])))
            np.save(args.out / f"trace_{trial}.npy", trace)
            if trial == 0:
                from PIL import Image

                Image.fromarray(
                    np.concatenate([frames[i][..., :3] for i in (0, 90, 210, 280, 349)], axis=1)
                ).resize((960, 192)).save(args.out / "student_head.png")
            print(json.dumps(outcomes[-1]), flush=True)
        dump(
            args.out / "evaluation.json",
            {
                "scope": manifest["scope"],
                "identical_reset_repeats": True,
                "checkpoint": str(args.checkpoint.resolve()),
                "blank_images": args.blank_images,
                "training_action_rmse_rad": ckpt["training_action_rmse_rad"],
                "episodes": outcomes,
            },
        )
    finally:
        r.renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("collect")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("train")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--blank-images", action="store_true")
    args = parser.parse_args()
    if getattr(args, "steps", 1) < 1 or getattr(args, "episodes", 1) < 1:
        parser.error("steps and episodes must be positive")
    {"collect": collect, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
