"""MLP BC matched to Codex's DP3 run: same 3 demonstrations, observations, action, budget, stop.

Stages (each writes under ``--out`` only; Codex's DP3 folder is read, never written):

``plan``      fix the settings from Codex's DP3 plan (history, seeds, optimiser, schedule,
              updates, batch, episode-balanced sampling) and the file hashes, before anything runs.
``replay``    positive control: the recorded final commands through Codex's own ``Archive``;
              every pre-action observation must equal Codex's saved DP3 features bit for bit,
              and all three must pass ``Outcome`` (contact_acquisition_guard_v2). Else stop.
``train``     MLP on Codex's saved features; limits normalisation (to [-1, 1]) computed from
              the training data only, as DP3's LinearNormalizer(mode='limits'); last update kept.
``evaluate``  the last checkpoint; each condition x Codex's evaluation seeds; budget = the
              episode's recorded ticks; stop early only at a quality violation; success judged
              by ``Outcome`` at the end, exactly as Codex's DP3 evaluation.

Deliberate, stated differences from DP3: an MLP (3 x 512) instead of the diffusion U-Net; one
action per tick from the two-frame history instead of 8 executed of a 16-step chunk; no EMA
(the last update). Observations, the 27 final actuator commands (clip only, nothing added),
initial states, budgets and the stop rule are DP3's.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

DP3 = Path("/data1/yspark/projects/dexprompt/stage1-dp3-v1")
WIDTH, DEPTH = 512, 3
ARM_DEVIATION, FINGER_DEVIATION = 0.05, 0.10
RANGE_EPS = 1e-4


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    temporary.replace(path)


def dp3_inputs():
    package = DP3 / "package"
    return (json.loads((package / "plan.json").read_text()),
            json.loads((package / "episodes.json").read_text()), package)


def plan(out: Path) -> None:
    config, rows, package = dp3_inputs()
    if (out / "config.json").exists():
        raise SystemExit("config.json exists; a fixed run is not rewritten")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "config.json", {
        "fixed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "does an MLP BC complete the three DP3 training conditions under DP3's "
                    "observations, action, budget and stop rule?",
        "scope": "execution comparison on a small set; not unseen-object generalisation",
        "episodes": [r["name"] for r in rows],
        "dp3_files_sha256": {
            "plan.json": sha(package / "plan.json"),
            "episodes.json": sha(package / "episodes.json"),
            "selection.json": sha(package / "selection.json"),
            **{f"features/{r['name']}.npz": sha(DP3 / "run" / "features" / f"{r['name']}.npz")
               for r in rows}},
        "observation": "Codex Archive.observation(): point_cloud 640x3 + agent_pos 125, "
                       f"history {config['n_obs_steps']} (DP3 sequence_indices padding)",
        "action": "27 final actuator commands; clip to ctrlrange only; nothing added",
        "model": {"kind": "MLP", "width": WIDTH, "depth": DEPTH, "activation": "ELU",
                  "input": "flattened history x (640*3 + 125)", "output": 27,
                  "normalisation": "limits to [-1, 1] from training data only "
                                   f"(point cloud per xyz, state and action per dim, range floor "
                                   f"{RANGE_EPS})"},
        "training": {k: config[k] for k in ("seed", "batch_size", "learning_rate",
                                            "weight_decay", "warmup", "updates")}
        | {"optimiser": "AdamW", "schedule": "linear warmup then cosine (DP3)",
           "grad_clip": 1.0, "loss": "MSE on normalised action at the current tick",
           "sampling": "episode-balanced, tick uniform (DP3 batch())",
           "checkpoint": "last update, no selection"},
        "evaluation": {"seeds": config["evaluation_seeds"],
                       "budget": "episode frames", "stop": "quality violation only",
                       "success": "Outcome.summary() at the end (contact_acquisition_guard_v2)",
                       "first_deviation": f"|q - teacher right_state| > {ARM_DEVIATION} rad on "
                                          f"an arm joint or {FINGER_DEVIATION} rad on a finger",
                       "note": "the MLP is deterministic; seeds set as in DP3 for equal budget"},
        "differences_from_dp3": ["MLP instead of diffusion U-Net",
                                 "one action per tick instead of 8 of a 16-step chunk",
                                 "no EMA; last update"],
        "code_sha256": sha(__file__)})


def replay(out: Path) -> None:
    from stage1_dp3.environment import SIGNALS, Archive, Outcome

    config, rows, package = dp3_inputs()
    reports, passed = [], True
    for row in rows:
        sim = Archive(package, row)
        judge = Outcome(row["quality_limits"], sim.height)
        with np.load(package / row["arrays"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(DP3 / "run" / "features" / f"{row['name']}.npz", allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        features_identical, state_gap, gaps = True, 0.0, {k: 0.0 for k in SIGNALS}
        for i, command in enumerate(arrays["action"]):
            obs = sim.observation()
            features_identical &= bool(np.array_equal(obs["agent_pos"], saved["agent_pos"][i])
                                       and np.array_equal(obs["point_cloud"],
                                                          saved["point_cloud"][i]))
            state_gap = max(state_gap, float(np.max(np.abs(
                sim.data.qpos[sim.addresses].astype(np.float32) - arrays["right_state"][i]))))
            signals = sim.step(command)
            judge.update(signals)
            for key in SIGNALS:
                gaps[key] = max(gaps[key], float(np.max(np.abs(
                    np.asarray(signals[key], float) - np.asarray(arrays[key][i], float)))))
        outcome = judge.summary()
        ok = features_identical and state_gap == 0.0 and all(v == 0.0 for v in gaps.values()) \
            and outcome["success"]
        passed &= ok
        reports.append({"name": row["name"], "passed": ok,
                        "features_identical_to_dp3": features_identical,
                        "state_max_abs": state_gap, "signals_max_abs": gaps,
                        "phase_free_outcome": outcome})
        print(json.dumps(reports[-1] | {"signals_max_abs": max(gaps.values())}), flush=True)
    dump(out / "replay.json", {"passed": passed, "episodes": reports})
    if not passed:
        raise SystemExit("positive control failed; no training")


def limits(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = values.min(axis=axis), values.max(axis=axis)
    span = np.where(hi - lo < RANGE_EPS, 1.0, hi - lo)
    scale = 2.0 / span
    offset = -1.0 - scale * lo
    offset = np.where(hi - lo < RANGE_EPS, -scale * (hi + lo) / 2, offset)
    return scale.astype(np.float32), offset.astype(np.float32)


def model(in_dim: int):
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, size = [], in_dim
            for _ in range(DEPTH):
                layers += [nn.Linear(size, WIDTH), nn.ELU()]
                size = WIDTH
            layers.append(nn.Linear(size, 27))
            self.net = nn.Sequential(*layers)
            for name, n in (("pc", 3), ("state", 125), ("act", 27)):
                self.register_buffer(f"{name}_scale", torch.ones(n))
                self.register_buffer(f"{name}_offset", torch.zeros(n))

        def forward(self, points, state):
            # points: B x H x 640 x 3, state: B x H x 125 -> normalised action (B x 27)
            p = points * self.pc_scale + self.pc_offset
            s = state * self.state_scale + self.state_offset
            return self.net(torch.cat([p.flatten(1), s.flatten(1)], dim=1))

        def unnormalise(self, action):
            return (action - self.act_offset) / self.act_scale

    return MLP()


def features(rows):
    data = []
    for row in rows:
        with np.load(DP3 / "run" / "features" / f"{row['name']}.npz", allow_pickle=False) as z:
            data.append({k: z[k].astype(np.float32) for k in ("point_cloud", "agent_pos",
                                                              "action")})
    return data


def train(out: Path) -> None:
    import torch
    from stage1_dp3.experiment import sequence_indices

    config = json.loads((out / "config.json").read_text())
    dp3, rows, _ = dp3_inputs()
    t = config["training"]
    history = dp3["n_obs_steps"]
    torch.set_num_threads(1)
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    data = features(rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model(history * (640 * 3 + 125)).to(device)
    for name, values, axis in (
            ("pc", np.concatenate([d["point_cloud"].reshape(-1, 3) for d in data]), 0),
            ("state", np.concatenate([d["agent_pos"] for d in data]), 0),
            ("act", np.concatenate([d["action"] for d in data]), 0)):
        scale, offset = limits(values, axis)
        getattr(net, f"{name}_scale").copy_(torch.as_tensor(scale))
        getattr(net, f"{name}_offset").copy_(torch.as_tensor(offset))
    optimiser = torch.optim.AdamW(net.net.parameters(), lr=t["learning_rate"],
                                  weight_decay=t["weight_decay"])

    def batch(size):
        points, states, actions = [], [], []
        for _ in range(size):
            ep = data[int(rng.integers(len(data)))]
            tick = int(rng.integers(len(ep["action"])))
            oi, _ = sequence_indices(tick, len(ep["action"]), dp3["horizon"], history)
            points.append(ep["point_cloud"][oi])
            states.append(ep["agent_pos"][oi])
            actions.append(ep["action"][tick])
        as_t = lambda v: torch.as_tensor(np.stack(v), device=device)  # noqa: E731
        return as_t(points), as_t(states), as_t(actions)

    started, log = time.monotonic(), []
    for step in range(1, t["updates"] + 1):
        factor = min(step / t["warmup"], 1.0)
        if step > t["warmup"]:
            factor *= 0.5 * (1 + math.cos(math.pi * (step - t["warmup"])
                                          / (t["updates"] - t["warmup"])))
        for group in optimiser.param_groups:
            group["lr"] = t["learning_rate"] * factor
        points, states, actions = batch(t["batch_size"])
        target = actions * net.act_scale + net.act_offset
        loss = torch.mean((net(points, states) - target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.net.parameters(), 1.0, error_if_nonfinite=True)
        optimiser.step()
        if step == 1 or step % 250 == 0:
            log.append({"update": step, "loss": float(loss),
                        "seconds": round(time.monotonic() - started, 1)})
            print(json.dumps(log[-1]), flush=True)
    torch.save({"model": net.cpu().state_dict(), "config": config, "updates": t["updates"],
                "history": history}, out / "final.pt")
    dump(out / "training-result.json", {"completed": True, "updates": t["updates"],
                                        "final_loss": log[-1]["loss"], "log": log,
                                        "seconds": round(time.monotonic() - started, 1),
                                        "device": device,
                                        "checkpoint_sha256": sha(out / "final.pt")})


def evaluate(out: Path) -> None:
    import torch
    from stage1_dp3.environment import Archive, Outcome

    config = json.loads((out / "config.json").read_text())
    dp3, rows, package = dp3_inputs()
    saved = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    if saved["updates"] != config["training"]["updates"] or saved["config"] != config:
        raise SystemExit("not the prespecified final checkpoint")
    history_len = saved["history"]
    net = model(history_len * (640 * 3 + 125))
    net.load_state_dict(saved["model"])
    net.eval()
    torch.set_num_threads(1)
    results = []
    for row in rows:
        with np.load(package / row["arrays"], allow_pickle=False) as z:
            teacher = z["right_state"].astype(float)
        for seed in config["evaluation"]["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            sim = Archive(package, row)
            judge = Outcome(row["quality_limits"], sim.height)
            history = deque([sim.observation() for _ in range(history_len)], maxlen=history_len)
            clipped, deviation, measured, actions = 0, None, [], []
            started = time.monotonic()
            for tick in range(row["frames"]):
                current = sim.observation()
                if tick:
                    history.append(current)
                q = sim.data.qpos[sim.addresses].copy()
                if deviation is None and tick < len(teacher):
                    gap = np.abs(q - teacher[tick])
                    if gap[:7].max() > ARM_DEVIATION or gap[7:].max() > FINGER_DEVIATION:
                        deviation = {"tick": tick, "arm_max": float(gap[:7].max()),
                                     "finger_max": float(gap[7:].max())}
                points = torch.as_tensor(np.stack([h["point_cloud"] for h in history])[None])
                states = torch.as_tensor(np.stack([h["agent_pos"] for h in history])[None])
                with torch.no_grad():
                    command = net.unnormalise(net(points, states))[0].numpy().astype(np.float64)
                bounded = np.clip(command, sim.low, sim.high)
                clipped += int(np.any(bounded != command))
                measured.append(q)
                actions.append(bounded)
                judge.update(sim.step(bounded))
                if judge.failure:
                    break
            result = {"name": row["name"], "object": row["object_id"], "seed": seed,
                      "saturated_ticks": clipped, "first_deviation": deviation,
                      "seconds": round(time.monotonic() - started, 1), **judge.summary()}
            results.append(result)
            np.savez_compressed(out / f"rollout-{row['name']}-{seed}.npz",
                                right_state=np.stack(measured), action=np.stack(actions))
            dump(out / "evaluation.json", {
                "completed": len(results) == len(rows) * len(config["evaluation"]["seeds"]),
                "successes": sum(r["success"] for r in results), "attempts": len(results),
                "checkpoint_sha256": sha(out / "final.pt"), "episodes": results})
            print(json.dumps(result), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "replay", "train", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"plan": plan, "replay": replay, "train": train, "evaluate": evaluate}[args.stage](
        args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
