"""Goal-conditioned BC on the replay-augmented teacher episodes (settings from config.json)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from stage1_bc.archive import SIGNALS, load_row_scene
from stage1_bc.obs import OBS_DIM, ActionMap, Static, observe
from stage1_bc.policy import Actor, save


def episode_frames(package: Path, augment: Path, row: dict):
    """(obs, q, command, phase) per tick; obs uses only pre-action information."""
    with np.load(package / row["arrays"]) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(augment / "augment" / f"{row['name']}.npz") as z:
        pre = {k: z[k] for k in z.files}
    scene = load_row_scene(package, row)
    static = Static.from_row(row, scene)
    reference = scene.height_reference
    obs, previous = [], None
    for i in range(len(arrays["action"])):
        state = {k: pre[k][i] for k in pre}
        obs.append(observe(static, state, previous, reference, i))
        previous = {k: arrays[k][i] for k in SIGNALS}
    return (np.stack(obs), pre["q"], arrays["action"], arrays["phase"],
            (scene.low, scene.high))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--augment", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads((args.run / "config.json").read_text())
    bc = config["bc"]
    torch.manual_seed(bc["seed"])
    np.random.seed(bc["seed"])
    rows = {r["name"]: r for r in (json.loads(line) for line in
                                   (args.package / "episodes.jsonl").read_text().splitlines())}
    obs, q, command, phase, bounds = [], [], [], [], None
    for name in config["training_episodes"]:
        o, qq, c, ph, bounds = episode_frames(args.package, args.augment, rows[name])
        obs.append(o), q.append(qq), command.append(c), phase.append(ph)
    obs, q, command, phase = map(np.concatenate, (obs, q, command, phase))
    assert obs.shape[1] == OBS_DIM
    mode = config.get("action_mode", "relative")
    if mode == "absolute":
        offset = command.mean(axis=0)
        scale = np.maximum(command.std(axis=0), 1e-3)
    else:
        offset = None
        scale = np.maximum((command - q).std(axis=0), 1e-3)
    amap = ActionMap(scale, *bounds, offset=offset)
    target = amap.target(q, command).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = Actor(OBS_DIM, 27, bc["width"]).to(device)
    actor.norm.set(obs.mean(0), obs.std(0))
    optimiser = torch.optim.Adam(actor.net.parameters(), lr=bc["lr"],
                                 weight_decay=bc["weight_decay"])
    x = torch.as_tensor(obs, device=device)
    y = torch.as_tensor(target, device=device)
    log, started = [], time.monotonic()
    for step in range(1, bc["steps"] + 1):
        batch = torch.randint(0, len(x), (bc["batch"],), device=device)
        loss = torch.mean((actor(x[batch]) - y[batch]) ** 2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 200 == 0 or step == 1:
            log.append({"step": step, "loss": float(loss), "seconds":
                        round(time.monotonic() - started, 1)})
            print(json.dumps(log[-1]), flush=True)
    with torch.no_grad():
        predicted = torch.cat([actor(x[i:i + 65536]) for i in range(0, len(x), 65536)])
        residual = (predicted - y).cpu().numpy()
        actor.log_std.copy_(torch.as_tensor(np.log(np.maximum(residual.std(0), 1e-3))))
    by_phase = {p: float(np.mean(residual[phase == p] ** 2)) for p in np.unique(phase)}
    save(args.run / "bc.pt", actor.cpu(), {
        "obs_dim": OBS_DIM, "act_dim": 27, "width": bc["width"], "scale": scale.tolist(),
        "action_mode": mode, "offset": None if offset is None else offset.tolist(),
        "low": bounds[0].tolist(), "high": bounds[1].tolist(), "config": config,
        "frames": int(len(x))})
    (args.run / "bc-train.json").write_text(json.dumps({
        "frames": int(len(x)), "episodes": len(config["training_episodes"]),
        "final_loss": log[-1]["loss"], "log": log,
        "training_mse_by_phase_diagnostic_only": by_phase,
        "seconds": round(time.monotonic() - started, 1), "device": device}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
