"""Why a stage-1 BC policy does not complete a training condition: link, open loop, first ticks.

For each development condition (all in training):

1. teacher-through-evaluation -- the recorded teacher commands sent through the very
   evaluation path (observation, ActionMap round trip, clip, collector-exact step, Judge); it
   must reproduce the teacher, otherwise the data/execution link is broken;
2. open loop -- the BC command on the teacher's own pre-action observations (from the replay
   augment) against the recorded command, per tick and joint group;
3. closed loop, first ticks -- the BC rollout's command and joints against the teacher's,
   and each finger command's distance from the open command.

Short rollouts only; nothing is trained, nothing is tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EARLY = 60


def one(package: str, augment: str, checkpoint: str, row: dict) -> dict:
    import torch

    from icra.dataset import QualityLimits
    from stage1_bc.archive import SIGNALS, Judge, load_row_scene
    from stage1_bc.obs import ActionMap, Static, observe
    from stage1_bc.policy import load

    torch.set_num_threads(1)
    package, augment = Path(package), Path(augment)
    actor, _, meta = load(checkpoint)
    actor.eval()
    amap = ActionMap.from_meta(meta)
    with np.load(package / row["arrays"]) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(augment / "augment" / f"{row['name']}.npz") as z:
        pre = {k: z[k] for k in z.files}
    scene = load_row_scene(package, row)
    static = Static.from_row(row, scene)
    teacher_cmd = arrays["action"]
    result = {"name": row["name"]}

    # 1. teacher commands through the evaluation path
    judge = Judge(QualityLimits(), scene.height_reference)
    identical = True
    for i in range(len(teacher_cmd)):
        state = scene.state()
        command = amap.command(state["q"], amap.target(state["q"], teacher_cmd[i]))
        identical &= bool(np.array_equal(command, teacher_cmd[i]))
        signals = scene.step(command)
        identical &= all(np.array_equal(np.asarray(signals[k]), np.asarray(arrays[k][i]))
                         for k in SIGNALS)
        if judge.update(signals) or judge.success:
            break
    result["teacher_through_evaluation"] = {"success": judge.success,
                                            "failure": judge.failure, "ticks": judge.ticks,
                                            "commands_and_signals_identical": identical}

    # 2. open loop on the teacher's own observations
    obs, previous = [], None
    for i in range(len(teacher_cmd)):
        obs.append(observe(static, {k: pre[k][i] for k in pre}, previous,
                           scene.height_reference, i))
        previous = {k: arrays[k][i] for k in SIGNALS}
    with torch.no_grad():
        u = actor(torch.as_tensor(np.stack(obs))).numpy()
    predicted = np.stack([amap.command(pre["q"][i], u[i]) for i in range(len(u))])
    error = np.abs(predicted - teacher_cmd)
    phases = arrays["phase"]
    result["open_loop"] = {
        "arm_mean_rad": float(error[:, :7].mean()), "finger_mean_rad": float(error[:, 7:].mean()),
        "first_ticks_arm_max": float(error[:EARLY, :7].max()),
        "first_ticks_finger_max": float(error[:EARLY, 7:].max()),
        "by_phase": {str(p): {"arm": float(error[phases == p, :7].mean()),
                              "finger": float(error[phases == p, 7:].mean())}
                     for p in np.unique(phases)}}

    # 3. closed loop, first ticks
    scene.reset()
    previous, trace = None, []
    for i in range(EARLY):
        state = scene.state()
        o = observe(static, state, previous, scene.height_reference, i)
        with torch.no_grad():
            ui = actor(torch.as_tensor(o)[None])[0].numpy()
        command = amap.command(state["q"], ui)
        arm_gap = float(np.abs(command[:7] - teacher_cmd[i, :7]).max())
        trace.append({"tick": i,
                      "command_vs_teacher_arm": arm_gap,
                      "command_vs_teacher_finger": float(np.abs(command[7:]
                                                                - teacher_cmd[i, 7:]).max()),
                      "q_vs_teacher_arm": float(np.abs(state["q"][:7] - pre["q"][i, :7]).max()),
                      "q_vs_teacher_finger": float(np.abs(state["q"][7:] - pre["q"][i, 7:]).max()),
                      "finger_command_from_open": float(np.abs(command[7:]
                                                               - static.open_cmd).max()),
                      "teacher_finger_command_from_open": float(np.abs(teacher_cmd[i, 7:]
                                                                       - static.open_cmd).max())})
        previous = scene.step(command)
    result["closed_loop_first_ticks"] = trace[::5]
    return result


def main(argv=None) -> int:
    import concurrent.futures as futures

    import torch

    from stage1_bc.policy import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--augment", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    config = json.loads((args.run / "config.json").read_text())
    rows = {r["name"]: r for r in (json.loads(line) for line in
                                   (args.package / "episodes.jsonl").read_text().splitlines())}
    names = config["development_conditions"]
    _, _, meta = load(args.checkpoint)
    scale = np.asarray(meta["scale"])
    torch.set_num_threads(1)
    with futures.ProcessPoolExecutor(min(args.workers, len(names))) as pool:
        results = list(pool.map(one, [str(args.package)] * len(names),
                                [str(args.augment)] * len(names),
                                [str(args.checkpoint)] * len(names), [rows[n] for n in names]))
    summary = {"action_scale": {"arm": scale[:7].round(5).tolist(),
                                "finger_min": float(scale[7:].min()),
                                "finger_max": float(scale[7:].max())},
               "results": results}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    for r in results:
        t, o = r["teacher_through_evaluation"], r["open_loop"]
        print(json.dumps({"name": r["name"], "teacher_path_success": t["success"],
                          "teacher_path_identical": t["commands_and_signals_identical"],
                          "open_loop_first_ticks_finger_max": round(o["first_ticks_finger_max"], 4),
                          "open_loop_first_ticks_arm_max": round(o["first_ticks_arm_max"], 4),
                          "closed_tick_20": r["closed_loop_first_ticks"][4]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
