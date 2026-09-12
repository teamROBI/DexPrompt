"""Forward-pass sensitivity of bc-v1 / bc-v2 along teacher trajectories (no training, no physics)."""
import json
from pathlib import Path

import numpy as np
import torch

from stage1_bc.archive import SIGNALS, load_row_scene
from stage1_bc.obs import ActionMap, Static, observe
from stage1_bc.policy import load

D = Path("/data1/yspark/projects/dexprompt/stage1-goal-bc-v1")
package, augment = D / "package", D / "runs/replay-all"
rows = {r["name"]: r for r in map(json.loads, (package / "episodes.jsonl").read_text().splitlines())}
names = json.loads((D / "runs/bc-v2/config.json").read_text())["development_conditions"]
rng = np.random.default_rng(0)
EPS, TICKS = 0.005, (0, 5, 10, 20, 40, 200)
result = {}
torch.set_num_threads(4)
for run in ("bc-v1", "bc-v2"):
    actor, _, meta = load(D / "runs" / run / "bc.pt")
    actor.eval()
    amap = ActionMap.from_meta(meta)
    gains = {"arm_q->command": [], "finger_q->command": [], "time+1tick->command": []}
    for name in names:
        row = rows[name]
        with np.load(package / row["arrays"]) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(augment / "augment" / f"{name}.npz") as z:
            pre = {k: z[k] for k in z.files}
        scene = load_row_scene(package, row)
        static = Static.from_row(row, scene)
        for t in TICKS:
            state = {k: pre[k][t] for k in pre}
            previous = None if t == 0 else {k: arrays[k][t - 1] for k in SIGNALS}

            def cmd(st, tick):
                o = observe(static, st, previous, scene.height_reference, tick)
                with torch.no_grad():
                    u = actor(torch.as_tensor(o)[None])[0].numpy()
                return amap.command(st["q"], u)
            base = cmd(state, t)
            for group, sl in (("arm_q->command", slice(0, 7)), ("finger_q->command", slice(7, 27))):
                for _ in range(5):
                    d = rng.normal(size=sl.stop - sl.start)
                    d *= EPS / np.linalg.norm(d)
                    q = state["q"].copy()
                    q[sl] += d
                    moved = cmd({**state, "q": q}, t)
                    gains[group].append(float(np.linalg.norm(moved - base) / EPS))
            gains["time+1tick->command"].append(float(np.linalg.norm(cmd(state, t + 1) - base)))
    result[run] = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                       "max": float(np.max(v))} for k, v in gains.items()}
    result[run]["teacher_command_step_median"] = None
(D / "runs/analysis/sensitivity.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))
