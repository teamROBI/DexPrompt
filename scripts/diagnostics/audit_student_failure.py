"""Audit saved student commands and training contracts, without loading a policy.

CPU only: replay saved commands in the archived simulator; compare the resulting
signals to the original evaluation. No new sampling, training or controller change.
"""

# ruff: noqa: E402 -- simulator backend and project import path precede imports.

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from replay_hx5_student import restore_recorder

from icra.dataset import file_sha256
from icra.execution import ContactMeter, aggregate
from icra.simulation import forward, restore
from icra.student_cache import StudentCache
from icra.student_data import EpisodeChunks, fit_normalizer
from icra.student_inference import StudentActor


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ObservationProbe(torch.nn.Module):
    """Exercise the real StudentActor preprocessing without a visual forward pass."""

    def __init__(self, config):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = config
        self.last_observation = None

    def predict_action(self, observation, *, generator=None):
        self.last_observation = observation
        # Distinct values expose a chunk offset or order error in StudentActor.
        return torch.arange(self.config["action_horizon"]).float()[None, :, None].expand(1, -1, 27)


def replay_commands(recorder, initial, entry, commands):
    restore(recorder.model, recorder.data, initial)
    body = recorder.object_body(entry["object_id"])
    groups = entry["provenance"]["required_contact_groups"]
    distractors = entry["provenance"]["execution"].get("distractor_ids", [])
    if distractors or entry["condition"]["kind"] != "object":
        raise ValueError("this diagnostic currently requires object tasks without distractors")
    meter = ContactMeter(recorder, body, groups, [], None)
    rows, measured, contacts = [], [], []
    initial_ctrl = recorder.targets().astype(float)
    initial_self_contacts = [
        float(-c.dist)
        for c in recorder.data.contact
        if c.dist < 0
        and all(
            recorder.model.body(recorder.model.geom_bodyid[g]).name.startswith(
                ("arm_r_", "finger_r_")
            )
            for g in (c.geom1, c.geom2)
        )
    ]
    ratio = 0.02 / recorder.model.opt.timestep
    if not np.isclose(ratio, round(ratio), atol=1e-12, rtol=0):
        raise ValueError("control interval is not an integer number of physics steps")
    for tick, command in enumerate(commands):
        measured.append(recorder.measured().copy())
        recorder.command(command)
        samples = []
        for substep in range(round(ratio)):
            recorder.mj.mj_step(recorder.model, recorder.data)
            forward(recorder.model, recorder.data)
            samples.append(meter.read())
            for contact in recorder.data.contact:
                if contact.dist >= 0:
                    continue
                names = [
                    recorder.model.body(recorder.model.geom_bodyid[g]).name
                    for g in (contact.geom1, contact.geom2)
                ]
                if not any(n.startswith(("arm_r_", "finger_r_")) for n in names):
                    continue
                contacts.append(
                    {
                        "tick": tick,
                        "substep": substep,
                        "bodies": names,
                        "depth_m": float(-contact.dist),
                    }
                )
        rows.append(aggregate(samples))
    signals = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    return np.asarray(measured), initial_ctrl, signals, contacts, initial_self_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("audit requires a fresh output directory")
    torch.set_num_threads(1)
    manifest = json.loads((args.data / "manifest.json").read_text())
    run = json.loads(args.run.read_text())
    source_matches = {p: file_sha256(ROOT / p) == sha for p, sha in run["source_sha256"].items()}
    assert all(source_matches.values()), "training source changed"
    assert file_sha256(args.data / "manifest.json") == run["dataset_sha256"]
    assert manifest["joint_names"] == run["joint_names"]
    assert manifest["control_dt"] == run["control_dt"] == 0.02
    assert manifest["action_representation"] == run["action_representation"]
    # The cache was sealed and fully checked at training and server preflight.
    # Here headers and manifest bindings are checked; do not claim a fresh full hash audit.
    cache = StudentCache(args.cache, args.data, manifest, verify_hashes=False)
    normalizer = fit_normalizer(args.data, manifest, cache=cache)
    assert normalizer == run["normalizer"], "recomputed normalizer differs"
    cfg = run["config"]
    dataset = EpisodeChunks(
        args.data,
        manifest,
        "train",
        cfg["action_horizon"],
        cfg["image_size"],
        cache=cache,
        observation_version=cfg["observation_version"],
        goal_image_size=cfg["goal_image_size"],
        goal_reference_policy=cfg["goal_reference_policy"],
    )
    report = {
        "training_started": False,
        "source_matches": source_matches,
        "normalizer_exact": True,
        "cache_full_hash_recheck": False,
        "units": "radians; tick indices are zero-based",
        "episodes": [],
    }
    args.out.mkdir(parents=True)
    all_traces, teacher_actions = {}, {}
    for index, entry in enumerate(dataset.entries):
        eid = entry["episode_id"]
        episode = cache.open_episode(eid)
        evaluation = args.evaluation / eid
        with np.load(evaluation / "initial_observation.npz") as z:
            observed = {k: z[k] for k in z.files}
        pixel_diffs = {
            k: {
                "equal": bool(np.array_equal(v, episode[k][0])),
                "max_abs": float(np.max(np.abs(v.astype(float) - episode[k][0]))),
            }
            for k, v in observed.items()
        }
        # Follow the causal memory along recorded states; compare at camera-reference
        # acquisition boundaries, the first frame, a later frame and the last frame.
        probe = ObservationProbe(cfg)
        actor = StudentActor(
            probe,
            {"run": run},
            joint_names=manifest["joint_names"],
            control_dt=0.02,
            execute_steps=1,
        )
        first = int(dataset._ends[index - 1]) if index else 0
        sample_frames = {dataset.starts[index], len(episode["action"]) - 1, 100}
        sample_frames.update(x for x in dataset.reference_indices[index] if x is not None)
        tensor_checks = []
        for tick in range(len(episode["action"])):
            obs = {k: episode[k][tick] for k in observed}
            actor.goal_memory.update(obs, entry["condition"])
            if tick not in sample_frames:
                continue
            actor.condition_key = json.dumps(entry["condition"], sort_keys=True)
            actor.act(obs, entry["condition"])
            sample = dataset[first + tick - dataset.starts[index]]
            delta = {
                k: float((v - probe.last_observation[k][0]).abs().max())
                for k, v in sample["obs"].items()
            }
            assert max(delta.values()) == 0, (eid, tick, delta)
            expected = episode["action"][
                np.minimum(
                    np.arange(tick, tick + cfg["action_horizon"]), len(episode["action"]) - 1
                )
            ]
            assert np.array_equal(sample["action"].numpy(), expected.astype(np.float32))
            tensor_checks.append({"tick": tick, "max_abs": max(delta.values())})
        # Check the real receding-horizon queue independently of observation acquisition.
        actor.reset()
        actor.execute_steps = cfg["execute_steps"]
        sequence = [float(actor.act(observed, entry["condition"])[0]) for _ in range(16)]
        assert sequence == [float(i % cfg["execute_steps"]) for i in range(16)]
        recorder, initial, _ = restore_recorder(args.data, manifest, entry, render=False)
        assert all(
            recorder.model.actuator_trnid[a, 0] == recorder.model.joint(n).id
            for a, n in zip(recorder.actuators, manifest["joint_names"], strict=True)
        )
        execution = entry["provenance"]["execution"]
        contract = {
            "model_sha256": file_sha256(args.data / execution["model"]),
            "initial_state_sha256": execution["initial_state_sha256"],
            "condition": entry["condition"],
            "limits": entry["quality"]["limits"],
            "required_contact_groups": entry["provenance"]["required_contact_groups"],
            "tick_budget": len(episode["action"]),
        }
        item = {
            "episode_id": eid,
            "contract": contract,
            "contract_sha256": digest_json(contract),
            "initial_input": pixel_diffs,
            "preprocessing_checks": tensor_checks,
            "chunk_sequence": sequence,
            "reference_indices": dataset.reference_indices[index],
            "rollouts": [],
        }
        teacher_actions[eid] = episode["action"]
        for trace_path in sorted(evaluation.glob("replay_*.npz")):
            with np.load(trace_path) as z:
                trace = {k: z[k] for k in z.files}
            command = trace["action"]
            measured, initial_ctrl, signals, contacts, initial_self = replay_commands(
                recorder, initial, entry, command
            )
            replay_delta = {
                k: float(np.max(np.abs(signals[k].astype(float) - v)))
                for k, v in trace.items()
                if k != "action"
            }
            assert max(replay_delta.values()) < 1e-6, (eid, trace_path, replay_delta)
            teacher = episode["action"][: len(command)]
            jump = np.diff(np.vstack([initial_ctrl, command]), axis=0)
            teacher_jump = np.diff(np.vstack([initial_ctrl, teacher]), axis=0)
            per_joint = []
            for j, name in enumerate(manifest["joint_names"]):
                per_joint.append(
                    {
                        "joint": name,
                        "first_student_minus_teacher_rad": float(command[0, j] - teacher[0, j]),
                        "first_student_minus_measured_rad": float(command[0, j] - measured[0, j]),
                        "student_max_command_step_rad": float(np.abs(jump[:, j]).max()),
                        "teacher_max_command_step_rad": float(np.abs(teacher_jump[:, j]).max()),
                        "max_command_minus_measured_rad": float(
                            np.abs(command[:, j] - measured[:, j]).max()
                        ),
                    }
                )
            item["rollouts"].append(
                {
                    "trace": trace_path.name,
                    "ticks": len(command),
                    "replay_signal_max_abs": replay_delta,
                    "first_negative_contact": contacts[0] if contacts else None,
                    "initial_self_contact_depths_m": initial_self,
                    "first_external_contact": next(
                        (
                            c
                            for c in contacts
                            if any(b == "world" or b.startswith("object_") for b in c["bodies"])
                        ),
                        None,
                    ),
                    "deepest_contact": max(contacts, key=lambda c: c["depth_m"])
                    if contacts
                    else None,
                    "per_joint": per_joint,
                }
            )
            all_traces[eid, trace_path.name] = trace
            np.savez_compressed(
                args.out / f"{eid}-{trace_path.stem}.npz", action=command, pre_action_state=measured
            )
        prefix = max(t["ticks"] for t in item["rollouts"])
        teacher_measured, _, _, _, _ = replay_commands(
            recorder, initial, entry, episode["action"][:prefix]
        )
        alignment_error = float(np.abs(teacher_measured - episode["right_state"][:prefix]).max())
        assert alignment_error < 1e-6, (eid, alignment_error)
        item["teacher_pre_action_alignment"] = {
            "ticks": prefix,
            "max_abs_rad": alignment_error,
            "control_dt_max_abs": float(np.abs(episode["control_dt"] - 0.02).max()),
            "timestamp_step_max_abs": float(np.abs(np.diff(episode["timestamp"]) - 0.02).max()),
        }
        report["episodes"].append(item)
        print(json.dumps({"completed": eid, "rollouts": len(item["rollouts"])}), flush=True)
    pairs = []
    ids = list(teacher_actions)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            n = min(len(teacher_actions[left]), len(teacher_actions[right]))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "teacher_first_action_max_abs": float(
                        np.max(np.abs(teacher_actions[left][0] - teacher_actions[right][0]))
                    ),
                    "teacher_prefix_max_abs": float(
                        np.max(np.abs(teacher_actions[left][:n] - teacher_actions[right][:n]))
                    ),
                    "traces_equal": {
                        name: all(
                            np.array_equal(v, all_traces[right, name][k])
                            for k, v in all_traces[left, name].items()
                        )
                        for name in ("replay_000.npz", "replay_001.npz", "replay_002.npz")
                    },
                }
            )
    report["pairs"] = pairs
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Audit complete; no policy forward pass or training performed.", flush=True)


if __name__ == "__main__":
    main()
