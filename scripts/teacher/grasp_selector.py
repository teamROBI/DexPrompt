# ruff: noqa: E402 -- direct execution requires bootstrap before project imports.
"""A minimal grasp-candidate selector over the execution ledger -- wiring, not a result.

Scores a (object, scene, candidate) for full home->approach->close->lift->hold
success, with stage-conditional scores (approach; close given approach; carry
given close). Inputs are only what is known before execution:

* object shape: a deterministic surface sample of the exact mesh (object frame,
  metres), encoded by a small point encoder;
* scene: placement yaw, table offset, object dimensions, distractor count and
  their fixed offsets from the placement file;
* candidate: grasp/squeeze/open joint targets, grasp type, required contact links.

The robot's initial state is the same home pose in every recorded row, so it is
recorded as constant rather than fed as a feature. Labels come from the ledger's
stage labels with their observation masks; an unobserved stage contributes no
loss. Folds are leave-one-geometry-group-out; normalisation statistics come from
the training fold only and travel in the checkpoint.

``check``     Real ledger batch -> forward -> masked loss -> ranking -> checkpoint
              save/load -> evaluate. Fails if backward or an optimiser step runs.
``train``     Implemented and resumable; refuses to run without ``--confirm-training``.
``evaluate``  Metrics on observed rows of the held-out group, and ranking coverage:
              how many top-k candidates have an observed label at all. Unobserved
              candidates are never counted as successes or failures.

    third_party/dexonomy/.venv/bin/python scripts/teacher/grasp_selector.py check \\
        --ledger <ledger dir> --out <dir>
"""

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
import math
import time
from pathlib import Path

import numpy as np

POINTS = 512
POINT_SEED = 0
HEADS = ("full", "approach", "close", "carry")
MAX_DISTRACTORS = 2


def sha(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- features ------------------------------------------------------------------------


def surface_points(mesh_path: str, cache: Path, digest: str | None = None) -> np.ndarray:
    """A fixed-seed surface sample of the exact mesh, cached by mesh hash.

    With ``digest`` (the ledger's recorded mesh hash) a cached sample is used without
    the mesh file, so a training bundle can move to a machine without the assets.
    """
    import trimesh

    digest = digest or sha(mesh_path)
    cached = cache / f"{digest}.npy"
    if cached.is_file():
        return np.load(cached)
    mesh = trimesh.load(mesh_path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, POINTS, seed=POINT_SEED)
    points = np.asarray(points, np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cached, points)
    return points


def link_mask(required) -> np.ndarray:
    mask = np.zeros(20, np.float32)
    for group in required or []:
        for name in group:
            index = int(name.removeprefix("finger_r_link")) - 1
            if 0 <= index < 20:
                mask[index] = 1.0
    return mask


def distractor_offsets(inputs: dict) -> np.ndarray:
    out = np.zeros((MAX_DISTRACTORS, 3), np.float32)
    path = inputs.get("placement_file")
    if path and Path(path).is_file():
        offsets = json.loads(Path(path).read_text())["offsets"]
        for i, name in enumerate(n for n in inputs.get("distractors", []) if n in offsets):
            if i < MAX_DISTRACTORS:
                out[i] = [*offsets[name], 1.0]
    return out.reshape(-1)


def scene_vector(inputs: dict) -> np.ndarray:
    yaw = math.radians(float(inputs.get("object_yaw_degrees") or 0.0))
    offset = inputs.get("object_offset") or [0.0, 0.0]
    dims = np.asarray(inputs.get("object_dims_mm") or [0, 0, 0], np.float32) / 1000.0
    return np.concatenate([[math.sin(yaw), math.cos(yaw)], np.asarray(offset, np.float32),
                           dims, [float(len(inputs.get("distractors") or []))],
                           distractor_offsets(inputs)]).astype(np.float32)


def candidate_vector(features: dict, grasp_type: str, types: list[str]) -> np.ndarray:
    onehot = np.zeros(len(types), np.float32)
    if grasp_type in types:
        onehot[types.index(grasp_type)] = 1.0
    return np.concatenate([np.asarray(features["grasp_qpos"], np.float32),
                           np.asarray(features["squeeze_qpos"], np.float32),
                           np.asarray(features["open_qpos"], np.float32),
                           link_mask(features["required"]), onehot]).astype(np.float32)


def label_arrays(labels: dict) -> tuple[np.ndarray, np.ndarray]:
    """Targets and observation masks for the four heads; unobserved -> mask 0."""
    y = np.zeros(len(HEADS), np.float32)
    m = np.zeros(len(HEADS), np.float32)
    for i, head in enumerate(HEADS):
        stage = labels[head]
        if stage["observed"] and stage["succeeded"] is not None:
            y[i], m[i] = float(bool(stage["succeeded"])), 1.0
    return y, m


class Dataset:
    """Independent, labelled executions of one physics cohort, joined to their candidates."""

    def __init__(self, ledger: Path, cache: Path, physics: str = "A",
                 centre_single_only: bool = False, exclude: set | None = None):
        self.ledger = ledger
        executions = read_jsonl(ledger / "executions.jsonl")
        candidates = read_jsonl(ledger / "candidates.jsonl")
        self.pool = [c for c in candidates if c["physics"] == physics]
        index = {(c["object"], c["candidate_sha256"]): c for c in self.pool}
        self.types = sorted({c["grasp_type"] for c in self.pool})
        rows, skipped = [], {"replicate": 0, "unlabelled": 0, "other_cohort": 0,
                             "no_candidate_features": 0, "no_mesh": 0,
                             "placement_or_clutter": 0, "excluded_candidate": 0}
        for r in executions:
            if r["cohort"]["physics"] != physics:
                skipped["other_cohort"] += 1
            elif centre_single_only and (r["inputs"]["placement"] != "centre"
                                         or r["inputs"]["distractors"]):
                skipped["placement_or_clutter"] += 1
            elif exclude and r["inputs"]["candidate_sha256"] in exclude:
                skipped["excluded_candidate"] += 1
            elif r.get("replicate_of") is not None:
                skipped["replicate"] += 1
            elif not r["labels"]["full"]["observed"]:
                skipped["unlabelled"] += 1
            elif (r["inputs"]["object"], r["inputs"]["candidate_sha256"]) not in index:
                skipped["no_candidate_features"] += 1
            elif not r["inputs"].get("mesh_path"):
                skipped["no_mesh"] += 1
            else:
                rows.append(r)
        self.rows, self.skipped = rows, skipped
        self.index, self.cache = index, cache
        self.groups = sorted({r["inputs"]["geometry_group"] for r in rows})

    def encode(self, inputs: dict, candidate: dict):
        points = surface_points(inputs["mesh_path"], self.cache, inputs.get("mesh_sha256"))
        return (points, scene_vector(inputs),
                candidate_vector(candidate["features"], candidate["grasp_type"], self.types))

    def tensors(self, rows: list[dict]):
        import torch

        pts, scn, cnd, ys, ms = [], [], [], [], []
        for r in rows:
            candidate = self.index[(r["inputs"]["object"], r["inputs"]["candidate_sha256"])]
            p, s, c = self.encode(r["inputs"], candidate)
            y, m = label_arrays(r["labels"])
            pts.append(p), scn.append(s), cnd.append(c), ys.append(y), ms.append(m)
        return tuple(torch.as_tensor(np.stack(x)) for x in (pts, scn, cnd, ys, ms))

    def folds(self) -> list[dict]:
        out = []
        for held in self.groups:
            train = [r for r in self.rows if r["inputs"]["geometry_group"] != held]
            dev = [r for r in self.rows if r["inputs"]["geometry_group"] == held]
            availability = {}
            for name, part in (("train", train), ("dev", dev)):
                ys = np.stack([label_arrays(r["labels"])[0] for r in part]) if part else None
                mask = np.stack([label_arrays(r["labels"])[1] for r in part]) if part else None
                availability[name] = {
                    head: {"observed": int(mask[:, i].sum()) if part else 0,
                           "positive": int((ys[:, i] * mask[:, i]).sum()) if part else 0}
                    for i, head in enumerate(HEADS)}
            out.append({"held_out_group": held, "train_rows": [r["row_id"] for r in train],
                        "dev_rows": [r["row_id"] for r in dev], "availability": availability,
                        "dev_evaluable": {h: 0 < availability["dev"][h]["positive"]
                                          < availability["dev"][h]["observed"] for h in HEADS}})
        return out


# -- model ---------------------------------------------------------------------------


def build_model(scene_dim: int, candidate_dim: int, width: int = 128):
    import torch
    from torch import nn

    class Selector(nn.Module):
        def __init__(self):
            super().__init__()
            self.points = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, width),
                                        nn.ReLU())
            self.body = nn.Sequential(nn.Linear(width + scene_dim + candidate_dim, 2 * width),
                                      nn.ReLU(), nn.Linear(2 * width, width), nn.ReLU())
            self.heads = nn.Linear(width, len(HEADS))
            self.register_buffer("scene_mean", torch.zeros(scene_dim))
            self.register_buffer("scene_std", torch.ones(scene_dim))
            self.register_buffer("candidate_mean", torch.zeros(candidate_dim))
            self.register_buffer("candidate_std", torch.ones(candidate_dim))

        def forward(self, points, scene, candidate):
            shape = self.points(points).max(dim=1).values
            scene = (scene - self.scene_mean) / self.scene_std
            candidate = (candidate - self.candidate_mean) / self.candidate_std
            return self.heads(self.body(torch.cat([shape, scene, candidate], dim=-1)))

    return Selector()


def masked_loss(logits, targets, masks):
    import torch

    per = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    observed = masks.sum(dim=0).clamp_min(1.0)
    return ((per * masks).sum(dim=0) / observed).sum()


def set_normalisation(model, scene, candidate) -> None:
    import torch

    with torch.no_grad():
        for name, x in (("scene", scene), ("candidate", candidate)):
            std = x.std(dim=0, unbiased=False)
            getattr(model, f"{name}_mean").copy_(x.mean(dim=0))
            getattr(model, f"{name}_std").copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))


def parameter_hash(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, model, contract: dict, optimiser=None, epoch: int = 0) -> None:
    import torch

    torch.save({"model": model.state_dict(), "contract": contract,
                "optimiser": optimiser.state_dict() if optimiser else None, "epoch": epoch}, path)


def load_checkpoint(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    c = payload["contract"]
    model = build_model(c["scene_dim"], c["candidate_dim"], c.get("width", 128))
    model.load_state_dict(payload["model"])
    return model, payload


# -- ranking and evaluation ------------------------------------------------------------


def rank_pool(model, data: Dataset, obj: str, reference_inputs: dict) -> list[dict]:
    """Scores for every pooled candidate of ``obj`` in one reference scene."""
    import torch

    pool = [c for c in data.pool if c["object"] == obj]
    encoded = [data.encode(reference_inputs, c) for c in pool]
    with torch.no_grad():
        logits = model(torch.as_tensor(np.stack([e[0] for e in encoded])),
                       torch.as_tensor(np.stack([e[1] for e in encoded])),
                       torch.as_tensor(np.stack([e[2] for e in encoded])))
    score = torch.sigmoid(logits[:, 0]).numpy()
    order = np.argsort(-score, kind="stable")
    return [{"candidate_sha256": pool[i]["candidate_sha256"], "score": float(score[i]),
             "label_observed": pool[i]["label_observed"], "any_accepted": pool[i]["any_accepted"]}
            for i in order]


def coverage(ranking: list[dict], k: int) -> dict:
    top = ranking[:k]
    observed = [r for r in top if r["label_observed"]]
    return {"k": k, "observed_in_top_k": len(observed),
            "accepted_among_observed": sum(bool(r["any_accepted"]) for r in observed),
            "unobserved_in_top_k": k - len(observed),
            "note": "unobserved candidates are neither successes nor failures"}


def baselines(data: Dataset, obj: str, k: int) -> dict:
    pool = [c for c in data.pool if c["object"] == obj]
    rng = np.random.default_rng(0)
    random_order = [pool[i] for i in rng.permutation(len(pool))]
    hash_order = sorted(pool, key=lambda c: c["candidate_sha256"])

    def as_ranking(order):
        return [{"candidate_sha256": c["candidate_sha256"], "label_observed": c["label_observed"],
                 "any_accepted": c["any_accepted"]} for c in order]
    return {"random_seed0": coverage(as_ranking(random_order), k),
            "source_hash_order": coverage(as_ranking(hash_order), k)}


def evaluate_rows(model, data: Dataset, rows: list[dict]) -> dict:
    import torch

    if not rows:
        return {"rows": 0}
    pts, scn, cnd, ys, ms = data.tensors(rows)
    with torch.no_grad():
        logits = model(pts, scn, cnd)
        loss = float(masked_loss(logits, ys, ms))
    prob = torch.sigmoid(logits).numpy()
    out = {"rows": len(rows), "masked_loss": loss, "heads": {}}
    for i, head in enumerate(HEADS):
        mask = ms[:, i].numpy() > 0
        y = ys[:, i].numpy()[mask]
        p = prob[:, i][mask]
        positives, negatives = p[y == 1], p[y == 0]
        auc = None
        if len(positives) and len(negatives):
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in positives for b in negatives]))
        out["heads"][head] = {"observed": int(mask.sum()), "positive": int(y.sum()), "auroc": auc}
    return out


# -- commands ------------------------------------------------------------------------


class NoTraining(RuntimeError):
    pass


def forbid_training():
    """Make any gradient step during a check an error rather than a silent update."""
    import torch

    def refuse(*_, **__):
        raise NoTraining("backward/optimizer.step called during a no-training check")
    originals = (torch.Tensor.backward, torch.optim.Optimizer.step)
    torch.Tensor.backward = refuse
    torch.optim.Optimizer.step = refuse
    return originals


def restore_training(originals) -> None:
    import torch

    torch.Tensor.backward, torch.optim.Optimizer.step = originals


# Anything only known after execution. None of these may appear among the inputs.
OUTCOME_WORDS = ("status", "accepted", "reason", "success", "succeeded", "end_phase", "contact",
                 "force", "height", "rise", "episode", "outcome", "label", "margin", "penetration")
# Input fields whose names contain one of those words but are fixed before execution:
# the teacher template's required contact groups are part of the candidate itself.
PRE_EXECUTION_FIELDS = {"required_contact_groups"}


def input_checks(args, data: Dataset, rows: list[dict], model, tensors, logits) -> dict:
    """Dimensions, missing values, determinism and outcome leakage of the inputs."""
    import copy

    import torch

    pts, scn, cnd = tensors
    out = {"input_dims": {"points": list(pts.shape[1:]), "scene": int(scn.shape[1]),
                          "candidate": int(cnd.shape[1]), "grasp_types": data.types}}
    out["missing_values"] = {
        "object_dims": sum(1 for r in rows if not r["inputs"].get("object_dims_mm")),
        "distractor_placement_file": sum(1 for r in rows if r["inputs"].get("distractors")
                                         and not r["inputs"].get("placement_file")),
        "grasp_type_not_in_vocabulary": sum(
            1 for r in rows
            if data.index[(r["inputs"]["object"], r["inputs"]["candidate_sha256"])]["grasp_type"]
            not in data.types),
        "non_finite_inputs": int(sum(int((~torch.isfinite(t)).sum()) for t in tensors))}
    # Same input, same result: rebuild the data and run the model again.
    again = Dataset(args.ledger, args.out / "point-cache")
    rows_again = [r for r in again.rows if r["row_id"] in {x["row_id"] for x in rows}]
    tensors_again = again.tensors(rows_again)[:3]
    with torch.no_grad():
        logits_again = model(*tensors_again)
    out["deterministic"] = {
        "inputs_bitwise_equal": all(torch.equal(a, b) for a, b in zip(tensors, tensors_again,
                                                                     strict=True)),
        "outputs_bitwise_equal": bool(torch.equal(logits.detach(), logits_again))}
    # Outcome leakage: changing every label must not change a single input value.
    mutated = copy.deepcopy(rows)
    for r in mutated:
        r["labels"] = {k: ({"observed": True, "succeeded": not bool((v or {}).get("succeeded"))}
                           if isinstance(v, dict) else None) for k, v in r["labels"].items()}
        r["source"]["episode_sha256"] = "0" * 64
    tensors_mutated = data.tensors(mutated)[:3]
    forbidden = sorted({k for r in rows for k in r["inputs"]
                        if k not in PRE_EXECUTION_FIELDS
                        and any(w in k.lower() for w in OUTCOME_WORDS)})
    candidate_keys = sorted({k for c in data.pool for k in c["features"]})
    out["leakage"] = {
        "inputs_unchanged_when_labels_change": all(
            torch.equal(a, b) for a, b in zip(tensors, tensors_mutated, strict=True)),
        "outcome_named_input_fields": forbidden,
        "candidate_feature_fields": candidate_keys,
        "candidate_features_pre_execution": set(candidate_keys) == {
            "grasp_qpos", "squeeze_qpos", "open_qpos", "required"}}
    out["input_checks_passed"] = bool(
        out["missing_values"]["non_finite_inputs"] == 0
        and out["deterministic"]["inputs_bitwise_equal"]
        and out["deterministic"]["outputs_bitwise_equal"]
        and out["leakage"]["inputs_unchanged_when_labels_change"]
        and not forbidden and out["leakage"]["candidate_features_pre_execution"])
    return out


def contract_for(data: Dataset, fold: dict, scene_dim: int, candidate_dim: int) -> dict:
    return {"physics": "A", "types": data.types, "scene_dim": scene_dim,
            "candidate_dim": candidate_dim, "width": 128, "heads": list(HEADS),
            "points": POINTS, "point_seed": POINT_SEED, "held_out_group": fold["held_out_group"],
            "train_rows": fold["train_rows"], "dev_rows": fold["dev_rows"],
            "ledger_executions_sha256": sha(data.ledger / "executions.jsonl"),
            "ledger_candidates_sha256": sha(data.ledger / "candidates.jsonl"),
            "normalisation": "train fold only",
            "robot_initial_state": "constant home; not a feature"}


def check(args) -> int:
    import torch

    torch.manual_seed(0)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = Dataset(args.ledger, out / "point-cache")
    folds = data.folds()
    fold = folds[0]
    train = [r for r in data.rows if r["row_id"] in set(fold["train_rows"])]
    dev = [r for r in data.rows if r["row_id"] in set(fold["dev_rows"])]
    started = time.perf_counter()
    pts, scn, cnd, ys, ms = data.tensors(train)
    feature_seconds = time.perf_counter() - started
    model = build_model(scn.shape[1], cnd.shape[1])
    set_normalisation(model, scn, cnd)
    before = parameter_hash(model)
    originals = forbid_training()
    report = {"physics": "A", "rows_used": len(data.rows), "skipped": data.skipped,
              "geometry_groups": data.groups, "types": data.types, "folds": [
                  {k: f[k] for k in ("held_out_group", "availability", "dev_evaluable")}
                  | {"train": len(f["train_rows"]), "dev": len(f["dev_rows"])} for f in folds]}
    try:
        started = time.perf_counter()
        logits = model(pts, scn, cnd)
        report["forward_ms_per_row"] = (time.perf_counter() - started) * 1000 / len(train)
        loss = masked_loss(logits, ys, ms)
        report["loss"] = float(loss)
        report["finite"] = bool(torch.isfinite(logits).all() and torch.isfinite(loss))
        # An unobserved target must not move the loss.
        flipped = ys.clone()
        flipped[ms == 0] = 1 - flipped[ms == 0]
        report["unobserved_labels_ignored"] = bool(
            abs(float(masked_loss(logits, flipped, ms)) - float(loss)) < 1e-6)
        try:
            loss.backward()
            report["backward_blocked"] = False
        except NoTraining:
            report["backward_blocked"] = True
        rankings = {}
        for obj in sorted({c["object"] for c in data.pool}):
            reference = next((r["inputs"] for r in data.rows if r["inputs"]["object"] == obj
                              and r["inputs"]["placement"] == "centre"
                              and not r["inputs"]["distractors"]), None)
            if reference is None:
                continue
            ranking = rank_pool(model, data, obj, reference)
            rankings[obj] = {"pool": len(ranking), "untrained_model": coverage(ranking, 8),
                             **baselines(data, obj, 8)}
        report["ranking_path"] = rankings
        contract = contract_for(data, fold, scn.shape[1], cnd.shape[1])
        path = out / "untrained-check.pt"
        save_checkpoint(path, model, contract)
        loaded, payload = load_checkpoint(path)
        report["checkpoint_roundtrip_equal"] = parameter_hash(loaded) == before
        report["evaluate_path"] = evaluate_rows(loaded, data, dev)
        report.update(input_checks(args, data, train, model, (pts, scn, cnd), logits))
        (out / "rankings-untrained.json").write_text(json.dumps(
            {"note": "untrained random-initialised model: a path check, not a selector result",
             "rankings": rankings}, indent=1) + "\n")
    finally:
        restore_training(originals)
    report["parameters_unchanged"] = parameter_hash(model) == before
    report["optimizer_steps"] = 0
    report["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["feature_seconds_train_fold"] = feature_seconds
    report["inputs"] = {"executions_sha256": sha(args.ledger / "executions.jsonl"),
                        "candidates_sha256": sha(args.ledger / "candidates.jsonl"),
                        "model_parameter_sha256": before}
    report["passed"] = bool(report["finite"] and report["unobserved_labels_ignored"]
                            and report["backward_blocked"] and report["parameters_unchanged"]
                            and report["checkpoint_roundtrip_equal"]
                            and report.get("input_checks_passed"))
    report["note"] = ("untrained weights: rankings and dev numbers test the path only and are not "
                      "a selector result")
    (out / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "rows_used", "skipped", "loss", "finite",
                                             "unobserved_labels_ignored", "backward_blocked",
                                             "parameters_unchanged", "checkpoint_roundtrip_equal",
                                             "model_parameters", "forward_ms_per_row")}, indent=1))
    return 0 if report["passed"] else 1


def train(args) -> int:
    """Train one held-out-group fold. Not run in this preparation."""
    import torch

    if not args.confirm_training:
        raise SystemExit("training is not authorised here; pass --confirm-training explicitly")
    torch.manual_seed(args.seed)
    data = Dataset(args.ledger, args.out / "point-cache")
    fold = next(f for f in data.folds() if f["held_out_group"] == args.held_out)
    train_rows = [r for r in data.rows if r["row_id"] in set(fold["train_rows"])]
    dev_rows = [r for r in data.rows if r["row_id"] in set(fold["dev_rows"])]
    pts, scn, cnd, ys, ms = data.tensors(train_rows)
    checkpoint = args.out / f"selector-{args.held_out}.pt"
    if args.resume and checkpoint.is_file():
        model, payload = load_checkpoint(checkpoint)
        start = payload["epoch"]
    else:
        model = build_model(scn.shape[1], cnd.shape[1])
        set_normalisation(model, scn, cnd)
        payload, start = None, 0
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    if payload and payload.get("optimiser"):
        optimiser.load_state_dict(payload["optimiser"])
    contract = contract_for(data, fold, scn.shape[1], cnd.shape[1])
    for epoch in range(start, args.epochs):
        optimiser.zero_grad()
        loss = masked_loss(model(pts, scn, cnd), ys, ms)
        loss.backward()
        optimiser.step()
        save_checkpoint(checkpoint, model, contract, optimiser, epoch + 1)
    (args.out / f"evaluate-{args.held_out}.json").write_text(
        json.dumps(evaluate_rows(model, data, dev_rows), indent=1) + "\n")
    return 0


TRAINING = {"width": 128, "optimiser": "Adam", "lr": 1e-3, "weight_decay": 0.0,
            "batch": "full", "updates": 400, "seed": 0,
            "loss": "masked BCE over full/approach/close/carry heads (unobserved masked out)",
            "checkpoint": "the final update; no early stopping and no checkpoint selection",
            "score": "sigmoid of the full-execution head"}


def export_bundle(args) -> int:
    """Training tensors per held-out geometry group, portable without the meshes."""
    import torch

    exclude = set(json.loads(args.exclude.read_text())) if args.exclude else set()
    data = Dataset(args.ledger, args.out / "point-cache", centre_single_only=True,
                   exclude=exclude)
    if any(r["inputs"]["candidate_sha256"] in exclude for r in data.rows):
        raise SystemExit("an excluded (evaluation) candidate reached the training rows")
    written = {}
    for fold in data.folds():
        train_rows = [r for r in data.rows if r["row_id"] in set(fold["train_rows"])]
        pts, scn, cnd, ys, ms = data.tensors(train_rows)
        contract = contract_for(data, fold, scn.shape[1], cnd.shape[1])
        contract.update(training=TRAINING, rows="A, centre single object, independent labelled",
                        excluded_candidates=len(exclude), skipped=data.skipped)
        name = f"bundle-{fold['held_out_group'].split('+')[0]}.pt"
        torch.save({"tensors": (pts, scn, cnd, ys, ms), "contract": contract,
                    "availability": fold["availability"]}, args.out / name)
        written[fold["held_out_group"]] = {"file": name, "train_rows": len(train_rows),
                                          "sha256": sha(args.out / name)}
    (args.out / "bundles.json").write_text(json.dumps(
        {"training": TRAINING, "bundles": written, "skipped": data.skipped,
         "types": data.types}, indent=1) + "\n")
    print(json.dumps(written, indent=1))
    return 0


def train_bundle(args) -> int:
    """Train one exported fold with the fixed settings; the final update is kept."""
    import torch

    if not args.confirm_training:
        raise SystemExit("training is not authorised here; pass --confirm-training explicitly")
    payload = torch.load(args.bundle, map_location="cpu", weights_only=False)
    pts, scn, cnd, ys, ms = payload["tensors"]
    contract = payload["contract"]
    torch.manual_seed(TRAINING["seed"])
    model = build_model(scn.shape[1], cnd.shape[1], TRAINING["width"])
    set_normalisation(model, scn, cnd)
    optimiser = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"],
                                 weight_decay=TRAINING["weight_decay"])
    history = []
    for update in range(TRAINING["updates"]):
        optimiser.zero_grad()
        loss = masked_loss(model(pts, scn, cnd), ys, ms)
        loss.backward()
        optimiser.step()
        if update % 50 == 0 or update == TRAINING["updates"] - 1:
            history.append({"update": update + 1, "train_loss": float(loss)})
    out = args.out / f"selector-{Path(args.bundle).stem.removeprefix('bundle-')}.pt"
    save_checkpoint(out, model, contract, optimiser, TRAINING["updates"])
    (out.with_suffix(".json")).write_text(json.dumps(
        {"bundle": str(args.bundle), "bundle_sha256": sha(args.bundle), "training": TRAINING,
         "history": history, "checkpoint_sha256": sha(out)}, indent=1) + "\n")
    print(json.dumps(history[-1]))
    return 0


def evaluate(args) -> int:
    model, payload = load_checkpoint(args.checkpoint)
    data = Dataset(args.ledger, args.out / "point-cache")
    rows = [r for r in data.rows if r["row_id"] in set(payload["contract"]["dev_rows"])]
    result = evaluate_rows(model, data, rows)
    (args.out / "evaluate.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-bundle")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude", type=Path, help="JSON list of candidate hashes kept out")
    p = sub.add_parser("train-bundle")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--confirm-training", action="store_true")
    for name in ("check", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--ledger", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "train":
            p.add_argument("--held-out", required=True)
            p.add_argument("--epochs", type=int, default=200)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--confirm-training", action="store_true")
        if name == "evaluate":
            p.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"check": check, "train": train, "evaluate": evaluate,
            "export-bundle": export_bundle, "train-bundle": train_bundle}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
