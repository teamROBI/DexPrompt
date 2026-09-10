"""Audit the frozen sample without treating bypassed filters as acceptance."""
from collections import Counter

from screen import ASSETS, OUT, ROOT, dump, json, mujoco, no_renderer, np, objects, sha
from screen import checked_candidate, read_candidates, serialize
from icra.grasp_env import _clear_of_table, _open_hand_fits
from icra.probe import Probe
from collect_episodes import in_object_frame


def measure(probe, pose, fingers):
    count = probe.contacts(pose, fingers)
    depths = {"table": [], "object": []}
    bodies = {"table": set(), "object": set()}
    for contact in probe.data.contact[:probe.data.ncon]:
        if contact.dist > 0:
            continue
        pair = {int(probe.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        hands = pair & probe.hand_bodies
        for label, body in (("table", probe.table_body), ("object", probe.object_body)):
            if body in pair and hands:
                depths[label].append(float(-contact.dist * 1000))
                bodies[label].update(mujoco.mj_id2name(probe.model, mujoco.mjtObj.mjOBJ_BODY, b)
                                     for b in hands)
    return {"object_contacts": count, "on_table": probe.on_table,
            "max_depth_mm": {k: max(v, default=0.0) for k, v in depths.items()},
            "bodies": {k: sorted(v) for k, v in bodies.items()}}


def inspect(row):
    metadata = "passed"
    try:
        checked_candidate(row)
    except (ValueError, KeyError, OSError) as error:
        metadata = str(error)
    raw = np.load(row["source"], allow_pickle=True).item()
    probe = Probe(ROOT / row["directory"])
    pose = in_object_frame(raw)
    squeeze = measure(probe, pose, np.asarray(row["squeeze_qpos"])[7:])
    opened = measure(probe, pose, np.asarray(row["open_qpos"]))
    # Independent coordinate identity: the probe plane maps to world z=0,
    # and its normal maps to world +z under the recorded stable object pose.
    asset = objects.load(ROOT / row["directory"])
    pos, quat = objects.resting(asset, 0)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    rotation = rotation.reshape(3, 3)
    table = probe.table_body
    plane_world = rotation @ probe.data.xpos[table] + pos
    normal_world = rotation @ probe.data.xmat[table].reshape(3, 3)[:, 2]
    assert abs(plane_world[2]) < 1e-8
    assert np.allclose(normal_world, [0, 0, 1], atol=1e-8)
    return {"source": row["source"], "object": row["object_name"],
            "metadata": metadata, "squeeze": squeeze, "open": opened,
            "static_pass": _clear_of_table(probe, raw) and
                           _open_hand_fits(probe, raw, np.asarray(row["open_qpos"])),
            "table_frame_check": "passed"}


def main():
    mujoco.Renderer = no_renderer
    plan = json.loads((OUT / "plan.json").read_text())
    baseline = json.loads((ROOT / "out/rl-approach/candidate.json").read_text())["candidates"][0]
    control = inspect(baseline)
    assert control["metadata"] == "passed" and control["static_pass"], control
    records = []
    for obj in plan["objects"]:
        # Deliberately bypass static filters ONLY to inspect rejected records.
        candidates = read_candidates(OUT / "subsets" / obj["object"], ASSETS,
                                     require_clear=False, require_approachable=False)
        for candidate in candidates:
            row = json.loads(json.dumps(serialize(candidate)))
            row["source_sha256"] = sha(row["source"])
            records.append(inspect(row))
        print(json.dumps({"object": obj["object"], "inspected": len(candidates)}), flush=True)
    assert len(records) == 100
    summary = []
    for obj in plan["objects"]:
        rows = [r for r in records if r["object"] == obj["object"]]
        depths = [r["squeeze"]["max_depth_mm"]["table"] for r in rows
                  if r["squeeze"]["on_table"]]
        summary.append({"object": obj["object"], "metadata": dict(Counter(r["metadata"] for r in rows)),
                        "squeeze_table_count": len(depths),
                        "squeeze_table_depth_mm_min_median_max":
                        [min(depths), float(np.median(depths)), max(depths)] if depths else [],
                        "open_object_count": sum(r["open"]["object_contacts"] > 0 for r in rows),
                        "open_table_count": sum(r["open"]["on_table"] for r in rows)})
    assert all(sha(r["source"]) == r["source_sha256"] for r in plan["input_candidates"])
    dump(OUT / "diagnostics.json", {"positive_control": control, "objects": summary,
                                   "records": records, "input_hashes_unchanged": True,
                                   "scope": "Static diagnostic; no IK, rollout, or learning"})
    print(json.dumps({"positive_control": control, "objects": summary}), flush=True)


if __name__ == "__main__":
    main()
