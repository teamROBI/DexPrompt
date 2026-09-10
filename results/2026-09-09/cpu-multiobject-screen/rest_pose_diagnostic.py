"""Post-hoc sensitivity check on the SAME 100 files; never a ready-data manifest."""
from screen import ASSETS, OUT, dump, json, mujoco, no_renderer, np, objects
from screen import read_candidates
from icra.probe import Probe
from collect_episodes import in_object_frame


def main():
    mujoco.Renderer = no_renderer
    plan = json.loads((OUT / "plan.json").read_text())
    records = []
    summaries = []
    for entry in plan["objects"]:
        name = entry["object"]
        directory = ASSETS / name
        asset = objects.load(directory)
        probe = Probe(directory)
        candidates = read_candidates(OUT / "subsets" / name, ASSETS,
                                     require_clear=False, require_approachable=False)
        by_pose = []
        for index in range(len(asset.rest)):
            pos, quat = objects.resting(asset, index)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
            rotation = rotation.reshape(3, 3)
            probe.model.body_pos[probe.table_body] = rotation.T @ np.array([0, 0, -pos[2]])
            normal = rotation.T @ np.array([0., 0., 1.])
            plane_quat = np.zeros(4)
            mujoco.mju_quatZ2Vec(plane_quat, normal)
            probe.model.body_quat[probe.table_body] = plane_quat
            kept = []
            for candidate in candidates:
                raw = np.load(candidate.source, allow_pickle=True).item()
                pose = in_object_frame(raw)
                probe.contacts(pose, candidate.squeeze_qpos[7:])
                squeezed_clear = not probe.on_table
                contacts = probe.contacts(pose, candidate.open_qpos)
                opened_clear = contacts == 0 and not probe.on_table
                if squeezed_clear and opened_clear:
                    kept.append(candidate.source)
            records.append({"object": name, "rest_index": index,
                            "position": pos.tolist(), "quaternion": quat.tolist(),
                            "passed_sources": kept})
            by_pose.append(len(kept))
        assert by_pose[0] == 0
        distinct = {s for r in records if r["object"] == name for s in r["passed_sources"]}
        summaries.append({"object": name, "input": len(candidates),
                          "static_pass_by_rest_pose": by_pose,
                          "distinct_candidates_passing_any_pose": len(distinct)})
    dump(OUT / "rest-pose-diagnostic.json", {
        "scope": "Post-hoc alternative stable-pose diagnostic. Original pose-0 result is unchanged. "
                 "No arm IK, trajectory, stability rollout, or learning. Not training-ready.",
        "objects": summaries, "records": records})
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
