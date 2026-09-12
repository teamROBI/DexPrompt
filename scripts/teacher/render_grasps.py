"""Render successful grasps as video, so a number in a table can be looked at.

``generate_grasps.py`` reports how many grasps survived evaluation. That count is
only as trustworthy as the pipeline behind it, and this project has twice had a
pipeline report contact where the hand was 45 mm from the object. Watching the
hand close is the cheap check that the number means what it says.

Each clip replays exactly what earned the success: the evaluation trajectory from
the pregrasp pose through the grasp to the squeeze, and then the shake -- the
object pulled in all six axis directions in turn, which is what ``eval_fc`` does
to decide whether a grasp holds. Contact points and contact forces are drawn, so
a grasp holding by friction on two pads looks different from one caged by five.

Rendering is offscreen on the GPU. GLFW needs a display, and MuJoCo's EGL context
raises when it is *freed* on this machine -- at teardown, after every frame has
rendered correctly -- so the backend is set here and the teardown is swallowed by
``icra.scene.renderer``.

Run with Dexonomy's interpreter::

    third_party/dexonomy/.venv/bin/python scripts/teacher/render_grasps.py
    third_party/dexonomy/.venv/bin/python scripts/teacher/render_grasps.py --type palmar_pinch --count 4
"""

from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import os
import subprocess
import sys
from pathlib import Path

# Set before mujoco is imported anywhere: the renderer picks its backend at
# import time, and this machine has neither a working EGL nor a display.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from icra import paths  # noqa: E402

#: Directions the object is pulled in, matching ``MuJoCo_EvalEnv.eval_fc``.
SHAKE = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


#: Direction the camera looks from, in world coordinates: front-right and
#: slightly above. How *far* along it is computed per clip from the size of what
#: is being shown.
CAMERA_DIRECTION = np.array([0.72, -0.62, 0.32])

#: Multiple of the subject's radius to stand back by. Enough to keep the hand in
#: frame as it closes, without leaving the object a few pixels across.
CAMERA_DISTANCE = 1.45

#: Grey-blue for the hand, warm for the object, so which is which is obvious in a
#: still frame. The hand colour is the one this project's Dexonomy hand config
#: already declares.
HAND_RGBA = (0.55, 0.60, 0.70, 1.0)
OBJECT_RGBA = (0.85, 0.55, 0.25, 1.0)


def dress(env, hand_prefix: str = "hand-") -> None:
    """Frame, light and colour the scene before anything is rendered.

    Dexonomy's debug renderer exists to show contacts during development, not to
    produce something to look at: it puts one light at ``[0, -1, 2]`` and a fixed
    camera 1.5 m from the origin, so a hand-sized subject lands a few dozen pixels
    across and unlit geometry comes out black. All three are properties of the
    compiled model, so they can be set afterwards without touching Dexonomy.

    It also turns on the geom group the object lives in; see below.

    The camera is framed on the hand and object together, at whatever distance
    fits them, rather than on a fixed point at a fixed range. Call it with the
    hand already at the grasp pose, so that is what gets framed.
    """
    import mujoco

    model = env._model
    name = lambda geom: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])

    # MuJoCo renders geom groups 0-2 and hides 3-5. Objects here carry only
    # collision geometry, written into group 3 because that is what Dexonomy's
    # own object XMLs use, so with the default option the object is simply absent
    # from the picture -- a hand closing on nothing, with the contact markers the
    # only hint that anything is there.
    env._debug_option.geomgroup[:] = 1

    # A headlight, which needs no position and cannot leave the subject in shadow.
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]

    for geom in range(model.ngeom):
        body = name(geom) or ""
        if body == "mocap_body":
            continue
        model.geom_rgba[geom] = HAND_RGBA if body.startswith(hand_prefix) else OBJECT_RGBA

    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "closeup")
    if camera < 0:
        return

    # Frame on the geometry that will be on screen. Body origins are not enough:
    # a link's origin is at its joint, so the hand's origins span a good deal less
    # than the hand does, and framing on them crops the fingers off. Each geom's
    # own bounding sphere gives the real extent.
    low, high = [], []
    for geom in range(model.ngeom):
        if (name(geom) or "") == "mocap_body" or not model.geom_group[geom] in (0, 1, 2, 3):
            continue
        centre, bound = env._data.geom_xpos[geom], model.geom_rbound[geom]
        low.append(centre - bound)
        high.append(centre + bound)
    low, high = np.asarray(low).min(axis=0), np.asarray(high).max(axis=0)
    target = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low) / 2), 0.05)

    direction = CAMERA_DIRECTION / np.linalg.norm(CAMERA_DIRECTION)
    eye = target + direction * radius * CAMERA_DISTANCE

    # A MuJoCo camera looks down its own -Z with +Y up, so build that frame from
    # the view direction and world up, then hand it over as a quaternion.
    back = direction
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.stack([right, up, back], axis=1).flatten())
    model.cam_pos[camera] = eye
    model.cam_quat[camera] = quaternion


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="icra")
    parser.add_argument(
        "--type", dest="types", action="append",
        help="grasp type to render; repeatable, default a spread over all of them",
    )
    parser.add_argument("--object", dest="objects", action="append")
    parser.add_argument("--count", type=int, default=6, help="clips to render")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=paths.OUT / "grasp_clips")
    parser.add_argument("--dexonomy", type=Path, default=paths.DEXONOMY)
    return parser.parse_args(argv)


def choose(root: Path, types, objects, count: int) -> list[Path]:
    """Successful grasps, spread across types rather than taken in order.

    Taking the first N would render six clips of whichever type sorts first,
    which says nothing about the rest.
    """
    by_type: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.npy")):
        relative = path.relative_to(root).parts
        if len(relative) < 2:
            continue
        template, obj = relative[0], relative[1]
        if types and template not in types:
            continue
        if objects and obj not in objects:
            continue
        by_type.setdefault(template, []).append(path)

    chosen, round_number = [], 0
    while len(chosen) < count and by_type:
        for template in sorted(by_type):
            if round_number < len(by_type[template]):
                chosen.append(by_type[template][round_number])
                if len(chosen) == count:
                    return chosen
        round_number += 1
        if all(round_number >= len(v) for v in by_type.values()):
            break
    return chosen


def encode(frames: list, destination: Path, fps: int) -> Path:
    """Write an mp4 through ffmpeg, falling back to a GIF if that is not there."""
    import imageio.v2 as imageio

    if not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode:
        height, width = frames[0].shape[:2]
        # Even dimensions, which yuv420p requires and 480x640 already satisfies.
        process = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
             str(destination)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        if process.wait() == 0:
            return destination
    fallback = destination.with_suffix(".gif")
    imageio.mimsave(fallback, frames, duration=1.0 / fps)
    return fallback


def render(
    path: Path, destination: Path, fps: int, dexonomy: Path
) -> tuple[Path, int, float] | None:
    """One clip. Returns its path, frame count, and how far the object moved."""
    import mujoco
    from dexonomy.op.eval_data import get_planner
    from dexonomy.sim.basic import HandCfg
    from dexonomy.sim.mujoco_env import MuJoCo_EvalCfg, MuJoCo_EvalEnv
    from dexonomy.util.file_util import load_json, load_scene_cfg
    from dexonomy.util.np_util import np_get_delta_pose

    grasp = np.load(path, allow_pickle=True).item()
    # Saved relative to the Dexonomy checkout, which is not where this runs from.
    scene = load_scene_cfg(str(dexonomy / str(grasp["scene_path"])))
    for entry in scene["scene"].values():
        if entry["type"] != "rigid_object":
            continue
        # The same density rewrite op=eval does, so the object weighs what the
        # run assumed rather than whatever its mesh volume implies.
        info = load_json(entry["info_path"])
        volume = info["mass"] / (info["density"] * info["scale"] ** 3)
        entry["density"] = 0.1 / (volume * np.prod(entry["scale"]))

    env = MuJoCo_EvalEnv(
        hand_cfg=HandCfg(str(paths.DEXONOMY_HAND / "right.xml"), freejoint=True),
        scene_cfg=scene,
        sim_cfg=MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]),
        debug_render=True,
        debug_view=False,
    )
    planner = get_planner(scene["task"])
    control, control_type, external, target = planner.plan_trajectory(
        np.copy(env.get_active_obj_pose()),
        grasp["pregrasp_qpos"], grasp["grasp_qpos"], grasp["squeeze_qpos"],
        grasp.get("robot_pose"),
    )

    # Framed on the grasp itself, which is the moment the clip is about; the
    # frames that framing produced are then thrown away.
    env.reset_qpos(np.asarray(grasp["grasp_qpos"])[0])
    mujoco.mj_forward(env._model, env._data)
    dress(env)
    env._debug_img.clear()

    worst = 0.0
    for direction in SHAKE:
        env.reset_qpos(control[0])
        env.run_trajectory(control, control_type, external)
        env.set_active_obj_ext_fdir(np.array(direction, dtype=float))
        for _ in range(10):
            env.step_sim(substep=50)
            moved, _ = np_get_delta_pose(target, env.get_active_obj_pose())
            worst = max(worst, moved)

    frames = env._debug_img
    if not frames:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    return encode(frames, destination, fps), len(frames), worst


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dexonomy / "output" / f"{args.exp_name}_hx5_d20" / "succ_grasp"
    problems = paths.check(root)
    if problems:
        print("\n".join(problems))
        print("\nrun scripts/teacher/generate_grasps.py first")
        return 1

    chosen = choose(root, args.types, args.objects, args.count)
    if not chosen:
        print("no successful grasps match that selection")
        return 1

    # debug_render writes a debug.xml into the working directory; keep that with
    # the clips rather than in the repo root or the Dexonomy checkout.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(args.out)

    print(f"rendering {len(chosen)} clip(s) into {args.out}\n")
    for path in chosen:
        template, obj = path.relative_to(root).parts[:2]
        name = f"{template}__{obj}__{path.stem}.mp4"
        result = render(path, args.out / name, args.fps, args.dexonomy.resolve())
        if result is None:
            print(f"  {template}/{obj}: renderer produced no frames")
            continue
        written, count, worst = result
        print(f"  {written.name}   {count} frames, object moved at most "
              f"{worst * 1000:.1f} mm under shaking")
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
