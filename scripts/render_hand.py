"""Render the hand at a few poses, so "do the fingers move" can be looked at.

One panel per pose, from a camera aimed at the right hand. Contacts can be disabled to
separate what the kinematics can represent from what the collision geometry allows --
which is the distinction that matters for this model, since with contacts off every
joint reaches its target to 0.0002 rad and with them on ten of twenty saturate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from icra.hand import Hand


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--side", default="r")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--settle-steps", type=int, default=4000)
    parser.add_argument("--no-contacts", action="store_true")
    return parser.parse_args(argv)


def curl_targets(hand: Hand, fraction: float) -> dict[str, float]:
    """Curl the flexion joints, hold the spread axes at zero."""
    targets = {}
    for joint in hand.joint_names:
        low, high = hand.joint_range(joint)
        targets[joint] = 0.0 if low < 0 < high else low + fraction * (high - low)
    return targets


def render(hand: Hand, width: int, height: int) -> np.ndarray:
    """A view of the hand, framed from the wrist body so it fills the image."""
    renderer = mujoco.Renderer(hand.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Framed on the fingertips' own centroid, not the wrist: the wrist sits 12 cm above
    # the fingers, and aiming there put most of the frame on the torso.
    tips = [
        mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_BODY, f"finger_{hand.side}_link{i}")
        for i in range(1, 21)
    ]
    points = np.array([hand.data.xpos[i] for i in tips if i >= 0])
    camera.lookat[:] = points.mean(axis=0)
    # Distance from the hand's own extent, so the framing survives a different pose.
    camera.distance = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 1.6
    # Viewed from the robot's own right (-y), where nothing is in the way. Aiming from
    # the front put the lift column between the camera and the hand.
    camera.azimuth = -90.0
    camera.elevation = -12.0

    # The model is uniformly dark and ships a single light, so a rendered hand is nearly
    # unreadable. Brighten the scene rather than the material: the geometry is the subject.
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    renderer.update_scene(hand.data, camera=camera, scene_option=scene_option)
    for light in range(renderer.scene.nlight):
        renderer.scene.lights[light].diffuse[:] = 0.9
        renderer.scene.lights[light].ambient[:] = 0.5
    image = renderer.render()
    renderer.close()
    return image


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    panels = []
    for label, fraction in (("open", 0.0), ("half", 0.5), ("fist", 1.0)):
        hand = Hand.create(args.side)
        if args.no_contacts:
            hand.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        hand.command(curl_targets(hand, fraction))
        hand.settle(args.settle_steps)
        image = render(hand, args.width, args.height)
        suffix = "_nocontact" if args.no_contacts else ""
        Image.fromarray(image).save(args.out / f"hand_{label}{suffix}.png")
        panels.append(image)
        residual = np.mean(
            [
                abs(hand.read(j) - curl_targets(hand, fraction)[j])
                for j in hand.joint_names
            ]
        )
        print(f"  {label:<5} mean|residual|={residual:.4f} rad  ncon={hand.data.ncon}")

    strip = np.concatenate(panels, axis=1)
    name = f"hand_strip{'_nocontact' if args.no_contacts else ''}.png"
    Image.fromarray(strip).save(args.out / name)
    print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
