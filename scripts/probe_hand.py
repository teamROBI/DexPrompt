"""Command the hand to poses and measure whether it gets there.

"Do the fingers move?" has two answers and they come apart. A joint can be free to move
and still not reach its target, because the finger actuators are limited to ±2 Nm at
kp=20 against joints with damping 1.0. So this reports, per joint, the commanded target,
where it settled, and the residual -- and reads the residual against the actuator's own
saturation rather than calling anything "fine".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icra.hand import Hand


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="r", choices=["r", "l"])
    parser.add_argument("--scene", default="model/scene_ffw_sh5.xml")
    parser.add_argument("--settle-steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("out"))
    return parser.parse_args(argv)


def flexion_targets(hand: Hand, fraction: float) -> dict[str, float]:
    """Close every flexion joint to ``fraction`` of its range, leaving spread at rest.

    A joint whose range straddles zero is a spread/abduction axis (or the thumb's
    rotation); those are held at 0 so the measurement is about curling, not splaying.
    """
    targets = {}
    for joint in hand.joint_names:
        low, high = hand.joint_range(joint)
        if low < 0 < high:
            targets[joint] = 0.0
        else:
            targets[joint] = low + fraction * (high - low)
    return targets


def measure(hand: Hand, targets: dict[str, float], steps: int) -> list[dict[str, object]]:
    hand.command(targets)
    hand.settle(steps)
    rows = []
    for joint, target in targets.items():
        actual = hand.read(joint)
        force = float(hand.data.actuator_force[hand.actuator_id(joint)])
        rows.append(
            {
                "joint": joint,
                "target": round(target, 4),
                "actual": round(actual, 4),
                "residual": round(actual - target, 4),
                "force": round(force, 3),
                # The finger actuators are clamped to +/-2.0 Nm. A joint sitting at the
                # clamp has run out of authority, which is a different failure from a
                # joint that is stuck.
                "saturated": abs(force) >= 1.99,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {}
    for label, fraction in (("open", 0.0), ("half", 0.5), ("closed", 1.0)):
        hand = Hand.create(args.side, args.scene)
        targets = flexion_targets(hand, fraction)
        rows = measure(hand, targets, args.settle_steps)
        residuals = np.array([abs(r["residual"]) for r in rows])
        report[label] = {
            "max_residual": round(float(residuals.max()), 4),
            "mean_residual": round(float(residuals.mean()), 4),
            "saturated_joints": [r["joint"] for r in rows if r["saturated"]],
            "joints": rows,
        }
        print(
            f"{label:<7} mean|residual|={residuals.mean():.4f} rad  "
            f"max={residuals.max():.4f} rad  "
            f"saturated={sum(1 for r in rows if r['saturated'])}/{len(rows)}"
        )

    # Per-finger, individually: does one finger curl without the others?
    resting = Hand.create(args.side, args.scene)
    resting.settle(args.settle_steps)
    baseline = {j: resting.read(j) for j in resting.joint_names}
    per_finger = {}
    for finger, joints in hand.fingers.items():
        hand = Hand.create(args.side, args.scene)
        targets = {j: 0.0 for j in hand.joint_names}
        for joint in joints:
            low, high = hand.joint_range(joint)
            targets[joint] = high if low >= 0 else 0.0
        rows = measure(hand, targets, args.settle_steps)
        # Against the *uncommanded baseline*, not against zero. Comparing to zero counted
        # every joint that simply does not rest at zero as cross-talk, and reported 3-4
        # phantom leaks per finger.
        moved = {
            r["joint"]: r["actual"]
            for r in rows
            if abs(r["actual"] - baseline[r["joint"]]) > 0.05
        }
        leaked = sorted(j for j in moved if j not in joints)
        per_finger[finger] = {
            "commanded": joints,
            "moved": sorted(moved),
            "leaked_to_other_fingers": leaked,
        }
        print(f"  {finger:<7} moved {len(moved)} joints, leak={len(leaked)}")
    report["per_finger"] = per_finger

    (args.out / "hand_probe.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out / 'hand_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
