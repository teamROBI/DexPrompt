"""Which refined grasps could be executed on a table at all.

DexGraspNet 2.0 generates per object, floating, and then keeps only what is
collision-free once the object is placed in a scene. Our refinement is floating
for the same reason -- with a table in the scene config op=init rejects
everything -- so the same filter is what bridges the two, and without it 96 per
cent of what is generated puts the hand through the table.

This is the cheap half: the hand alone against the object and the table it
stands on, one forward pass per candidate, no arm and no inverse kinematics. It
sweeps the same rolls and depths collect_episodes would, so a grasp is kept if
*any* of them clears the table and still closes on the object.
"""
import sys, glob, shutil, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import collect_episodes as ce

src = Path(sys.argv[1])
objects = Path(sys.argv[2])
out = Path(sys.argv[3])
shutil.rmtree(out, ignore_errors=True)

probes: dict[str, ce.Probe] = {}
tally = collections.Counter()
paths = sorted(glob.glob(str(src / "**/*.npy"), recursive=True))
print(f"{len(paths)} grasps to check\n")
for n, path in enumerate(paths):
    relative = Path(path).relative_to(src)
    name = relative.parts[1]
    if not (objects / name).exists():
        tally["object not in the scene"] += 1
        continue
    probe = probes.get(name)
    if probe is None:
        probe = probes[name] = ce.Probe(objects / name)
    grasp = np.load(path, allow_pickle=True).item()
    # The squeeze shape, which is what collect_episodes scores at and what the
    # arm actually closes to. Dexonomy refines to a 1 mm contact distance and
    # stores the tighter close separately, so counting contacts at grasp_qpos
    # reads a good grasp as closing on nothing: eight of eight sampled were 0
    # contacts there and 5 to 66 at squeeze_qpos.
    fingers = np.asarray(np.asarray(grasp["squeeze_qpos"])[0][7:], dtype=float)
    best = 0
    for roll in np.linspace(0.0, 2 * np.pi, ce.ROLL_STEPS, endpoint=False):
        for depth in np.linspace(-ce.DEPTH_SPAN, ce.DEPTH_SPAN, ce.DEPTH_STEPS):
            about = ce.in_object_frame(grasp, float(roll), float(depth), None)
            touching = probe.contacts(about, fingers)
            if probe.on_table:
                continue
            best = max(best, touching)
    if best >= ce.MIN_CONTACTS:
        tally["kept"] += 1
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, grasp)
    else:
        tally["blocked by the table or closes on nothing"] += 1
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(paths)}  kept {tally['kept']}", flush=True)

print()
for key, count in tally.most_common():
    print(f"{count:6d}  {key}")
print(f"\nkept {tally['kept']} of {len(paths)} "
      f"({100*tally['kept']/max(len(paths),1):.1f}%) -> {out}")
